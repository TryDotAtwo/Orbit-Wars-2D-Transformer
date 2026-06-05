use std::env;
use std::fs;
use std::path::PathBuf;
use std::thread;
use std::time::Instant;

mod v8_cuda;

use orbit_wars_core::{
    decode_action_slots, is_terminal, ActionSlotOutput, AgentConfig, Fleet, MoveCommand, Planet,
    SimulationState, V8Model,
};

const DEFAULT_GAMES: usize = 16;
const DEFAULT_PLAYERS: usize = 2;
const DEFAULT_OUTPUT: &str = "dashboard/public/telemetry/latest.json";
const PLAYER_COUNT_FOUR: usize = 4;
const PLAYER_IDS: [i32; PLAYER_COUNT_FOUR] = [0, 1, 2, 3];
const GPU_SIM_PLANETS: usize = 64;
const MAP_SEED_BASE: u64 = 0x4f_57_4d_41_50;
const MAP_GENERATION_SEED_FACTOR: u64 = 1_000_003;
const MAP_GAME_SEED_FACTOR: u64 = 9_176;
const MAP_RNG_MULTIPLIER: u64 = 6_364_136_223_846_793_005;
const MAP_RNG_INCREMENT: u64 = 1_442_695_040_888_963_407;
const MAP_RNG_FLOAT_SCALE: f32 = 16_777_216.0;
const MAP_GENERATION_ATTEMPT_LIMIT: usize = 5_000;
const MAP_MIN_PLANET_GROUPS: usize = 5;
const MAP_MAX_PLANET_GROUPS: usize = 10;
const MAP_MIN_STATIC_GROUPS: usize = 3;
const MAP_PLANET_CLEARANCE: f32 = 7.0;
const MAP_HOME_PLANET_SHIPS: f32 = 10.0;
const MAP_STATIC_AXIS_CLEARANCE: f32 = 5.0;
const MAP_ORBITING_MIN_COORDINATE_OFFSET: f32 = 15.0;
const MAP_ORBITING_EDGE_CLEARANCE: f32 = 5.0;
const MAP_SUN_SPAWN_CLEARANCE: f32 = 10.0;
const MAP_STATIC_SHIP_MIN: usize = 5;
const MAP_STATIC_SHIP_MAX: usize = 99;
const MAP_ORBITING_SHIP_MIN: usize = 5;
const MAP_ORBITING_SHIP_MAX: usize = 30;
const MAP_PRODUCTION_MIN: usize = 1;
const MAP_PRODUCTION_MAX: usize = 5;
const MAP_GROUP_SIZE: usize = 4;

#[derive(Clone, Debug)]
struct Cli {
    games: usize,
    games_per_model: usize,
    players: usize,
    replay_stride: usize,
    workers: usize,
    cpu_workers: usize,
    cuda_v8_smoke: bool,
    cuda_v8_forward_smoke: bool,
    cuda_sim_parity_smoke: bool,
    cuda_v8: bool,
    gpu_sim: bool,
    output: PathBuf,
    model: Option<PathBuf>,
    model_list: Option<PathBuf>,
}

#[derive(Clone, Debug)]
struct ReplayFrame {
    step: usize,
    planets: Vec<Planet>,
    fleets: Vec<Fleet>,
    actions_by_player: Vec<Vec<MoveCommand>>,
    model_ids_by_player: Vec<usize>,
}

#[derive(Clone, Debug, Default)]
struct PlayerStats {
    wins: usize,
    draws: usize,
    losses: usize,
    launch_actions: usize,
    launched_ships: i64,
    captures: usize,
    fleet_hits: usize,
    hit_ships: i64,
    sun_destroyed_fleets: usize,
    sun_destroyed_ships: i64,
}

#[derive(Clone, Debug)]
struct GameResult {
    winner: Option<usize>,
    frames: Vec<ReplayFrame>,
    player_stats: Vec<PlayerStats>,
}

#[derive(Clone, Debug)]
struct MapRng {
    state: u64,
}

#[derive(Clone, Debug)]
struct ActiveCudaGame {
    state: SimulationState,
    frames: Vec<ReplayFrame>,
    player_stats: Vec<PlayerStats>,
    done: bool,
}

#[derive(Clone, Debug)]
struct ActiveModelGame {
    state: SimulationState,
    frames: Vec<ReplayFrame>,
    participant_models: Vec<usize>,
    player_stats: Vec<PlayerStats>,
    done: bool,
    winner_player: Option<usize>,
}

fn main() -> Result<(), String> {
    let cli = parse_cli(env::args().skip(1))?;
    if cli.cuda_v8_smoke {
        let cuda = v8_cuda::V8Cuda::open_from_environment()?;
        cuda.status()?;
        println!("{{\"event\":\"cuda_v8_smoke\",\"status\":\"ok\"}}");
        return Ok(());
    }
    let config = AgentConfig::default();
    if cli.cuda_sim_parity_smoke {
        return cuda_sim_parity_smoke(&config);
    }
    if cli.players == 0 || cli.players > config.max_players {
        return Err(format!("players_out_of_range={}", cli.players));
    }

    if let Some(model_list) = cli.model_list.as_ref() {
        if !cli.cuda_v8 {
            return Err("model_list_requires_cuda_v8".to_string());
        }
        let model_paths = read_model_list(model_list)?;
        let models = model_paths
            .iter()
            .map(|path| {
                V8Model::from_bytes(&fs::read(path).map_err(|error| format!("model_read_failed={}: {error}", path.display()))?)
                    .map_err(|error| format!("model_parse_failed={}: {error:?}", path.display()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let started = Instant::now();
        let results = run_cuda_model_tournament(
            &models,
            cli.games_per_model,
            cli.players,
            cli.replay_stride,
            cli.workers,
            cli.cpu_workers,
            cli.gpu_sim,
            &config,
        )?;
        let elapsed = started.elapsed().as_secs_f32().max(1.0e-6);
        write_dashboard_telemetry(
            &cli.output,
            &results,
            cli.players,
            cli.replay_stride,
            elapsed,
            &config,
            true,
            true,
            cli.workers,
        )?;
        println!(
            "{{\"event\":\"arena_complete\",\"models\":{},\"games_per_model\":{},\"games\":{},\"players\":{},\"seconds\":{:.3},\"telemetry\":\"{}\"}}",
            models.len(),
            cli.games_per_model,
            results.len(),
            cli.players,
            elapsed,
            cli.output.display()
        );
        return Ok(());
    }

    let model = match cli.model.as_ref() {
        Some(path) => Some(
            V8Model::from_bytes(&fs::read(path).map_err(|error| format!("model_read_failed={error}"))?)
                .map_err(|error| format!("model_parse_failed={error:?}"))?,
        ),
        None => None,
    };
    if cli.cuda_v8_forward_smoke {
        let Some(model) = model.as_ref() else {
            return Err("cuda_v8_forward_smoke_requires_model".to_string());
        };
        return cuda_v8_forward_smoke(model, &config);
    }
    let started = Instant::now();
    let model_enabled = model.is_some();
    let results = if cli.cuda_v8 {
        let Some(model) = model.as_ref() else {
            return Err("cuda_v8_requires_model".to_string());
        };
        run_games_cuda(cli.games, cli.players, cli.replay_stride, cli.workers, &config, model)?
    } else {
        run_games_parallel(cli.games, cli.players, cli.replay_stride, cli.workers, &config, model.as_ref())?
    };
    let elapsed = started.elapsed().as_secs_f32().max(1.0e-6);
    write_dashboard_telemetry(
        &cli.output,
        &results,
        cli.players,
        cli.replay_stride,
        elapsed,
        &config,
        model_enabled,
        cli.cuda_v8,
        cli.workers,
    )?;
    println!(
        "{{\"event\":\"arena_complete\",\"games\":{},\"players\":{},\"seconds\":{:.3},\"telemetry\":\"{}\"}}",
        cli.games,
        cli.players,
        elapsed,
        cli.output.display()
    );
    Ok(())
}

fn parse_cli(args: impl Iterator<Item = String>) -> Result<Cli, String> {
    let mut cli = Cli {
        games: DEFAULT_GAMES,
        games_per_model: 20,
        players: DEFAULT_PLAYERS,
        replay_stride: 1,
        workers: 1,
        cpu_workers: 1,
        cuda_v8_smoke: false,
        cuda_v8_forward_smoke: false,
        cuda_sim_parity_smoke: false,
        cuda_v8: false,
        gpu_sim: false,
        output: PathBuf::from(DEFAULT_OUTPUT),
        model: None,
        model_list: None,
    };
    let mut pending_key: Option<String> = None;
    for arg in args {
        if let Some(key) = pending_key.take() {
            apply_arg(&mut cli, &key, &arg)?;
            continue;
        }
        if let Some((key, value)) = arg.split_once('=') {
            apply_arg(&mut cli, key, value)?;
        } else if arg == "--cuda-v8-smoke" {
            cli.cuda_v8_smoke = true;
        } else if arg == "--cuda-v8-forward-smoke" {
            cli.cuda_v8_forward_smoke = true;
        } else if arg == "--cuda-sim-parity-smoke" {
            cli.cuda_sim_parity_smoke = true;
        } else if arg == "--cuda-v8" {
            cli.cuda_v8 = true;
        } else if arg == "--gpu-sim" {
            cli.gpu_sim = true;
        } else if matches!(arg.as_str(), "--games" | "--games-per-model" | "--players" | "--replay-stride" | "--workers" | "--cpu-workers" | "--output" | "--model" | "--model-list") {
            pending_key = Some(arg);
        } else {
            return Err(format!("unknown_argument={arg}"));
        }
    }
    if let Some(key) = pending_key {
        return Err(format!("missing_value_for={key}"));
    }
    Ok(cli)
}

fn apply_arg(cli: &mut Cli, key: &str, value: &str) -> Result<(), String> {
    match key {
        "--games" => cli.games = value.parse().map_err(|_| format!("bad_games={value}"))?,
        "--games-per-model" => cli.games_per_model = value.parse().map_err(|_| format!("bad_games_per_model={value}"))?,
        "--players" => cli.players = value.parse().map_err(|_| format!("bad_players={value}"))?,
        "--workers" => {
            cli.workers = value.parse().map_err(|_| format!("bad_workers={value}"))?;
            if cli.workers == 0 {
                return Err("bad_workers=0".to_string());
            }
        }
        "--cpu-workers" => {
            cli.cpu_workers = value.parse().map_err(|_| format!("bad_cpu_workers={value}"))?;
            if cli.cpu_workers == 0 {
                return Err("bad_cpu_workers=0".to_string());
            }
        }
        "--replay-stride" => {
            cli.replay_stride = value.parse().map_err(|_| format!("bad_replay_stride={value}"))?;
            if cli.replay_stride == 0 {
                return Err("bad_replay_stride=0".to_string());
            }
        }
        "--output" => cli.output = PathBuf::from(value),
        "--model" => cli.model = Some(PathBuf::from(value)),
        "--model-list" => cli.model_list = Some(PathBuf::from(value)),
        _ => return Err(format!("unknown_argument={key}")),
    }
    Ok(())
}

fn run_games_parallel(
    games: usize,
    player_count: usize,
    replay_stride: usize,
    workers: usize,
    config: &AgentConfig,
    model: Option<&V8Model>,
) -> Result<Vec<GameResult>, String> {
    if games == 0 {
        return Ok(Vec::new());
    }
    if workers <= 1 || games == 1 {
        let mut results = Vec::with_capacity(games);
        for game_index in 0..games {
            results.push(run_game(game_index, player_count, replay_stride, config, model)?);
        }
        return Ok(results);
    }
    let worker_count = workers.min(games);
    let mut results: Vec<Option<GameResult>> = vec![None; games];
    thread::scope(|scope| {
        let mut handles = Vec::with_capacity(worker_count);
        for worker_index in 0..worker_count {
            handles.push(scope.spawn(move || {
                let mut worker_results = Vec::new();
                let mut game_index = worker_index;
                while game_index < games {
                    let result = run_game(game_index, player_count, replay_stride, config, model);
                    worker_results.push((game_index, result));
                    game_index += worker_count;
                }
                worker_results
            }));
        }
        for handle in handles {
            for (game_index, result) in handle.join().map_err(|_| "arena_worker_panicked".to_string())? {
                results[game_index] = Some(result?);
            }
        }
        Ok::<(), String>(())
    })?;
    results
        .into_iter()
        .enumerate()
        .map(|(index, result)| result.ok_or_else(|| format!("arena_missing_game_result={index}")))
        .collect()
}

fn run_games_cuda(
    games: usize,
    player_count: usize,
    replay_stride: usize,
    workers: usize,
    config: &AgentConfig,
    model: &V8Model,
) -> Result<Vec<GameResult>, String> {
    let cuda = v8_cuda::V8Cuda::open_from_environment()?;
    cuda.status()?;
    let cuda_model = cuda.create_model(model)?;
    let mut results = Vec::with_capacity(games);
    let batch_games = workers.max(1).min(games.max(1));
    let mut start = 0usize;
    while start < games {
        let end = (start + batch_games).min(games);
        results.extend(run_cuda_game_batch(
            start..end,
            player_count,
            replay_stride,
            config,
            model,
            &cuda_model,
        )?);
        start = end;
    }
    Ok(results)
}

fn read_model_list(path: &PathBuf) -> Result<Vec<PathBuf>, String> {
    let text = fs::read_to_string(path).map_err(|error| format!("model_list_read_failed={error}"))?;
    let base = path.parent().unwrap_or_else(|| std::path::Path::new("."));
    let mut paths = Vec::new();
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let entry = trimmed.split_whitespace().last().unwrap_or(trimmed);
        let candidate = PathBuf::from(entry);
        paths.push(if candidate.is_absolute() { candidate } else { base.join(candidate) });
    }
    if paths.is_empty() {
        return Err("empty_model_list".to_string());
    }
    Ok(paths)
}

fn tournament_schedule(model_count: usize, players: usize, games_per_model: usize) -> Vec<Vec<usize>> {
    let mut games = Vec::new();
    if model_count == 0 || players == 0 || games_per_model == 0 {
        return games;
    }
    for round in 0..games_per_model {
        let offset = (round * 7) % model_count;
        let mut order = (0..model_count)
            .map(|index| (index + offset) % model_count)
            .collect::<Vec<_>>();
        if round % 2 == 1 {
            order.reverse();
        }
        for chunk in order.chunks(players) {
            if chunk.len() == players {
                games.push(chunk.to_vec());
            }
        }
    }
    games
}

fn grouped_tournament_schedule(model_count: usize, players: usize, games_per_model: usize) -> Vec<Vec<usize>> {
    let mut games = Vec::new();
    if model_count == 0 || players == 0 || games_per_model == 0 {
        return games;
    }
    for group in (0..model_count).collect::<Vec<_>>().chunks(players) {
        if group.len() != players {
            continue;
        }
        for round in 0..games_per_model {
            let mut participants = group.to_vec();
            participants.rotate_left(round % players);
            if round % 2 == 1 {
                participants.reverse();
            }
            games.push(participants);
        }
    }
    games
}

fn run_cuda_model_tournament(
    models: &[V8Model],
    games_per_model: usize,
    player_count: usize,
    replay_stride: usize,
    workers: usize,
    cpu_workers: usize,
    gpu_sim: bool,
    config: &AgentConfig,
) -> Result<Vec<GameResult>, String> {
    let cuda = v8_cuda::V8Cuda::open_from_environment()?;
    cuda.status()?;
    let cuda_models = models
        .iter()
        .map(|model| cuda.create_model(model))
        .collect::<Result<Vec<_>, _>>()?;
    let schedule = if gpu_sim {
        grouped_tournament_schedule(models.len(), player_count, games_per_model)
    } else {
        tournament_schedule(models.len(), player_count, games_per_model)
    };
    let batch_games = if gpu_sim {
        games_per_model.max(1).min(schedule.len().max(1))
    } else {
        workers.max(1).min(schedule.len().max(1))
    };
    let mut results = Vec::with_capacity(schedule.len());
    let mut start = 0usize;
    while start < schedule.len() {
        let end = (start + batch_games).min(schedule.len());
        results.extend(run_cuda_model_game_batch(
            start,
            &schedule[start..end],
            models,
            &cuda_models,
            &cuda,
            player_count,
            replay_stride,
            cpu_workers,
            gpu_sim,
            config,
        )?);
        start = end;
    }
    Ok(results)
}

fn run_game(
    game_index: usize,
    player_count: usize,
    replay_stride: usize,
    config: &AgentConfig,
    model: Option<&V8Model>,
) -> Result<GameResult, String> {
    let mut state = seeded_state(game_index, player_count);
    let mut frames = Vec::new();
    let mut player_stats = vec![PlayerStats::default(); player_count];

    while !is_terminal(&state, config) {
        let actions = (0..player_count)
            .map(|player| player_actions(&state, player as i32, config, model))
            .collect::<Result<Vec<_>, _>>()?;
        if state.step % replay_stride == 0 {
            frames.push(frame_from_state(&state, &actions));
        }
        let events = state
            .step_turn_with_events(&actions, config)
            .map_err(|error| format!("simulation_error={error:?}"))?;
        for player in 0..player_count {
            let event = events.player(player as i32);
            player_stats[player].launch_actions += event.launched_fleet_count;
            player_stats[player].launched_ships += event.launched_ship_count;
            player_stats[player].captures += event.captured_planet_count;
            player_stats[player].fleet_hits += event.hit_fleet_count;
            player_stats[player].hit_ships += event.hit_ship_count;
            player_stats[player].sun_destroyed_fleets += event.sun_destroyed_fleet_count;
            player_stats[player].sun_destroyed_ships += event.sun_destroyed_ship_count;
        }
        if is_terminal(&state, config) {
            frames.push(frame_from_state(&state, &Vec::new()));
        }
    }

    let winner = winner_index(&state, config, player_count);
    for player in 0..player_count {
        match winner {
            Some(value) if value == player => player_stats[player].wins += 1,
            Some(_) => player_stats[player].losses += 1,
            None => player_stats[player].draws += 1,
        }
    }
    Ok(GameResult { winner, frames, player_stats })
}

#[allow(dead_code)]
fn run_game_cuda(
    game_index: usize,
    player_count: usize,
    replay_stride: usize,
    config: &AgentConfig,
    model: &V8Model,
    cuda_model: &v8_cuda::V8CudaModel<'_>,
) -> Result<GameResult, String> {
    let mut state = seeded_state(game_index, player_count);
    let mut frames = Vec::new();
    let mut player_stats = vec![PlayerStats::default(); player_count];

    while !is_terminal(&state, config) {
        let actions = player_actions_cuda_batch(&state, player_count, config, model, cuda_model)?;
        if state.step % replay_stride == 0 {
            frames.push(frame_from_state(&state, &actions));
        }
        let events = state
            .step_turn_with_events(&actions, config)
            .map_err(|error| format!("simulation_error={error:?}"))?;
        for player in 0..player_count {
            let event = events.player(player as i32);
            player_stats[player].launch_actions += event.launched_fleet_count;
            player_stats[player].launched_ships += event.launched_ship_count;
            player_stats[player].captures += event.captured_planet_count;
            player_stats[player].fleet_hits += event.hit_fleet_count;
            player_stats[player].hit_ships += event.hit_ship_count;
            player_stats[player].sun_destroyed_fleets += event.sun_destroyed_fleet_count;
            player_stats[player].sun_destroyed_ships += event.sun_destroyed_ship_count;
        }
    }
    if is_terminal(&state, config) {
        frames.push(frame_from_state(&state, &Vec::new()));
    }
    let winner = winner_index(&state, config, player_count);
    for player in 0..player_count {
        match winner {
            Some(value) if value == player => player_stats[player].wins += 1,
            Some(_) => player_stats[player].losses += 1,
            None => player_stats[player].draws += 1,
        }
    }
    Ok(GameResult { winner, frames, player_stats })
}

fn run_cuda_game_batch(
    game_range: std::ops::Range<usize>,
    player_count: usize,
    replay_stride: usize,
    config: &AgentConfig,
    model: &V8Model,
    cuda_model: &v8_cuda::V8CudaModel<'_>,
) -> Result<Vec<GameResult>, String> {
    let mut games = game_range
        .map(|game_index| ActiveCudaGame {
            state: seeded_state(game_index, player_count),
            frames: Vec::new(),
            player_stats: vec![PlayerStats::default(); player_count],
            done: false,
        })
        .collect::<Vec<_>>();

    while games.iter().any(|game| !game.done) {
        let mut requests = Vec::new();
        let mut token_batch = Vec::new();
        for (game_index, game) in games.iter().enumerate() {
            if game.done {
                continue;
            }
            for player in 0..player_count {
                requests.push((game_index, player));
                token_batch.push(model.tokenize(
                    player as i32,
                    game.state.step,
                    game.state.angular_velocity,
                    &game.state.planets,
                    &game.state.fleets,
                ));
            }
        }
        if token_batch.is_empty() {
            break;
        }
        let slots_by_request = cuda_model.forward_tokens(model, &token_batch)?;
        let mut actions_by_game = vec![vec![Vec::<MoveCommand>::new(); player_count]; games.len()];
        for ((game_index, player), slots) in requests.into_iter().zip(slots_by_request.iter()) {
            actions_by_game[game_index][player] = decode_action_slots(
                &games[game_index].state.planets,
                &games[game_index].state.initial_planets,
                &[],
                player as i32,
                games[game_index].state.angular_velocity,
                slots,
                config,
            )
            .map_err(|error| format!("decode_error={error:?}"))?;
        }

        for (game_index, game) in games.iter_mut().enumerate() {
            if game.done {
                continue;
            }
            let actions = &actions_by_game[game_index];
            if game.state.step % replay_stride == 0 {
                game.frames.push(frame_from_state(&game.state, actions));
            }
            let events = game
                .state
                .step_turn_with_events(actions, config)
                .map_err(|error| format!("simulation_error={error:?}"))?;
            for player in 0..player_count {
                let event = events.player(player as i32);
                game.player_stats[player].launch_actions += event.launched_fleet_count;
                game.player_stats[player].launched_ships += event.launched_ship_count;
                game.player_stats[player].captures += event.captured_planet_count;
                game.player_stats[player].fleet_hits += event.hit_fleet_count;
                game.player_stats[player].hit_ships += event.hit_ship_count;
                game.player_stats[player].sun_destroyed_fleets += event.sun_destroyed_fleet_count;
                game.player_stats[player].sun_destroyed_ships += event.sun_destroyed_ship_count;
            }
            if is_terminal(&game.state, config) {
                game.frames.push(frame_from_state(&game.state, &Vec::new()));
                game.done = true;
            }
        }
    }

    let mut results = Vec::with_capacity(games.len());
    for mut game in games {
        let winner = winner_index(&game.state, config, player_count);
        for player in 0..player_count {
            match winner {
                Some(value) if value == player => game.player_stats[player].wins += 1,
                Some(_) => game.player_stats[player].losses += 1,
                None => game.player_stats[player].draws += 1,
            }
        }
        results.push(GameResult {
            winner,
            frames: game.frames,
            player_stats: game.player_stats,
        });
    }
    Ok(results)
}

fn run_cuda_model_game_batch(
    game_offset: usize,
    schedule: &[Vec<usize>],
    models: &[V8Model],
    cuda_models: &[v8_cuda::V8CudaModel<'_>],
    cuda: &v8_cuda::V8Cuda,
    player_count: usize,
    replay_stride: usize,
    cpu_workers: usize,
    gpu_sim: bool,
    config: &AgentConfig,
) -> Result<Vec<GameResult>, String> {
    let mut games = schedule
        .iter()
        .enumerate()
        .map(|(index, participants)| ActiveModelGame {
            state: seeded_state(game_offset + index, player_count),
            frames: Vec::new(),
            participant_models: participants.clone(),
            player_stats: vec![PlayerStats::default(); player_count],
            done: false,
            winner_player: None,
        })
        .collect::<Vec<_>>();
    let mut gpu_sim_groups = if gpu_sim {
        Some(create_gpu_sim_groups(cuda, &games, config, player_count, 4096, 8)?)
    } else {
        None
    };

    while games.iter().any(|game| !game.done) {
        let mut requests_by_model = vec![Vec::<(usize, usize)>::new(); models.len()];
        for (game_index, game) in games.iter().enumerate() {
            if game.done {
                continue;
            }
            for player in 0..player_count {
                requests_by_model[game.participant_models[player]].push((game_index, player));
            }
        }
        if let Some(groups) = gpu_sim_groups.as_mut() {
            let done_changed = step_active_model_games_gpu_resident(
                groups,
                &mut games,
                &requests_by_model,
                cuda_models,
                replay_stride,
                player_count,
            )?;
            if done_changed {
                *groups = create_gpu_sim_groups(cuda, &games, config, player_count, 4096, 8)?;
            }
            continue;
        }

        let mut tokens_by_model = vec![Vec::new(); models.len()];
        for (_game_index, game) in games.iter().enumerate() {
            if game.done {
                continue;
            }
            for player in 0..player_count {
                let model_id = game.participant_models[player];
                tokens_by_model[model_id].push(models[model_id].tokenize(
                    player as i32,
                    game.state.step,
                    game.state.angular_velocity,
                    &game.state.planets,
                    &game.state.fleets,
                ));
            }
        }

        let mut actions_by_game = vec![vec![Vec::<MoveCommand>::new(); player_count]; games.len()];
        for model_id in 0..models.len() {
            if tokens_by_model[model_id].is_empty() {
                continue;
            }
            let slots = cuda_models[model_id].forward_tokens(&models[model_id], &tokens_by_model[model_id])?;
            for ((game_index, player), slot_rows) in requests_by_model[model_id].iter().copied().zip(slots.iter()) {
                actions_by_game[game_index][player] = decode_action_slots(
                    &games[game_index].state.planets,
                    &games[game_index].state.initial_planets,
                    &[],
                    player as i32,
                    games[game_index].state.angular_velocity,
                    slot_rows,
                    config,
                )
                .map_err(|error| format!("decode_error={error:?}"))?;
            }
        }

        if let Some(groups) = gpu_sim_groups.as_mut() {
            let done_changed = step_active_model_games_gpu_sim(
                groups,
                &mut games,
                &actions_by_game,
                replay_stride,
                config,
                player_count,
            )?;
            if done_changed {
                *groups = create_gpu_sim_groups(cuda, &games, config, player_count, 4096, 8)?;
            }
        } else {
            step_active_model_games(
                &mut games,
                &actions_by_game,
                replay_stride,
                config,
                player_count,
                cpu_workers,
            )?;
        }
    }

    let mut results = Vec::with_capacity(games.len());
    for mut game in games {
        let winner_player = game.winner_player.or_else(|| winner_index(&game.state, config, player_count));
        let winner_model = winner_player.map(|player| game.participant_models[player]);
        let mut model_stats = vec![PlayerStats::default(); models.len()];
        for player in 0..player_count {
            match winner_player {
                Some(value) if value == player => game.player_stats[player].wins += 1,
                Some(_) => game.player_stats[player].losses += 1,
                None => game.player_stats[player].draws += 1,
            }
            let model_id = game.participant_models[player];
            add_stats(&mut model_stats[model_id], &game.player_stats[player]);
        }
        results.push(GameResult {
            winner: winner_model,
            frames: game.frames,
            player_stats: model_stats,
        });
    }
    Ok(results)
}

struct GpuSimGroup<'a> {
    sim_state: v8_cuda::CudaSimState<'a>,
    game_indices: Vec<usize>,
    planet_counts: Vec<usize>,
    planet_count: usize,
    max_fleets: usize,
    max_actions_per_player: usize,
}

fn step_active_model_games(
    games: &mut [ActiveModelGame],
    actions_by_game: &[Vec<Vec<MoveCommand>>],
    replay_stride: usize,
    config: &AgentConfig,
    player_count: usize,
    cpu_workers: usize,
) -> Result<(), String> {
    let active_count = games.iter().filter(|game| !game.done).count();
    if active_count == 0 {
        return Ok(());
    }
    let thread_count = cpu_workers.max(1).min(games.len().max(1));
    if thread_count <= 1 || games.len() <= 1 {
        return step_active_model_games_range(games, actions_by_game, 0, replay_stride, config, player_count);
    }
    let chunk_size = (games.len() + thread_count - 1) / thread_count;
    thread::scope(|scope| {
        let mut handles = Vec::new();
        for (chunk_index, chunk) in games.chunks_mut(chunk_size).enumerate() {
            let chunk_start = chunk_index * chunk_size;
            handles.push(scope.spawn(move || {
                step_active_model_games_range(
                    chunk,
                    actions_by_game,
                    chunk_start,
                    replay_stride,
                    config,
                    player_count,
                )
            }));
        }
        for handle in handles {
            handle
                .join()
                .map_err(|_| "cpu_step_worker_panicked".to_string())??;
        }
        Ok::<(), String>(())
    })
}

fn step_active_model_games_range(
    games: &mut [ActiveModelGame],
    actions_by_game: &[Vec<Vec<MoveCommand>>],
    game_offset: usize,
    replay_stride: usize,
    config: &AgentConfig,
    player_count: usize,
) -> Result<(), String> {
    for (local_index, game) in games.iter_mut().enumerate() {
        if game.done {
            continue;
        }
        let actions = &actions_by_game[game_offset + local_index];
        if game.state.step % replay_stride == 0 {
            game.frames.push(frame_from_model_state(
                &game.state,
                actions,
                &game.participant_models,
            ));
        }
        let events = game
            .state
            .step_turn_with_events(actions, config)
            .map_err(|error| format!("simulation_error={error:?}"))?;
        for player in 0..player_count {
            let event = events.player(player as i32);
            game.player_stats[player].launch_actions += event.launched_fleet_count;
            game.player_stats[player].launched_ships += event.launched_ship_count;
            game.player_stats[player].captures += event.captured_planet_count;
            game.player_stats[player].fleet_hits += event.hit_fleet_count;
            game.player_stats[player].hit_ships += event.hit_ship_count;
            game.player_stats[player].sun_destroyed_fleets += event.sun_destroyed_fleet_count;
            game.player_stats[player].sun_destroyed_ships += event.sun_destroyed_ship_count;
        }
        if is_terminal(&game.state, config) {
            game.frames.push(frame_from_model_state(
                &game.state,
                &Vec::new(),
                &game.participant_models,
            ));
            game.done = true;
            game.winner_player = winner_index(&game.state, config, player_count);
        }
    }
    Ok(())
}

fn step_active_model_games_gpu_sim(
    groups: &mut [GpuSimGroup<'_>],
    games: &mut [ActiveModelGame],
    actions_by_game: &[Vec<Vec<MoveCommand>>],
    replay_stride: usize,
    config: &AgentConfig,
    player_count: usize,
) -> Result<bool, String> {
    let mut done_changed = false;
    for group in groups {
        if group.game_indices.is_empty() {
            continue;
        }
        let step = games[group.game_indices[0]].state.step;
        let mut actions = vec![
            v8_cuda::OrbitWarsCudaAction::default();
            group.game_indices.len() * player_count * group.max_actions_per_player
        ];
        let mut action_counts = vec![0i32; group.game_indices.len() * player_count];
        for (local_game, &game_index) in group.game_indices.iter().enumerate() {
            let game = &games[game_index];
            if game.state.step != step {
                return Err("gpu_sim_group_step_mismatch".to_string());
            }
            if game.state.step % replay_stride == 0 {
                games[game_index].frames.push(frame_from_model_state(
                    &game.state,
                    &actions_by_game[game_index],
                    &game.participant_models,
                ));
            }
            for player in 0..player_count {
                let player_actions = &actions_by_game[game_index][player];
                if player_actions.len() > group.max_actions_per_player {
                    return Err(format!("cuda_sim_action_capacity_exceeded={}", player_actions.len()));
                }
                action_counts[local_game * player_count + player] = player_actions.len() as i32;
                for (action_index, action) in player_actions.iter().copied().enumerate() {
                    let offset = (local_game * player_count + player) * group.max_actions_per_player + action_index;
                    actions[offset] = v8_cuda::OrbitWarsCudaAction::from(action);
                }
            }
        }
        group.sim_state.step(&actions, &action_counts, step)?;

        let mut planets = vec![v8_cuda::OrbitWarsCudaPlanet::default(); group.game_indices.len() * group.planet_count];
        let mut fleets = vec![v8_cuda::OrbitWarsCudaFleet::default(); group.game_indices.len() * group.max_fleets];
        let mut next_fleet_ids = vec![0i32; group.game_indices.len()];
        let mut stats = vec![v8_cuda::OrbitWarsCudaSimStats::default(); group.game_indices.len() * player_count];
        group.sim_state.read(&mut planets, &mut fleets, &mut next_fleet_ids, &mut stats)?;

        for (local_game, &game_index) in group.game_indices.iter().enumerate() {
            let game = &mut games[game_index];
            let planet_start = local_game * group.planet_count;
            let fleet_start = local_game * group.max_fleets;
            let real_planet_count = group.planet_counts[local_game];
            game.state.planets = planets[planet_start..planet_start + real_planet_count]
                .iter()
                .copied()
                .map(Planet::from)
                .collect();
            game.state.fleets = fleets[fleet_start..fleet_start + group.max_fleets]
                .iter()
                .copied()
                .filter(|fleet| fleet.alive != 0)
                .map(Fleet::from)
                .collect();
            game.state.next_fleet_id = next_fleet_ids[local_game];
            game.state.step += 1;
            for player in 0..player_count {
                let event = stats[local_game * player_count + player];
                game.player_stats[player].launch_actions += event.launched_fleet_count as usize;
                game.player_stats[player].launched_ships += event.launched_ship_count as i64;
                game.player_stats[player].captures += event.captured_planet_count as usize;
                game.player_stats[player].fleet_hits += event.hit_fleet_count as usize;
                game.player_stats[player].hit_ships += event.hit_ship_count as i64;
                game.player_stats[player].sun_destroyed_fleets += event.sun_destroyed_fleet_count as usize;
                game.player_stats[player].sun_destroyed_ships += event.sun_destroyed_ship_count as i64;
            }
            if is_terminal(&game.state, config) {
                game.frames.push(frame_from_model_state(
                    &game.state,
                    &Vec::new(),
                    &game.participant_models,
                ));
                game.done = true;
                game.winner_player = winner_index(&game.state, config, player_count);
                done_changed = true;
            }
        }
    }
    Ok(done_changed)
}

fn step_active_model_games_gpu_resident(
    groups: &mut [GpuSimGroup<'_>],
    games: &mut [ActiveModelGame],
    requests_by_model: &[Vec<(usize, usize)>],
    cuda_models: &[v8_cuda::V8CudaModel<'_>],
    replay_stride: usize,
    player_count: usize,
) -> Result<bool, String> {
    let Some(group) = groups.first_mut() else {
        return Ok(false);
    };
    if group.game_indices.is_empty() {
        return Ok(false);
    }
    let Some(step) = group
        .game_indices
        .iter()
        .filter_map(|&game_index| {
            let game = &games[game_index];
            if game.done {
                None
            } else {
                Some(game.state.step)
            }
        })
        .next()
    else {
        return Ok(false);
    };
    let mut local_by_game = vec![usize::MAX; games.len()];
    for (local_game, &game_index) in group.game_indices.iter().enumerate() {
        local_by_game[game_index] = local_game;
        if games[game_index].done {
            continue;
        }
        if games[game_index].state.step != step {
            return Err("gpu_resident_group_step_mismatch".to_string());
        }
        if replay_stride > 0 && games[game_index].state.step % replay_stride == 0 {
            games[game_index].frames.push(frame_from_model_state(
                &games[game_index].state,
                &Vec::new(),
                &games[game_index].participant_models,
            ));
        }
    }
    group.sim_state.clear_actions()?;
    for (model_id, requests) in requests_by_model.iter().enumerate() {
        if requests.is_empty() {
            continue;
        }
        let mut request_games = Vec::with_capacity(requests.len());
        let mut request_players = Vec::with_capacity(requests.len());
        for &(game_index, player) in requests {
            let local_game = local_by_game
                .get(game_index)
                .copied()
                .unwrap_or(usize::MAX);
            if local_game == usize::MAX {
                continue;
            }
            request_games.push(local_game as i32);
            request_players.push(player as i32);
        }
        cuda_models[model_id].resident_decode(&group.sim_state, &request_games, &request_players, step)?;
    }
    group.sim_state.step_device_actions(step)?;

    let mut statuses = vec![v8_cuda::OrbitWarsCudaGameStatus::default(); group.game_indices.len()];
    let mut stats = vec![v8_cuda::OrbitWarsCudaSimStats::default(); group.game_indices.len() * player_count];
    group.sim_state.read_status_stats(&mut statuses, &mut stats)?;
    let needs_full_readback = group
        .game_indices
        .iter()
        .enumerate()
        .any(|(local_game, &game_index)| {
            if games[game_index].done {
                return false;
            }
            let status = statuses[local_game];
            status.done != 0 || (replay_stride > 0 && (step + 1) % replay_stride == 0)
        });
    let mut planets = Vec::new();
    let mut fleets = Vec::new();
    let mut next_fleet_ids = Vec::new();
    if needs_full_readback {
        planets = vec![v8_cuda::OrbitWarsCudaPlanet::default(); group.game_indices.len() * group.planet_count];
        fleets = vec![v8_cuda::OrbitWarsCudaFleet::default(); group.game_indices.len() * group.max_fleets];
        next_fleet_ids = vec![0i32; group.game_indices.len()];
        let mut full_stats = vec![v8_cuda::OrbitWarsCudaSimStats::default(); group.game_indices.len() * player_count];
        group.sim_state.read(&mut planets, &mut fleets, &mut next_fleet_ids, &mut full_stats)?;
    }

    for (local_game, &game_index) in group.game_indices.iter().enumerate() {
        let game = &mut games[game_index];
        if game.done {
            continue;
        }
        if needs_full_readback {
            let planet_start = local_game * group.planet_count;
            let fleet_start = local_game * group.max_fleets;
            let real_planet_count = group.planet_counts[local_game];
            game.state.planets = planets[planet_start..planet_start + real_planet_count]
                .iter()
                .copied()
                .map(Planet::from)
                .collect();
            game.state.fleets = fleets[fleet_start..fleet_start + group.max_fleets]
                .iter()
                .copied()
                .filter(|fleet| fleet.alive != 0)
                .map(Fleet::from)
                .collect();
            game.state.next_fleet_id = next_fleet_ids[local_game];
        } else {
            game.state.fleets.clear();
        }
        game.state.step = statuses[local_game].step.max((step + 1) as i32) as usize;
        for player in 0..player_count {
            let event = stats[local_game * player_count + player];
            game.player_stats[player].launch_actions += event.launched_fleet_count as usize;
            game.player_stats[player].launched_ships += event.launched_ship_count as i64;
            game.player_stats[player].captures += event.captured_planet_count as usize;
            game.player_stats[player].fleet_hits += event.hit_fleet_count as usize;
            game.player_stats[player].hit_ships += event.hit_ship_count as i64;
            game.player_stats[player].sun_destroyed_fleets += event.sun_destroyed_fleet_count as usize;
            game.player_stats[player].sun_destroyed_ships += event.sun_destroyed_ship_count as i64;
        }
        if replay_stride > 0 && game.state.step % replay_stride == 0 && statuses[local_game].done == 0 {
            game.frames.push(frame_from_model_state(
                &game.state,
                &Vec::new(),
                &game.participant_models,
            ));
        }
        if statuses[local_game].done != 0 {
            game.frames.push(frame_from_model_state(
                &game.state,
                &Vec::new(),
                &game.participant_models,
            ));
            if !game.done {
                game.done = true;
                game.winner_player = if statuses[local_game].winner >= 0 {
                    Some(statuses[local_game].winner as usize)
                } else {
                    None
                };
            }
        }
    }
    Ok(false)
}

fn create_gpu_sim_groups<'a>(
    cuda: &'a v8_cuda::V8Cuda,
    games: &[ActiveModelGame],
    config: &AgentConfig,
    player_count: usize,
    max_fleets: usize,
    max_actions_per_player: usize,
) -> Result<Vec<GpuSimGroup<'a>>, String> {
    let game_indices = games
        .iter()
        .enumerate()
        .filter_map(|(game_index, game)| if game.done { None } else { Some(game_index) })
        .collect::<Vec<_>>();
    if game_indices.is_empty() {
        return Ok(Vec::new());
    }
    let sim_config = v8_cuda::OrbitWarsCudaSimConfig::new(
        game_indices.len(),
        GPU_SIM_PLANETS,
        max_fleets,
        player_count,
        max_actions_per_player,
        games[game_indices[0]].state.step,
        games[game_indices[0]].state.angular_velocity,
        config,
    );
    let sim_state = cuda.create_sim_state(sim_config)?;
    let mut planets = vec![v8_cuda::OrbitWarsCudaPlanet::default(); game_indices.len() * GPU_SIM_PLANETS];
    let mut initial_planets = vec![v8_cuda::OrbitWarsCudaPlanet::default(); game_indices.len() * GPU_SIM_PLANETS];
    let mut fleets = vec![v8_cuda::OrbitWarsCudaFleet::default(); game_indices.len() * max_fleets];
    let mut next_fleet_ids = Vec::with_capacity(game_indices.len());
    let mut angular_velocities = Vec::with_capacity(game_indices.len());
    let mut planet_counts = Vec::with_capacity(game_indices.len());
    for (local_game, &game_index) in game_indices.iter().enumerate() {
        let state = &games[game_index].state;
        if state.planets.len() > GPU_SIM_PLANETS {
            return Err(format!("cuda_sim_planet_capacity_exceeded={}", state.planets.len()));
        }
        planet_counts.push(state.planets.len());
        angular_velocities.push(state.angular_velocity);
        for (planet_index, planet) in state.planets.iter().copied().enumerate() {
            planets[local_game * GPU_SIM_PLANETS + planet_index] = v8_cuda::OrbitWarsCudaPlanet::from(planet);
        }
        for (planet_index, planet) in state.initial_planets.iter().copied().enumerate() {
            initial_planets[local_game * GPU_SIM_PLANETS + planet_index] = v8_cuda::OrbitWarsCudaPlanet::from(planet);
        }
        for planet_index in state.planets.len()..GPU_SIM_PLANETS {
            let offset = local_game * GPU_SIM_PLANETS + planet_index;
            planets[offset].id = -1000 - planet_index as i32;
            planets[offset].owner = -9;
            initial_planets[offset].id = planets[offset].id;
            initial_planets[offset].owner = -9;
        }
        if state.fleets.len() > max_fleets {
            return Err(format!("cuda_sim_fleet_capacity_exceeded={}", state.fleets.len()));
        }
        for (fleet_index, fleet) in state.fleets.iter().copied().enumerate() {
            fleets[local_game * max_fleets + fleet_index] = v8_cuda::OrbitWarsCudaFleet::from(fleet);
        }
        next_fleet_ids.push(state.next_fleet_id);
    }
    sim_state.load_with_angular_velocities(&planets, &initial_planets, &fleets, &next_fleet_ids, &angular_velocities)?;
    Ok(vec![GpuSimGroup {
        sim_state,
        game_indices,
        planet_counts,
        planet_count: GPU_SIM_PLANETS,
        max_fleets,
        max_actions_per_player,
    }])
}

fn add_stats(target: &mut PlayerStats, source: &PlayerStats) {
    target.wins += source.wins;
    target.draws += source.draws;
    target.losses += source.losses;
    target.launch_actions += source.launch_actions;
    target.launched_ships += source.launched_ships;
    target.captures += source.captures;
    target.fleet_hits += source.fleet_hits;
    target.hit_ships += source.hit_ships;
    target.sun_destroyed_fleets += source.sun_destroyed_fleets;
    target.sun_destroyed_ships += source.sun_destroyed_ships;
}

fn winner_index(state: &SimulationState, config: &AgentConfig, player_count: usize) -> Option<usize> {
    let scores = state.scores(config);
    let best = scores.iter().take(player_count).map(|score| score.ships).max().unwrap_or(0);
    let winners = scores
        .iter()
        .take(player_count)
        .filter(|score| score.ships == best)
        .map(|score| score.player as usize)
        .collect::<Vec<_>>();
    if winners.len() == 1 { Some(winners[0]) } else { None }
}

fn player_actions(
    state: &SimulationState,
    player: i32,
    config: &AgentConfig,
    model: Option<&V8Model>,
) -> Result<Vec<MoveCommand>, String> {
    match model {
        Some(model) => {
            let slots = model
                .action_slots(
                    player,
                    state.step,
                    state.angular_velocity,
                    &state.planets,
                    &state.fleets,
                )
                .map_err(|error| format!("model_forward_failed={error:?}"))?;
            decode_action_slots(
                &state.planets,
                &state.initial_planets,
                &[],
                player,
                state.angular_velocity,
                &slots,
                config,
            )
            .map_err(|error| format!("decode_error={error:?}"))
        }
        None => heuristic_actions(state, player, config),
    }
}

#[allow(dead_code)]
fn player_actions_cuda_batch(
    state: &SimulationState,
    player_count: usize,
    config: &AgentConfig,
    model: &V8Model,
    cuda_model: &v8_cuda::V8CudaModel<'_>,
) -> Result<Vec<Vec<MoveCommand>>, String> {
    let mut tokens = Vec::with_capacity(player_count);
    for player in 0..player_count {
        tokens.push(model.tokenize(
            player as i32,
            state.step,
            state.angular_velocity,
            &state.planets,
            &state.fleets,
        ));
    }
    let slots_by_player = cuda_model.forward_tokens(model, &tokens)?;
    let mut actions = Vec::with_capacity(player_count);
    for (player, slots) in slots_by_player.iter().enumerate() {
        actions.push(
            decode_action_slots(
                &state.planets,
                &state.initial_planets,
                &[],
                player as i32,
                state.angular_velocity,
                slots,
                config,
            )
            .map_err(|error| format!("decode_error={error:?}"))?,
        );
    }
    Ok(actions)
}

fn cuda_v8_forward_smoke(model: &V8Model, config: &AgentConfig) -> Result<(), String> {
    let cuda = v8_cuda::V8Cuda::open_from_environment()?;
    cuda.status()?;
    let cuda_model = cuda.create_model(model)?;
    let state = seeded_state(0, config.max_players.min(DEFAULT_PLAYERS));
    let cpu_slots = model
        .action_slots(
            0,
            state.step,
            state.angular_velocity,
            &state.planets,
            &state.fleets,
        )
        .map_err(|error| format!("model_forward_failed={error:?}"))?;
    let tokens = model.tokenize(
        0,
        state.step,
        state.angular_velocity,
        &state.planets,
        &state.fleets,
    );
    let gpu = cuda_model.forward_tokens(model, &[tokens])?;
    let gpu_slots = gpu
        .first()
        .ok_or_else(|| "cuda_forward_empty_output".to_string())?;
    let diff = max_slot_diff(&cpu_slots, gpu_slots);
    println!(
        "{{\"event\":\"cuda_v8_forward_smoke\",\"slots\":{},\"max_abs_diff\":{:.8},\"status\":\"ok\"}}",
        gpu_slots.len(),
        diff
    );
    Ok(())
}

fn cuda_sim_parity_smoke(config: &AgentConfig) -> Result<(), String> {
    let cuda = v8_cuda::V8Cuda::open_from_environment()?;
    cuda.status()?;
    let mut cpu_state = seeded_state(0, PLAYER_COUNT_FOUR);
    let mut gpu_state = cpu_state.clone();
    let max_fleets = 4096usize;
    let max_actions_per_player = 8usize;
    let sim_config = v8_cuda::OrbitWarsCudaSimConfig::new(
        1,
        gpu_state.planets.len(),
        max_fleets,
        PLAYER_COUNT_FOUR,
        max_actions_per_player,
        gpu_state.step,
        gpu_state.angular_velocity,
        config,
    );
    let sim_state = cuda.create_sim_state(sim_config)?;
    cuda_sim_load_state(&sim_state, &gpu_state, max_fleets)?;
    let mut max_planet_diff = 0.0f32;
    let mut max_fleet_diff = 0.0f32;
    let mut checked_steps = 0usize;
    for _ in 0..50 {
        let actions = parity_actions(&cpu_state, PLAYER_COUNT_FOUR);
        let cpu_events = cpu_state
            .step_turn_with_events(&actions, config)
            .map_err(|error| format!("cpu_sim_error={error:?}"))?;
        let gpu_stats = cuda_sim_step_state_persistent(
            &sim_state,
            &mut gpu_state,
            &actions,
            max_fleets,
            max_actions_per_player,
        )?;
        let (planet_diff, fleet_diff) = compare_states(&cpu_state, &gpu_state)?;
        max_planet_diff = max_planet_diff.max(planet_diff);
        max_fleet_diff = max_fleet_diff.max(fleet_diff);
        compare_step_stats(&cpu_events, &gpu_stats, PLAYER_COUNT_FOUR)?;
        checked_steps += 1;
    }
    println!(
        "{{\"event\":\"cuda_sim_parity_smoke\",\"steps\":{},\"max_planet_diff\":{:.8},\"max_fleet_diff\":{:.8},\"fleets\":{},\"status\":\"ok\"}}",
        checked_steps,
        max_planet_diff,
        max_fleet_diff,
        cpu_state.fleets.len()
    );
    Ok(())
}

fn parity_actions(state: &SimulationState, player_count: usize) -> Vec<Vec<MoveCommand>> {
    let mut actions = vec![Vec::new(); player_count];
    for player in 0..player_count {
        let Some(source) = state
            .planets
            .iter()
            .filter(|planet| planet.owner == player as i32 && planet.ships >= 20.0)
            .max_by(|left, right| left.ships.total_cmp(&right.ships))
        else {
            continue;
        };
        let angle = (50.0 - source.y).atan2(50.0 - source.x);
        actions[player].push(MoveCommand {
            from_planet_id: source.id,
            direction_angle: angle,
            ship_count: (source.ships.floor() as i32 / 2).max(1),
        });
    }
    actions
}

fn cuda_sim_load_state(
    sim_state: &v8_cuda::CudaSimState<'_>,
    state: &SimulationState,
    max_fleets: usize,
) -> Result<(), String> {
    let planets = state
        .planets
        .iter()
        .copied()
        .map(v8_cuda::OrbitWarsCudaPlanet::from)
        .collect::<Vec<_>>();
    let initial_planets = state
        .initial_planets
        .iter()
        .copied()
        .map(v8_cuda::OrbitWarsCudaPlanet::from)
        .collect::<Vec<_>>();
    let mut fleets = vec![v8_cuda::OrbitWarsCudaFleet::default(); max_fleets];
    if state.fleets.len() > max_fleets {
        return Err(format!("cuda_sim_fleet_capacity_exceeded={}", state.fleets.len()));
    }
    for (index, fleet) in state.fleets.iter().copied().enumerate() {
        fleets[index] = v8_cuda::OrbitWarsCudaFleet::from(fleet);
    }
    sim_state.load(&planets, &initial_planets, &fleets, &[state.next_fleet_id])
}

fn cuda_sim_step_state_persistent(
    sim_state: &v8_cuda::CudaSimState<'_>,
    state: &mut SimulationState,
    actions_by_player: &[Vec<MoveCommand>],
    max_fleets: usize,
    max_actions_per_player: usize,
) -> Result<Vec<v8_cuda::OrbitWarsCudaSimStats>, String> {
    let mut actions = vec![v8_cuda::OrbitWarsCudaAction::default(); PLAYER_COUNT_FOUR * max_actions_per_player];
    let mut action_counts = vec![0i32; PLAYER_COUNT_FOUR];
    for player in 0..PLAYER_COUNT_FOUR {
        let count = actions_by_player.get(player).map(|items| items.len()).unwrap_or(0);
        if count > max_actions_per_player {
            return Err(format!("cuda_sim_action_capacity_exceeded={count}"));
        }
        action_counts[player] = count as i32;
        for (index, action) in actions_by_player[player].iter().copied().enumerate() {
            actions[player * max_actions_per_player + index] = v8_cuda::OrbitWarsCudaAction::from(action);
        }
    }
    sim_state.step(&actions, &action_counts, state.step)?;

    let mut planets = vec![v8_cuda::OrbitWarsCudaPlanet::default(); state.planets.len()];
    let mut fleets = vec![v8_cuda::OrbitWarsCudaFleet::default(); max_fleets];
    let mut next_fleet_ids = vec![0i32; 1];
    let mut stats = vec![v8_cuda::OrbitWarsCudaSimStats::default(); PLAYER_COUNT_FOUR];
    sim_state.read(&mut planets, &mut fleets, &mut next_fleet_ids, &mut stats)?;
    state.planets = planets.into_iter().map(Planet::from).collect();
    state.fleets = fleets
        .into_iter()
        .filter(|fleet| fleet.alive != 0)
        .map(Fleet::from)
        .collect();
    state.next_fleet_id = next_fleet_ids[0];
    state.step += 1;
    Ok(stats)
}

fn cuda_sim_step_state(
    cuda: &v8_cuda::V8Cuda,
    state: &mut SimulationState,
    actions_by_player: &[Vec<MoveCommand>],
    config: &AgentConfig,
    max_fleets: usize,
    max_actions_per_player: usize,
) -> Result<Vec<v8_cuda::OrbitWarsCudaSimStats>, String> {
    let planet_count = state.planets.len();
    let mut planets = state
        .planets
        .iter()
        .copied()
        .map(v8_cuda::OrbitWarsCudaPlanet::from)
        .collect::<Vec<_>>();
    let initial_planets = state
        .initial_planets
        .iter()
        .copied()
        .map(v8_cuda::OrbitWarsCudaPlanet::from)
        .collect::<Vec<_>>();
    let mut fleets = vec![v8_cuda::OrbitWarsCudaFleet::default(); max_fleets];
    if state.fleets.len() > max_fleets {
        return Err(format!("cuda_sim_fleet_capacity_exceeded={}", state.fleets.len()));
    }
    for (index, fleet) in state.fleets.iter().copied().enumerate() {
        fleets[index] = v8_cuda::OrbitWarsCudaFleet::from(fleet);
    }
    let mut actions = vec![v8_cuda::OrbitWarsCudaAction::default(); PLAYER_COUNT_FOUR * max_actions_per_player];
    let mut action_counts = vec![0i32; PLAYER_COUNT_FOUR];
    for player in 0..PLAYER_COUNT_FOUR {
        let count = actions_by_player.get(player).map(|items| items.len()).unwrap_or(0);
        if count > max_actions_per_player {
            return Err(format!("cuda_sim_action_capacity_exceeded={count}"));
        }
        action_counts[player] = count as i32;
        for (index, action) in actions_by_player[player].iter().copied().enumerate() {
            actions[player * max_actions_per_player + index] = v8_cuda::OrbitWarsCudaAction::from(action);
        }
    }
    let mut next_fleet_ids = vec![state.next_fleet_id];
    let mut stats = vec![v8_cuda::OrbitWarsCudaSimStats::default(); PLAYER_COUNT_FOUR];
    let sim_config = v8_cuda::OrbitWarsCudaSimConfig::new(
        1,
        planet_count,
        max_fleets,
        PLAYER_COUNT_FOUR,
        max_actions_per_player,
        state.step,
        state.angular_velocity,
        config,
    );
    cuda.sim_step(
        &mut planets,
        &initial_planets,
        &mut fleets,
        &mut next_fleet_ids,
        &actions,
        &action_counts,
        &mut stats,
        sim_config,
    )?;
    state.planets = planets.into_iter().map(Planet::from).collect();
    state.fleets = fleets
        .into_iter()
        .filter(|fleet| fleet.alive != 0)
        .map(Fleet::from)
        .collect();
    state.next_fleet_id = next_fleet_ids[0];
    state.step += 1;
    Ok(stats)
}

fn compare_states(left: &SimulationState, right: &SimulationState) -> Result<(f32, f32), String> {
    if left.planets.len() != right.planets.len() {
        return Err(format!("planet_len_mismatch={}!={}", left.planets.len(), right.planets.len()));
    }
    let mut planet_diff = 0.0f32;
    for (left, right) in left.planets.iter().zip(right.planets.iter()) {
        if left.id != right.id || left.owner != right.owner {
            return Err(format!("planet_identity_mismatch={} owner {}!={}", left.id, left.owner, right.owner));
        }
        planet_diff = planet_diff
            .max((left.x - right.x).abs())
            .max((left.y - right.y).abs())
            .max((left.ships - right.ships).abs())
            .max((left.velocity_x - right.velocity_x).abs())
            .max((left.velocity_y - right.velocity_y).abs());
    }
    let mut left_fleets = left.fleets.clone();
    let mut right_fleets = right.fleets.clone();
    left_fleets.sort_by_key(|fleet| fleet.id);
    right_fleets.sort_by_key(|fleet| fleet.id);
    if left_fleets.len() != right_fleets.len() {
        return Err(format!("fleet_len_mismatch={}!={}", left_fleets.len(), right_fleets.len()));
    }
    let mut fleet_diff = 0.0f32;
    for (left, right) in left_fleets.iter().zip(right_fleets.iter()) {
        if left.id != right.id || left.owner != right.owner || left.from_planet_id != right.from_planet_id {
            return Err(format!("fleet_identity_mismatch={}!={}", left.id, right.id));
        }
        fleet_diff = fleet_diff
            .max((left.x - right.x).abs())
            .max((left.y - right.y).abs())
            .max((left.angle - right.angle).abs())
            .max((left.ships - right.ships).abs());
    }
    if planet_diff > 1.0e-3 || fleet_diff > 1.0e-3 {
        return Err(format!("cuda_sim_state_diff planet={planet_diff:.6} fleet={fleet_diff:.6}"));
    }
    Ok((planet_diff, fleet_diff))
}

fn compare_step_stats(
    cpu_events: &orbit_wars_core::SimulationStepEvents,
    gpu_stats: &[v8_cuda::OrbitWarsCudaSimStats],
    player_count: usize,
) -> Result<(), String> {
    for player in 0..player_count {
        let cpu = cpu_events.player(player as i32);
        let gpu = gpu_stats.get(player).copied().unwrap_or_default();
        let cpu_tuple = (
            cpu.launched_fleet_count as i32,
            cpu.launched_ship_count as i32,
            cpu.hit_fleet_count as i32,
            cpu.hit_ship_count as i32,
            cpu.sun_destroyed_fleet_count as i32,
            cpu.sun_destroyed_ship_count as i32,
            cpu.out_of_bounds_destroyed_fleet_count as i32,
            cpu.out_of_bounds_destroyed_ship_count as i32,
            cpu.captured_planet_count as i32,
        );
        let gpu_tuple = (
            gpu.launched_fleet_count,
            gpu.launched_ship_count,
            gpu.hit_fleet_count,
            gpu.hit_ship_count,
            gpu.sun_destroyed_fleet_count,
            gpu.sun_destroyed_ship_count,
            gpu.out_of_bounds_destroyed_fleet_count,
            gpu.out_of_bounds_destroyed_ship_count,
            gpu.captured_planet_count,
        );
        if cpu_tuple != gpu_tuple {
            return Err(format!("cuda_sim_stats_mismatch player={player} cpu={cpu_tuple:?} gpu={gpu_tuple:?}"));
        }
    }
    Ok(())
}

fn max_slot_diff(left: &[ActionSlotOutput], right: &[ActionSlotOutput]) -> f32 {
    let mut max_diff = 0.0f32;
    for (left, right) in left.iter().zip(right.iter()) {
        max_diff = max_diff.max((left.fire_logit - right.fire_logit).abs());
        for index in 0..64 {
            if left.source_logits[index].is_finite() && right.source_logits[index].is_finite() {
                max_diff = max_diff.max((left.source_logits[index] - right.source_logits[index]).abs());
            }
            if left.target_logits[index].is_finite() && right.target_logits[index].is_finite() {
                max_diff = max_diff.max((left.target_logits[index] - right.target_logits[index]).abs());
            }
        }
        for index in 0..16 {
            max_diff = max_diff.max((left.amount_logits[index] - right.amount_logits[index]).abs());
        }
    }
    max_diff
}

fn heuristic_actions(
    state: &SimulationState,
    player: i32,
    config: &AgentConfig,
) -> Result<Vec<MoveCommand>, String> {
    let mut slots = vec![empty_slot(); config.action_slots];
    let Some((source_row, source)) = state
        .planets
        .iter()
        .enumerate()
        .filter(|(_, planet)| planet.owner == player && planet.ships >= 12.0)
        .max_by(|(_, left), (_, right)| left.ships.total_cmp(&right.ships))
    else {
        return Ok(Vec::new());
    };
    let Some((target_row, _target)) = state
        .planets
        .iter()
        .enumerate()
        .filter(|(_, planet)| planet.id != source.id && planet.owner != player)
        .min_by(|(_, left), (_, right)| source.distance_squared_to(left).total_cmp(&source.distance_squared_to(right)))
    else {
        return Ok(Vec::new());
    };
    slots[0].fire_logit = 8.0;
    slots[0].source_logits[source_row] = 10.0;
    slots[0].target_logits[target_row] = 10.0;
    slots[0].amount_logits[2] = 10.0;
    decode_action_slots(
        &state.planets,
        &state.initial_planets,
        &[],
        player,
        state.angular_velocity,
        &slots,
        config,
    )
    .map_err(|error| format!("decode_error={error:?}"))
}

fn empty_slot() -> ActionSlotOutput {
    ActionSlotOutput {
        fire_logit: -8.0,
        source_logits: [f32::NEG_INFINITY; 64],
        target_logits: [f32::NEG_INFINITY; 64],
        amount_logits: [f32::NEG_INFINITY; 16],
    }
}

fn seeded_state(game_index: usize, player_count: usize) -> SimulationState {
    let config = AgentConfig::default();
    let seed = map_seed(1, game_index).unwrap_or(MAP_SEED_BASE ^ game_index as u64);
    let mut rng = MapRng::new(seed);
    let angular_velocity = rng.range_f32(0.025, 0.05);
    let mut planets = generate_official_like_planets(&config, &mut rng).unwrap_or_else(|_| fallback_planets(player_count));
    if assign_home_planets(&mut planets, player_count, &mut rng).is_err() {
        return SimulationState::new(fallback_planets(player_count), angular_velocity);
    }
    let mut state = SimulationState::new(planets, angular_velocity);
    state.next_fleet_id = 1;
    state
}

fn fallback_planets(player_count: usize) -> Vec<Planet> {
    let mut planets = vec![
        planet(0, 0, 18.0, 18.0, 2.4, 80.0, 4.0),
        planet(1, 1, 82.0, 82.0, 2.4, 80.0, 4.0),
        planet(2, -1, 50.0, 18.0, 1.8, 32.0, 3.0),
        planet(3, -1, 18.0, 50.0, 1.8, 32.0, 3.0),
        planet(4, -1, 82.0, 50.0, 1.8, 32.0, 3.0),
        planet(5, -1, 50.0, 82.0, 1.8, 32.0, 3.0),
        planet(6, -1, 38.0, 38.0, 1.3, 18.0, 2.0),
        planet(7, -1, 62.0, 62.0, 1.3, 18.0, 2.0),
    ];
    if player_count >= 4 {
        planets.push(planet(8, 2, 18.0, 82.0, 2.4, 80.0, 4.0));
        planets.push(planet(9, 3, 82.0, 18.0, 2.4, 80.0, 4.0));
    }
    planets
}

fn generate_official_like_planets(config: &AgentConfig, rng: &mut MapRng) -> Result<Vec<Planet>, String> {
    let target_groups = rng.range_usize(MAP_MIN_PLANET_GROUPS, MAP_MAX_PLANET_GROUPS);
    let target_planets = target_groups
        .checked_mul(MAP_GROUP_SIZE)
        .ok_or_else(|| "map_target_planet_count_overflow".to_string())?;
    let mut planets = Vec::with_capacity(target_planets);
    let mut next_id = 0i32;
    let mut static_groups = 0usize;
    for _ in 0..MAP_GENERATION_ATTEMPT_LIMIT {
        if static_groups >= MAP_MIN_STATIC_GROUPS {
            break;
        }
        if let Some(group) = generate_static_planet_group(next_id, &planets, config, rng)? {
            planets.extend(group);
            next_id += MAP_GROUP_SIZE as i32;
            static_groups += 1;
        }
    }
    let mut attempts = 0usize;
    let mut has_orbiting = false;
    while planets.len() < target_planets || (!has_orbiting && attempts < MAP_GENERATION_ATTEMPT_LIMIT) {
        attempts += 1;
        if attempts >= MAP_GENERATION_ATTEMPT_LIMIT {
            break;
        }
        let Some(group) = generate_orbiting_or_static_planet_group(next_id, &planets, config, rng)? else {
            continue;
        };
        if group.iter().any(|planet| planet_orbits(planet, config)) {
            has_orbiting = true;
        }
        planets.extend(group);
        next_id += MAP_GROUP_SIZE as i32;
    }
    if planets.len() < MAP_MIN_PLANET_GROUPS * MAP_GROUP_SIZE {
        return Err("map_generation_too_few_planets".to_string());
    }
    Ok(planets)
}

fn generate_static_planet_group(
    next_id: i32,
    existing: &[Planet],
    config: &AgentConfig,
    rng: &mut MapRng,
) -> Result<Option<Vec<Planet>>, String> {
    let production = rng.range_usize(MAP_PRODUCTION_MIN, MAP_PRODUCTION_MAX) as f32;
    let radius = planet_radius_from_production(production);
    let angle = rng.range_f32(0.0, std::f32::consts::FRAC_PI_2);
    let min_orbital = config.rotation_radius_limit - radius;
    let max_orbital = (config.board_size - config.board_center - radius) / angle.cos().max(angle.sin());
    if min_orbital > max_orbital {
        return Ok(None);
    }
    let orbital_radius = rng.range_f32(min_orbital, max_orbital);
    let x = config.board_center + orbital_radius * angle.cos();
    let y = config.board_center + orbital_radius * angle.sin();
    if x + radius > config.board_size
        || x - radius < 0.0
        || y + radius > config.board_size
        || y - radius < 0.0
        || x - config.board_center < radius + MAP_STATIC_AXIS_CLEARANCE
        || y - config.board_center < radius + MAP_STATIC_AXIS_CLEARANCE
    {
        return Ok(None);
    }
    let ships = rng
        .range_usize(MAP_STATIC_SHIP_MIN, MAP_STATIC_SHIP_MAX)
        .min(rng.range_usize(MAP_STATIC_SHIP_MIN, MAP_STATIC_SHIP_MAX)) as f32;
    let group = symmetric_planet_group(next_id, x, y, radius, ships, production, config)?;
    Ok(planet_group_has_clearance(&group, existing).then_some(group))
}

fn generate_orbiting_or_static_planet_group(
    next_id: i32,
    existing: &[Planet],
    config: &AgentConfig,
    rng: &mut MapRng,
) -> Result<Option<Vec<Planet>>, String> {
    let production = rng.range_usize(MAP_PRODUCTION_MIN, MAP_PRODUCTION_MAX) as f32;
    let radius = planet_radius_from_production(production);
    let x = rng.range_f32(
        config.board_center + MAP_ORBITING_MIN_COORDINATE_OFFSET,
        config.board_size - radius - MAP_ORBITING_EDGE_CLEARANCE,
    );
    let y = rng.range_f32(
        config.board_center + MAP_ORBITING_MIN_COORDINATE_OFFSET,
        config.board_size - radius - MAP_ORBITING_EDGE_CLEARANCE,
    );
    let orbital_radius = distance_xy(x, y, config.board_center, config.board_center);
    if orbital_radius < config.sun_radius + radius + MAP_SUN_SPAWN_CLEARANCE {
        return Ok(None);
    }
    if orbital_radius + radius >= config.rotation_radius_limit
        && (x + radius > config.board_size || x - radius < 0.0 || y + radius > config.board_size || y - radius < 0.0)
    {
        return Ok(None);
    }
    let ships = rng.range_usize(MAP_ORBITING_SHIP_MIN, MAP_ORBITING_SHIP_MAX) as f32;
    let group = symmetric_planet_group(next_id, x, y, radius, ships, production, config)?;
    Ok((planet_group_has_clearance(&group, existing) && planet_group_orbit_static_cross_check(&group, existing, config))
        .then_some(group))
}

fn symmetric_planet_group(
    next_id: i32,
    x: f32,
    y: f32,
    radius: f32,
    ships: f32,
    production: f32,
    config: &AgentConfig,
) -> Result<Vec<Planet>, String> {
    Ok(vec![
        planet(next_id, -1, y, x, radius, ships, production),
        planet(next_id + 1, -1, config.board_size - x, y, radius, ships, production),
        planet(next_id + 2, -1, x, config.board_size - y, radius, ships, production),
        planet(next_id + 3, -1, config.board_size - y, config.board_size - x, radius, ships, production),
    ])
}

fn planet_group_has_clearance(group: &[Planet], existing: &[Planet]) -> bool {
    group.iter().all(|candidate| {
        existing.iter().all(|planet| {
            distance_xy(candidate.x, candidate.y, planet.x, planet.y)
                >= candidate.radius + planet.radius + MAP_PLANET_CLEARANCE
        })
    })
}

fn planet_group_orbit_static_cross_check(group: &[Planet], existing: &[Planet], config: &AgentConfig) -> bool {
    group.iter().all(|candidate| {
        existing.iter().all(|planet| {
            if planet_orbits(candidate, config) == planet_orbits(planet, config) {
                return true;
            }
            let candidate_orbital = distance_xy(candidate.x, candidate.y, config.board_center, config.board_center);
            let planet_orbital = distance_xy(planet.x, planet.y, config.board_center, config.board_center);
            (candidate_orbital - planet_orbital).abs() >= candidate.radius + planet.radius + MAP_PLANET_CLEARANCE
        })
    })
}

fn assign_home_planets(planets: &mut [Planet], player_count: usize, rng: &mut MapRng) -> Result<(), String> {
    if planets.len() < MAP_GROUP_SIZE || planets.len() % MAP_GROUP_SIZE != 0 {
        return Err("map_home_group_unavailable".to_string());
    }
    let group_count = planets.len() / MAP_GROUP_SIZE;
    let base = rng
        .range_usize(0, group_count - 1)
        .checked_mul(MAP_GROUP_SIZE)
        .ok_or_else(|| "map_home_group_index_overflow".to_string())?;
    if player_count == 2 {
        planets[base].owner = 0;
        planets[base].ships = MAP_HOME_PLANET_SHIPS;
        planets[base + 3].owner = 1;
        planets[base + 3].ships = MAP_HOME_PLANET_SHIPS;
    } else if player_count == PLAYER_COUNT_FOUR {
        for player_slot in 0..PLAYER_COUNT_FOUR {
            planets[base + player_slot].owner = PLAYER_IDS[player_slot];
            planets[base + player_slot].ships = MAP_HOME_PLANET_SHIPS;
        }
    } else {
        return Err(format!("invalid_player_count={player_count}"));
    }
    Ok(())
}

fn planet_radius_from_production(production: f32) -> f32 {
    1.0 + production.ln()
}

fn planet_orbits(planet: &Planet, config: &AgentConfig) -> bool {
    distance_xy(planet.x, planet.y, config.board_center, config.board_center) + planet.radius
        < config.rotation_radius_limit
}

fn distance_xy(left_x: f32, left_y: f32, right_x: f32, right_y: f32) -> f32 {
    let dx = left_x - right_x;
    let dy = left_y - right_y;
    (dx * dx + dy * dy).sqrt()
}

fn map_seed(generation: usize, game_index: usize) -> Result<u64, String> {
    let generation_part = (generation as u64)
        .checked_mul(MAP_GENERATION_SEED_FACTOR)
        .ok_or_else(|| "map_generation_seed_overflow".to_string())?;
    let game_part = (game_index as u64)
        .checked_mul(MAP_GAME_SEED_FACTOR)
        .ok_or_else(|| "map_game_seed_overflow".to_string())?;
    Ok(MAP_SEED_BASE ^ generation_part ^ game_part)
}

impl MapRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(MAP_RNG_MULTIPLIER)
            .wrapping_add(MAP_RNG_INCREMENT);
        self.state
    }

    fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / MAP_RNG_FLOAT_SCALE
    }

    fn range_f32(&mut self, min: f32, max: f32) -> f32 {
        min + (max - min) * self.next_f32()
    }

    fn range_usize(&mut self, min: usize, max: usize) -> usize {
        let span = max - min + 1;
        min + (self.next_u64() as usize % span)
    }
}

fn planet(id: i32, owner: i32, x: f32, y: f32, radius: f32, ships: f32, production: f32) -> Planet {
    Planet {
        id,
        owner,
        x,
        y,
        radius,
        ships,
        production,
        velocity_x: 0.0,
        velocity_y: 0.0,
    }
}

fn frame_from_state(state: &SimulationState, actions_by_player: &[Vec<MoveCommand>]) -> ReplayFrame {
    ReplayFrame {
        step: state.step,
        planets: state.planets.clone(),
        fleets: state.fleets.clone(),
        actions_by_player: actions_by_player.to_vec(),
        model_ids_by_player: Vec::new(),
    }
}

fn frame_from_model_state(
    state: &SimulationState,
    actions_by_player: &[Vec<MoveCommand>],
    model_ids_by_player: &[usize],
) -> ReplayFrame {
    ReplayFrame {
        step: state.step,
        planets: state.planets.clone(),
        fleets: state.fleets.clone(),
        actions_by_player: actions_by_player.to_vec(),
        model_ids_by_player: model_ids_by_player.to_vec(),
    }
}

fn write_dashboard_telemetry(
    path: &PathBuf,
    results: &[GameResult],
    player_count: usize,
    replay_stride: usize,
    elapsed: f32,
    config: &AgentConfig,
    model_enabled: bool,
    cuda_v8: bool,
    workers: usize,
) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|error| format!("telemetry_dir_failed={error}"))?;
    }
    let games = results.len().max(1);
    let win_count = results.iter().filter(|result| result.winner == Some(0)).count();
    let draw_count = results.iter().filter(|result| result.winner.is_none()).count();
    let win_rate = win_count as f32 / games as f32;
    let latest_frames = results.last().map(|result| result.frames.as_slice()).unwrap_or(&[]);
    let stats_count = results
        .iter()
        .map(|result| result.player_stats.len())
        .max()
        .unwrap_or(player_count);
    let mut player_stats = vec![PlayerStats::default(); stats_count];
    for result in results {
        for (index, stats) in result.player_stats.iter().enumerate() {
            add_stats(&mut player_stats[index], stats);
        }
    }
    let launch_actions = player_stats.iter().map(|stats| stats.launch_actions).sum::<usize>();
    let launched_ships = player_stats.iter().map(|stats| stats.launched_ships).sum::<i64>();
    let captures = player_stats.iter().map(|stats| stats.captures).sum::<usize>();
    let fleet_hits = player_stats.iter().map(|stats| stats.fleet_hits).sum::<usize>();
    let hit_ships = player_stats.iter().map(|stats| stats.hit_ships).sum::<i64>();
    let sun_fleets = player_stats.iter().map(|stats| stats.sun_destroyed_fleets).sum::<usize>();
    let sun_ships = player_stats.iter().map(|stats| stats.sun_destroyed_ships).sum::<i64>();
    let games_per_model_report = player_stats
        .iter()
        .map(|stats| stats.wins + stats.draws + stats.losses)
        .max()
        .unwrap_or(games);
    let turns = results
        .iter()
        .map(|result| result.frames.last().map(|frame| frame.step).unwrap_or(0))
        .sum::<usize>()
        .max(1);
    let model_action_calls = if model_enabled { turns * player_count } else { 0 };
    let max_inference_batch_size = if model_enabled {
        player_count * if cuda_v8 { workers.max(1) } else { 1 }
    } else {
        0
    };
    let model_mode = if cuda_v8 {
        "owv8_cuda_model"
    } else if model_enabled {
        "owv8_cpu_model"
    } else {
        "heuristic"
    };
    let source_message = if cuda_v8 {
        "v8 native arena; OWV8 CUDA forward connected"
    } else if model_enabled {
        "v8 native arena; OWV8 CPU forward connected"
    } else {
        "v8 native arena; heuristic fallback"
    };
    let json = format!(
        concat!(
            "{{\n",
            "  \"sourceMessage\":\"{}\",\n",
            "  \"runId\":\"v8-arena-smoke\",\n",
            "  \"activeGeneration\":1,\n",
            "  \"runProfile\":\"v8_arena_smoke\",\n",
            "  \"trainingMode\":\"v8_action_slots_arena\",\n",
            "  \"strictGameRules\":true,\n",
            "  \"episodeSteps\":{},\n",
            "  \"expectedFramesPerGame\":{},\n",
            "  \"populationSize\":{},\n",
            "  \"eliteCount\":1,\n",
            "  \"gamesPerModel\":{},\n",
            "  \"replaysPerModel\":1,\n",
            "  \"generationReplayGameCount\":1,\n",
            "  \"playersPerGame\":{},\n",
            "  \"simultaneousGames\":1,\n",
            "  \"inferenceBatchCalls\":{},\n",
            "  \"maxInferenceBatchSize\":{},\n",
            "  \"mapSource\":\"official_like_native_seeded_map\",\n",
            "  \"turnLoop\":\"orbit_wars_core_step_turn_with_events\",\n",
            "  \"fullReplay\":{},\n",
            "  \"replayFrameStride\":{},\n",
            "  \"storedReplayGames\":1,\n",
            "  \"gamesPerSecond\":{:.3},\n",
            "  \"turnsPerSecond\":{:.3},\n",
            "  \"gpuUtilization\":0,\n",
            "  \"cpuWorkers\":1,\n",
            "  \"evalQueue\":0,\n",
            "  \"submissionGate\":\"manual\",\n",
            "  \"validationErrors\":[],\n",
            "  \"warnings\":[],\n",
            "  \"metrics\":[{}],\n",
            "  \"generationWinRates\":[{}],\n",
            "  \"models\":[{}],\n",
            "  \"frames\":[{}],\n",
            "  \"replayChunks\":[],\n",
            "  \"replayGames\":[{{\"generation\":1,\"modelId\":\"V8-Arena\",\"gameIndex\":{},\"opponentId\":\"{}\",\"reward\":0,\"frames\":[{}]}}]\n",
            "}}\n"
        ),
        source_message,
        config.episode_steps,
        config.episode_steps + 1,
            stats_count,
        games_per_model_report,
        player_count,
        model_action_calls,
        max_inference_batch_size,
        if replay_stride == 1 { "true" } else { "false" },
        replay_stride,
        games as f32 / elapsed,
        turns as f32 / elapsed,
        metric_json(1, win_rate, games, model_action_calls, max_inference_batch_size, launch_actions, launched_ships, captures, fleet_hits, hit_ships, sun_fleets, sun_ships, elapsed),
        generation_win_rate_json(1, games, win_count, draw_count, games.saturating_sub(win_count + draw_count), win_rate),
        models_json(&player_stats, model_mode),
        frames_json(latest_frames),
        games.saturating_sub(1),
        model_mode,
        frames_json(latest_frames)
    );
    fs::write(path, json).map_err(|error| format!("telemetry_write_failed={error}"))
}

fn metric_json(
    generation: usize,
    win_rate: f32,
    games: usize,
    model_action_calls: usize,
    max_inference_batch_size: usize,
    launch_actions: usize,
    launched_ships: i64,
    captures: usize,
    fleet_hits: usize,
    hit_ships: i64,
    sun_fleets: usize,
    sun_ships: i64,
    elapsed: f32,
) -> String {
    format!(
        "{{\"generation\":{},\"winRate\":{:.6},\"gamesPerSecond\":{:.3},\"turnsPerSecond\":0.0,\"p95LatencyMs\":0.0,\"gpuUtilization\":0,\"evaluatedGames\":{},\"sampledReplayGames\":1,\"modelActionCalls\":{},\"launchActions\":{},\"launchedShips\":{},\"captures\":{},\"fleetHits\":{},\"hitShips\":{},\"sunDestroyedFleets\":{},\"sunDestroyedShips\":{},\"avgFleetSize\":0.0,\"avgLaunchActionsPerTurn\":0.0,\"avgLaunchedShipsPerTurn\":0.0,\"avgModelActionMs\":0.0,\"inferenceBatchCalls\":{},\"maxInferenceBatchSize\":{},\"simultaneousGames\":1,\"modelActionSeconds\":0.0,\"simulationStepSeconds\":{:.6},\"evaluationSeconds\":{:.6},\"replaySeconds\":0.0,\"replayWriteSeconds\":0.0,\"generationValidationGames\":0,\"generationValidationSeconds\":0.0,\"backpropSamples\":0,\"backpropModels\":0,\"backpropSeconds\":0.0,\"reproductionSeconds\":0.0,\"generationSeconds\":{:.6}}}",
        generation, win_rate, games as f32 / elapsed, games, model_action_calls, launch_actions, launched_ships, captures, fleet_hits, hit_ships, sun_fleets, sun_ships, model_action_calls, max_inference_batch_size, elapsed, elapsed, elapsed
    )
}

fn generation_win_rate_json(generation: usize, games: usize, wins: usize, draws: usize, losses: usize, win_rate: f32) -> String {
    format!(
        "{{\"validationGeneration\":{},\"evaluatedGeneration\":{},\"modelCount\":1,\"games\":{},\"wins\":{},\"draws\":{},\"losses\":{},\"winRate\":{:.6}}}",
        generation, generation, games, wins, draws, losses, win_rate
    )
}

fn models_json(stats: &[PlayerStats], parent: &str) -> String {
    stats
        .iter()
        .enumerate()
        .map(|(index, stats)| {
            format!(
                "{{\"id\":\"P-{}\",\"parent\":\"{}\",\"rating\":{},\"wins\":{},\"draws\":{},\"losses\":{},\"games\":{},\"captures\":{},\"fleetHits\":{},\"hitShips\":{},\"sunDestroyedFleets\":{},\"sunDestroyedShips\":{},\"outOfBoundsFleets\":0,\"outOfBoundsShips\":0,\"launchActions\":{},\"launchedShips\":{},\"avgFleetSize\":0.0,\"mutation\":\"none\",\"selected\":{}}}",
                index,
                parent,
                600 + stats.wins as i32 * 10 - stats.losses as i32 * 10,
                stats.wins,
                stats.draws,
                stats.losses,
                stats.wins + stats.draws + stats.losses,
                stats.captures,
                stats.fleet_hits,
                stats.hit_ships,
                stats.sun_destroyed_fleets,
                stats.sun_destroyed_ships,
                stats.launch_actions,
                stats.launched_ships,
                if index == 0 { "true" } else { "false" }
            )
        })
        .collect::<Vec<_>>()
        .join(",")
}

fn frames_json(frames: &[ReplayFrame]) -> String {
    frames.iter().map(frame_json).collect::<Vec<_>>().join(",")
}

fn frame_json(frame: &ReplayFrame) -> String {
    format!(
        "{{\"step\":{},\"planets\":[{}],\"fleets\":[{}],\"comets\":[],\"modelIds\":[{}],\"actions\":[{}]}}",
        frame.step,
        frame.planets.iter().map(planet_json).collect::<Vec<_>>().join(","),
        frame.fleets.iter().map(fleet_json).collect::<Vec<_>>().join(","),
        frame.model_ids_by_player
            .iter()
            .map(|model_id| model_id.to_string())
            .collect::<Vec<_>>()
            .join(","),
        actions_json(&frame.actions_by_player)
    )
}

fn planet_json(planet: &Planet) -> String {
    format!(
        "{{\"id\":{},\"owner\":{},\"x\":{:.3},\"y\":{:.3},\"radius\":{:.3},\"ships\":{:.3},\"production\":{:.3}}}",
        planet.id, planet.owner, planet.x, planet.y, planet.radius, planet.ships, planet.production
    )
}

fn fleet_json(fleet: &Fleet) -> String {
    format!(
        "{{\"id\":{},\"owner\":{},\"x\":{:.3},\"y\":{:.3},\"angle\":{:.6},\"ships\":{:.3}}}",
        fleet.id, fleet.owner, fleet.x, fleet.y, fleet.angle, fleet.ships
    )
}

fn actions_json(actions_by_player: &[Vec<MoveCommand>]) -> String {
    actions_by_player
        .iter()
        .map(|actions| {
            format!(
                "[{}]",
                actions
                    .iter()
                    .map(action_json)
                    .collect::<Vec<_>>()
                    .join(",")
            )
        })
        .collect::<Vec<_>>()
        .join(",")
}

fn action_json(action: &MoveCommand) -> String {
    format!(
        "{{\"source\":{},\"angle\":{:.6},\"ships\":{}}}",
        action.from_planet_id, action.direction_angle, action.ship_count
    )
}
