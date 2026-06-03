use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, OpenOptions};
use std::io::{self, Read, Write};
use std::path::Path;
use std::process::Command;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use orbit_wars_core::{
    decode_model_outputs_with_trace, encode_state, is_terminal, true2d_model_binary_bytes,
    ActionOutput, AgentConfig, DecodedMoveCommand, Fleet, MoveCommand, Planet, PlayerStepEvents,
    RowFeature, SimulationState, SimulationStepEvents, True2DTransformer, True2DTransformerShape,
};

mod cuda_true2d;

const PLAYER_ZERO: i32 = 0;
const PLAYER_ONE: i32 = 1;
const PLAYER_TWO: i32 = 2;
const PLAYER_THREE: i32 = 3;
const PLAYER_COUNT_TWO: usize = 2;
const PLAYER_COUNT_FOUR: usize = 4;
const PLAYER_IDS: [i32; PLAYER_COUNT_FOUR] = [PLAYER_ZERO, PLAYER_ONE, PLAYER_TWO, PLAYER_THREE];
const FIRST_GENERATION: usize = 1;
const FIRST_USER_ARGUMENT_INDEX: usize = 1;
const NEXT_ARGUMENT_OFFSET: usize = 1;
const MAX_PLAYER_SLOTS: usize = PLAYER_COUNT_FOUR;
const P95_PERCENTILE_NUMERATOR: usize = 95;
const PERCENT_DENOMINATOR: usize = 100;
const MILLISECONDS_PER_SECOND: f64 = 1000.0;
const BASELINE_RATING: i32 = 600;
const UNMEASURED_GPU_UTILIZATION_PERCENT: usize = 0;
const TRAINING_CPU_WORKER_COUNT: usize = 12;
const EMPTY_EVAL_QUEUE_DEPTH: usize = 0;
const NO_SIMULTANEOUS_GAMES: usize = 0;
const MODEL_SEED_OFFSET: u64 = 1;
const CROSSOVER_SEED_OFFSET: u64 = 10;
const MUTATION_SEED_OFFSET: u64 = 100;
const CUDA_SMOKE_MODEL_ZERO_SEED: u64 = 19;
const CUDA_SMOKE_MODEL_ONE_SEED: u64 = 23;
const GENERATIONS_EQUALS_PREFIX: &str = "--generations=";
const PLAYERS_EQUALS_PREFIX: &str = "--players=";
const RESUME_CHECKPOINT_EQUALS_PREFIX: &str = "--resume-checkpoint=";
const DASHBOARD_STRICT_ARGUMENT: &str = "--dashboard-strict";
const RUN_PROFILE_SMOKE: &str = "smoke_64";
const RUN_PROFILE_DASHBOARD_STRICT: &str = "dashboard_strict_500";
const RUN_PROFILE_FULL: &str = "full_500";
const TRAINING_BACKEND_CUDA: &str = "cuda";
const TRAINING_BACKEND_CPU: &str = "cpu";
const TRAINING_PHASE_INITIALIZING: &str = "initializing";
const TRAINING_PHASE_EVALUATION: &str = "evaluation";
const TRAINING_PHASE_BACKPROP: &str = "backprop";
const TRAINING_PHASE_REPLAY: &str = "replay_write";
const TRAINING_PHASE_VALIDATION: &str = "generation_tournament";
const TRAINING_PHASE_REPRODUCTION: &str = "reproduction";
const TRAINER_LOG_PREFIX: &str = "trainer_log";
const NATIVE_SEEDED_REFERENCE_MAP_SOURCE: &str = "native_seeded_reference_map";
const OFFICIAL_TURN_LOOP_NAME: &str = "official_order_native_reference";
const GPU_UTILIZATION_UNMEASURED_WARNING: &str = "gpu_utilization_unmeasured_by_native_trainer";
const TRAINING_PROGRESS_PENDING_WARNING: &str = "training_generation_in_progress_metrics_pending";
const TRAINING_LIVE_REPLAY_PROGRESS_WARNING: &str = "training_generation_live_replay_partial";
const TELEMETRY_WRITE_ATTEMPTS: usize = 3;
const TELEMETRY_WRITE_RETRY_DELAY_MS: u64 = 25;
const LIVE_REPLAY_FORMAT: &str = "orbit_live_replay_v1";
const LIVE_REPLAY_BINARY_MAGIC: &[u8] = b"OWLIVE1\n";
const LIVE_REPLAY_RECORD_HEADER: u8 = 1;
const LIVE_REPLAY_RECORD_FRAME: u8 = 2;
const LIVE_REPLAY_RECORD_RESULT: u8 = 3;
const TRAINER_CHECKPOINT_MAGIC: &[u8] = b"OWTRAIN1";
const TRAINER_CHECKPOINT_VERSION: u32 = 1;
const GENERATION_TOP_MODELS_MAGIC: &[u8] = b"OWTOP4_1";
const KAGGLE_SUBMISSION_MAIN_FILE: &str = "main.py";
const KAGGLE_SUBMISSION_LIBRARY_FILE: &str = "liborbit_wars_agent.so";
const KAGGLE_SUBMISSION_MODEL_FILE: &str = "model.bin";
const KAGGLE_SUBMISSION_MESSAGE_FILE: &str = "submission.message.txt";
const KAGGLE_SUBMISSION_READY_FILE: &str = "submission.ready";
const TAR_COMMAND: &str = "tar";
const TAR_CREATE_GZIP_ARG: &str = "-czf";
const TAR_DIRECTORY_ARG: &str = "-C";
const COMMAND_OUTPUT_SNIPPET_BYTES: usize = 4096;
const DEFAULT_FULL_REPLAY_ACTUAL_GAMES: usize = 10;
const DEFAULT_FULL_REPLAY_PARTICIPANT_VIEWS: usize =
    DEFAULT_FULL_REPLAY_ACTUAL_GAMES * PLAYER_COUNT_FOUR;
const MAP_SEED_BASE: u64 = 0x4f_57_4d_41_50;
const MAP_GENERATION_SEED_FACTOR: u64 = 1_000_003;
const MAP_GAME_SEED_FACTOR: u64 = 9_176;
const MAP_RNG_MULTIPLIER: u64 = 6_364_136_223_846_793_005;
const MAP_RNG_INCREMENT: u64 = 1_442_695_040_888_963_407;
const MAP_RNG_FLOAT_SCALE: f32 = 16_777_216.0;
const MAP_GENERATION_ATTEMPT_LIMIT: usize = 5_000;
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
struct EvolutionConfig {
    population_size: usize,
    elite_count: usize,
    games_per_model: usize,
    episode_steps: usize,
    active_player_count: usize,
    profile_name: &'static str,
    strict_game_rules: bool,
}

#[derive(Clone, Debug)]
struct TrainerCli {
    smoke: bool,
    dashboard_strict: bool,
    use_cuda: bool,
    full_replays: bool,
    player_count_override: Option<usize>,
    generation_override: Option<usize>,
    resume_checkpoint: Option<String>,
}

#[derive(Debug)]
struct TrainerCheckpoint {
    run_id: String,
    next_generation: usize,
    population: Vec<True2DTransformer>,
    champion_archive: Vec<GenerationChampion>,
}

#[derive(Clone, Debug)]
struct EvaluatedModel {
    model_index: usize,
    reward: i32,
    gameplay: GameplayStats,
}

#[derive(Clone, Debug)]
struct ReproductionParent {
    model: True2DTransformer,
    score: EvaluatedModel,
}

#[derive(Debug)]
struct PopulationEvaluation {
    models: Vec<EvaluatedModel>,
    game_latencies_ms: Vec<f32>,
    replay_games: Vec<ReplayGame>,
    training_samples: TrainingSamples,
    elapsed_seconds: f32,
    compute_stats: ComputeStats,
    pipeline_stats: PipelineStats,
}

#[derive(Clone, Debug)]
struct ReplayWriteStats {
    elapsed_seconds: f32,
    compute_stats: ComputeStats,
    pipeline_stats: PipelineStats,
}

#[derive(Debug)]
struct LiveReplayStorage {
    file_path: String,
    public_path: String,
    frame_count: usize,
    game_count: usize,
    active_player_count: usize,
}

#[derive(Clone, Copy)]
struct LiveReplayTelemetry<'a> {
    evolution_config: &'a EvolutionConfig,
    metrics: &'a [MetricRecord],
    generation_validation_history: &'a [GenerationValidationRecord],
    replay_history: &'a [ReplayChunkRecord],
    agent_config: &'a AgentConfig,
    cli: &'a TrainerCli,
    run_id: &'a str,
}

#[derive(Clone, Copy, Debug, Default)]
struct BackpropStats {
    sample_count: usize,
    model_count: usize,
    elapsed_seconds: f32,
}

#[derive(Clone, Debug)]
struct MetricRecord {
    generation: usize,
    win_rate: f32,
    games_per_second: f32,
    turns_per_second: f32,
    p95_latency_ms: f32,
    gpu_utilization: usize,
    evaluated_games: usize,
    sampled_replay_games: usize,
    model_action_calls: usize,
    launch_actions: usize,
    launched_ships: i64,
    captures: usize,
    fleet_hits: usize,
    hit_ships: i64,
    sun_destroyed_fleets: usize,
    sun_destroyed_ships: i64,
    avg_fleet_size: f32,
    avg_launch_actions_per_turn: f32,
    avg_launched_ships_per_turn: f32,
    avg_model_action_ms: f32,
    inference_batch_calls: usize,
    max_inference_batch_size: usize,
    simultaneous_games: usize,
    model_action_seconds: f32,
    simulation_step_seconds: f32,
    evaluation_seconds: f32,
    replay_seconds: f32,
    replay_write_seconds: f32,
    generation_validation_games: usize,
    generation_validation_seconds: f32,
    backprop_samples: usize,
    backprop_models: usize,
    backprop_seconds: f32,
    reproduction_seconds: f32,
    generation_seconds: f32,
}

#[derive(Clone, Copy, Debug, Default)]
struct ComputeStats {
    game_count: usize,
    turn_count: usize,
    model_action_call_count: usize,
    launch_action_count: usize,
    launched_ship_count: i64,
    model_action_ms: f32,
    simulation_step_ms: f32,
    total_game_ms: f32,
}

#[derive(Clone, Copy, Debug, Default)]
struct PipelineStats {
    inference_batch_call_count: usize,
    max_inference_batch_size: usize,
    simultaneous_game_count: usize,
}

#[derive(Clone, Copy, Debug, Default)]
struct GameplayStats {
    game_count: usize,
    win_count: usize,
    draw_count: usize,
    loss_count: usize,
    capture_count: usize,
    fleet_hit_count: usize,
    hit_ship_count: i64,
    sun_destroyed_fleet_count: usize,
    sun_destroyed_ship_count: i64,
    out_of_bounds_destroyed_fleet_count: usize,
    out_of_bounds_destroyed_ship_count: i64,
    launch_action_count: usize,
    launched_ship_count: i64,
    fleet_ship_observation_sum: f32,
    fleet_observation_count: usize,
}

#[derive(Clone, Copy, Debug)]
struct GameAssignment {
    model_indices: [usize; MAX_PLAYER_SLOTS],
}

#[derive(Clone, Debug)]
struct ActiveGame {
    assignment: GameAssignment,
    state: SimulationState,
    frames: Vec<ReplayFrame>,
    stats: ComputeStats,
    gameplay: [GameplayStats; MAX_PLAYER_SLOTS],
    rewards: [i32; MAX_PLAYER_SLOTS],
    fleet_action_traces: BTreeMap<i32, ActionTrace>,
    completed: bool,
    started_at: Instant,
}

#[derive(Clone, Debug)]
struct MapRng {
    state: u64,
}

#[derive(Clone, Debug)]
struct ActionRequest {
    game_index: usize,
    player_slot: usize,
    player_id: i32,
    model_index: usize,
    step: usize,
    planets: Vec<Planet>,
    initial_planets: Vec<Planet>,
    angular_velocity: f32,
    rows: Vec<RowFeature>,
}

#[derive(Debug, Default)]
struct PopulationForwardScratch {
    input_rows: Vec<f32>,
    model_indices: Vec<usize>,
}

impl PopulationForwardScratch {
    fn pack_chunk(
        &mut self,
        request_chunk: &[ActionRequest],
        config: &AgentConfig,
    ) -> Result<(), String> {
        self.input_rows.clear();
        self.model_indices.clear();
        let input_capacity = request_chunk
            .len()
            .checked_mul(config.max_rows)
            .and_then(|value| value.checked_mul(config.true2d_input_features))
            .ok_or_else(|| "population_forward_scratch_input_capacity_overflow".to_string())?;
        self.input_rows.reserve(input_capacity);
        self.model_indices.reserve(request_chunk.len());
        for request in request_chunk {
            cuda_true2d::pack_rows_into(&request.rows, config, &mut self.input_rows)?;
            self.model_indices.push(request.model_index);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ActionTrace {
    sample_index: usize,
    source_row: usize,
    target_slot: usize,
    target_row: usize,
}

#[derive(Debug)]
struct ResolvedActions {
    actions_by_game: Vec<Vec<Vec<MoveCommand>>>,
    traces_by_game: Vec<Vec<Vec<ActionTrace>>>,
}

#[derive(Debug)]
struct GameBatchResult {
    games: Vec<GameRunResult>,
    pipeline_stats: PipelineStats,
    training_samples: TrainingSamples,
}

#[derive(Clone, Debug)]
struct GameRunResult {
    rewards: [i32; MAX_PLAYER_SLOTS],
    frames: Vec<ReplayFrame>,
    stats: ComputeStats,
    gameplay: [GameplayStats; MAX_PLAYER_SLOTS],
}

#[derive(Clone, Debug)]
struct ReplayFrame {
    step: usize,
    planets: Vec<Planet>,
    fleets: Vec<Fleet>,
    comet_groups: Vec<ReplayCometGroup>,
}

#[derive(Clone, Debug)]
struct ReplayCometGroup {
    planet_ids: Vec<i32>,
    paths: Vec<Vec<(f32, f32)>>,
    path_index: isize,
}

#[derive(Clone, Debug)]
struct ReplayGame {
    generation: usize,
    model_index: usize,
    opponent_label: String,
    game_index: usize,
    reward: i32,
    frames: Arc<Vec<ReplayFrame>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ReplayChunkRecord {
    generation: usize,
    game_count: usize,
    actual_game_count: usize,
}

#[derive(Clone, Copy, Debug)]
struct LiveReplayCompactionStats {
    game_count: usize,
    compact_bytes: u64,
    source: &'static str,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct TrainingSampleMetadata {
    generation: usize,
    game_index: usize,
    player_slot: usize,
    player_id: i32,
    model_index: usize,
    step: usize,
}

#[derive(Debug, Default)]
struct TrainingSamples {
    metadata: Vec<TrainingSampleMetadata>,
    rewards: Vec<i32>,
    sun_target_penalties: Vec<f32>,
    input_rows: Vec<f32>,
    output_rows: Vec<f32>,
}

#[derive(Debug)]
struct BackpropTrainingData {
    input_rows: Vec<f32>,
    output_rows: Vec<f32>,
    rewards: Vec<i32>,
    sun_target_penalties: Vec<f32>,
    model_indices: Vec<usize>,
    model_count: usize,
}

#[derive(Clone, Debug)]
struct GenerationChampion {
    generation: usize,
    model: True2DTransformer,
}

#[derive(Clone, Debug, Default)]
struct GenerationValidationAccumulator {
    model_count: usize,
    game_count: usize,
    win_count: usize,
    draw_count: usize,
    loss_count: usize,
}

#[derive(Clone, Debug)]
struct GenerationValidationRecord {
    validation_generation: usize,
    evaluated_generation: usize,
    model_count: usize,
    game_count: usize,
    win_count: usize,
    draw_count: usize,
    loss_count: usize,
    win_rate: f32,
}

#[derive(Clone, Debug)]
struct GenerationValidationBatch {
    records: Vec<GenerationValidationRecord>,
    champion_scores: Vec<EvaluatedModel>,
    game_count: usize,
    elapsed_seconds: f32,
}

#[derive(Clone, Debug)]
struct KaggleStageOutcome {
    archive_path: String,
    message_path: String,
    ready_path: String,
    model_index: usize,
    win_rate: f32,
    reward: i32,
    game_count: usize,
    win_count: usize,
    draw_count: usize,
    loss_count: usize,
}

fn main() -> Result<(), String> {
    let args = std::env::args().collect::<Vec<_>>();
    if args.iter().any(|arg| arg == "--reference-scenarios") {
        print_reference_scenarios()?;
        return Ok(());
    }

    if args.iter().any(|arg| arg == "--cuda-true2d-smoke") {
        run_cuda_true2d_smoke()?;
        return Ok(());
    }

    let cli = parse_trainer_cli(&args)?;
    let mut agent_config = AgentConfig::default();
    let generation_count = cli
        .generation_override
        .unwrap_or(agent_config.training_generations_per_run);
    let active_player_count = cli
        .player_count_override
        .unwrap_or(agent_config.training_players_per_game);
    validate_active_player_count(active_player_count)?;
    let evolution_config = if cli.smoke {
        EvolutionConfig {
            population_size: agent_config.trainer_smoke_population_size,
            elite_count: agent_config.trainer_smoke_elite_count,
            games_per_model: agent_config.trainer_smoke_games_per_model,
            episode_steps: agent_config.trainer_smoke_episode_steps,
            active_player_count,
            profile_name: RUN_PROFILE_SMOKE,
            strict_game_rules: false,
        }
    } else if cli.dashboard_strict {
        EvolutionConfig {
            population_size: agent_config.dashboard_strict_population_size,
            elite_count: agent_config.dashboard_strict_elite_count,
            games_per_model: agent_config.dashboard_strict_games_per_model,
            episode_steps: agent_config.episode_steps,
            active_player_count,
            profile_name: RUN_PROFILE_DASHBOARD_STRICT,
            strict_game_rules: true,
        }
    } else {
        EvolutionConfig {
            population_size: agent_config.training_population_size,
            elite_count: agent_config.training_elite_count,
            games_per_model: agent_config.default_self_play_games,
            episode_steps: agent_config.episode_steps,
            active_player_count,
            profile_name: RUN_PROFILE_FULL,
            strict_game_rules: true,
        }
    };
    agent_config.episode_steps = evolution_config.episode_steps;
    if cli.dashboard_strict {
        agent_config.dashboard_replays_per_model = agent_config.dashboard_strict_replays_per_model;
        agent_config.dashboard_replay_frame_stride =
            agent_config.dashboard_full_replay_frame_stride;
        agent_config.dashboard_replay_max_frames = agent_config.episode_steps + 1;
    } else if cli.full_replays {
        agent_config.dashboard_replay_frame_stride =
            agent_config.dashboard_full_replay_frame_stride;
        agent_config.dashboard_replay_max_frames = agent_config.episode_steps + 1;
    }
    agent_config
        .validate()
        .map_err(|error| format!("config_invalid={error}"))?;

    let shape = True2DTransformerShape {
        layers: agent_config.default_model_layers,
        row_count: agent_config.true2d_internal_rows,
    };
    let checkpoint = cli
        .resume_checkpoint
        .as_ref()
        .map(|path| load_trainer_checkpoint(path, shape, &agent_config, &evolution_config))
        .transpose()?;
    let mut population = if let Some(checkpoint) = checkpoint.as_ref() {
        checkpoint.population.clone()
    } else {
        (0..evolution_config.population_size)
            .map(|index| {
                True2DTransformer::seeded(shape, index as u64 + MODEL_SEED_OFFSET, &agent_config)
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| format!("model_seed_failed={error:?}"))?
    };
    let cuda_backend = if cli.use_cuda {
        Some(cuda_true2d::CudaTrue2D::open_from_environment()?)
    } else {
        None
    };
    if agent_config.training_backprop_enabled && cuda_backend.is_none() {
        return Err("training_backprop_requires_cuda_backend".to_string());
    }
    let run_id = checkpoint
        .as_ref()
        .map(|checkpoint| checkpoint.run_id.clone())
        .unwrap_or(run_id()?);
    log_trainer_event(format!(
        "event=start; run_id={run_id}; profile={}; backend={}; players_per_game={}; generations={}; episode_steps={}; population={}; elite={}; games_per_model={}; replay_interval={}; generation_replay_games={}; validation_interval={}; cuda_chunk_requests={}",
        evolution_config.profile_name,
        training_backend(&cli),
        evolution_config.active_player_count,
        generation_count,
        evolution_config.episode_steps,
        evolution_config.population_size,
        evolution_config.elite_count,
        evolution_config.games_per_model,
        agent_config.generation_replay_interval,
        agent_config.dashboard_generation_replay_game_count,
        agent_config.generation_validation_interval,
        agent_config.cuda_true2d_max_requests_per_forward
    ))?;
    let mut last_results = Vec::new();
    let mut metric_history = Vec::new();
    let mut replay_history = Vec::new();
    let mut champion_archive = checkpoint
        .as_ref()
        .map(|checkpoint| checkpoint.champion_archive.clone())
        .unwrap_or_default();
    let mut generation_validation_history = Vec::new();
    let start_generation = checkpoint
        .as_ref()
        .map(|checkpoint| checkpoint.next_generation)
        .unwrap_or(FIRST_GENERATION);
    write_progress_telemetry(
        start_generation,
        TRAINING_PHASE_INITIALIZING,
        &evolution_config,
        &metric_history,
        &generation_validation_history,
        &replay_history,
        &agent_config,
        &cli,
        &run_id,
    )?;
    save_trainer_checkpoint(
        trainer_checkpoint_save_path(&cli, &agent_config),
        &run_id,
        start_generation,
        &population,
        &champion_archive,
    )?;
    for generation in start_generation..=generation_count {
        let generation_started_at = Instant::now();
        log_trainer_phase_start(
            &run_id,
            generation,
            TRAINING_PHASE_EVALUATION,
            &evolution_config,
            &agent_config,
            &cli,
        )?;
        write_progress_telemetry(
            generation,
            TRAINING_PHASE_EVALUATION,
            &evolution_config,
            &metric_history,
            &generation_validation_history,
            &replay_history,
            &agent_config,
            &cli,
            &run_id,
        )?;
        let capture_generation_replays =
            scheduled_generation(generation, agent_config.generation_replay_interval);
        let live_replay_telemetry = LiveReplayTelemetry {
            evolution_config: &evolution_config,
            metrics: &metric_history,
            generation_validation_history: &generation_validation_history,
            replay_history: &replay_history,
            agent_config: &agent_config,
            cli: &cli,
            run_id: &run_id,
        };
        let evaluation = evaluate_population(
            generation,
            &population,
            &evolution_config,
            &agent_config,
            cuda_backend.as_ref(),
            capture_generation_replays,
            Some(&live_replay_telemetry),
        )?;
        log_trainer_event(format!(
            "event=phase_done; run_id={run_id}; generation={generation}; phase={}; seconds={:.6}; evaluated_games={}; model_action_calls={}; inference_batch_calls={}; max_inference_batch_size={}; simultaneous_games={}; training_samples={}; training_input_floats={}; training_output_floats={}",
            TRAINING_PHASE_EVALUATION,
            evaluation.elapsed_seconds,
            evaluation.compute_stats.game_count,
            evaluation.compute_stats.model_action_call_count,
            evaluation.pipeline_stats.inference_batch_call_count,
            evaluation.pipeline_stats.max_inference_batch_size,
            evaluation.pipeline_stats.simultaneous_game_count,
            evaluation.training_samples.sample_count(),
            evaluation.training_samples.input_float_count(),
            evaluation.training_samples.output_float_count()
        ))?;
        last_results = evaluation.models.clone();
        let backprop_stats = if agent_config.training_backprop_enabled {
            log_trainer_phase_start(
                &run_id,
                generation,
                TRAINING_PHASE_BACKPROP,
                &evolution_config,
                &agent_config,
                &cli,
            )?;
            write_progress_telemetry(
                generation,
                TRAINING_PHASE_BACKPROP,
                &evolution_config,
                &metric_history,
                &generation_validation_history,
                &replay_history,
                &agent_config,
                &cli,
                &run_id,
            )?;
            let stats = train_elite_models_with_backprop(
                &mut population,
                &last_results,
                &evaluation.training_samples,
                &evolution_config,
                &agent_config,
                cuda_backend.as_ref(),
            )?;
            log_trainer_event(format!(
                "event=phase_done; run_id={run_id}; generation={generation}; phase={}; seconds={:.6}; trained_models={}; training_samples={}; batch_samples={}; learning_rate={:.8}",
                TRAINING_PHASE_BACKPROP,
                stats.elapsed_seconds,
                stats.model_count,
                stats.sample_count,
                agent_config.training_backprop_batch_samples,
                agent_config.training_backprop_learning_rate
            ))?;
            stats
        } else {
            BackpropStats::default()
        };
        archive_generation_champions(
            &mut champion_archive,
            generation,
            &last_results,
            &population,
            &agent_config,
        )?;
        write_generation_top_models_artifact(
            &run_id,
            generation,
            &last_results,
            &population,
            agent_config.generation_archive_champions_per_generation,
        )?;
        let (replay_write_stats, replay_write_seconds) = if capture_generation_replays {
            log_trainer_phase_start(
                &run_id,
                generation,
                TRAINING_PHASE_REPLAY,
                &evolution_config,
                &agent_config,
                &cli,
            )?;
            write_progress_telemetry(
                generation,
                TRAINING_PHASE_REPLAY,
                &evolution_config,
                &metric_history,
                &generation_validation_history,
                &replay_history,
                &agent_config,
                &cli,
                &run_id,
            )?;
            let replay_write_started_at = Instant::now();
            match compact_live_slot_to_owlive_from_replay_games(
                &run_id,
                generation,
                &evaluation.replay_games,
                evolution_config.active_player_count,
            ) {
                Ok(Some(stats)) => log_trainer_event(format!(
                    "event=live_replay_compacted; run_id={run_id}; generation={generation}; source={}; target=owlive; games={}; bytes={}",
                    stats.source, stats.game_count, stats.compact_bytes
                ))?,
                Ok(None) => {}
                Err(error) => log_trainer_event(format!(
                    "event=warning; run_id={run_id}; generation={generation}; phase={}; warning=live_replay_compaction_failed; error={error}",
                    TRAINING_PHASE_REPLAY
                ))?,
            }
            write_replay_chunk(&run_id, generation, &evaluation.replay_games)?;
            let write_seconds = elapsed_seconds(replay_write_started_at.elapsed());
            replay_history.extend(replay_chunk_records_from_replay_games(
                &evaluation.replay_games,
            ));
            let replay_stats = replay_write_stats_from_evaluation(&evaluation.replay_games);
            log_trainer_event(format!(
                "event=phase_done; run_id={run_id}; generation={generation}; phase={}; seconds={:.6}; replay_games={}; replay_write_seconds={:.6}",
                TRAINING_PHASE_REPLAY,
                replay_stats.elapsed_seconds,
                evaluation.replay_games.len(),
                write_seconds
            ))?;
            (replay_stats, write_seconds)
        } else {
            (empty_replay_write_stats(), 0.0)
        };
        let generation_validation =
            if scheduled_generation(generation, agent_config.generation_validation_interval) {
                log_trainer_phase_start(
                    &run_id,
                    generation,
                    TRAINING_PHASE_VALIDATION,
                    &evolution_config,
                    &agent_config,
                    &cli,
                )?;
                write_progress_telemetry(
                    generation,
                    TRAINING_PHASE_VALIDATION,
                    &evolution_config,
                    &metric_history,
                    &generation_validation_history,
                    &replay_history,
                    &agent_config,
                    &cli,
                    &run_id,
                )?;
                validate_generation_archive(
                    generation,
                    &champion_archive,
                    &evolution_config,
                    &agent_config,
                    cuda_backend.as_ref(),
                )?
            } else {
                empty_generation_validation()
            };
        if scheduled_generation(generation, agent_config.generation_validation_interval) {
            log_trainer_event(format!(
                "event=phase_done; run_id={run_id}; generation={generation}; phase={}; seconds={:.6}; tournament_games={}",
                TRAINING_PHASE_VALIDATION,
                generation_validation.elapsed_seconds,
                generation_validation.game_count
            ))?;
            if agent_config.generation_tournament_kaggle_stage_for_host_submit {
                log_trainer_event(format!(
                    "event=kaggle_stage_start; run_id={run_id}; generation={generation}; source=generation_tournament_top1_winrate"
                ))?;
                match stage_generation_tournament_top_winrate_model_for_host_submit(
                    generation,
                    &run_id,
                    &champion_archive,
                    &generation_validation.champion_scores,
                    &agent_config,
                ) {
                    Ok(outcome) => log_trainer_event(format!(
                        "event=kaggle_stage_done; run_id={run_id}; generation={generation}; model={}; win_rate={:.3}; record={}-{}-{}; reward={}; games={}; archive={}; message={}; ready={}",
                        outcome.model_index,
                        outcome.win_rate,
                        outcome.win_count,
                        outcome.draw_count,
                        outcome.loss_count,
                        outcome.reward,
                        outcome.game_count,
                        outcome.archive_path,
                        outcome.message_path,
                        outcome.ready_path
                    ))?,
                    Err(error) => log_trainer_event(format!(
                        "event=warning; run_id={run_id}; generation={generation}; phase={}; warning=kaggle_stage_failed; error={error}",
                        TRAINING_PHASE_VALIDATION
                    ))?,
                }
            }
        }
        let reproduction_started_at = Instant::now();
        if scheduled_generation(generation, agent_config.generation_validation_interval) {
            population = reproduce_population_from_champion_scores(
                &mut champion_archive,
                &generation_validation.champion_scores,
                &mut replay_history,
                &run_id,
                &evolution_config,
                &agent_config,
            )?;
        } else {
            population =
                reproduce_population(&population, &last_results, &evolution_config, &agent_config)?;
        }
        let reproduction_seconds = elapsed_seconds(reproduction_started_at.elapsed());
        log_trainer_event(format!(
            "event=phase_done; run_id={run_id}; generation={generation}; phase={}; seconds={:.6}; source={}_top{}",
            TRAINING_PHASE_REPRODUCTION,
            reproduction_seconds,
            if scheduled_generation(generation, agent_config.generation_validation_interval) {
                "generation_tournament"
            } else {
                "current_generation"
            },
            evolution_config.elite_count
        ))?;
        generation_validation_history.extend(generation_validation.records);
        let generation_seconds = elapsed_seconds(generation_started_at.elapsed());
        let metric = metric_record(
            generation,
            &last_results,
            &evaluation,
            &replay_write_stats,
            replay_write_seconds,
            generation_validation.game_count,
            generation_validation.elapsed_seconds,
            backprop_stats,
            reproduction_seconds,
            generation_seconds,
        );
        let generation_best = last_results
            .first()
            .ok_or_else(|| "evaluation produced no models".to_string())?;
        log_trainer_event(format!(
            "event=generation_done; run_id={run_id}; generation={generation}; seconds={:.6}; win_rate={:.3}; games_per_second={:.3}; turns_per_second={:.3}; p95_latency_ms={:.3}; backprop_samples={}; best_model={}; best_reward={}",
            metric.generation_seconds,
            metric.win_rate,
            metric.games_per_second,
            metric.turns_per_second,
            metric.p95_latency_ms,
            backprop_stats.sample_count,
            generation_best.model_index,
            generation_best.reward
        ))?;
        write_generation_log_artifact(
            &run_id,
            generation,
            &metric,
            &last_results,
            &evaluation.replay_games,
            agent_config.generation_archive_champions_per_generation,
        )?;
        metric_history.push(metric);
        write_telemetry(
            generation,
            &last_results,
            &evolution_config,
            &metric_history,
            &generation_validation_history,
            &replay_history,
            &agent_config,
            &cli,
            &run_id,
        )?;
        save_trainer_checkpoint(
            trainer_checkpoint_save_path(&cli, &agent_config),
            &run_id,
            generation + 1,
            &population,
            &champion_archive,
        )?;
    }

    let best = last_results
        .first()
        .ok_or_else(|| "evaluation produced no models".to_string())?;
    log_trainer_event(format!(
        "event=complete; run_id={run_id}; profile={}; backend={}; generations={}; best_model={}; best_reward={}",
        evolution_config.profile_name,
        training_backend(&cli),
        generation_count,
        best.model_index,
        best.reward
    ))?;
    println!(
        "status=ok; profile={}; smoke={}; dashboard_strict={}; cuda={}; full_replays={}; players_per_game={}; generations={}; episode_steps={}; population={}; elite={}; best_model={}; best_reward={}",
        evolution_config.profile_name,
        cli.smoke,
        cli.dashboard_strict,
        cli.use_cuda,
        cli.full_replays,
        evolution_config.active_player_count,
        generation_count,
        evolution_config.episode_steps,
        evolution_config.population_size,
        evolution_config.elite_count,
        best.model_index,
        best.reward
    );
    flush_stdout()?;
    Ok(())
}

fn training_backend(cli: &TrainerCli) -> &'static str {
    if cli.use_cuda {
        TRAINING_BACKEND_CUDA
    } else {
        TRAINING_BACKEND_CPU
    }
}

fn log_trainer_phase_start(
    run_id: &str,
    generation: usize,
    phase: &str,
    evolution_config: &EvolutionConfig,
    agent_config: &AgentConfig,
    cli: &TrainerCli,
) -> Result<(), String> {
    log_trainer_event(format!(
        "event=phase_start; run_id={run_id}; generation={generation}; phase={phase}; profile={}; backend={}; players_per_game={}; episode_steps={}; population={}; games_per_model={}; cuda_chunk_requests={}",
        evolution_config.profile_name,
        training_backend(cli),
        evolution_config.active_player_count,
        evolution_config.episode_steps,
        evolution_config.population_size,
        evolution_config.games_per_model,
        agent_config.cuda_true2d_max_requests_per_forward
    ))
}

fn log_trainer_event(message: String) -> Result<(), String> {
    let mut stdout = io::stdout().lock();
    writeln!(stdout, "{TRAINER_LOG_PREFIX}; {message}")
        .map_err(|error| format!("trainer_log_write_failed={error}"))?;
    stdout
        .flush()
        .map_err(|error| format!("trainer_log_flush_failed={error}"))
}

fn flush_stdout() -> Result<(), String> {
    let mut stdout = io::stdout().lock();
    stdout
        .flush()
        .map_err(|error| format!("trainer_stdout_flush_failed={error}"))
}

fn trainer_checkpoint_save_path<'a>(cli: &'a TrainerCli, config: &'a AgentConfig) -> &'a str {
    cli.resume_checkpoint
        .as_deref()
        .unwrap_or(config.trainer_checkpoint_path)
}

fn parse_trainer_cli(args: &[String]) -> Result<TrainerCli, String> {
    let mut cli = TrainerCli {
        smoke: false,
        dashboard_strict: false,
        use_cuda: false,
        full_replays: false,
        player_count_override: None,
        generation_override: None,
        resume_checkpoint: None,
    };
    let mut argument_index = FIRST_USER_ARGUMENT_INDEX;
    while argument_index < args.len() {
        let argument = args[argument_index].as_str();
        match argument {
            "--smoke" => {
                cli.smoke = true;
                argument_index += 1;
            }
            DASHBOARD_STRICT_ARGUMENT => {
                cli.dashboard_strict = true;
                argument_index += 1;
            }
            "--cuda" => {
                cli.use_cuda = true;
                argument_index += 1;
            }
            "--full-replays" => {
                cli.full_replays = true;
                argument_index += 1;
            }
            "--resume-checkpoint" => {
                let value_index = argument_index + NEXT_ARGUMENT_OFFSET;
                let value = args
                    .get(value_index)
                    .ok_or_else(|| "missing_resume_checkpoint_argument".to_string())?;
                cli.resume_checkpoint = Some(value.clone());
                argument_index = value_index + NEXT_ARGUMENT_OFFSET;
            }
            "--players" => {
                let value_index = argument_index + NEXT_ARGUMENT_OFFSET;
                let value = args
                    .get(value_index)
                    .ok_or_else(|| "missing_players_argument".to_string())?;
                cli.player_count_override = Some(parse_player_count(value)?);
                argument_index = value_index + NEXT_ARGUMENT_OFFSET;
            }
            "--generations" => {
                let value_index = argument_index + NEXT_ARGUMENT_OFFSET;
                let value = args
                    .get(value_index)
                    .ok_or_else(|| "missing_generations_argument".to_string())?;
                cli.generation_override = Some(parse_positive_usize("generations", value)?);
                argument_index = value_index + NEXT_ARGUMENT_OFFSET;
            }
            value if value.starts_with(GENERATIONS_EQUALS_PREFIX) => {
                let raw_value = value
                    .strip_prefix(GENERATIONS_EQUALS_PREFIX)
                    .ok_or_else(|| "missing_generations_argument".to_string())?;
                cli.generation_override = Some(parse_positive_usize("generations", raw_value)?);
                argument_index += 1;
            }
            value if value.starts_with(PLAYERS_EQUALS_PREFIX) => {
                let raw_value = value
                    .strip_prefix(PLAYERS_EQUALS_PREFIX)
                    .ok_or_else(|| "missing_players_argument".to_string())?;
                cli.player_count_override = Some(parse_player_count(raw_value)?);
                argument_index += 1;
            }
            value if value.starts_with(RESUME_CHECKPOINT_EQUALS_PREFIX) => {
                let raw_value = value
                    .strip_prefix(RESUME_CHECKPOINT_EQUALS_PREFIX)
                    .ok_or_else(|| "missing_resume_checkpoint_argument".to_string())?;
                cli.resume_checkpoint = Some(raw_value.to_string());
                argument_index += 1;
            }
            unknown => {
                return Err(format!("unknown_argument={unknown}"));
            }
        }
    }
    if cli.smoke && cli.dashboard_strict {
        return Err("smoke_and_dashboard_strict_are_mutually_exclusive".to_string());
    }
    Ok(cli)
}

fn parse_positive_usize(name: &str, value: &str) -> Result<usize, String> {
    let parsed = value
        .parse::<usize>()
        .map_err(|error| format!("invalid_{name}={value}; error={error}"))?;
    if parsed == 0 {
        return Err(format!("invalid_{name}=0"));
    }
    Ok(parsed)
}

fn parse_player_count(value: &str) -> Result<usize, String> {
    let parsed = parse_positive_usize("players", value)?;
    validate_active_player_count(parsed)?;
    Ok(parsed)
}

fn save_trainer_checkpoint(
    path: &str,
    run_id: &str,
    next_generation: usize,
    population: &[True2DTransformer],
    champion_archive: &[GenerationChampion],
) -> Result<(), String> {
    let path = Path::new(path);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|error| format!("checkpoint_dir_failed={error}"))?;
    }
    let mut bytes = Vec::new();
    bytes.extend_from_slice(TRAINER_CHECKPOINT_MAGIC);
    push_u32(&mut bytes, TRAINER_CHECKPOINT_VERSION);
    push_checkpoint_string(&mut bytes, run_id);
    push_usize_as_u64(&mut bytes, next_generation);
    push_transformer_population(&mut bytes, population)?;
    push_usize_as_u64(&mut bytes, champion_archive.len());
    for champion in champion_archive {
        push_usize_as_u64(&mut bytes, champion.generation);
        push_transformer(&mut bytes, &champion.model)?;
    }
    let tmp_path = path.with_extension("tmp");
    fs::write(&tmp_path, bytes).map_err(|error| format!("checkpoint_write_failed={error}"))?;
    fs::rename(&tmp_path, path).map_err(|error| format!("checkpoint_rename_failed={error}"))
}

fn load_trainer_checkpoint(
    path: &str,
    expected_shape: True2DTransformerShape,
    config: &AgentConfig,
    evolution_config: &EvolutionConfig,
) -> Result<TrainerCheckpoint, String> {
    let mut bytes = Vec::new();
    fs::File::open(path)
        .map_err(|error| format!("checkpoint_open_failed={error}"))?
        .read_to_end(&mut bytes)
        .map_err(|error| format!("checkpoint_read_failed={error}"))?;
    let mut cursor = 0usize;
    read_magic(&bytes, &mut cursor)?;
    let version = read_u32(&bytes, &mut cursor)?;
    if version != TRAINER_CHECKPOINT_VERSION {
        return Err(format!("checkpoint_version_unsupported={version}"));
    }
    let run_id = read_string(&bytes, &mut cursor)?;
    let next_generation = read_usize(&bytes, &mut cursor, "checkpoint_next_generation")?;
    let checkpoint_population_size =
        read_usize(&bytes, &mut cursor, "checkpoint_population_count")?;
    let mut population = read_transformer_population(
        &bytes,
        &mut cursor,
        checkpoint_population_size,
        expected_shape,
        config,
    )?;
    population = adapt_checkpoint_population(population, evolution_config)?;
    if checkpoint_population_size != population.len() {
        log_trainer_event(format!(
            "event=checkpoint_population_resized; run_id={run_id}; from={checkpoint_population_size}; to={}",
            population.len()
        ))?;
    }
    let champion_count = read_usize(&bytes, &mut cursor, "checkpoint_champion_count")?;
    let mut champion_archive = Vec::with_capacity(champion_count);
    for _ in 0..champion_count {
        let generation = read_usize(&bytes, &mut cursor, "checkpoint_champion_generation")?;
        let model = read_transformer(&bytes, &mut cursor, expected_shape, config)?;
        champion_archive.push(GenerationChampion { generation, model });
    }
    if cursor != bytes.len() {
        return Err("checkpoint_trailing_bytes".to_string());
    }
    Ok(TrainerCheckpoint {
        run_id,
        next_generation,
        population,
        champion_archive,
    })
}

fn push_transformer_population(
    bytes: &mut Vec<u8>,
    population: &[True2DTransformer],
) -> Result<(), String> {
    push_usize_as_u64(bytes, population.len());
    for model in population {
        push_transformer(bytes, model)?;
    }
    Ok(())
}

fn read_transformer_population(
    bytes: &[u8],
    cursor: &mut usize,
    model_count: usize,
    expected_shape: True2DTransformerShape,
    config: &AgentConfig,
) -> Result<Vec<True2DTransformer>, String> {
    let mut population = Vec::with_capacity(model_count);
    for _ in 0..model_count {
        population.push(read_transformer(bytes, cursor, expected_shape, config)?);
    }
    Ok(population)
}

fn adapt_checkpoint_population(
    mut population: Vec<True2DTransformer>,
    evolution_config: &EvolutionConfig,
) -> Result<Vec<True2DTransformer>, String> {
    if population.len() == evolution_config.population_size {
        return Ok(population);
    }
    if population.len() > evolution_config.population_size {
        population.truncate(evolution_config.population_size);
        return Ok(population);
    }
    Err(format!(
        "checkpoint_population_too_small={}; expected={}",
        population.len(),
        evolution_config.population_size
    ))
}

fn push_transformer(bytes: &mut Vec<u8>, model: &True2DTransformer) -> Result<(), String> {
    push_usize_as_u64(bytes, model.shape.layers);
    push_usize_as_u64(bytes, model.shape.row_count);
    push_usize_as_u64(bytes, model.weights.len());
    for weight in &model.weights {
        if !weight.is_finite() {
            return Err("checkpoint_non_finite_weight".to_string());
        }
        bytes.extend_from_slice(&weight.to_le_bytes());
    }
    Ok(())
}

fn read_transformer(
    bytes: &[u8],
    cursor: &mut usize,
    expected_shape: True2DTransformerShape,
    config: &AgentConfig,
) -> Result<True2DTransformer, String> {
    let layers = read_usize(bytes, cursor, "checkpoint_model_layers")?;
    let row_count = read_usize(bytes, cursor, "checkpoint_model_row_count")?;
    let shape = True2DTransformerShape { layers, row_count };
    if shape != expected_shape {
        return Err("checkpoint_model_shape_mismatch".to_string());
    }
    let weight_count = read_usize(bytes, cursor, "checkpoint_model_weight_count")?;
    let mut weights = Vec::with_capacity(weight_count);
    for _ in 0..weight_count {
        weights.push(read_f32(bytes, cursor)?);
    }
    True2DTransformer::from_weights(shape, weights, config)
        .map_err(|error| format!("checkpoint_model_invalid={error:?}"))
}

fn read_magic(bytes: &[u8], cursor: &mut usize) -> Result<(), String> {
    let magic = read_bytes(bytes, cursor, TRAINER_CHECKPOINT_MAGIC.len())?;
    if magic != TRAINER_CHECKPOINT_MAGIC {
        return Err("checkpoint_magic_mismatch".to_string());
    }
    Ok(())
}

fn push_checkpoint_string(bytes: &mut Vec<u8>, value: &str) {
    push_usize_as_u64(bytes, value.len());
    bytes.extend_from_slice(value.as_bytes());
}

fn read_string(bytes: &[u8], cursor: &mut usize) -> Result<String, String> {
    let len = read_usize(bytes, cursor, "checkpoint_string_len")?;
    let raw = read_bytes(bytes, cursor, len)?;
    String::from_utf8(raw.to_vec()).map_err(|error| format!("checkpoint_string_invalid={error}"))
}

fn push_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn read_u32(bytes: &[u8], cursor: &mut usize) -> Result<u32, String> {
    let raw = read_bytes(bytes, cursor, 4)?;
    Ok(u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]))
}

fn push_usize_as_u64(bytes: &mut Vec<u8>, value: usize) {
    bytes.extend_from_slice(&(value as u64).to_le_bytes());
}

fn read_usize(bytes: &[u8], cursor: &mut usize, name: &str) -> Result<usize, String> {
    let raw = read_bytes(bytes, cursor, 8)?;
    let value = u64::from_le_bytes([
        raw[0], raw[1], raw[2], raw[3], raw[4], raw[5], raw[6], raw[7],
    ]);
    usize::try_from(value).map_err(|_| format!("{name}_overflow={value}"))
}

fn read_f32(bytes: &[u8], cursor: &mut usize) -> Result<f32, String> {
    let raw = read_bytes(bytes, cursor, 4)?;
    let value = f32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]);
    if !value.is_finite() {
        return Err("checkpoint_non_finite_weight".to_string());
    }
    Ok(value)
}

fn read_bytes<'a>(bytes: &'a [u8], cursor: &mut usize, len: usize) -> Result<&'a [u8], String> {
    let end = cursor
        .checked_add(len)
        .ok_or_else(|| "checkpoint_cursor_overflow".to_string())?;
    if end > bytes.len() {
        return Err("checkpoint_unexpected_eof".to_string());
    }
    let slice = &bytes[*cursor..end];
    *cursor = end;
    Ok(slice)
}

fn validate_active_player_count(player_count: usize) -> Result<(), String> {
    if player_count == PLAYER_COUNT_TWO || player_count == PLAYER_COUNT_FOUR {
        Ok(())
    } else {
        Err(format!(
            "invalid_players_per_game={player_count}; allowed={},{}",
            PLAYER_COUNT_TWO, PLAYER_COUNT_FOUR
        ))
    }
}

fn run_cuda_true2d_smoke() -> Result<(), String> {
    let config = AgentConfig::default();
    let shape = True2DTransformerShape {
        layers: config.default_model_layers,
        row_count: config.true2d_internal_rows,
    };
    let model = True2DTransformer::seeded(shape, CUDA_SMOKE_MODEL_ZERO_SEED, &config)
        .map_err(|error| format!("cuda_smoke_model_seed_failed={error:?}"))?;
    let second_model = True2DTransformer::seeded(shape, CUDA_SMOKE_MODEL_ONE_SEED, &config)
        .map_err(|error| format!("cuda_smoke_second_model_seed_failed={error:?}"))?;
    let cuda = cuda_true2d::CudaTrue2D::open_from_environment()?;
    let state = seeded_state(&config, PLAYER_COUNT_TWO, FIRST_GENERATION, 0)?;
    let encoded_zero = encode_state(
        &state.planets,
        &state.initial_planets,
        &[],
        PLAYER_ZERO,
        &config,
    )
    .map_err(|error| format!("cuda_smoke_encode_zero_failed={error:?}"))?;
    let encoded_one = encode_state(
        &state.planets,
        &state.initial_planets,
        &[],
        PLAYER_ONE,
        &config,
    )
    .map_err(|error| format!("cuda_smoke_encode_one_failed={error:?}"))?;
    let smoke_rows = [&encoded_zero.rows, &encoded_one.rows];
    if config.cuda_true2d_smoke_games != smoke_rows.len() {
        return Err("cuda_true2d_smoke_game_count_mismatch".to_string());
    }
    let mut input_rows = Vec::new();
    for rows in smoke_rows {
        input_rows.extend(cuda_true2d::pack_rows(rows, &config)?);
    }
    let gpu_outputs =
        cuda.forward_packed(&input_rows, &model, config.cuda_true2d_smoke_games, &config)?;
    let cpu_zero = model
        .forward(&encoded_zero.rows, &config)
        .map_err(|error| format!("cuda_smoke_cpu_zero_failed={error:?}"))?;
    let cpu_one = model
        .forward(&encoded_one.rows, &config)
        .map_err(|error| format!("cuda_smoke_cpu_one_failed={error:?}"))?;
    let expected = cpu_zero
        .iter()
        .chain(cpu_one.iter())
        .copied()
        .collect::<Vec<_>>();
    let max_abs_diff = action_output_max_abs_diff(&gpu_outputs, &expected)?;
    if max_abs_diff > config.cuda_true2d_forward_tolerance {
        return Err(format!(
            "cuda_true2d_smoke_diff_exceeded={max_abs_diff:.8}; tolerance={:.8}",
            config.cuda_true2d_forward_tolerance
        ));
    }
    let second_cpu_one = second_model
        .forward(&encoded_one.rows, &config)
        .map_err(|error| format!("cuda_smoke_second_cpu_one_failed={error:?}"))?;
    let many_expected = cpu_zero
        .iter()
        .chain(second_cpu_one.iter())
        .copied()
        .collect::<Vec<_>>();
    let many_population = vec![model.clone(), second_model];
    let many_model_indices = vec![0, 1];
    let many_gpu_outputs = cuda.forward_population_packed(
        &input_rows,
        &many_model_indices,
        &many_population,
        &config,
    )?;
    let many_max_abs_diff = action_output_max_abs_diff(&many_gpu_outputs, &many_expected)?;
    if many_max_abs_diff > config.cuda_true2d_forward_tolerance {
        return Err(format!(
            "cuda_true2d_many_smoke_diff_exceeded={many_max_abs_diff:.8}; tolerance={:.8}",
            config.cuda_true2d_forward_tolerance
        ));
    }
    cuda.upload_population(
        &many_population,
        config.cuda_true2d_max_requests_per_forward,
        &config,
    )?;
    let resident_gpu_outputs = cuda.forward_population_resident_packed(
        &input_rows,
        &many_model_indices,
        &many_population,
        &config,
    )?;
    let resident_max_abs_diff = action_output_max_abs_diff(&resident_gpu_outputs, &many_expected)?;
    if resident_max_abs_diff > config.cuda_true2d_forward_tolerance {
        return Err(format!(
            "cuda_true2d_resident_smoke_diff_exceeded={resident_max_abs_diff:.8}; tolerance={:.8}",
            config.cuda_true2d_forward_tolerance
        ));
    }
    println!(
        "status=ok; cuda_true2d_smoke=true; games={}; rows={}; max_abs_diff={max_abs_diff:.8}; many_max_abs_diff={many_max_abs_diff:.8}; resident_max_abs_diff={resident_max_abs_diff:.8}",
        config.cuda_true2d_smoke_games, config.max_rows
    );
    Ok(())
}

fn action_output_max_abs_diff(
    left: &[ActionOutput],
    right: &[ActionOutput],
) -> Result<f32, String> {
    if left.len() != right.len() {
        return Err("action_output_compare_len_mismatch".to_string());
    }
    let mut max_abs_diff = 0.0f32;
    for (left_output, right_output) in left.iter().zip(right.iter()) {
        for (left_target, right_target) in
            left_output.targets.iter().zip(right_output.targets.iter())
        {
            max_abs_diff = max_abs_diff
                .max((left_target.target_fraction - right_target.target_fraction).abs())
                .max((left_target.send_fraction - right_target.send_fraction).abs());
        }
    }
    Ok(max_abs_diff)
}

fn print_reference_scenarios() -> Result<(), String> {
    let scenarios = [
        reference_user_combat()?,
        reference_swept_rotating_planet()?,
        reference_fast_planet_before_bounds()?,
        reference_fast_planet_before_sun()?,
        reference_reinforce()?,
        reference_launch_moves_same_turn()?,
        reference_sun_tangent_survives()?,
    ];
    println!("[");
    for (index, scenario) in scenarios.iter().enumerate() {
        let separator = if index + 1 == scenarios.len() {
            ""
        } else {
            ","
        };
        println!("  {}{}", scenario, separator);
    }
    println!("]");
    Ok(())
}

fn reference_user_combat() -> Result<String, String> {
    let mut config = AgentConfig::default();
    config.fleet_speed_max = 5.0;
    let mut state = SimulationState::new(
        vec![reference_planet(0, -1, 80.0, 80.0, 5.0, 10.0, 0.0)],
        0.01,
    );
    state.step = 1;
    state.next_fleet_id = 4;
    state.fleets = vec![
        reference_fleet(0, 0, 76.0, 80.0, 0.0, 1, 41.0),
        reference_fleet(1, 1, 76.0, 80.0, 0.0, 2, 20.0),
        reference_fleet(2, 1, 76.0, 80.0, 0.0, 2, 20.0),
        reference_fleet(3, 2, 76.0, 80.0, 0.0, 3, 42.0),
    ];
    state
        .step_turn(&[Vec::new(), Vec::new(), Vec::new(), Vec::new()], &config)
        .map_err(|error| format!("reference_user_combat_failed={error:?}"))?;
    Ok(reference_json("combat_user_example", &state))
}

fn reference_swept_rotating_planet() -> Result<String, String> {
    let mut config = AgentConfig::default();
    config.fleet_speed_max = 2.0;
    let mut state = SimulationState::new(
        vec![reference_planet(0, -1, 50.0, 52.0, 1.0, 10.0, 0.0)],
        std::f32::consts::PI,
    );
    state.step = 1;
    state.next_fleet_id = 0;
    state.fleets = vec![reference_fleet(0, 0, 49.0, 50.0, 0.0, 1, 1_000.0)];
    state
        .step_turn(&[Vec::new(), Vec::new()], &config)
        .map_err(|error| format!("reference_swept_rotating_planet_failed={error:?}"))?;
    Ok(reference_json("swept_rotating_planet", &state))
}

fn reference_fast_planet_before_bounds() -> Result<String, String> {
    let config = AgentConfig::default();
    let mut state = SimulationState::new(
        vec![reference_planet(0, 1, 98.0, 50.0, 2.0, 50.0, 1.0)],
        0.01,
    );
    state.step = 1;
    state.next_fleet_id = 1;
    state.fleets = vec![reference_fleet(0, 0, 95.0, 50.0, 0.0, 99, 1_000.0)];
    state
        .step_turn(&[Vec::new(), Vec::new()], &config)
        .map_err(|error| format!("reference_fast_planet_before_bounds_failed={error:?}"))?;
    Ok(reference_json("fast_planet_before_bounds", &state))
}

fn reference_fast_planet_before_sun() -> Result<String, String> {
    let config = AgentConfig::default();
    let mut state = SimulationState::new(
        vec![reference_planet(0, 1, 62.0, 50.0, 2.0, 50.0, 1.0)],
        0.0,
    );
    state.step = 1;
    state.next_fleet_id = 1;
    state.fleets = vec![reference_fleet(
        0,
        0,
        65.0,
        50.0,
        std::f32::consts::PI,
        99,
        1_000.0,
    )];
    state
        .step_turn(&[Vec::new(), Vec::new()], &config)
        .map_err(|error| format!("reference_fast_planet_before_sun_failed={error:?}"))?;
    Ok(reference_json("fast_planet_before_sun", &state))
}

fn reference_reinforce() -> Result<String, String> {
    let config = AgentConfig::default();
    let mut state = SimulationState::new(
        vec![reference_planet(0, 0, 80.0, 50.0, 3.0, 10.0, 1.0)],
        0.01,
    );
    state.step = 1;
    state.next_fleet_id = 1;
    state.fleets = vec![reference_fleet(0, 0, 76.0, 50.0, 0.0, 99, 25.0)];
    state
        .step_turn(&[Vec::new(), Vec::new()], &config)
        .map_err(|error| format!("reference_reinforce_failed={error:?}"))?;
    Ok(reference_json("combat_reinforce", &state))
}

fn reference_launch_moves_same_turn() -> Result<String, String> {
    let config = AgentConfig::default();
    let mut state = SimulationState::new(
        vec![reference_planet(0, 0, 10.0, 10.0, 1.0, 20.0, 1.0)],
        0.01,
    );
    state.step = 1;
    state.next_fleet_id = 100;
    state
        .step_turn(
            &[
                vec![MoveCommand {
                    from_planet_id: 0,
                    direction_angle: 0.0,
                    ship_count: 5,
                }],
                Vec::new(),
            ],
            &config,
        )
        .map_err(|error| format!("reference_launch_moves_same_turn_failed={error:?}"))?;
    Ok(reference_json("launch_moves_same_turn", &state))
}

fn reference_sun_tangent_survives() -> Result<String, String> {
    let mut config = AgentConfig::default();
    config.fleet_speed_max = 20.0;
    let mut state = SimulationState::new(
        vec![reference_planet(0, 0, 10.0, 10.0, 1.0, 20.0, 1.0)],
        0.01,
    );
    state.step = 1;
    state.next_fleet_id = 1;
    state.fleets = vec![reference_fleet(0, 0, 40.0, 40.0, 0.0, 0, 1_000.0)];
    state
        .step_turn(&[Vec::new(), Vec::new()], &config)
        .map_err(|error| format!("reference_sun_tangent_survives_failed={error:?}"))?;
    Ok(reference_json("sun_tangent_survives", &state))
}

fn reference_json(name: &str, state: &SimulationState) -> String {
    format!(
        "{{\"name\":\"{}\",\"planets\":[{}],\"fleets\":[{}]}}",
        name,
        state
            .planets
            .iter()
            .map(reference_planet_json)
            .collect::<Vec<_>>()
            .join(","),
        state
            .fleets
            .iter()
            .map(reference_fleet_json)
            .collect::<Vec<_>>()
            .join(",")
    )
}

fn reference_planet_json(planet: &Planet) -> String {
    format!(
        "[{},{},{:.6},{:.6},{:.6},{:.6},{:.6}]",
        planet.id, planet.owner, planet.x, planet.y, planet.radius, planet.ships, planet.production
    )
}

fn reference_fleet_json(fleet: &orbit_wars_core::Fleet) -> String {
    format!(
        "[{},{},{:.6},{:.6},{:.6},{},{:.6}]",
        fleet.id, fleet.owner, fleet.x, fleet.y, fleet.angle, fleet.from_planet_id, fleet.ships
    )
}

fn reference_planet(
    id: i32,
    owner: i32,
    x: f32,
    y: f32,
    radius: f32,
    ships: f32,
    production: f32,
) -> Planet {
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

fn reference_fleet(
    id: i32,
    owner: i32,
    x: f32,
    y: f32,
    angle: f32,
    from_planet_id: i32,
    ships: f32,
) -> orbit_wars_core::Fleet {
    orbit_wars_core::Fleet {
        id,
        owner,
        x,
        y,
        angle,
        from_planet_id,
        ships,
    }
}

fn evaluate_population(
    generation: usize,
    population: &[True2DTransformer],
    evolution_config: &EvolutionConfig,
    agent_config: &AgentConfig,
    cuda_backend: Option<&cuda_true2d::CudaTrue2D>,
    capture_replays: bool,
    live_replay_telemetry: Option<&LiveReplayTelemetry<'_>>,
) -> Result<PopulationEvaluation, String> {
    let evaluation_started_at = Instant::now();
    require_population_for_player_count(population.len(), evolution_config.active_player_count)?;
    upload_population_to_cuda(cuda_backend, population, agent_config)?;
    let assignments = game_assignments(
        population.len(),
        evolution_config.games_per_model,
        evolution_config.active_player_count,
        agent_config,
    )?;
    let captured_replay_game_count = if capture_replays {
        assignments
            .len()
            .min(agent_config.dashboard_generation_replay_game_count)
    } else {
        0
    };
    let batch_result = run_games_batched(
        generation,
        &assignments,
        population,
        agent_config,
        cuda_backend,
        captured_replay_game_count,
        agent_config.training_capture_trajectories,
        evolution_config.active_player_count,
        live_replay_telemetry,
    )?;
    let mut rewards = vec![0i32; population.len()];
    let mut gameplay_by_model = vec![GameplayStats::default(); population.len()];
    let mut game_latencies_ms = Vec::with_capacity(batch_result.games.len());
    let mut compute_stats = ComputeStats::default();
    for (assignment, game_result) in assignments.iter().zip(batch_result.games.iter()) {
        game_latencies_ms.push(game_result.stats.total_game_ms);
        compute_stats.add_game(game_result.stats);
        for player_slot in 0..evolution_config.active_player_count {
            let model_index = assignment.model_indices[player_slot];
            rewards[model_index] += game_result.rewards[player_slot];
            gameplay_by_model[model_index].add_gameplay(game_result.gameplay[player_slot]);
        }
    }

    let mut evaluated = rewards
        .into_iter()
        .enumerate()
        .map(|(model_index, reward)| EvaluatedModel {
            model_index,
            reward,
            gameplay: gameplay_by_model[model_index],
        })
        .collect::<Vec<_>>();
    evaluated.sort_by(|left, right| {
        right
            .reward
            .cmp(&left.reward)
            .then_with(|| left.model_index.cmp(&right.model_index))
    });
    Ok(PopulationEvaluation {
        models: evaluated,
        game_latencies_ms,
        replay_games: replay_games_from_evaluation(
            generation,
            &assignments,
            &batch_result.games,
            evolution_config.active_player_count,
            captured_replay_game_count,
        ),
        training_samples: batch_result.training_samples,
        elapsed_seconds: elapsed_seconds(evaluation_started_at.elapsed()),
        compute_stats,
        pipeline_stats: batch_result.pipeline_stats,
    })
}

fn require_population_for_player_count(
    population_size: usize,
    active_player_count: usize,
) -> Result<(), String> {
    if population_size < active_player_count {
        return Err(format!(
            "population_size_less_than_players_per_game={population_size}; players={active_player_count}"
        ));
    }
    Ok(())
}

fn game_assignments(
    population_size: usize,
    games_per_model: usize,
    active_player_count: usize,
    config: &AgentConfig,
) -> Result<Vec<GameAssignment>, String> {
    if config.max_players != MAX_PLAYER_SLOTS {
        return Err("trainer_requires_four_owner_slots".to_string());
    }
    validate_active_player_count(active_player_count)?;
    require_population_for_player_count(population_size, active_player_count)?;
    if population_size % active_player_count != 0 {
        return Err(format!(
            "population_size_not_divisible_by_players={population_size}; players={active_player_count}"
        ));
    }
    let game_count = population_size
        .checked_mul(games_per_model)
        .ok_or_else(|| "self_play_participation_count_overflow".to_string())?
        / active_player_count;
    let group_count = population_size / active_player_count;
    let stride = assignment_permutation_stride(population_size, active_player_count)?;
    let mut assignments = Vec::with_capacity(game_count);
    for participation_round in 0..games_per_model {
        for group_index in 0..group_count {
            let mut model_indices = [0usize; MAX_PLAYER_SLOTS];
            for player_slot in 0..active_player_count {
                let rank = group_index * active_player_count + player_slot;
                let model_index = (rank * stride + participation_round) % population_size;
                if model_indices[..player_slot].contains(&model_index) {
                    return Err(format!(
                        "duplicate_model_in_assignment={model_index}; round={participation_round}; group={group_index}"
                    ));
                }
                model_indices[player_slot] = model_index;
            }
            assignments.push(GameAssignment { model_indices });
        }
    }
    Ok(assignments)
}

fn expected_self_play_game_count(
    population_size: usize,
    games_per_model: usize,
    active_player_count: usize,
) -> Result<usize, String> {
    validate_active_player_count(active_player_count)?;
    require_population_for_player_count(population_size, active_player_count)?;
    if population_size % active_player_count != 0 {
        return Err(format!(
            "population_size_not_divisible_by_players={population_size}; players={active_player_count}"
        ));
    }
    Ok(population_size
        .checked_mul(games_per_model)
        .ok_or_else(|| "self_play_participation_count_overflow".to_string())?
        / active_player_count)
}

fn assignment_permutation_stride(
    population_size: usize,
    active_player_count: usize,
) -> Result<usize, String> {
    let first_candidate = active_player_count + 1;
    let last_candidate = first_candidate
        .checked_add(population_size)
        .ok_or_else(|| "assignment_stride_search_overflow".to_string())?;
    for candidate in first_candidate..last_candidate {
        if greatest_common_divisor(candidate, population_size) == 1 {
            return Ok(candidate);
        }
    }
    Err(format!(
        "assignment_coprime_stride_missing={population_size}; players={active_player_count}"
    ))
}

fn greatest_common_divisor(mut left: usize, mut right: usize) -> usize {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

fn opponent_label_for_slot(
    assignment: &GameAssignment,
    active_player_count: usize,
    player_slot: usize,
) -> String {
    assignment.model_indices[..active_player_count]
        .iter()
        .enumerate()
        .filter(|(slot_index, _)| *slot_index != player_slot)
        .map(|(_, model_index)| format!("M-{model_index:03}"))
        .collect::<Vec<_>>()
        .join("/")
}

fn replay_games_from_evaluation(
    generation: usize,
    assignments: &[GameAssignment],
    games: &[GameRunResult],
    active_player_count: usize,
    captured_replay_game_count: usize,
) -> Vec<ReplayGame> {
    if captured_replay_game_count == 0 {
        return Vec::new();
    }
    let captured_game_count = captured_replay_game_count.min(assignments.len());
    let mut replay_games = Vec::with_capacity(captured_game_count * active_player_count);
    for (source_game_index, (assignment, game_result)) in assignments
        .iter()
        .zip(games.iter())
        .take(captured_game_count)
        .enumerate()
    {
        let shared_frames = Arc::new(game_result.frames.clone());
        for player_slot in 0..active_player_count {
            let model_index = assignment.model_indices[player_slot];
            replay_games.push(ReplayGame {
                generation,
                model_index,
                opponent_label: opponent_label_for_slot(
                    assignment,
                    active_player_count,
                    player_slot,
                ),
                game_index: source_game_index,
                reward: game_result.rewards[player_slot],
                frames: Arc::clone(&shared_frames),
            });
        }
    }
    replay_games
}

fn metric_record(
    generation: usize,
    evaluated: &[EvaluatedModel],
    evaluation: &PopulationEvaluation,
    replay_write_stats: &ReplayWriteStats,
    replay_write_seconds: f32,
    generation_validation_games: usize,
    generation_validation_seconds: f32,
    backprop_stats: BackpropStats,
    reproduction_seconds: f32,
    generation_seconds: f32,
) -> MetricRecord {
    let win_rate = evaluated
        .first()
        .map(|model| model.gameplay.win_count as f32 / model.gameplay.game_count.max(1) as f32)
        .unwrap_or(0.0);
    let game_count = evaluation.game_latencies_ms.len() as f32;
    let mut combined_compute_stats = evaluation.compute_stats;
    combined_compute_stats.add_game(replay_write_stats.compute_stats);
    let mut combined_pipeline_stats = evaluation.pipeline_stats;
    combined_pipeline_stats.add(replay_write_stats.pipeline_stats);
    let mut combined_gameplay = GameplayStats::default();
    for model in evaluated {
        combined_gameplay.add_gameplay(model.gameplay);
    }
    let action_call_count = combined_compute_stats.model_action_call_count.max(1) as f32;
    let turn_count = combined_compute_stats.turn_count.max(1) as f32;
    MetricRecord {
        generation,
        win_rate,
        games_per_second: game_count / evaluation.elapsed_seconds,
        turns_per_second: combined_compute_stats.turn_count as f32 / generation_seconds,
        p95_latency_ms: percentile_latency_ms(&evaluation.game_latencies_ms),
        gpu_utilization: UNMEASURED_GPU_UTILIZATION_PERCENT,
        evaluated_games: evaluation.compute_stats.game_count,
        sampled_replay_games: replay_write_stats.compute_stats.game_count,
        model_action_calls: combined_compute_stats.model_action_call_count,
        launch_actions: combined_compute_stats.launch_action_count,
        launched_ships: combined_compute_stats.launched_ship_count,
        captures: combined_gameplay.capture_count,
        fleet_hits: combined_gameplay.fleet_hit_count,
        hit_ships: combined_gameplay.hit_ship_count,
        sun_destroyed_fleets: combined_gameplay.sun_destroyed_fleet_count,
        sun_destroyed_ships: combined_gameplay.sun_destroyed_ship_count,
        avg_fleet_size: combined_gameplay.avg_fleet_size(),
        avg_launch_actions_per_turn: combined_compute_stats.launch_action_count as f32 / turn_count,
        avg_launched_ships_per_turn: combined_compute_stats.launched_ship_count as f32 / turn_count,
        avg_model_action_ms: combined_compute_stats.model_action_ms / action_call_count,
        inference_batch_calls: combined_pipeline_stats.inference_batch_call_count,
        max_inference_batch_size: combined_pipeline_stats.max_inference_batch_size,
        simultaneous_games: combined_pipeline_stats.simultaneous_game_count,
        model_action_seconds: seconds_from_millis(combined_compute_stats.model_action_ms),
        simulation_step_seconds: seconds_from_millis(combined_compute_stats.simulation_step_ms),
        evaluation_seconds: evaluation.elapsed_seconds,
        replay_seconds: replay_write_stats.elapsed_seconds,
        replay_write_seconds,
        generation_validation_games,
        generation_validation_seconds,
        backprop_samples: backprop_stats.sample_count,
        backprop_models: backprop_stats.model_count,
        backprop_seconds: backprop_stats.elapsed_seconds,
        reproduction_seconds,
        generation_seconds,
    }
}

fn percentile_latency_ms(latencies_ms: &[f32]) -> f32 {
    if latencies_ms.is_empty() {
        return 0.0;
    }
    let mut sorted = latencies_ms.to_vec();
    sorted.sort_by(|left, right| left.total_cmp(right));
    let rank_index =
        sorted.len().saturating_sub(1) * P95_PERCENTILE_NUMERATOR / PERCENT_DENOMINATOR;
    sorted[rank_index]
}

fn duration_to_millis(duration: Duration) -> f32 {
    (duration.as_secs_f64() * MILLISECONDS_PER_SECOND) as f32
}

fn elapsed_seconds(duration: Duration) -> f32 {
    duration.as_secs_f32().max(f32::EPSILON)
}

fn seconds_from_millis(milliseconds: f32) -> f32 {
    milliseconds / MILLISECONDS_PER_SECOND as f32
}

impl ComputeStats {
    fn add_game(&mut self, game_stats: ComputeStats) {
        self.game_count += game_stats.game_count;
        self.turn_count += game_stats.turn_count;
        self.model_action_call_count += game_stats.model_action_call_count;
        self.launch_action_count += game_stats.launch_action_count;
        self.launched_ship_count += game_stats.launched_ship_count;
        self.model_action_ms += game_stats.model_action_ms;
        self.simulation_step_ms += game_stats.simulation_step_ms;
        self.total_game_ms += game_stats.total_game_ms;
    }
}

impl PipelineStats {
    fn add(&mut self, stats: PipelineStats) {
        self.inference_batch_call_count += stats.inference_batch_call_count;
        self.max_inference_batch_size = self
            .max_inference_batch_size
            .max(stats.max_inference_batch_size);
        self.simultaneous_game_count = self
            .simultaneous_game_count
            .max(stats.simultaneous_game_count);
    }

    fn record_batch(&mut self, batch_size: usize) {
        self.inference_batch_call_count += 1;
        self.max_inference_batch_size = self.max_inference_batch_size.max(batch_size);
    }
}

impl TrainingSamples {
    fn with_capacity(sample_capacity: usize, config: &AgentConfig) -> Result<Self, String> {
        let input_capacity = sample_capacity
            .checked_mul(training_sample_input_stride(config)?)
            .ok_or_else(|| "training_sample_input_capacity_overflow".to_string())?;
        let output_capacity = sample_capacity
            .checked_mul(training_sample_output_stride(config)?)
            .ok_or_else(|| "training_sample_output_capacity_overflow".to_string())?;
        let sun_target_penalty_capacity = sample_capacity
            .checked_mul(training_sun_target_penalty_stride(config)?)
            .ok_or_else(|| "training_sun_target_penalty_capacity_overflow".to_string())?;
        Ok(Self {
            metadata: Vec::with_capacity(sample_capacity),
            rewards: Vec::new(),
            sun_target_penalties: Vec::with_capacity(sun_target_penalty_capacity),
            input_rows: Vec::with_capacity(input_capacity),
            output_rows: Vec::with_capacity(output_capacity),
        })
    }

    fn sample_count(&self) -> usize {
        self.metadata.len()
    }

    fn input_float_count(&self) -> usize {
        self.input_rows.len()
    }

    fn output_float_count(&self) -> usize {
        self.output_rows.len()
    }

    fn record(
        &mut self,
        generation: usize,
        request: &ActionRequest,
        outputs: &[ActionOutput],
        config: &AgentConfig,
    ) -> Result<(), String> {
        if outputs.len() != config.max_rows {
            return Err("training_sample_output_row_count_mismatch".to_string());
        }
        cuda_true2d::pack_rows_into(&request.rows, config, &mut self.input_rows)?;
        for output in outputs {
            for target in output.targets {
                self.output_rows.push(target.target_fraction);
                self.output_rows.push(target.send_fraction);
            }
        }
        let penalty_stride = training_sun_target_penalty_stride(config)?;
        self.sun_target_penalties
            .resize(self.sun_target_penalties.len() + penalty_stride, 0.0);
        self.metadata.push(TrainingSampleMetadata {
            generation,
            game_index: request.game_index,
            player_slot: request.player_slot,
            player_id: request.player_id,
            model_index: request.model_index,
            step: request.step,
        });
        Ok(())
    }

    fn assign_rewards(&mut self, games: &[GameRunResult]) -> Result<(), String> {
        self.rewards.clear();
        self.rewards.reserve(self.metadata.len());
        for sample in &self.metadata {
            let game = games.get(sample.game_index).ok_or_else(|| {
                format!(
                    "training_sample_game_index_out_of_range={}",
                    sample.game_index
                )
            })?;
            if sample.player_slot >= MAX_PLAYER_SLOTS {
                return Err(format!(
                    "training_sample_player_slot_out_of_range={}",
                    sample.player_slot
                ));
            }
            self.rewards.push(game.rewards[sample.player_slot]);
        }
        Ok(())
    }

    fn add_sun_target_penalty(
        &mut self,
        sample_index: usize,
        source_row: usize,
        target_slot: usize,
        penalty: f32,
        config: &AgentConfig,
    ) -> Result<(), String> {
        if sample_index >= self.sample_count() {
            return Err(format!(
                "sun_penalty_sample_index_out_of_range={sample_index}"
            ));
        }
        if source_row >= config.max_rows {
            return Err(format!("sun_penalty_source_row_out_of_range={source_row}"));
        }
        if target_slot >= config.true2d_action_targets_per_source {
            return Err(format!(
                "sun_penalty_target_slot_out_of_range={target_slot}"
            ));
        }
        if !penalty.is_finite() {
            return Err("sun_penalty_non_finite".to_string());
        }
        let penalty_stride = training_sun_target_penalty_stride(config)?;
        let source_offset = source_row
            .checked_mul(config.true2d_action_targets_per_source)
            .ok_or_else(|| "sun_penalty_source_offset_overflow".to_string())?;
        let offset = sample_index
            .checked_mul(penalty_stride)
            .and_then(|value| value.checked_add(source_offset))
            .and_then(|value| value.checked_add(target_slot))
            .ok_or_else(|| "sun_penalty_offset_overflow".to_string())?;
        let Some(target_penalty) = self.sun_target_penalties.get_mut(offset) else {
            return Err("sun_penalty_offset_out_of_range".to_string());
        };
        *target_penalty += penalty;
        Ok(())
    }
}

fn training_sample_input_stride(config: &AgentConfig) -> Result<usize, String> {
    config
        .max_rows
        .checked_mul(config.true2d_input_features)
        .ok_or_else(|| "training_sample_input_stride_overflow".to_string())
}

fn training_sample_output_stride(config: &AgentConfig) -> Result<usize, String> {
    config
        .max_rows
        .checked_mul(config.true2d_output_features)
        .ok_or_else(|| "training_sample_output_stride_overflow".to_string())
}

fn training_sun_target_penalty_stride(config: &AgentConfig) -> Result<usize, String> {
    config
        .max_rows
        .checked_mul(config.true2d_action_targets_per_source)
        .ok_or_else(|| "training_sun_target_penalty_stride_overflow".to_string())
}

fn training_sample_capacity(
    game_count: usize,
    active_player_count: usize,
    config: &AgentConfig,
) -> Result<usize, String> {
    game_count
        .checked_mul(active_player_count)
        .and_then(|value| value.checked_mul(config.episode_steps))
        .ok_or_else(|| "training_sample_capacity_overflow".to_string())
}

impl GameplayStats {
    fn add_gameplay(&mut self, gameplay: GameplayStats) {
        self.game_count += gameplay.game_count;
        self.win_count += gameplay.win_count;
        self.draw_count += gameplay.draw_count;
        self.loss_count += gameplay.loss_count;
        self.capture_count += gameplay.capture_count;
        self.fleet_hit_count += gameplay.fleet_hit_count;
        self.hit_ship_count += gameplay.hit_ship_count;
        self.sun_destroyed_fleet_count += gameplay.sun_destroyed_fleet_count;
        self.sun_destroyed_ship_count += gameplay.sun_destroyed_ship_count;
        self.out_of_bounds_destroyed_fleet_count += gameplay.out_of_bounds_destroyed_fleet_count;
        self.out_of_bounds_destroyed_ship_count += gameplay.out_of_bounds_destroyed_ship_count;
        self.launch_action_count += gameplay.launch_action_count;
        self.launched_ship_count += gameplay.launched_ship_count;
        self.fleet_ship_observation_sum += gameplay.fleet_ship_observation_sum;
        self.fleet_observation_count += gameplay.fleet_observation_count;
    }

    fn add_step_events(&mut self, events: PlayerStepEvents) {
        self.capture_count += events.captured_planet_count;
        self.fleet_hit_count += events.hit_fleet_count;
        self.hit_ship_count += events.hit_ship_count;
        self.sun_destroyed_fleet_count += events.sun_destroyed_fleet_count;
        self.sun_destroyed_ship_count += events.sun_destroyed_ship_count;
        self.out_of_bounds_destroyed_fleet_count += events.out_of_bounds_destroyed_fleet_count;
        self.out_of_bounds_destroyed_ship_count += events.out_of_bounds_destroyed_ship_count;
        self.launch_action_count += events.launched_fleet_count;
        self.launched_ship_count += events.launched_ship_count;
    }

    fn add_reward(&mut self, reward: i32, config: &AgentConfig) {
        self.game_count += 1;
        if reward == config.training_reward_win {
            self.win_count += 1;
        } else if reward == config.training_reward_loss {
            self.loss_count += 1;
        } else {
            self.draw_count += 1;
        }
    }

    fn add_fleet_observations(&mut self, state: &SimulationState, player: i32) {
        for fleet in state.fleets.iter().filter(|fleet| fleet.owner == player) {
            self.fleet_ship_observation_sum += fleet.ships;
            self.fleet_observation_count += 1;
        }
    }

    fn win_rate(&self) -> f32 {
        self.win_count as f32 / self.game_count.max(1) as f32
    }

    fn avg_fleet_size(&self) -> f32 {
        if self.fleet_observation_count == 0 {
            0.0
        } else {
            self.fleet_ship_observation_sum / self.fleet_observation_count as f32
        }
    }
}

fn tournament_submit_candidate(champion_scores: &[EvaluatedModel]) -> Option<&EvaluatedModel> {
    champion_scores.iter().max_by(|left, right| {
        left.gameplay
            .win_rate()
            .partial_cmp(&right.gameplay.win_rate())
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.reward.cmp(&right.reward))
            .then_with(|| right.model_index.cmp(&left.model_index))
    })
}

fn run_id() -> Result<String, String> {
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| format!("run_id_time_failed={error}"))?;
    Ok(duration.as_secs().to_string())
}

fn empty_replay_write_stats() -> ReplayWriteStats {
    ReplayWriteStats {
        elapsed_seconds: 0.0,
        compute_stats: ComputeStats::default(),
        pipeline_stats: PipelineStats::default(),
    }
}

fn replay_write_stats_from_evaluation(replay_games: &[ReplayGame]) -> ReplayWriteStats {
    let mut compute_stats = ComputeStats::default();
    compute_stats.game_count = replay_games.len();
    ReplayWriteStats {
        elapsed_seconds: 0.0,
        compute_stats,
        pipeline_stats: PipelineStats::default(),
    }
}

fn train_elite_models_with_backprop(
    population: &mut [True2DTransformer],
    evaluated: &[EvaluatedModel],
    training_samples: &TrainingSamples,
    evolution_config: &EvolutionConfig,
    agent_config: &AgentConfig,
    cuda_backend: Option<&cuda_true2d::CudaTrue2D>,
) -> Result<BackpropStats, String> {
    if evaluated.len() < evolution_config.elite_count {
        return Err("backprop_elite_count_exceeds_evaluated_population".to_string());
    }
    let cuda = cuda_backend.ok_or_else(|| "backprop_requires_cuda_backend".to_string())?;
    let started_at = Instant::now();
    let mut selected_model_slots = vec![None; population.len()];
    let mut selected_model_indices = Vec::with_capacity(evolution_config.elite_count);
    for evaluated_model in evaluated.iter().take(evolution_config.elite_count) {
        let selected = selected_model_slots
            .get_mut(evaluated_model.model_index)
            .ok_or_else(|| {
                format!(
                    "backprop_selected_model_out_of_range={}",
                    evaluated_model.model_index
                )
            })?;
        if selected.is_some() {
            return Err(format!(
                "backprop_duplicate_selected_model={}",
                evaluated_model.model_index
            ));
        }
        *selected = Some(selected_model_indices.len());
        selected_model_indices.push(evaluated_model.model_index);
    }
    let training_data = training_samples_for_selected_models(
        training_samples,
        &selected_model_slots,
        agent_config,
    )?;
    let mut selected_population = Vec::with_capacity(selected_model_indices.len());
    for model_index in selected_model_indices.iter().copied() {
        selected_population.push(population[model_index].clone());
    }
    cuda.train_full_attention_population(
        &training_data.input_rows,
        &training_data.output_rows,
        &training_data.rewards,
        &training_data.sun_target_penalties,
        &training_data.model_indices,
        &mut selected_population,
        agent_config,
    )?;
    for (compact_index, model_index) in selected_model_indices.iter().copied().enumerate() {
        population[model_index] = selected_population[compact_index].clone();
    }
    Ok(BackpropStats {
        sample_count: training_data.rewards.len(),
        model_count: training_data.model_count,
        elapsed_seconds: elapsed_seconds(started_at.elapsed()),
    })
}

fn training_samples_for_selected_models(
    training_samples: &TrainingSamples,
    selected_model_slots: &[Option<usize>],
    config: &AgentConfig,
) -> Result<BackpropTrainingData, String> {
    let sample_count = training_samples.sample_count();
    if training_samples.rewards.len() != sample_count {
        return Err("backprop_reward_count_mismatch".to_string());
    }
    let input_stride = training_sample_input_stride(config)?;
    let output_stride = training_sample_output_stride(config)?;
    let sun_target_penalty_stride = training_sun_target_penalty_stride(config)?;
    if training_samples.input_rows.len() != sample_count * input_stride {
        return Err("backprop_input_float_count_mismatch".to_string());
    }
    if training_samples.output_rows.len() != sample_count * output_stride {
        return Err("backprop_output_float_count_mismatch".to_string());
    }
    if training_samples.sun_target_penalties.len() != sample_count * sun_target_penalty_stride {
        return Err("backprop_sun_target_penalty_count_mismatch".to_string());
    }

    let compact_model_count = selected_model_slots
        .iter()
        .filter(|model_slot| model_slot.is_some())
        .count();
    if compact_model_count == 0 {
        return Err("backprop_selected_models_empty".to_string());
    }
    let mut included_models = vec![false; compact_model_count];
    let mut input_rows = Vec::new();
    let mut output_rows = Vec::new();
    let mut rewards = Vec::new();
    let mut sun_target_penalties = Vec::new();
    let mut model_indices = Vec::new();
    for (sample_index, metadata) in training_samples.metadata.iter().enumerate() {
        let model_slot = selected_model_slots
            .get(metadata.model_index)
            .ok_or_else(|| {
                format!(
                    "backprop_sample_model_out_of_range={}",
                    metadata.model_index
                )
            })?;
        let compact_model_index = match *model_slot {
            Some(model_index) => model_index,
            None => continue,
        };
        let included_model = included_models
            .get_mut(compact_model_index)
            .ok_or_else(|| format!("backprop_compact_model_out_of_range={compact_model_index}"))?;
        *included_model = true;
        let input_start = sample_index * input_stride;
        let input_end = input_start + input_stride;
        let output_start = sample_index * output_stride;
        let output_end = output_start + output_stride;
        let penalty_start = sample_index * sun_target_penalty_stride;
        let penalty_end = penalty_start + sun_target_penalty_stride;
        input_rows.extend_from_slice(&training_samples.input_rows[input_start..input_end]);
        output_rows.extend_from_slice(&training_samples.output_rows[output_start..output_end]);
        rewards.push(training_samples.rewards[sample_index]);
        sun_target_penalties
            .extend_from_slice(&training_samples.sun_target_penalties[penalty_start..penalty_end]);
        model_indices.push(compact_model_index);
    }
    if rewards.is_empty() {
        return Err("backprop_training_samples_empty".to_string());
    }
    Ok(BackpropTrainingData {
        input_rows,
        output_rows,
        rewards,
        sun_target_penalties,
        model_indices,
        model_count: included_models.iter().filter(|selected| **selected).count(),
    })
}

fn reproduce_population(
    population: &[True2DTransformer],
    evaluated: &[EvaluatedModel],
    evolution_config: &EvolutionConfig,
    agent_config: &AgentConfig,
) -> Result<Vec<True2DTransformer>, String> {
    if evaluated.len() < evolution_config.elite_count {
        return Err("elite_count exceeds evaluated population".to_string());
    }
    let parents = evaluated
        .iter()
        .take(evolution_config.elite_count)
        .map(|result| ReproductionParent {
            model: population[result.model_index].clone(),
            score: result.clone(),
        })
        .collect::<Vec<_>>();
    reproduce_from_weighted_parents(&parents, evolution_config, agent_config)
}

fn reproduce_population_from_champion_scores(
    champion_archive: &mut Vec<GenerationChampion>,
    champion_scores: &[EvaluatedModel],
    replay_history: &mut Vec<ReplayChunkRecord>,
    run_id: &str,
    evolution_config: &EvolutionConfig,
    agent_config: &AgentConfig,
) -> Result<Vec<True2DTransformer>, String> {
    if champion_scores.len() < evolution_config.elite_count {
        return Err(format!(
            "generation_tournament_elite_count_exceeds_scores={}; scores={}",
            evolution_config.elite_count,
            champion_scores.len()
        ));
    }
    let selected_indices = champion_scores
        .iter()
        .take(evolution_config.elite_count)
        .map(|score| score.model_index)
        .collect::<BTreeSet<_>>();
    let retained_generations = selected_indices
        .iter()
        .map(|index| champion_archive[*index].generation)
        .collect::<BTreeSet<_>>();
    let parents = champion_scores
        .iter()
        .take(evolution_config.elite_count)
        .map(|score| ReproductionParent {
            model: champion_archive[score.model_index].model.clone(),
            score: score.clone(),
        })
        .collect::<Vec<_>>();
    *champion_archive = champion_scores
        .iter()
        .take(evolution_config.elite_count)
        .map(|score| champion_archive[score.model_index].clone())
        .collect::<Vec<_>>();
    prune_replay_history_and_files(replay_history, run_id, &retained_generations)?;
    reproduce_from_weighted_parents(&parents, evolution_config, agent_config)
}

fn reproduce_from_weighted_parents(
    parents: &[ReproductionParent],
    evolution_config: &EvolutionConfig,
    agent_config: &AgentConfig,
) -> Result<Vec<True2DTransformer>, String> {
    if parents.is_empty() {
        return Err("reproduction_parents_empty".to_string());
    }
    let parent_slots = reproduction_parent_slot_counts(parents, evolution_config, agent_config)?;
    let mut next_population = parents
        .iter()
        .map(|parent| parent.model.clone())
        .collect::<Vec<_>>();
    let child_parent_sequence = reproduction_child_parent_sequence(&parent_slots)?;
    for (sequence_index, left_parent_index) in child_parent_sequence.into_iter().enumerate() {
        if next_population.len() >= evolution_config.population_size {
            break;
        }
        let child_index = next_population.len();
        let right_parent_index = if parents.len() == 1 {
            left_parent_index
        } else {
            (left_parent_index + sequence_index + 1) % parents.len()
        };
        let left = &parents[left_parent_index].model;
        let right = &parents[right_parent_index].model;
        let crossed =
            True2DTransformer::crossover(left, right, child_index as u64 + CROSSOVER_SEED_OFFSET)
                .map_err(|error| format!("crossover_failed={error:?}"))?;
        next_population.push(crossed.mutate(
            child_index as u64 + MUTATION_SEED_OFFSET,
            agent_config.default_mutation_sigma,
        ));
    }
    if next_population.len() != evolution_config.population_size {
        return Err(format!(
            "weighted_reproduction_population_mismatch={}; expected={}",
            next_population.len(),
            evolution_config.population_size
        ));
    }
    Ok(next_population)
}

fn reproduction_parent_slot_counts(
    parents: &[ReproductionParent],
    evolution_config: &EvolutionConfig,
    agent_config: &AgentConfig,
) -> Result<Vec<usize>, String> {
    if parents.len() != evolution_config.elite_count {
        return Err(format!(
            "reproduction_parent_count_mismatch={}; expected={}",
            parents.len(),
            evolution_config.elite_count
        ));
    }
    let min_slots = agent_config
        .training_reproduction_min_slots_per_elite
        .min(evolution_config.population_size / parents.len())
        .max(1);
    let max_slots = agent_config.training_reproduction_max_slots_per_elite;
    let minimum_population = min_slots
        .checked_mul(parents.len())
        .ok_or_else(|| "reproduction_minimum_population_overflow".to_string())?;
    let maximum_population = max_slots
        .checked_mul(parents.len())
        .ok_or_else(|| "reproduction_maximum_population_overflow".to_string())?;
    if minimum_population > evolution_config.population_size {
        return Err(format!(
            "reproduction_minimum_slots_exceed_population={minimum_population}; population={}",
            evolution_config.population_size
        ));
    }
    if maximum_population < evolution_config.population_size {
        return Err(format!(
            "reproduction_maximum_slots_below_population={maximum_population}; population={}",
            evolution_config.population_size
        ));
    }

    let mut slots = vec![min_slots; parents.len()];
    let mut remaining_slots = evolution_config.population_size - minimum_population;
    if remaining_slots == 0 {
        return Ok(slots);
    }

    let weights = parents
        .iter()
        .map(|parent| reproduction_parent_weight(&parent.score, agent_config))
        .collect::<Result<Vec<_>, _>>()?;
    let total_weight = weights.iter().sum::<f32>();
    if !total_weight.is_finite() || total_weight <= 0.0 {
        return Err("reproduction_weight_sum_invalid".to_string());
    }
    let mut remainders = Vec::with_capacity(parents.len());
    for (index, weight) in weights.iter().enumerate() {
        let available_slots = max_slots - min_slots;
        let ideal_extra = remaining_slots as f32 * *weight / total_weight;
        let extra_slots = (ideal_extra.floor() as usize).min(available_slots);
        slots[index] += extra_slots;
        remainders.push((index, ideal_extra - extra_slots as f32, *weight));
    }
    let assigned_extra_slots = slots.iter().map(|slot| slot - min_slots).sum::<usize>();
    remaining_slots = remaining_slots
        .checked_sub(assigned_extra_slots)
        .ok_or_else(|| "reproduction_remaining_slots_underflow".to_string())?;
    remainders.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| right.2.total_cmp(&left.2))
            .then_with(|| left.0.cmp(&right.0))
    });
    while remaining_slots > 0 {
        let mut progressed = false;
        for (index, _, _) in &remainders {
            if slots[*index] < max_slots {
                slots[*index] += 1;
                remaining_slots -= 1;
                progressed = true;
                if remaining_slots == 0 {
                    break;
                }
            }
        }
        if !progressed {
            return Err("reproduction_slots_capped_before_population_filled".to_string());
        }
    }
    Ok(slots)
}

fn reproduction_parent_weight(
    score: &EvaluatedModel,
    agent_config: &AgentConfig,
) -> Result<f32, String> {
    let game_count = score.gameplay.game_count as f32;
    let smoothed_winrate = (score.gameplay.win_count as f32
        + agent_config.training_reproduction_winrate_smoothing_wins)
        / (game_count + agent_config.training_reproduction_winrate_smoothing_games);
    let reward_score = if score.gameplay.game_count == 0 {
        0.0
    } else {
        let min_reward = game_count * agent_config.training_reward_loss as f32;
        let max_reward = game_count * agent_config.training_reward_win as f32;
        (score.reward as f32 - min_reward) / (max_reward - min_reward)
    };
    let weight = smoothed_winrate + reward_score.max(0.0);
    if !weight.is_finite() || weight <= 0.0 {
        return Err("reproduction_parent_weight_invalid".to_string());
    }
    Ok(weight)
}

fn reproduction_child_parent_sequence(parent_slots: &[usize]) -> Result<Vec<usize>, String> {
    let mut remaining_children = parent_slots
        .iter()
        .map(|slots| {
            slots
                .checked_sub(1)
                .ok_or_else(|| "reproduction_parent_slot_below_elite_copy".to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    let child_count = remaining_children.iter().sum::<usize>();
    let mut sequence = Vec::with_capacity(child_count);
    while sequence.len() < child_count {
        let mut progressed = false;
        for (parent_index, remaining) in remaining_children.iter_mut().enumerate() {
            if *remaining > 0 {
                sequence.push(parent_index);
                *remaining -= 1;
                progressed = true;
            }
        }
        if !progressed {
            return Err("reproduction_child_sequence_stalled".to_string());
        }
    }
    Ok(sequence)
}

fn prune_replay_history_and_files(
    replay_history: &mut Vec<ReplayChunkRecord>,
    run_id: &str,
    retained_generations: &BTreeSet<usize>,
) -> Result<(), String> {
    let existing_generations = replay_history
        .iter()
        .map(|record| record.generation)
        .collect::<BTreeSet<_>>();
    replay_history.retain(|record| retained_generations.contains(&record.generation));
    for generation in existing_generations {
        if !retained_generations.contains(&generation) {
            remove_replay_artifacts(run_id, generation)?;
        }
    }
    Ok(())
}

fn remove_replay_artifacts(run_id: &str, generation: usize) -> Result<(), String> {
    let paths = [
        replay_chunk_path(run_id, generation),
        live_replay_file_path(run_id, generation),
        legacy_live_replay_slot_file_path(run_id, generation),
        generation_top_models_path(run_id, generation),
    ];
    for path_string in paths {
        let path = Path::new(&path_string);
        if path.exists() {
            fs::remove_file(path)
                .map_err(|error| format!("replay_prune_failed={path_string}; error={error}"))?;
        }
    }
    Ok(())
}

fn write_generation_top_models_artifact(
    run_id: &str,
    generation: usize,
    evaluated: &[EvaluatedModel],
    population: &[True2DTransformer],
    model_count: usize,
) -> Result<(), String> {
    if evaluated.len() < model_count {
        return Err(format!(
            "top_model_artifact_count_exceeds_evaluated={model_count}; evaluated={}",
            evaluated.len()
        ));
    }
    let path_string = generation_top_models_path(run_id, generation);
    let path = Path::new(&path_string);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|error| format!("top_models_dir_failed={error}"))?;
    }
    let mut bytes = Vec::new();
    bytes.extend_from_slice(GENERATION_TOP_MODELS_MAGIC);
    push_usize_as_u64(&mut bytes, generation);
    push_usize_as_u64(&mut bytes, model_count);
    for evaluated_model in evaluated.iter().take(model_count) {
        push_usize_as_u64(&mut bytes, evaluated_model.model_index);
        push_i32(&mut bytes, evaluated_model.reward);
        push_transformer(&mut bytes, &population[evaluated_model.model_index])?;
    }
    write_binary_with_retries(path, &bytes, "top_models_write_failed")
}

fn generation_top_models_path(run_id: &str, generation: usize) -> String {
    format!("dashboard/public/telemetry/generation_models/{run_id}/generation_{generation}_top4.owmodels")
}

fn generation_top_models_public_path(run_id: &str, generation: usize) -> String {
    format!("/telemetry/generation_models/{run_id}/generation_{generation}_top4.owmodels")
}

fn stage_generation_tournament_top_winrate_model_for_host_submit(
    generation: usize,
    run_id: &str,
    champion_archive: &[GenerationChampion],
    champion_scores: &[EvaluatedModel],
    agent_config: &AgentConfig,
) -> Result<KaggleStageOutcome, String> {
    let candidate = tournament_submit_candidate(champion_scores)
        .ok_or_else(|| "kaggle_stage_no_champion_scores".to_string())?;
    let champion = champion_archive.get(candidate.model_index).ok_or_else(|| {
        format!(
            "kaggle_stage_model_index_out_of_range={}; archive={}",
            candidate.model_index,
            champion_archive.len()
        )
    })?;
    let stage_dir = Path::new(agent_config.kaggle_submission_stage_dir)
        .join(run_id)
        .join(format!("generation_{generation}"));
    fs::create_dir_all(&stage_dir).map_err(|error| format!("kaggle_stage_dir_failed={error}"))?;

    let staged_main_path = stage_dir.join(KAGGLE_SUBMISSION_MAIN_FILE);
    let staged_library_path = stage_dir.join(KAGGLE_SUBMISSION_LIBRARY_FILE);
    let staged_model_path = stage_dir.join(KAGGLE_SUBMISSION_MODEL_FILE);
    let staged_message_path = stage_dir.join(KAGGLE_SUBMISSION_MESSAGE_FILE);
    let staged_ready_path = stage_dir.join(KAGGLE_SUBMISSION_READY_FILE);
    fs::copy(agent_config.kaggle_submission_main_path, &staged_main_path)
        .map_err(|error| format!("kaggle_stage_main_copy_failed={error}"))?;
    fs::copy(
        agent_config.kaggle_submission_library_path,
        &staged_library_path,
    )
    .map_err(|error| format!("kaggle_stage_library_copy_failed={error}"))?;
    let model_bytes = true2d_model_binary_bytes(&champion.model)
        .map_err(|error| format!("kaggle_stage_model_serialize_failed={error:?}"))?;
    fs::write(&staged_model_path, model_bytes)
        .map_err(|error| format!("kaggle_stage_model_write_failed={error}"))?;

    let archive_path = stage_dir.join(agent_config.kaggle_submission_archive_name);
    let tar_output = Command::new(TAR_COMMAND)
        .arg(TAR_CREATE_GZIP_ARG)
        .arg(&archive_path)
        .arg(TAR_DIRECTORY_ARG)
        .arg(&stage_dir)
        .arg(KAGGLE_SUBMISSION_MAIN_FILE)
        .arg(KAGGLE_SUBMISSION_LIBRARY_FILE)
        .arg(KAGGLE_SUBMISSION_MODEL_FILE)
        .output()
        .map_err(|error| format!("kaggle_stage_tar_command_failed={error}"))?;
    if !tar_output.status.success() {
        return Err(format!(
            "kaggle_stage_tar_failed; status={}; stdout={}; stderr={}",
            tar_output.status,
            command_output_snippet(&tar_output.stdout),
            command_output_snippet(&tar_output.stderr)
        ));
    }

    let submit_message = kaggle_submit_message(generation, run_id, candidate);
    fs::write(&staged_message_path, &submit_message)
        .map_err(|error| format!("kaggle_stage_message_write_failed={error}"))?;
    fs::write(
        &staged_ready_path,
        format!(
            "competition={}\narchive={}\nmessage_file={}\nrun_id={run_id}\ngeneration={generation}\nmodel_index={}\nwin_rate={:.6}\nrecord={}-{}-{}\nreward={}\n",
            agent_config.kaggle_competition_slug,
            archive_path.display(),
            staged_message_path.display(),
            candidate.model_index,
            candidate.gameplay.win_rate(),
            candidate.gameplay.win_count,
            candidate.gameplay.draw_count,
            candidate.gameplay.loss_count,
            candidate.reward
        ),
    )
    .map_err(|error| format!("kaggle_stage_ready_write_failed={error}"))?;

    Ok(KaggleStageOutcome {
        archive_path: archive_path.display().to_string(),
        message_path: staged_message_path.display().to_string(),
        ready_path: staged_ready_path.display().to_string(),
        model_index: candidate.model_index,
        win_rate: candidate.gameplay.win_rate(),
        reward: candidate.reward,
        game_count: candidate.gameplay.game_count,
        win_count: candidate.gameplay.win_count,
        draw_count: candidate.gameplay.draw_count,
        loss_count: candidate.gameplay.loss_count,
    })
}

fn kaggle_submit_message(generation: usize, run_id: &str, candidate: &EvaluatedModel) -> String {
    format!(
        "gen_tournament_top1 run={run_id} gen={generation} model={} winrate={:.3} record={}-{}-{} reward={}",
        candidate.model_index,
        candidate.gameplay.win_rate(),
        candidate.gameplay.win_count,
        candidate.gameplay.draw_count,
        candidate.gameplay.loss_count,
        candidate.reward
    )
}

fn command_output_snippet(bytes: &[u8]) -> String {
    let end = bytes.len().min(COMMAND_OUTPUT_SNIPPET_BYTES);
    let normalized = String::from_utf8_lossy(&bytes[..end])
        .chars()
        .map(|character| {
            if character.is_control() {
                ' '
            } else {
                character
            }
        })
        .collect::<String>()
        .replace(';', ",");
    let trimmed = normalized.trim();
    if trimmed.is_empty() {
        "empty".to_string()
    } else {
        trimmed.to_string()
    }
}

fn archive_generation_champions(
    champion_archive: &mut Vec<GenerationChampion>,
    generation: usize,
    evaluated: &[EvaluatedModel],
    population: &[True2DTransformer],
    agent_config: &AgentConfig,
) -> Result<(), String> {
    let champion_count = agent_config.generation_archive_champions_per_generation;
    if evaluated.len() < champion_count {
        return Err(format!(
            "generation_champion_count_exceeds_evaluated_models={champion_count}; evaluated={}",
            evaluated.len()
        ));
    }
    if population.len() > agent_config.generation_validation_max_models {
        return Err(format!(
            "population_size_exceeds_generation_validation_cap={}; cap={}",
            population.len(),
            agent_config.generation_validation_max_models
        ));
    }
    champion_archive.retain(|champion| champion.generation != generation);
    for evaluated_model in evaluated.iter().take(champion_count) {
        champion_archive.push(GenerationChampion {
            generation,
            model: population[evaluated_model.model_index].clone(),
        });
    }
    prune_generation_archive(champion_archive, generation, agent_config);
    Ok(())
}

fn prune_generation_archive(
    champion_archive: &mut Vec<GenerationChampion>,
    current_generation: usize,
    agent_config: &AgentConfig,
) {
    let first_retained_generation =
        current_generation.saturating_sub(agent_config.generation_archive_max_generations - 1);
    champion_archive.retain(|champion| champion.generation >= first_retained_generation);
    while champion_archive.len() > agent_config.generation_validation_max_models {
        champion_archive.remove(0);
    }
}

fn validate_generation_archive(
    validation_generation: usize,
    champion_archive: &[GenerationChampion],
    evolution_config: &EvolutionConfig,
    agent_config: &AgentConfig,
    cuda_backend: Option<&cuda_true2d::CudaTrue2D>,
) -> Result<GenerationValidationBatch, String> {
    let validation_started_at = Instant::now();
    if champion_archive.len() > agent_config.generation_validation_max_models {
        return Err(format!(
            "generation_validation_pool_exceeds_cap={}; cap={}",
            champion_archive.len(),
            agent_config.generation_validation_max_models
        ));
    }
    require_population_for_player_count(
        champion_archive.len(),
        evolution_config.active_player_count,
    )?;
    let assignments = game_assignments(
        champion_archive.len(),
        agent_config.generation_validation_games_per_model,
        evolution_config.active_player_count,
        agent_config,
    )?;
    let validation_population = champion_archive
        .iter()
        .map(|champion| champion.model.clone())
        .collect::<Vec<_>>();
    upload_population_to_cuda(cuda_backend, &validation_population, agent_config)?;
    let batch_result = run_games_batched(
        validation_generation,
        &assignments,
        &validation_population,
        agent_config,
        cuda_backend,
        0,
        false,
        evolution_config.active_player_count,
        None,
    )?;
    let mut accumulators = generation_validation_accumulators(champion_archive);
    let mut champion_rewards = vec![0i32; champion_archive.len()];
    let mut champion_gameplay = vec![GameplayStats::default(); champion_archive.len()];
    for (assignment, game_result) in assignments.iter().zip(batch_result.games.iter()) {
        for player_slot in 0..evolution_config.active_player_count {
            let model_index = assignment.model_indices[player_slot];
            let reward = game_result.rewards[player_slot];
            champion_rewards[model_index] += reward;
            champion_gameplay[model_index].add_reward(reward, agent_config);
            let evaluated_generation = champion_archive[model_index].generation;
            let accumulator = accumulators.get_mut(&evaluated_generation).ok_or_else(|| {
                format!("generation_validation_accumulator_missing={evaluated_generation}")
            })?;
            accumulator.game_count += 1;
            add_validation_reward(accumulator, reward, agent_config);
        }
    }
    let records = accumulators
        .into_iter()
        .map(|(evaluated_generation, accumulator)| {
            let win_rate = accumulator.win_count as f32 / accumulator.game_count.max(1) as f32;
            GenerationValidationRecord {
                validation_generation,
                evaluated_generation,
                model_count: accumulator.model_count,
                game_count: accumulator.game_count,
                win_count: accumulator.win_count,
                draw_count: accumulator.draw_count,
                loss_count: accumulator.loss_count,
                win_rate,
            }
        })
        .collect::<Vec<_>>();
    let mut champion_scores = champion_rewards
        .into_iter()
        .enumerate()
        .map(|(model_index, reward)| EvaluatedModel {
            model_index,
            reward,
            gameplay: champion_gameplay[model_index],
        })
        .collect::<Vec<_>>();
    champion_scores.sort_by(|left, right| {
        right
            .reward
            .cmp(&left.reward)
            .then_with(|| left.model_index.cmp(&right.model_index))
    });
    Ok(GenerationValidationBatch {
        records,
        champion_scores,
        game_count: assignments.len(),
        elapsed_seconds: elapsed_seconds(validation_started_at.elapsed()),
    })
}

fn empty_generation_validation() -> GenerationValidationBatch {
    GenerationValidationBatch {
        records: Vec::new(),
        champion_scores: Vec::new(),
        game_count: 0,
        elapsed_seconds: 0.0,
    }
}

fn scheduled_generation(generation: usize, interval: usize) -> bool {
    generation % interval == 0
}

fn generation_validation_accumulators(
    champion_archive: &[GenerationChampion],
) -> BTreeMap<usize, GenerationValidationAccumulator> {
    let mut accumulators = BTreeMap::new();
    for champion in champion_archive {
        accumulators
            .entry(champion.generation)
            .or_insert_with(GenerationValidationAccumulator::default)
            .model_count += 1;
    }
    accumulators
}

fn add_validation_reward(
    accumulator: &mut GenerationValidationAccumulator,
    reward: i32,
    agent_config: &AgentConfig,
) {
    if reward == agent_config.training_reward_win {
        accumulator.win_count += 1;
    } else if reward == agent_config.training_reward_loss {
        accumulator.loss_count += 1;
    } else {
        accumulator.draw_count += 1;
    }
}

fn upload_population_to_cuda(
    cuda_backend: Option<&cuda_true2d::CudaTrue2D>,
    population: &[True2DTransformer],
    config: &AgentConfig,
) -> Result<(), String> {
    if let Some(cuda) = cuda_backend {
        cuda.upload_population(
            population,
            config.cuda_true2d_max_requests_per_forward,
            config,
        )?;
    }
    Ok(())
}

fn run_games_batched(
    generation: usize,
    assignments: &[GameAssignment],
    population: &[True2DTransformer],
    config: &AgentConfig,
    cuda_backend: Option<&cuda_true2d::CudaTrue2D>,
    captured_replay_game_count: usize,
    capture_training_samples: bool,
    active_player_count: usize,
    live_replay_telemetry: Option<&LiveReplayTelemetry<'_>>,
) -> Result<GameBatchResult, String> {
    validate_active_player_count(active_player_count)?;
    let batch_started_at = Instant::now();
    let live_capture_count = live_replay_telemetry
        .map(|telemetry| live_replay_capture_count(assignments.len(), telemetry.agent_config))
        .unwrap_or(0);
    let mut training_samples = if capture_training_samples {
        TrainingSamples::with_capacity(
            training_sample_capacity(assignments.len(), active_player_count, config)?,
            config,
        )?
    } else {
        TrainingSamples::default()
    };
    let mut active_games = assignments
        .iter()
        .copied()
        .enumerate()
        .map(|(game_index, assignment)| -> Result<ActiveGame, String> {
            let mut game = ActiveGame {
                assignment,
                state: seeded_state(config, active_player_count, generation, game_index)?,
                frames: Vec::new(),
                stats: ComputeStats::default(),
                gameplay: [GameplayStats::default(); MAX_PLAYER_SLOTS],
                rewards: [config.training_reward_draw; MAX_PLAYER_SLOTS],
                fleet_action_traces: BTreeMap::new(),
                completed: false,
                started_at: Instant::now(),
            };
            if captures_live_replay(game_index, live_capture_count) {
                record_live_replay_frame(&mut game.frames, &game.state, config);
            } else if captures_generation_replay(game_index, captured_replay_game_count) {
                record_replay_frame(&mut game.frames, &game.state, config);
            }
            Ok(game)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut pipeline_stats = PipelineStats {
        simultaneous_game_count: active_games.len(),
        ..PipelineStats::default()
    };
    let mut requests_per_game = vec![0usize; active_games.len()];
    let mut population_forward_scratch = PopulationForwardScratch::default();
    let mut live_telemetry_warning_emitted = false;
    let mut live_storage_warning_emitted = false;
    let mut live_replay_storage = if let Some(telemetry) = live_replay_telemetry {
        if live_capture_count > 0 {
            match create_live_replay_storage(
                generation,
                &active_games,
                live_capture_count,
                active_player_count,
                telemetry.run_id,
                telemetry.agent_config,
            ) {
                Ok(mut storage) => {
                    for (game_index, game) in
                        active_games.iter().take(live_capture_count).enumerate()
                    {
                        if let Some(frame) = game.frames.last() {
                            if let Err(error) =
                                append_live_replay_frame(&mut storage, game_index, frame)
                            {
                                log_trainer_event(format!(
                                    "event=warning; generation={generation}; phase={}; warning=live_replay_storage_append_skipped; error={}",
                                    TRAINING_PHASE_EVALUATION, error
                                ))?;
                                live_storage_warning_emitted = true;
                                break;
                            }
                        }
                    }
                    Some(storage)
                }
                Err(error) => {
                    log_trainer_event(format!(
                        "event=warning; generation={generation}; phase={}; warning=live_replay_storage_create_skipped; error={}",
                        TRAINING_PHASE_EVALUATION, error
                    ))?;
                    live_storage_warning_emitted = true;
                    None
                }
            }
        } else {
            None
        }
    } else {
        None
    };
    if let Some(telemetry) = live_replay_telemetry {
        if live_capture_count > 0 {
            if let Err(error) = write_live_replay_telemetry(
                generation,
                0,
                live_capture_count * active_player_count,
                latest_live_replay_frame(&active_games, live_capture_count),
                active_games_compute_stats(&active_games),
                pipeline_stats,
                elapsed_seconds(batch_started_at.elapsed()),
                telemetry,
                live_replay_storage.as_ref(),
            ) {
                log_trainer_event(format!(
                    "event=warning; generation={generation}; phase={}; warning=live_replay_telemetry_write_skipped; error={}",
                    TRAINING_PHASE_EVALUATION, error
                ))?;
                live_telemetry_warning_emitted = true;
            }
        }
    }

    let mut turn_index = 0usize;
    while active_games.iter().any(|game| !game.completed) {
        let action_started_at = Instant::now();
        let action_requests = collect_action_requests(&active_games, config, active_player_count)?;
        let resolved_actions = resolve_action_requests(
            &action_requests,
            population,
            config,
            cuda_backend,
            &mut pipeline_stats,
            &mut population_forward_scratch,
            if capture_training_samples {
                Some(&mut training_samples)
            } else {
                None
            },
            generation,
            active_player_count,
        )?;
        let action_elapsed_ms = duration_to_millis(action_started_at.elapsed());
        let active_game_count = active_games
            .iter()
            .filter(|game| !game.completed)
            .count()
            .max(1) as f32;
        let action_elapsed_ms_per_game = action_elapsed_ms / active_game_count;
        requests_per_game.fill(0);
        for request in &action_requests {
            requests_per_game[request.game_index] += 1;
        }
        for (game_index, game) in active_games
            .iter_mut()
            .enumerate()
            .filter(|(_, game)| !game.completed)
        {
            game.stats.model_action_ms += action_elapsed_ms_per_game;
            game.stats.model_action_call_count += requests_per_game[game_index];
        }

        let sequential_live_count = live_capture_count.min(active_games.len());
        for game_index in 0..sequential_live_count {
            let player_actions = &resolved_actions.actions_by_game[game_index];
            let game = &mut active_games[game_index];
            if game.completed {
                continue;
            }
            let simulation_started_at = Instant::now();
            let events = game
                .state
                .step_turn_with_events(&player_actions, config)
                .map_err(|error| format!("step_failed={error:?}"))?;
            game.stats.simulation_step_ms += duration_to_millis(simulation_started_at.elapsed());
            game.stats.turn_count += 1;
            game.stats.launch_action_count += events_launch_count(&events);
            game.stats.launched_ship_count += events_launched_ships(&events);
            apply_step_events_to_gameplay(&mut game.gameplay, &events, active_player_count);
            if capture_training_samples {
                apply_sun_target_penalty_events(
                    game,
                    &events,
                    &resolved_actions.traces_by_game[game_index],
                    active_player_count,
                    &mut training_samples,
                    config,
                )?;
            }
            sample_fleet_sizes(&mut game.gameplay, &game.state, active_player_count);
            if captures_live_replay(game_index, live_capture_count) {
                let frame_recorded =
                    record_live_replay_frame(&mut game.frames, &game.state, config);
                if frame_recorded {
                    if let (Some(storage), Some(frame)) =
                        (live_replay_storage.as_mut(), game.frames.last())
                    {
                        if let Err(error) = append_live_replay_frame(storage, game_index, frame) {
                            if !live_storage_warning_emitted {
                                log_trainer_event(format!(
                                    "event=warning; generation={generation}; phase={}; warning=live_replay_storage_append_skipped; error={}",
                                    TRAINING_PHASE_EVALUATION, error
                                ))?;
                                live_storage_warning_emitted = true;
                            }
                        }
                    }
                }
            } else if captures_generation_replay(game_index, captured_replay_game_count) {
                record_replay_frame(&mut game.frames, &game.state, config);
            }
            if is_terminal(&game.state, config) {
                game.completed = true;
                game.rewards = final_rewards(&game.state, config, active_player_count);
                if captures_live_replay(game_index, live_capture_count) {
                    if let Some(storage) = live_replay_storage.as_ref() {
                        if let Err(error) = append_live_replay_result(
                            storage,
                            game_index,
                            &game.rewards,
                            active_player_count,
                        ) {
                            if !live_storage_warning_emitted {
                                log_trainer_event(format!(
                                    "event=warning; generation={generation}; phase={}; warning=live_replay_storage_result_skipped; error={}",
                                    TRAINING_PHASE_EVALUATION, error
                                ))?;
                                live_storage_warning_emitted = true;
                            }
                        }
                    }
                }
                for player_slot in 0..active_player_count {
                    game.gameplay[player_slot].add_reward(game.rewards[player_slot], config);
                }
                game.stats.game_count = 1;
                game.stats.total_game_ms = duration_to_millis(game.started_at.elapsed());
            }
        }
        let non_live_events = step_non_live_games_parallel(
            &mut active_games,
            &resolved_actions.actions_by_game,
            sequential_live_count,
            captured_replay_game_count,
            active_player_count,
            config,
        )?;
        if capture_training_samples {
            for (game_index, events) in non_live_events {
                let Some(game) = active_games.get_mut(game_index) else {
                    return Err("sun_penalty_game_index_out_of_range".to_string());
                };
                apply_sun_target_penalty_events(
                    game,
                    &events,
                    &resolved_actions.traces_by_game[game_index],
                    active_player_count,
                    &mut training_samples,
                    config,
                )?;
            }
        }
        turn_index += 1;
        if let Some(telemetry) = live_replay_telemetry {
            if should_write_live_replay(live_capture_count, turn_index, config) {
                if let Err(error) = write_live_replay_telemetry(
                    generation,
                    turn_index,
                    live_capture_count * active_player_count,
                    latest_live_replay_frame(&active_games, live_capture_count),
                    active_games_compute_stats(&active_games),
                    pipeline_stats,
                    elapsed_seconds(batch_started_at.elapsed()),
                    telemetry,
                    live_replay_storage.as_ref(),
                ) {
                    if !live_telemetry_warning_emitted {
                        log_trainer_event(format!(
                            "event=warning; generation={generation}; phase={}; warning=live_replay_telemetry_write_skipped; error={}",
                            TRAINING_PHASE_EVALUATION, error
                        ))?;
                        live_telemetry_warning_emitted = true;
                    }
                }
            }
        }
    }

    let games = active_games
        .into_iter()
        .map(|game| GameRunResult {
            rewards: game.rewards,
            frames: game.frames,
            stats: game.stats,
            gameplay: game.gameplay,
        })
        .collect::<Vec<_>>();
    training_samples.assign_rewards(&games)?;
    Ok(GameBatchResult {
        games,
        pipeline_stats,
        training_samples,
    })
}

fn live_replay_capture_count(total_game_count: usize, config: &AgentConfig) -> usize {
    if config.dashboard_live_replay_enabled {
        total_game_count.min(config.dashboard_live_replay_game_count)
    } else {
        0
    }
}

fn captures_live_replay(game_index: usize, live_capture_count: usize) -> bool {
    game_index < live_capture_count
}

fn captures_generation_replay(game_index: usize, captured_replay_game_count: usize) -> bool {
    game_index < captured_replay_game_count
}

fn should_write_live_replay(
    live_capture_count: usize,
    turn_index: usize,
    config: &AgentConfig,
) -> bool {
    live_capture_count > 0 && turn_index % config.dashboard_live_replay_update_interval_turns == 0
}

fn step_non_live_games_parallel(
    active_games: &mut [ActiveGame],
    action_batches: &[Vec<Vec<MoveCommand>>],
    start_game_index: usize,
    captured_replay_game_count: usize,
    active_player_count: usize,
    config: &AgentConfig,
) -> Result<Vec<(usize, SimulationStepEvents)>, String> {
    if start_game_index >= active_games.len() {
        return Ok(Vec::new());
    }
    if action_batches.len() != active_games.len() {
        return Err("parallel_step_action_batch_count_mismatch".to_string());
    }
    let game_count = active_games.len() - start_game_index;
    let worker_count = worker_count_for_len(game_count);
    let chunk_size = chunk_size_for_workers(game_count, worker_count);
    let game_tail = &mut active_games[start_game_index..];
    let action_tail = &action_batches[start_game_index..];
    thread::scope(|scope| {
        let mut handles = Vec::new();
        for (chunk_index, (game_chunk, action_chunk)) in game_tail
            .chunks_mut(chunk_size)
            .zip(action_tail.chunks(chunk_size))
            .enumerate()
        {
            let chunk_start_game_index = start_game_index + chunk_index * chunk_size;
            handles.push(scope.spawn(move || {
                let mut step_events = Vec::new();
                for (local_game_index, game) in game_chunk.iter_mut().enumerate() {
                    if game.completed {
                        continue;
                    }
                    let game_index = chunk_start_game_index + local_game_index;
                    let player_actions = &action_chunk[local_game_index];
                    if let Some(events) = step_non_live_game(
                        game,
                        game_index,
                        player_actions,
                        captured_replay_game_count,
                        active_player_count,
                        config,
                    )? {
                        step_events.push((game_index, events));
                    }
                }
                Ok::<Vec<(usize, SimulationStepEvents)>, String>(step_events)
            }));
        }
        let mut all_step_events = Vec::new();
        for handle in handles {
            all_step_events.extend(
                handle
                    .join()
                    .map_err(|_| "parallel_step_worker_panicked".to_string())??,
            );
        }
        Ok(all_step_events)
    })
}

fn step_non_live_game(
    game: &mut ActiveGame,
    game_index: usize,
    player_actions: &[Vec<MoveCommand>],
    captured_replay_game_count: usize,
    active_player_count: usize,
    config: &AgentConfig,
) -> Result<Option<SimulationStepEvents>, String> {
    let simulation_started_at = Instant::now();
    let events = game
        .state
        .step_turn_with_events(player_actions, config)
        .map_err(|error| format!("step_failed={error:?}"))?;
    game.stats.simulation_step_ms += duration_to_millis(simulation_started_at.elapsed());
    game.stats.turn_count += 1;
    game.stats.launch_action_count += events_launch_count(&events);
    game.stats.launched_ship_count += events_launched_ships(&events);
    apply_step_events_to_gameplay(&mut game.gameplay, &events, active_player_count);
    sample_fleet_sizes(&mut game.gameplay, &game.state, active_player_count);
    if captures_generation_replay(game_index, captured_replay_game_count) {
        record_replay_frame(&mut game.frames, &game.state, config);
    }
    if is_terminal(&game.state, config) {
        game.completed = true;
        game.rewards = final_rewards(&game.state, config, active_player_count);
        for player_slot in 0..active_player_count {
            game.gameplay[player_slot].add_reward(game.rewards[player_slot], config);
        }
        game.stats.game_count = 1;
        game.stats.total_game_ms = duration_to_millis(game.started_at.elapsed());
    }
    Ok(Some(events))
}

fn apply_sun_target_penalty_events(
    game: &mut ActiveGame,
    events: &SimulationStepEvents,
    action_traces_by_player: &[Vec<ActionTrace>],
    active_player_count: usize,
    training_samples: &mut TrainingSamples,
    config: &AgentConfig,
) -> Result<(), String> {
    let mut launched_trace_indices = vec![0usize; active_player_count];
    for launched in &events.launched_fleets {
        if launched.player < 0 {
            return Err(format!(
                "launched_fleet_negative_player={}",
                launched.player
            ));
        }
        let player_slot = launched.player as usize;
        if player_slot >= active_player_count {
            return Err(format!("launched_fleet_player_out_of_range={player_slot}"));
        }
        let trace_index = launched_trace_indices[player_slot];
        launched_trace_indices[player_slot] += 1;
        let Some(trace) = action_traces_by_player
            .get(player_slot)
            .and_then(|player_traces| player_traces.get(trace_index))
        else {
            return Err(format!(
                "launched_fleet_action_trace_missing=player:{player_slot};trace:{trace_index}"
            ));
        };
        game.fleet_action_traces.insert(launched.fleet_id, *trace);
    }

    if config.training_sun_target_penalty_scale > 0.0 {
        for destroyed in &events.sun_destroyed_fleets {
            if let Some(trace) = game.fleet_action_traces.remove(&destroyed.fleet_id) {
                training_samples.add_sun_target_penalty(
                    trace.sample_index,
                    trace.source_row,
                    trace.target_slot,
                    -config.training_sun_target_penalty_scale,
                    config,
                )?;
            }
        }
    }
    prune_inactive_fleet_action_traces(game);
    Ok(())
}

fn prune_inactive_fleet_action_traces(game: &mut ActiveGame) {
    let active_fleet_ids = game
        .state
        .fleets
        .iter()
        .map(|fleet| fleet.id)
        .collect::<BTreeSet<_>>();
    game.fleet_action_traces
        .retain(|fleet_id, _| active_fleet_ids.contains(fleet_id));
}

fn active_games_compute_stats(active_games: &[ActiveGame]) -> ComputeStats {
    let mut stats = ComputeStats::default();
    for game in active_games {
        stats.add_game(game.stats);
    }
    stats
}

fn latest_live_replay_frame(
    active_games: &[ActiveGame],
    live_capture_count: usize,
) -> Option<&ReplayFrame> {
    active_games
        .iter()
        .take(live_capture_count)
        .find_map(|game| game.frames.last())
}

fn worker_count_for_len(len: usize) -> usize {
    len.clamp(1, TRAINING_CPU_WORKER_COUNT)
}

fn chunk_size_for_workers(len: usize, worker_count: usize) -> usize {
    len.div_ceil(worker_count.max(1)).max(1)
}

fn player_has_owned_planet(planets: &[Planet], player_id: i32) -> bool {
    planets.iter().any(|planet| planet.owner == player_id)
}

fn collect_action_requests(
    active_games: &[ActiveGame],
    config: &AgentConfig,
    active_player_count: usize,
) -> Result<Vec<ActionRequest>, String> {
    if active_games.is_empty() {
        return Ok(Vec::new());
    }
    let worker_count = worker_count_for_len(active_games.len());
    let chunk_size = chunk_size_for_workers(active_games.len(), worker_count);
    let chunk_results = thread::scope(|scope| {
        let mut handles = Vec::new();
        for (chunk_index, game_chunk) in active_games.chunks(chunk_size).enumerate() {
            let start_game_index = chunk_index * chunk_size;
            handles.push(scope.spawn(move || {
                let mut requests = Vec::new();
                for (local_game_index, game) in game_chunk.iter().enumerate() {
                    if game.completed {
                        continue;
                    }
                    let game_index = start_game_index + local_game_index;
                    let alive_players = game.state.alive_players();
                    for (player_slot, player_id) in PLAYER_IDS
                        .iter()
                        .copied()
                        .take(active_player_count)
                        .enumerate()
                    {
                        if !alive_players.contains(&player_id) {
                            continue;
                        }
                        if !player_has_owned_planet(&game.state.planets, player_id) {
                            continue;
                        }
                        let encoded = encode_state(
                            &game.state.planets,
                            &game.state.initial_planets,
                            &[],
                            player_id,
                            config,
                        )
                        .map_err(|error| format!("encode_failed={error:?}"))?;
                        requests.push(ActionRequest {
                            game_index,
                            player_slot,
                            player_id,
                            model_index: game.assignment.model_indices[player_slot],
                            step: game.state.step,
                            planets: game.state.planets.clone(),
                            initial_planets: game.state.initial_planets.clone(),
                            angular_velocity: game.state.angular_velocity,
                            rows: encoded.rows.to_vec(),
                        });
                    }
                }
                Ok::<Vec<ActionRequest>, String>(requests)
            }));
        }
        handles
            .into_iter()
            .map(|handle| {
                handle
                    .join()
                    .map_err(|_| "collect_action_requests_worker_panicked".to_string())?
            })
            .collect::<Result<Vec<_>, _>>()
    })?;
    Ok(chunk_results.into_iter().flatten().collect())
}

fn resolve_action_requests(
    action_requests: &[ActionRequest],
    population: &[True2DTransformer],
    config: &AgentConfig,
    cuda_backend: Option<&cuda_true2d::CudaTrue2D>,
    pipeline_stats: &mut PipelineStats,
    population_forward_scratch: &mut PopulationForwardScratch,
    mut training_samples: Option<&mut TrainingSamples>,
    generation: usize,
    active_player_count: usize,
) -> Result<ResolvedActions, String> {
    let game_count = action_requests
        .iter()
        .map(|request| request.game_index)
        .max()
        .map(|index| index + 1)
        .unwrap_or(0);
    let mut actions_by_game = (0..game_count)
        .map(|_| {
            (0..active_player_count)
                .map(|_| Vec::new())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut traces_by_game = (0..game_count)
        .map(|_| {
            (0..active_player_count)
                .map(|_| Vec::new())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    if let Some(cuda) = cuda_backend {
        let outputs = batched_population_outputs(
            population,
            action_requests,
            config,
            cuda,
            pipeline_stats,
            population_forward_scratch,
        )?;
        let mut first_sample_index = None;
        if let Some(samples) = training_samples.as_deref_mut() {
            let sample_start = samples.sample_count();
            record_training_samples(
                samples,
                generation,
                action_requests.iter(),
                &outputs,
                config,
            )?;
            first_sample_index = Some(sample_start);
        }
        decode_outputs_into_actions(
            action_requests,
            &outputs,
            config,
            active_player_count,
            first_sample_index,
            &mut actions_by_game,
            &mut traces_by_game,
        )?;
        return Ok(ResolvedActions {
            actions_by_game,
            traces_by_game,
        });
    }

    let mut requests_by_model = BTreeMap::<usize, Vec<&ActionRequest>>::new();
    for request in action_requests {
        requests_by_model
            .entry(request.model_index)
            .or_default()
            .push(request);
    }

    for (model_index, requests) in requests_by_model {
        let outputs =
            batched_model_outputs(&population[model_index], &requests, config, pipeline_stats)?;
        let mut first_sample_index = None;
        if let Some(samples) = training_samples.as_deref_mut() {
            let sample_start = samples.sample_count();
            record_training_samples(
                samples,
                generation,
                requests.iter().copied(),
                &outputs,
                config,
            )?;
            first_sample_index = Some(sample_start);
        }
        let request_clones = requests
            .iter()
            .map(|request| (*request).clone())
            .collect::<Vec<_>>();
        decode_outputs_into_actions(
            &request_clones,
            &outputs,
            config,
            active_player_count,
            first_sample_index,
            &mut actions_by_game,
            &mut traces_by_game,
        )?;
    }
    Ok(ResolvedActions {
        actions_by_game,
        traces_by_game,
    })
}

fn record_training_samples<'a, I>(
    training_samples: &mut TrainingSamples,
    generation: usize,
    requests: I,
    outputs: &[ActionOutput],
    config: &AgentConfig,
) -> Result<(), String>
where
    I: ExactSizeIterator<Item = &'a ActionRequest>,
{
    let request_count = requests.len();
    let expected_output_count = request_count
        .checked_mul(config.max_rows)
        .ok_or_else(|| "training_sample_output_count_overflow".to_string())?;
    if outputs.len() != expected_output_count {
        return Err("training_sample_output_count_mismatch".to_string());
    }
    for (request_index, request) in requests.enumerate() {
        let start = request_index * config.max_rows;
        let end = start + config.max_rows;
        training_samples.record(generation, request, &outputs[start..end], config)?;
    }
    Ok(())
}

fn decode_outputs_into_actions(
    action_requests: &[ActionRequest],
    outputs: &[ActionOutput],
    config: &AgentConfig,
    active_player_count: usize,
    first_sample_index: Option<usize>,
    actions_by_game: &mut [Vec<Vec<MoveCommand>>],
    traces_by_game: &mut [Vec<Vec<ActionTrace>>],
) -> Result<(), String> {
    let expected_output_count = action_requests
        .len()
        .checked_mul(config.max_rows)
        .ok_or_else(|| "decode_output_count_overflow".to_string())?;
    if outputs.len() != expected_output_count {
        return Err("decode_output_count_mismatch".to_string());
    }
    if action_requests.is_empty() {
        return Ok(());
    }
    let worker_count = worker_count_for_len(action_requests.len());
    let chunk_size = chunk_size_for_workers(action_requests.len(), worker_count);
    let decoded_chunks = thread::scope(|scope| {
        let mut handles = Vec::new();
        for (chunk_index, request_chunk) in action_requests.chunks(chunk_size).enumerate() {
            let start_request_index = chunk_index * chunk_size;
            handles.push(scope.spawn(move || {
                let mut decoded = Vec::with_capacity(request_chunk.len());
                for (local_request_index, request) in request_chunk.iter().enumerate() {
                    if request.player_slot >= active_player_count {
                        return Err("decode_player_slot_out_of_range".to_string());
                    }
                    let request_index = start_request_index + local_request_index;
                    let start = request_index * config.max_rows;
                    let end = start + config.max_rows;
                    let decoded_commands = decode_model_outputs_with_trace(
                        &request.planets,
                        &request.initial_planets,
                        &[],
                        request.player_id,
                        request.angular_velocity,
                        request.step,
                        &outputs[start..end],
                        config,
                    )
                    .map_err(|error| format!("decode_failed={error:?}"))?;
                    let sample_index = decode_sample_index(first_sample_index, request_index);
                    let (game_actions, action_traces) =
                        action_trace_records(decoded_commands, sample_index);
                    decoded.push((
                        request.game_index,
                        request.player_slot,
                        game_actions,
                        action_traces,
                    ));
                }
                Ok::<Vec<(usize, usize, Vec<MoveCommand>, Vec<ActionTrace>)>, String>(decoded)
            }));
        }
        handles
            .into_iter()
            .map(|handle| {
                handle
                    .join()
                    .map_err(|_| "decode_outputs_worker_panicked".to_string())?
            })
            .collect::<Result<Vec<_>, _>>()
    })?;
    for (game_index, player_slot, game_actions, action_traces) in
        decoded_chunks.into_iter().flatten()
    {
        let Some(player_actions) = actions_by_game.get_mut(game_index) else {
            return Err("decode_game_index_out_of_range".to_string());
        };
        player_actions[player_slot] = game_actions;
        let Some(player_traces) = traces_by_game.get_mut(game_index) else {
            return Err("decode_trace_game_index_out_of_range".to_string());
        };
        player_traces[player_slot] = action_traces;
    }
    Ok(())
}

fn decode_sample_index(first_sample_index: Option<usize>, request_index: usize) -> Option<usize> {
    first_sample_index.map(|first_index| first_index + request_index)
}

fn action_trace_records(
    decoded_commands: Vec<DecodedMoveCommand>,
    sample_index: Option<usize>,
) -> (Vec<MoveCommand>, Vec<ActionTrace>) {
    let mut commands = Vec::with_capacity(decoded_commands.len());
    let mut traces = Vec::new();
    for decoded in decoded_commands {
        commands.push(decoded.command);
        if let Some(sample_index) = sample_index {
            traces.push(ActionTrace {
                sample_index,
                source_row: decoded.source_row,
                target_slot: decoded.target_slot,
                target_row: decoded.target_row,
            });
        }
    }
    (commands, traces)
}

fn record_replay_frame(
    frames: &mut Vec<ReplayFrame>,
    state: &SimulationState,
    config: &AgentConfig,
) {
    record_replay_frame_with_limits(
        frames,
        state,
        config.dashboard_replay_frame_stride,
        config.dashboard_replay_max_frames,
    );
}

fn record_live_replay_frame(
    frames: &mut Vec<ReplayFrame>,
    state: &SimulationState,
    config: &AgentConfig,
) -> bool {
    if state.step % config.dashboard_live_replay_frame_stride != 0 && !frames.is_empty() {
        return false;
    }
    frames.push(replay_frame_from_state(state));
    true
}

fn record_replay_frame_with_limits(
    frames: &mut Vec<ReplayFrame>,
    state: &SimulationState,
    frame_stride: usize,
    max_frames: usize,
) {
    if frames.len() >= max_frames {
        return;
    }
    if state.step % frame_stride != 0 && !frames.is_empty() {
        return;
    }
    frames.push(replay_frame_from_state(state));
}

fn replay_frame_from_state(state: &SimulationState) -> ReplayFrame {
    ReplayFrame {
        step: state.step,
        planets: state.planets.clone(),
        fleets: state.fleets.clone(),
        comet_groups: state
            .comet_groups
            .iter()
            .filter(|group| group.active)
            .map(|group| ReplayCometGroup {
                planet_ids: group.planet_ids.clone(),
                paths: group.paths.clone(),
                path_index: group.path_index,
            })
            .collect(),
    }
}

fn batched_model_outputs(
    model: &True2DTransformer,
    requests: &[&ActionRequest],
    config: &AgentConfig,
    pipeline_stats: &mut PipelineStats,
) -> Result<Vec<ActionOutput>, String> {
    if requests.is_empty() {
        return Ok(Vec::new());
    }
    pipeline_stats.record_batch(requests.len());
    let mut outputs = Vec::with_capacity(requests.len() * config.max_rows);
    for request in requests {
        outputs.extend(
            model
                .forward(&request.rows, config)
                .map_err(|error| format!("forward_failed={error:?}"))?,
        );
    }
    Ok(outputs)
}

fn batched_population_outputs(
    population: &[True2DTransformer],
    action_requests: &[ActionRequest],
    config: &AgentConfig,
    cuda: &cuda_true2d::CudaTrue2D,
    pipeline_stats: &mut PipelineStats,
    scratch: &mut PopulationForwardScratch,
) -> Result<Vec<ActionOutput>, String> {
    if action_requests.is_empty() {
        return Ok(Vec::new());
    }
    let request_chunk_size = config.cuda_true2d_max_requests_per_forward;
    if request_chunk_size == 0 {
        return Err("cuda_true2d_max_requests_per_forward_zero".to_string());
    }
    let mut outputs = Vec::with_capacity(action_requests.len() * config.max_rows);
    for request_chunk in action_requests.chunks(request_chunk_size) {
        pipeline_stats.record_batch(request_chunk.len());
        scratch.pack_chunk(request_chunk, config)?;
        outputs.extend(cuda.forward_population_resident_packed(
            &scratch.input_rows,
            &scratch.model_indices,
            population,
            config,
        )?);
    }
    Ok(outputs)
}

fn events_launch_count(events: &SimulationStepEvents) -> usize {
    events
        .by_player
        .values()
        .map(|player_events| player_events.launched_fleet_count)
        .sum()
}

fn events_launched_ships(events: &SimulationStepEvents) -> i64 {
    events
        .by_player
        .values()
        .map(|player_events| player_events.launched_ship_count)
        .sum()
}

fn apply_step_events_to_gameplay(
    gameplay: &mut [GameplayStats; MAX_PLAYER_SLOTS],
    events: &SimulationStepEvents,
    active_player_count: usize,
) {
    for (player_slot, player_id) in PLAYER_IDS
        .iter()
        .copied()
        .take(active_player_count)
        .enumerate()
    {
        gameplay[player_slot].add_step_events(events.player(player_id));
    }
}

fn sample_fleet_sizes(
    gameplay: &mut [GameplayStats; MAX_PLAYER_SLOTS],
    state: &SimulationState,
    active_player_count: usize,
) {
    for (player_slot, player_id) in PLAYER_IDS
        .iter()
        .copied()
        .take(active_player_count)
        .enumerate()
    {
        gameplay[player_slot].add_fleet_observations(state, player_id);
    }
}

fn final_rewards(
    state: &SimulationState,
    config: &AgentConfig,
    active_player_count: usize,
) -> [i32; MAX_PLAYER_SLOTS] {
    let scores = state.scores(config);
    let mut player_scores = [0i32; MAX_PLAYER_SLOTS];
    for score in scores {
        if let Some(player_score) = player_scores.get_mut(score.player as usize) {
            *player_score = score.ships;
        }
    }
    let active_scores = &player_scores[..active_player_count];
    let best_score = active_scores.iter().copied().max().unwrap_or(0);
    let best_count = active_scores
        .iter()
        .filter(|score| **score == best_score)
        .count();
    let mut rewards = [config.training_reward_draw; MAX_PLAYER_SLOTS];
    for (player_slot, score) in active_scores.iter().copied().enumerate() {
        rewards[player_slot] = if score == best_score && best_count == 1 {
            config.training_reward_win
        } else if score == best_score {
            config.training_reward_draw
        } else {
            config.training_reward_loss
        };
    }
    rewards
}

fn seeded_state(
    config: &AgentConfig,
    active_player_count: usize,
    generation: usize,
    game_index: usize,
) -> Result<SimulationState, String> {
    let seed = map_seed(generation, game_index)?;
    let mut rng = MapRng::new(seed);
    let angular_velocity = rng.range_f32(0.025, 0.05);
    let mut planets = generate_official_like_planets(config, &mut rng)?;
    assign_home_planets(&mut planets, active_player_count, config, &mut rng)?;
    let mut state = SimulationState::new(planets, angular_velocity);
    state.next_fleet_id = 1;
    Ok(state)
}

fn generate_official_like_planets(
    config: &AgentConfig,
    rng: &mut MapRng,
) -> Result<Vec<Planet>, String> {
    let target_groups = rng.range_usize(config.map_min_planet_groups, config.map_max_planet_groups);
    let target_planets = target_groups
        .checked_mul(MAP_GROUP_SIZE)
        .ok_or_else(|| "map_target_planet_count_overflow".to_string())?;
    let mut planets = Vec::with_capacity(target_planets);
    let mut next_id = 0i32;
    let mut static_groups = 0usize;
    for _ in 0..MAP_GENERATION_ATTEMPT_LIMIT {
        if static_groups >= config.map_min_static_groups {
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
    while planets.len() < target_planets
        || (!has_orbiting && attempts < MAP_GENERATION_ATTEMPT_LIMIT)
    {
        attempts += 1;
        if attempts >= MAP_GENERATION_ATTEMPT_LIMIT {
            break;
        }
        let Some(group) = generate_orbiting_or_static_planet_group(next_id, &planets, config, rng)?
        else {
            continue;
        };
        if group.iter().any(|planet| planet_orbits(planet, config)) {
            has_orbiting = true;
        }
        planets.extend(group);
        next_id += MAP_GROUP_SIZE as i32;
    }
    if planets.len() < config.map_min_planet_groups * MAP_GROUP_SIZE {
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
    let max_orbital =
        (config.board_size - config.board_center - radius) / angle.cos().max(angle.sin());
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
        || config.board_size - x - radius < 0.0
        || config.board_size - y - radius < 0.0
        || x - config.board_center < radius + MAP_STATIC_AXIS_CLEARANCE
        || y - config.board_center < radius + MAP_STATIC_AXIS_CLEARANCE
    {
        return Ok(None);
    }
    let ships = rng
        .range_usize(MAP_STATIC_SHIP_MIN, MAP_STATIC_SHIP_MAX)
        .min(rng.range_usize(MAP_STATIC_SHIP_MIN, MAP_STATIC_SHIP_MAX)) as f32;
    let group = symmetric_planet_group(next_id, x, y, radius, ships, production, config)?;
    if planet_group_has_clearance(&group, existing, config) {
        Ok(Some(group))
    } else {
        Ok(None)
    }
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
        && (x + radius > config.board_size
            || x - radius < 0.0
            || y + radius > config.board_size
            || y - radius < 0.0)
    {
        return Ok(None);
    }
    let ships = rng.range_usize(MAP_ORBITING_SHIP_MIN, MAP_ORBITING_SHIP_MAX) as f32;
    let group = symmetric_planet_group(next_id, x, y, radius, ships, production, config)?;
    if planet_group_has_clearance(&group, existing, config)
        && planet_group_orbit_static_cross_check(&group, existing, config)
    {
        Ok(Some(group))
    } else {
        Ok(None)
    }
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
    let ids = [
        next_id,
        next_id
            .checked_add(1)
            .ok_or_else(|| "map_planet_id_overflow".to_string())?,
        next_id
            .checked_add(2)
            .ok_or_else(|| "map_planet_id_overflow".to_string())?,
        next_id
            .checked_add(3)
            .ok_or_else(|| "map_planet_id_overflow".to_string())?,
    ];
    Ok(vec![
        planet_with_radius(ids[0], -1, y, x, radius, ships, production),
        planet_with_radius(
            ids[1],
            -1,
            config.board_size - x,
            y,
            radius,
            ships,
            production,
        ),
        planet_with_radius(
            ids[2],
            -1,
            x,
            config.board_size - y,
            radius,
            ships,
            production,
        ),
        planet_with_radius(
            ids[3],
            -1,
            config.board_size - y,
            config.board_size - x,
            radius,
            ships,
            production,
        ),
    ])
}

fn planet_with_radius(
    id: i32,
    owner: i32,
    x: f32,
    y: f32,
    radius: f32,
    ships: f32,
    production: f32,
) -> Planet {
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

fn planet_group_has_clearance(group: &[Planet], existing: &[Planet], config: &AgentConfig) -> bool {
    group.iter().all(|candidate| {
        existing.iter().all(|planet| {
            distance_xy(candidate.x, candidate.y, planet.x, planet.y)
                >= candidate.radius + planet.radius + config.map_planet_clearance
        })
    })
}

fn planet_group_orbit_static_cross_check(
    group: &[Planet],
    existing: &[Planet],
    config: &AgentConfig,
) -> bool {
    group.iter().all(|candidate| {
        existing.iter().all(|planet| {
            if planet_orbits(candidate, config) == planet_orbits(planet, config) {
                return true;
            }
            let candidate_orbital = distance_xy(
                candidate.x,
                candidate.y,
                config.board_center,
                config.board_center,
            );
            let planet_orbital =
                distance_xy(planet.x, planet.y, config.board_center, config.board_center);
            (candidate_orbital - planet_orbital).abs()
                >= candidate.radius + planet.radius + config.map_planet_clearance
        })
    })
}

fn assign_home_planets(
    planets: &mut [Planet],
    active_player_count: usize,
    config: &AgentConfig,
    rng: &mut MapRng,
) -> Result<(), String> {
    if planets.len() < MAP_GROUP_SIZE || planets.len() % MAP_GROUP_SIZE != 0 {
        return Err("map_home_group_unavailable".to_string());
    }
    let group_count = planets.len() / MAP_GROUP_SIZE;
    let home_group = rng.range_usize(0, group_count - 1);
    let base = home_group
        .checked_mul(MAP_GROUP_SIZE)
        .ok_or_else(|| "map_home_group_index_overflow".to_string())?;
    if active_player_count == PLAYER_COUNT_TWO {
        planets[base].owner = PLAYER_ZERO;
        planets[base].ships = config.map_home_planet_ships;
        planets[base + 3].owner = PLAYER_ONE;
        planets[base + 3].ships = config.map_home_planet_ships;
    } else if active_player_count == PLAYER_COUNT_FOUR {
        for player_slot in 0..PLAYER_COUNT_FOUR {
            planets[base + player_slot].owner = PLAYER_IDS[player_slot];
            planets[base + player_slot].ships = config.map_home_planet_ships;
        }
    } else {
        return Err(format!("invalid_active_player_count={active_player_count}"));
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

fn write_progress_telemetry(
    generation: usize,
    phase: &str,
    evolution_config: &EvolutionConfig,
    metrics: &[MetricRecord],
    generation_validation_history: &[GenerationValidationRecord],
    replay_history: &[ReplayChunkRecord],
    agent_config: &AgentConfig,
    cli: &TrainerCli,
    run_id: &str,
) -> Result<(), String> {
    let first_frames = "[]";
    let training_backend = if cli.use_cuda {
        TRAINING_BACKEND_CUDA
    } else {
        TRAINING_BACKEND_CPU
    };
    let training_mode = format!("{}_{}", evolution_config.profile_name, training_backend);
    let full_replay = cli.full_replays || cli.dashboard_strict;
    let expected_evaluation_games = expected_self_play_game_count(
        evolution_config.population_size,
        evolution_config.games_per_model,
        evolution_config.active_player_count,
    )?;
    let expected_tournament_games = expected_self_play_game_count(
        agent_config.generation_validation_max_models,
        agent_config.generation_validation_games_per_model,
        evolution_config.active_player_count,
    )?;
    let expected_simultaneous_games = match phase {
        TRAINING_PHASE_EVALUATION => expected_evaluation_games,
        TRAINING_PHASE_VALIDATION => expected_tournament_games,
        _ => NO_SIMULTANEOUS_GAMES,
    };
    let progress_metric = MetricRecord {
        generation,
        win_rate: 0.0,
        games_per_second: 0.0,
        turns_per_second: 0.0,
        p95_latency_ms: 0.0,
        gpu_utilization: UNMEASURED_GPU_UTILIZATION_PERCENT,
        evaluated_games: 0,
        sampled_replay_games: 0,
        model_action_calls: 0,
        launch_actions: 0,
        launched_ships: 0,
        captures: 0,
        fleet_hits: 0,
        hit_ships: 0,
        sun_destroyed_fleets: 0,
        sun_destroyed_ships: 0,
        avg_fleet_size: 0.0,
        avg_launch_actions_per_turn: 0.0,
        avg_launched_ships_per_turn: 0.0,
        avg_model_action_ms: 0.0,
        inference_batch_calls: 0,
        max_inference_batch_size: agent_config.cuda_true2d_max_requests_per_forward,
        simultaneous_games: expected_simultaneous_games,
        model_action_seconds: 0.0,
        simulation_step_seconds: 0.0,
        evaluation_seconds: 0.0,
        replay_seconds: 0.0,
        replay_write_seconds: 0.0,
        generation_validation_games: 0,
        generation_validation_seconds: 0.0,
        backprop_samples: 0,
        backprop_models: 0,
        backprop_seconds: 0.0,
        reproduction_seconds: 0.0,
        generation_seconds: 0.0,
    };
    let metric_json = progress_metrics_json(metrics, &progress_metric);
    let telemetry = format!(
        "{{\n  \"sourceMessage\": \"run_profile={profile}; phase={phase}; status=in_progress; players_per_game={players_per_game}; compute_timing=pending\",\n  \"runId\": \"{run_id}\",\n  \"activeGeneration\": {generation},\n  \"runProfile\": \"{profile}\",\n  \"trainingMode\": \"{training_mode}\",\n  \"strictGameRules\": {strict_game_rules},\n  \"episodeSteps\": {episode_steps},\n  \"expectedFramesPerGame\": {expected_frames_per_game},\n  \"populationSize\": {population_size},\n  \"eliteCount\": {elite_count},\n  \"gamesPerModel\": {games_per_model},\n  \"replaysPerModel\": {replays_per_model},\n  \"generationReplayGameCount\": {generation_replay_game_count},\n  \"playersPerGame\": {players_per_game},\n  \"simultaneousGames\": {simultaneous_games},\n  \"inferenceBatchCalls\": 0,\n  \"maxInferenceBatchSize\": {max_inference_batch_size},\n  \"mapSource\": \"{map_source}\",\n  \"turnLoop\": \"{turn_loop}\",\n  \"fullReplay\": {full_replay},\n  \"replayFrameStride\": {replay_frame_stride},\n  \"storedReplayGames\": {stored_replay_games},\n  \"gamesPerSecond\": 0.0,\n  \"turnsPerSecond\": 0.0,\n  \"gpuUtilization\": {gpu_utilization},\n  \"cpuWorkers\": {cpu_workers},\n  \"evalQueue\": {eval_queue},\n  \"submissionGate\": \"generation tournament top1 staged for host-side Kaggle submit when scheduled\",\n  \"validationErrors\": [],\n  \"warnings\": [\"{gpu_warning}\",\"{progress_warning}\"],\n  \"metrics\": [{metric_json}],\n  \"generationWinRates\": [{generation_win_rates_json}],\n  \"models\": [{models}],\n  \"frames\": {first_frames},\n  \"replayChunks\": [{replay_chunks_json}],\n  \"replayGames\": []\n}}\n",
        profile = evolution_config.profile_name,
        strict_game_rules = json_bool(evolution_config.strict_game_rules),
        episode_steps = agent_config.episode_steps,
        expected_frames_per_game = agent_config.episode_steps + 1,
        population_size = evolution_config.population_size,
        elite_count = evolution_config.elite_count,
        games_per_model = evolution_config.games_per_model,
        replays_per_model = agent_config.dashboard_replays_per_model,
        generation_replay_game_count = agent_config.dashboard_generation_replay_game_count,
        players_per_game = evolution_config.active_player_count,
        simultaneous_games = expected_simultaneous_games,
        max_inference_batch_size = agent_config.cuda_true2d_max_requests_per_forward,
        map_source = NATIVE_SEEDED_REFERENCE_MAP_SOURCE,
        turn_loop = OFFICIAL_TURN_LOOP_NAME,
        full_replay = json_bool(full_replay),
        replay_frame_stride = agent_config.dashboard_replay_frame_stride,
        gpu_utilization = UNMEASURED_GPU_UTILIZATION_PERCENT,
        cpu_workers = TRAINING_CPU_WORKER_COUNT,
        eval_queue = EMPTY_EVAL_QUEUE_DEPTH,
        gpu_warning = GPU_UTILIZATION_UNMEASURED_WARNING,
        progress_warning = TRAINING_PROGRESS_PENDING_WARNING,
        metric_json = metric_json,
        generation_win_rates_json = generation_win_rates_json(generation_validation_history),
        models = progress_models_json(evolution_config.population_size),
        first_frames = first_frames,
        replay_chunks_json = replay_chunks_json_preserving_existing(run_id, replay_history),
        stored_replay_games = replay_chunk_record_game_count(replay_history)
    );
    let path = Path::new("dashboard/public/telemetry/latest.json");
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|error| format!("telemetry_dir_failed={error}"))?;
    }
    write_text_with_retries(path, &telemetry, "telemetry_write_failed")
}

fn write_live_replay_telemetry(
    generation: usize,
    turn_index: usize,
    live_replay_game_count: usize,
    latest_frame: Option<&ReplayFrame>,
    compute_stats: ComputeStats,
    pipeline_stats: PipelineStats,
    elapsed_seconds: f32,
    telemetry_context: &LiveReplayTelemetry<'_>,
    live_replay_storage: Option<&LiveReplayStorage>,
) -> Result<(), String> {
    let agent_config = telemetry_context.agent_config;
    let evolution_config = telemetry_context.evolution_config;
    let first_frames = latest_frame
        .map(|frame| format!("[{}]", replay_frame_json(frame)))
        .unwrap_or_else(|| "[]".to_string());
    let live_replay_path = live_replay_storage
        .map(|storage| storage.public_path.as_str())
        .unwrap_or("");
    let live_replay_frame_count = live_replay_storage
        .map(|storage| storage.frame_count)
        .unwrap_or_default();
    let training_backend = if telemetry_context.cli.use_cuda {
        TRAINING_BACKEND_CUDA
    } else {
        TRAINING_BACKEND_CPU
    };
    let training_mode = format!("{}_{}", evolution_config.profile_name, training_backend);
    let turn_count = compute_stats.turn_count.max(1) as f32;
    let action_call_count = compute_stats.model_action_call_count.max(1) as f32;
    let live_full_replay = agent_config.dashboard_live_replay_frame_stride
        == agent_config.dashboard_full_replay_frame_stride;
    let progress_metric = MetricRecord {
        generation,
        win_rate: 0.0,
        games_per_second: compute_stats.game_count as f32 / elapsed_seconds,
        turns_per_second: compute_stats.turn_count as f32 / elapsed_seconds,
        p95_latency_ms: 0.0,
        gpu_utilization: UNMEASURED_GPU_UTILIZATION_PERCENT,
        evaluated_games: compute_stats.game_count,
        sampled_replay_games: live_replay_game_count,
        model_action_calls: compute_stats.model_action_call_count,
        launch_actions: compute_stats.launch_action_count,
        launched_ships: compute_stats.launched_ship_count,
        captures: 0,
        fleet_hits: 0,
        hit_ships: 0,
        sun_destroyed_fleets: 0,
        sun_destroyed_ships: 0,
        avg_fleet_size: 0.0,
        avg_launch_actions_per_turn: compute_stats.launch_action_count as f32 / turn_count,
        avg_launched_ships_per_turn: compute_stats.launched_ship_count as f32 / turn_count,
        avg_model_action_ms: compute_stats.model_action_ms / action_call_count,
        inference_batch_calls: pipeline_stats.inference_batch_call_count,
        max_inference_batch_size: pipeline_stats.max_inference_batch_size,
        simultaneous_games: pipeline_stats.simultaneous_game_count,
        model_action_seconds: seconds_from_millis(compute_stats.model_action_ms),
        simulation_step_seconds: seconds_from_millis(compute_stats.simulation_step_ms),
        evaluation_seconds: elapsed_seconds,
        replay_seconds: 0.0,
        replay_write_seconds: 0.0,
        generation_validation_games: 0,
        generation_validation_seconds: 0.0,
        backprop_samples: 0,
        backprop_models: 0,
        backprop_seconds: 0.0,
        reproduction_seconds: 0.0,
        generation_seconds: elapsed_seconds,
    };
    let metric_json = progress_metrics_json(telemetry_context.metrics, &progress_metric);
    let telemetry = format!(
        "{{\n  \"sourceMessage\": \"run_profile={profile}; phase={phase}; status=in_progress; live_replay_turn={turn}; live_replay_games={live_replay_games}; live_replay_storage={live_replay_format}; compute_timing=partial\",\n  \"runId\": \"{run_id}\",\n  \"activeGeneration\": {generation},\n  \"runProfile\": \"{profile}\",\n  \"trainingMode\": \"{training_mode}\",\n  \"strictGameRules\": {strict_game_rules},\n  \"episodeSteps\": {episode_steps},\n  \"expectedFramesPerGame\": {expected_frames_per_game},\n  \"populationSize\": {population_size},\n  \"eliteCount\": {elite_count},\n  \"gamesPerModel\": {games_per_model},\n  \"replaysPerModel\": {replays_per_model},\n  \"generationReplayGameCount\": {generation_replay_game_count},\n  \"playersPerGame\": {players_per_game},\n  \"simultaneousGames\": {simultaneous_games},\n  \"inferenceBatchCalls\": {inference_batch_calls},\n  \"maxInferenceBatchSize\": {max_inference_batch_size},\n  \"mapSource\": \"{map_source}\",\n  \"turnLoop\": \"{turn_loop}\",\n  \"fullReplay\": {full_replay},\n  \"replayFrameStride\": {replay_frame_stride},\n  \"liveReplayFormat\": \"{live_replay_format}\",\n  \"liveReplayPath\": \"{live_replay_path}\",\n  \"liveReplayFrameCount\": {live_replay_frame_count},\n  \"storedReplayGames\": {stored_replay_games},\n  \"gamesPerSecond\": {games_per_second:.3},\n  \"turnsPerSecond\": {turns_per_second:.3},\n  \"gpuUtilization\": {gpu_utilization},\n  \"cpuWorkers\": {cpu_workers},\n  \"evalQueue\": {eval_queue},\n  \"submissionGate\": \"generation tournament top1 staged for host-side Kaggle submit when scheduled\",\n  \"validationErrors\": [],\n  \"warnings\": [\"{gpu_warning}\",\"{live_warning}\"],\n  \"metrics\": [{metric_json}],\n  \"generationWinRates\": [{generation_win_rates_json}],\n  \"models\": [{models}],\n  \"frames\": {first_frames},\n  \"replayChunks\": [{replay_chunks_json}],\n  \"replayGames\": []\n}}\n",
        profile = evolution_config.profile_name,
        phase = TRAINING_PHASE_EVALUATION,
        turn = turn_index,
        live_replay_games = live_replay_game_count,
        run_id = telemetry_context.run_id,
        strict_game_rules = json_bool(evolution_config.strict_game_rules),
        episode_steps = agent_config.episode_steps,
        expected_frames_per_game = agent_config.episode_steps + 1,
        population_size = evolution_config.population_size,
        elite_count = evolution_config.elite_count,
        games_per_model = evolution_config.games_per_model,
        replays_per_model = agent_config.dashboard_replays_per_model,
        generation_replay_game_count = agent_config.dashboard_generation_replay_game_count,
        players_per_game = evolution_config.active_player_count,
        simultaneous_games = pipeline_stats.simultaneous_game_count,
        inference_batch_calls = pipeline_stats.inference_batch_call_count,
        max_inference_batch_size = pipeline_stats.max_inference_batch_size,
        map_source = NATIVE_SEEDED_REFERENCE_MAP_SOURCE,
        turn_loop = OFFICIAL_TURN_LOOP_NAME,
        full_replay = json_bool(live_full_replay),
        replay_frame_stride = agent_config.dashboard_live_replay_frame_stride,
        live_replay_format = LIVE_REPLAY_FORMAT,
        live_replay_path = live_replay_path,
        live_replay_frame_count = live_replay_frame_count,
        stored_replay_games = live_replay_game_count,
        games_per_second = progress_metric.games_per_second,
        turns_per_second = progress_metric.turns_per_second,
        gpu_utilization = UNMEASURED_GPU_UTILIZATION_PERCENT,
        cpu_workers = TRAINING_CPU_WORKER_COUNT,
        eval_queue = EMPTY_EVAL_QUEUE_DEPTH,
        gpu_warning = GPU_UTILIZATION_UNMEASURED_WARNING,
        live_warning = TRAINING_LIVE_REPLAY_PROGRESS_WARNING,
        metric_json = metric_json,
        generation_win_rates_json =
            generation_win_rates_json(telemetry_context.generation_validation_history),
        models = progress_models_json(evolution_config.population_size),
        first_frames = first_frames,
        replay_chunks_json = replay_chunks_json_preserving_existing(
            telemetry_context.run_id,
            telemetry_context.replay_history
        ),
    );
    let path = Path::new("dashboard/public/telemetry/latest.json");
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|error| format!("telemetry_dir_failed={error}"))?;
    }
    write_text_with_retries(path, &telemetry, "telemetry_write_failed")
}

fn progress_models_json(population_size: usize) -> String {
    (0..population_size)
        .map(|model_index| {
            format!(
                "{{\"id\":\"M-{model_index:03}\",\"parent\":\"pending\",\"rating\":{rating},\"wins\":0,\"draws\":0,\"losses\":0,\"games\":0,\"captures\":0,\"fleetHits\":0,\"hitShips\":0,\"sunDestroyedFleets\":0,\"sunDestroyedShips\":0,\"outOfBoundsFleets\":0,\"outOfBoundsShips\":0,\"launchActions\":0,\"launchedShips\":0,\"avgFleetSize\":0.0,\"mutation\":\"pending\",\"selected\":{selected}}}",
                rating = BASELINE_RATING,
                selected = json_bool(model_index == 0)
            )
        })
        .collect::<Vec<_>>()
        .join(",")
}

fn progress_metrics_json(metrics: &[MetricRecord], progress_metric: &MetricRecord) -> String {
    let mut excluded_generations = metrics
        .iter()
        .map(|metric| metric.generation)
        .collect::<Vec<_>>();
    excluded_generations.push(progress_metric.generation);
    let mut records = telemetry_array_items_excluding_generations("metrics", &excluded_generations);
    records.extend(dedup_metric_records_json(
        metrics,
        &[progress_metric.generation],
    ));
    records.join(",")
}

fn completed_metrics_json(metrics: &[MetricRecord]) -> String {
    let excluded_generations = metrics
        .iter()
        .map(|metric| metric.generation)
        .collect::<Vec<_>>();
    let mut records = telemetry_array_items_excluding_generations("metrics", &excluded_generations);
    records.extend(dedup_metric_records_json(metrics, &[]));
    records.join(",")
}

fn dedup_metric_records_json(
    metrics: &[MetricRecord],
    excluded_generations: &[usize],
) -> Vec<String> {
    let excluded = excluded_generations
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let mut by_generation = BTreeMap::new();
    for metric in metrics {
        if !excluded.contains(&metric.generation) {
            by_generation.insert(metric.generation, metric_json(metric));
        }
    }
    by_generation.into_values().collect()
}

fn generation_win_rates_json(records: &[GenerationValidationRecord]) -> String {
    dedup_generation_win_rate_items(telemetry_array_items("generationWinRates"), records).join(",")
}

fn dedup_generation_win_rate_items(
    existing_items: Vec<String>,
    records: &[GenerationValidationRecord],
) -> Vec<String> {
    let mut by_key = BTreeMap::new();
    for item in existing_items {
        if let Some(key) = json_generation_win_rate_key(&item) {
            by_key.insert(key, item);
        }
    }
    for record in records {
        by_key.insert(
            (record.validation_generation, record.evaluated_generation),
            generation_validation_json(record),
        );
    }
    by_key.into_values().collect()
}

fn replay_chunks_json_preserving_existing(
    run_id: &str,
    replay_history: &[ReplayChunkRecord],
) -> String {
    let excluded_generations = replay_history
        .iter()
        .map(|record| record.generation)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let mut items = replay_chunk_artifact_items(run_id, &excluded_generations);
    for item in telemetry_array_items_excluding_generations("replayChunks", &excluded_generations) {
        if let Some(generation) = json_generation_value(&item) {
            if items
                .iter()
                .any(|existing| json_generation_value(existing) == Some(generation))
            {
                continue;
            }
        }
        items.push(item);
    }
    let current = replay_chunks_json(run_id, replay_history);
    if !current.is_empty() {
        items.push(current);
    }
    items.join(",")
}

fn telemetry_array_items_excluding_generations(field: &str, generations: &[usize]) -> Vec<String> {
    let excluded = generations.iter().copied().collect::<BTreeSet<_>>();
    let mut non_generation_items = Vec::new();
    let mut by_generation = BTreeMap::new();
    for item in telemetry_array_items(field) {
        if let Some(generation) = json_generation_value(&item) {
            if !excluded.contains(&generation) {
                by_generation.insert(generation, item);
            }
        } else {
            non_generation_items.push(item);
        }
    }
    non_generation_items.extend(by_generation.into_values());
    non_generation_items
}

fn telemetry_array_items(field: &str) -> Vec<String> {
    let telemetry = match fs::read_to_string("dashboard/public/telemetry/latest.json") {
        Ok(telemetry) => telemetry,
        Err(_) => return Vec::new(),
    };
    extract_json_array(&telemetry, field)
        .map(split_json_array_items)
        .unwrap_or_default()
}

fn extract_json_array<'a>(source: &'a str, field: &str) -> Option<&'a str> {
    let marker = format!("\"{field}\"");
    let marker_index = source.find(&marker)?;
    let after_marker = &source[marker_index + marker.len()..];
    let bracket_offset = after_marker.find('[')?;
    let array_start = marker_index + marker.len() + bracket_offset;
    let mut depth = 0isize;
    let mut in_string = false;
    let mut escaped = false;
    for (offset, byte) in source[array_start..].bytes().enumerate() {
        if in_string {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                in_string = false;
            }
            continue;
        }
        match byte {
            b'"' => in_string = true,
            b'[' => depth += 1,
            b']' => {
                depth -= 1;
                if depth == 0 {
                    return Some(&source[array_start + 1..array_start + offset]);
                }
            }
            _ => {}
        }
    }
    None
}

fn split_json_array_items(items: &str) -> Vec<String> {
    let mut split = Vec::new();
    let mut item_start = 0usize;
    let mut depth = 0isize;
    let mut in_string = false;
    let mut escaped = false;
    for (index, byte) in items.bytes().enumerate() {
        if in_string {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                in_string = false;
            }
            continue;
        }
        match byte {
            b'"' => in_string = true,
            b'{' | b'[' => depth += 1,
            b'}' | b']' => depth -= 1,
            b',' if depth == 0 => {
                let item = items[item_start..index].trim();
                if !item.is_empty() {
                    split.push(item.to_string());
                }
                item_start = index + 1;
            }
            _ => {}
        }
    }
    let item = items[item_start..].trim();
    if !item.is_empty() {
        split.push(item.to_string());
    }
    split
}

fn json_generation_value(item: &str) -> Option<usize> {
    json_usize_field_value(item, "generation")
}

fn json_generation_win_rate_key(item: &str) -> Option<(usize, usize)> {
    Some((
        json_usize_field_value(item, "validationGeneration")?,
        json_usize_field_value(item, "evaluatedGeneration")?,
    ))
}

fn json_usize_field_value(item: &str, field: &str) -> Option<usize> {
    let marker = format!("\"{field}\"");
    let marker_index = item.find(&marker)?;
    let after_marker = &item[marker_index + marker.len()..];
    let colon_index = after_marker.find(':')?;
    let value = after_marker[colon_index + 1..].trim_start();
    let digit_count = value
        .bytes()
        .take_while(|byte| byte.is_ascii_digit())
        .count();
    if digit_count == 0 {
        return None;
    }
    value[..digit_count].parse().ok()
}

fn write_telemetry(
    generation: usize,
    evaluated: &[EvaluatedModel],
    evolution_config: &EvolutionConfig,
    metrics: &[MetricRecord],
    generation_validation_history: &[GenerationValidationRecord],
    replay_history: &[ReplayChunkRecord],
    agent_config: &AgentConfig,
    cli: &TrainerCli,
    run_id: &str,
) -> Result<(), String> {
    let first_frames = "[]";
    let latest_metric = metrics
        .last()
        .ok_or_else(|| "telemetry_metric_history_empty".to_string())?;
    let training_backend = if cli.use_cuda {
        TRAINING_BACKEND_CUDA
    } else {
        TRAINING_BACKEND_CPU
    };
    let training_mode = format!("{}_{}", evolution_config.profile_name, training_backend);
    let full_replay = cli.full_replays || cli.dashboard_strict;
    let telemetry = format!(
        "{{\n  \"sourceMessage\": \"run_profile={profile}; players_per_game={players_per_game}; simultaneous_games={simultaneous_games}; compute_timing=tracked; generation_tournament=enabled\",\n  \"runId\": \"{run_id}\",\n  \"activeGeneration\": {generation},\n  \"runProfile\": \"{profile}\",\n  \"trainingMode\": \"{training_mode}\",\n  \"strictGameRules\": {strict_game_rules},\n  \"episodeSteps\": {episode_steps},\n  \"expectedFramesPerGame\": {expected_frames_per_game},\n  \"populationSize\": {population_size},\n  \"eliteCount\": {elite_count},\n  \"gamesPerModel\": {games_per_model},\n  \"replaysPerModel\": {replays_per_model},\n  \"generationReplayGameCount\": {generation_replay_game_count},\n  \"playersPerGame\": {players_per_game},\n  \"simultaneousGames\": {simultaneous_games},\n  \"inferenceBatchCalls\": {inference_batch_calls},\n  \"maxInferenceBatchSize\": {max_inference_batch_size},\n  \"mapSource\": \"{map_source}\",\n  \"turnLoop\": \"{turn_loop}\",\n  \"fullReplay\": {full_replay},\n  \"replayFrameStride\": {replay_frame_stride},\n  \"storedReplayGames\": {stored_replay_games},\n  \"gamesPerSecond\": {games_per_second:.3},\n  \"turnsPerSecond\": {turns_per_second:.3},\n  \"gpuUtilization\": {gpu_utilization},\n  \"cpuWorkers\": {cpu_workers},\n  \"evalQueue\": {eval_queue},\n  \"submissionGate\": \"generation tournament top1 staged for host-side Kaggle submit when scheduled\",\n  \"validationErrors\": [],\n  \"warnings\": [\"{gpu_warning}\"],\n  \"metrics\": [{metrics_json}],\n  \"generationWinRates\": [{generation_win_rates_json}],\n  \"models\": [{models}],\n  \"frames\": {first_frames},\n  \"replayChunks\": [{replay_chunks_json}],\n  \"replayGames\": []\n}}\n",
        profile = evolution_config.profile_name,
        strict_game_rules = json_bool(evolution_config.strict_game_rules),
        episode_steps = agent_config.episode_steps,
        expected_frames_per_game = agent_config.episode_steps + 1,
        population_size = evolution_config.population_size,
        elite_count = evolution_config.elite_count,
        games_per_model = evolution_config.games_per_model,
        replays_per_model = agent_config.dashboard_replays_per_model,
        generation_replay_game_count = agent_config.dashboard_generation_replay_game_count,
        players_per_game = evolution_config.active_player_count,
        simultaneous_games = latest_metric.simultaneous_games,
        inference_batch_calls = latest_metric.inference_batch_calls,
        max_inference_batch_size = latest_metric.max_inference_batch_size,
        map_source = NATIVE_SEEDED_REFERENCE_MAP_SOURCE,
        turn_loop = OFFICIAL_TURN_LOOP_NAME,
        full_replay = json_bool(full_replay),
        replay_frame_stride = agent_config.dashboard_replay_frame_stride,
        stored_replay_games = replay_chunk_record_game_count(replay_history),
        games_per_second = latest_metric.games_per_second,
        turns_per_second = latest_metric.turns_per_second,
        gpu_utilization = latest_metric.gpu_utilization,
        cpu_workers = TRAINING_CPU_WORKER_COUNT,
        eval_queue = EMPTY_EVAL_QUEUE_DEPTH,
        gpu_warning = GPU_UTILIZATION_UNMEASURED_WARNING,
        metrics_json = completed_metrics_json(metrics),
        generation_win_rates_json = generation_win_rates_json(generation_validation_history),
        models = evaluated
            .iter()
            .map(|model| format!(
                "{{\"id\":\"M-{id:03}\",\"parent\":\"local\",\"rating\":{rating},\"wins\":{wins},\"draws\":{draws},\"losses\":{losses},\"games\":{games},\"captures\":{captures},\"fleetHits\":{fleet_hits},\"hitShips\":{hit_ships},\"sunDestroyedFleets\":{sun_destroyed_fleets},\"sunDestroyedShips\":{sun_destroyed_ships},\"outOfBoundsFleets\":{out_of_bounds_fleets},\"outOfBoundsShips\":{out_of_bounds_ships},\"launchActions\":{launch_actions},\"launchedShips\":{launched_ships},\"avgFleetSize\":{avg_fleet_size:.3},\"mutation\":\"self-play\",\"selected\":{selected}}}",
                id = model.model_index,
                rating = BASELINE_RATING + model.reward,
                wins = model.gameplay.win_count,
                draws = model.gameplay.draw_count,
                losses = model.gameplay.loss_count,
                games = model.gameplay.game_count,
                captures = model.gameplay.capture_count,
                fleet_hits = model.gameplay.fleet_hit_count,
                hit_ships = model.gameplay.hit_ship_count,
                sun_destroyed_fleets = model.gameplay.sun_destroyed_fleet_count,
                sun_destroyed_ships = model.gameplay.sun_destroyed_ship_count,
                out_of_bounds_fleets = model.gameplay.out_of_bounds_destroyed_fleet_count,
                out_of_bounds_ships = model.gameplay.out_of_bounds_destroyed_ship_count,
                launch_actions = model.gameplay.launch_action_count,
                launched_ships = model.gameplay.launched_ship_count,
                avg_fleet_size = model.gameplay.avg_fleet_size(),
                selected = json_bool(model_selected(evaluated, evolution_config, model.model_index))
            ))
            .collect::<Vec<_>>()
            .join(","),
        replay_chunks_json = replay_chunks_json_preserving_existing(run_id, replay_history)
    );
    let path = Path::new("dashboard/public/telemetry/latest.json");
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|error| format!("telemetry_dir_failed={error}"))?;
    }
    write_text_with_retries(path, &telemetry, "telemetry_write_failed")
}

fn write_text_with_retries(path: &Path, contents: &str, error_prefix: &str) -> Result<(), String> {
    let mut last_error = None;
    for attempt in 0..TELEMETRY_WRITE_ATTEMPTS {
        match write_text_atomic(path, contents) {
            Ok(()) => return Ok(()),
            Err(error) => {
                last_error = Some(error);
                if attempt + 1 < TELEMETRY_WRITE_ATTEMPTS {
                    std::thread::sleep(Duration::from_millis(TELEMETRY_WRITE_RETRY_DELAY_MS));
                }
            }
        }
    }
    let error = last_error
        .map(|error| error.to_string())
        .unwrap_or_else(|| "unknown".to_string());
    Err(format!("{error_prefix}={error}"))
}

fn write_text_atomic(path: &Path, contents: &str) -> Result<(), std::io::Error> {
    let tmp_path = path.with_extension("json.tmp");
    let _ = fs::remove_file(&tmp_path);
    fs::write(&tmp_path, contents)?;
    fs::rename(&tmp_path, path)?;
    Ok(())
}

fn create_live_replay_storage(
    generation: usize,
    active_games: &[ActiveGame],
    live_capture_count: usize,
    active_player_count: usize,
    run_id: &str,
    _agent_config: &AgentConfig,
) -> Result<LiveReplayStorage, String> {
    let file_path = live_replay_file_path(run_id, generation);
    let public_path = live_replay_public_path(run_id, generation);
    let path = Path::new(&file_path);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|error| format!("live_replay_dir_failed={error}"))?;
    }
    let captured_game_count = live_capture_count.min(active_games.len());
    let mut bytes = Vec::new();
    bytes.extend_from_slice(LIVE_REPLAY_BINARY_MAGIC);
    bytes.push(LIVE_REPLAY_RECORD_HEADER);
    push_usize_as_u32(&mut bytes, generation, "live_replay_generation")?;
    push_usize_as_u32(
        &mut bytes,
        active_player_count,
        "live_replay_active_player_count",
    )?;
    push_usize_as_u32(
        &mut bytes,
        captured_game_count,
        "live_replay_captured_game_count",
    )?;
    for (game_index, game) in active_games.iter().take(captured_game_count).enumerate() {
        push_usize_as_u32(&mut bytes, game_index, "live_replay_game_index")?;
        push_usize_as_u32(
            &mut bytes,
            active_player_count,
            "live_replay_participant_count",
        )?;
        for player_slot in 0..active_player_count {
            push_usize_as_u32(
                &mut bytes,
                game.assignment.model_indices[player_slot],
                "live_replay_model_index",
            )?;
            push_string(
                &mut bytes,
                &opponent_label_for_slot(&game.assignment, active_player_count, player_slot),
                "live_replay_opponent_label",
            )?;
        }
    }
    write_binary_with_retries(path, &bytes, "live_replay_write_failed")?;
    Ok(LiveReplayStorage {
        file_path,
        public_path,
        frame_count: 0,
        game_count: captured_game_count,
        active_player_count,
    })
}

fn append_live_replay_frame(
    storage: &mut LiveReplayStorage,
    game_index: usize,
    frame: &ReplayFrame,
) -> Result<(), String> {
    if game_index >= storage.game_count {
        return Err(format!("live_replay_game_index_out_of_range={game_index}"));
    }
    let mut bytes = Vec::new();
    bytes.push(LIVE_REPLAY_RECORD_FRAME);
    push_usize_as_u32(&mut bytes, game_index, "live_replay_frame_game")?;
    push_replay_frame_binary(&mut bytes, frame)?;
    append_binary_with_retries(
        Path::new(&storage.file_path),
        &bytes,
        "live_replay_append_frame_failed",
    )?;
    storage.frame_count += 1;
    Ok(())
}

fn append_live_replay_result(
    storage: &LiveReplayStorage,
    game_index: usize,
    rewards: &[i32; MAX_PLAYER_SLOTS],
    active_player_count: usize,
) -> Result<(), String> {
    if game_index >= storage.game_count {
        return Err(format!("live_replay_game_index_out_of_range={game_index}"));
    }
    let mut bytes = Vec::new();
    bytes.push(LIVE_REPLAY_RECORD_RESULT);
    push_usize_as_u32(&mut bytes, game_index, "live_replay_result_game")?;
    let reward_count = active_player_count.min(storage.active_player_count);
    push_usize_as_u32(&mut bytes, reward_count, "live_replay_result_reward_count")?;
    for reward in rewards.iter().take(reward_count) {
        push_i32(&mut bytes, *reward);
    }
    append_binary_with_retries(
        Path::new(&storage.file_path),
        &bytes,
        "live_replay_append_result_failed",
    )
}

fn push_replay_frame_binary(buffer: &mut Vec<u8>, frame: &ReplayFrame) -> Result<(), String> {
    push_usize_as_u32(buffer, frame.step, "live_replay_frame_step")?;
    push_usize_as_u32(buffer, frame.planets.len(), "live_replay_planet_count")?;
    for planet in &frame.planets {
        push_i32(buffer, planet.id);
        push_i32(buffer, planet.owner);
        push_f32(buffer, planet.x);
        push_f32(buffer, planet.y);
        push_f32(buffer, planet.radius);
        push_f32(buffer, planet.ships);
        push_f32(buffer, planet.production);
    }
    push_usize_as_u32(buffer, frame.fleets.len(), "live_replay_fleet_count")?;
    for fleet in &frame.fleets {
        push_i32(buffer, fleet.id);
        push_i32(buffer, fleet.owner);
        push_f32(buffer, fleet.x);
        push_f32(buffer, fleet.y);
        push_f32(buffer, fleet.angle);
        push_f32(buffer, fleet.ships);
    }
    push_usize_as_u32(
        buffer,
        frame.comet_groups.len(),
        "live_replay_comet_group_count",
    )?;
    for group in &frame.comet_groups {
        push_usize_as_u32(
            buffer,
            group.planet_ids.len(),
            "live_replay_comet_planet_count",
        )?;
        for planet_id in &group.planet_ids {
            push_i32(buffer, *planet_id);
        }
        push_usize_as_u32(buffer, group.paths.len(), "live_replay_comet_path_count")?;
        for path in &group.paths {
            push_usize_as_u32(buffer, path.len(), "live_replay_comet_path_point_count")?;
            for (x, y) in path {
                push_f32(buffer, *x);
                push_f32(buffer, *y);
            }
        }
        push_isize_as_i32(buffer, group.path_index, "live_replay_comet_path_index")?;
    }
    Ok(())
}

fn push_usize_as_u32(buffer: &mut Vec<u8>, value: usize, field: &str) -> Result<(), String> {
    let converted = u32::try_from(value).map_err(|_| format!("{field}_overflow"))?;
    buffer.extend_from_slice(&converted.to_le_bytes());
    Ok(())
}

fn push_isize_as_i32(buffer: &mut Vec<u8>, value: isize, field: &str) -> Result<(), String> {
    let converted = i32::try_from(value).map_err(|_| format!("{field}_overflow"))?;
    push_i32(buffer, converted);
    Ok(())
}

fn push_string(buffer: &mut Vec<u8>, value: &str, field: &str) -> Result<(), String> {
    push_usize_as_u32(buffer, value.len(), field)?;
    buffer.extend_from_slice(value.as_bytes());
    Ok(())
}

fn push_i32(buffer: &mut Vec<u8>, value: i32) {
    buffer.extend_from_slice(&value.to_le_bytes());
}

fn push_f32(buffer: &mut Vec<u8>, value: f32) {
    buffer.extend_from_slice(&value.to_le_bytes());
}

fn write_binary_with_retries(
    path: &Path,
    contents: &[u8],
    error_prefix: &str,
) -> Result<(), String> {
    let mut last_error = None;
    for attempt in 0..TELEMETRY_WRITE_ATTEMPTS {
        match fs::write(path, contents) {
            Ok(()) => return Ok(()),
            Err(error) => {
                last_error = Some(error);
                if attempt + 1 < TELEMETRY_WRITE_ATTEMPTS {
                    std::thread::sleep(Duration::from_millis(TELEMETRY_WRITE_RETRY_DELAY_MS));
                }
            }
        }
    }
    let error = last_error
        .map(|error| error.to_string())
        .unwrap_or_else(|| "unknown".to_string());
    Err(format!("{error_prefix}={error}"))
}

fn append_binary_with_retries(
    path: &Path,
    contents: &[u8],
    error_prefix: &str,
) -> Result<(), String> {
    let mut last_error = None;
    for attempt in 0..TELEMETRY_WRITE_ATTEMPTS {
        match OpenOptions::new()
            .append(true)
            .open(path)
            .and_then(|mut file| file.write_all(contents))
        {
            Ok(()) => return Ok(()),
            Err(error) => {
                last_error = Some(error);
                if attempt + 1 < TELEMETRY_WRITE_ATTEMPTS {
                    std::thread::sleep(Duration::from_millis(TELEMETRY_WRITE_RETRY_DELAY_MS));
                }
            }
        }
    }
    let error = last_error
        .map(|error| error.to_string())
        .unwrap_or_else(|| "unknown".to_string());
    Err(format!("{error_prefix}={error}"))
}

fn write_replay_chunk(
    run_id: &str,
    generation: usize,
    replay_games: &[ReplayGame],
) -> Result<(), String> {
    let path_string = replay_chunk_path(run_id, generation);
    let path = Path::new(&path_string);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|error| format!("replay_chunk_dir_failed={error}"))?;
    }
    let actual_games = replay_actual_games(replay_games);
    if !actual_games.is_empty() {
        for participants in &actual_games {
            let first = participants
                .first()
                .ok_or_else(|| "indexed_replay_empty_game".to_string())?;
            let game_path_string = replay_game_file_path(run_id, generation, first.game_index);
            let game_path = Path::new(&game_path_string);
            if let Some(parent) = game_path.parent() {
                fs::create_dir_all(parent)
                    .map_err(|error| format!("indexed_replay_game_dir_failed={error}"))?;
            }
            let game_chunk = format!(
                "{{\n  \"generation\": {generation},\n  \"gameIndex\": {},\n  \"format\": \"compact_replay_game_v1\",\n  \"game\": {}\n}}\n",
                first.game_index,
                compact_replay_actual_game_json(first, participants)
            );
            write_text_with_retries(game_path, &game_chunk, "indexed_replay_game_write_failed")?;
        }
        let chunk = format!(
            "{{\n  \"generation\": {generation},\n  \"format\": \"indexed_replay_v1\",\n  \"actualGameCount\": {},\n  \"gameCount\": {},\n  \"games\": [{}]\n}}\n",
            actual_games.len(),
            replay_games.len(),
            indexed_replay_games_json(run_id, generation, &actual_games)
        );
        return write_text_with_retries(path, &chunk, "replay_chunk_write_failed");
    }
    let chunk = format!(
        "{{\n  \"generation\": {generation},\n  \"format\": \"compact_replay_v2\",\n  \"games\": [{replay_games_json}]\n}}\n",
        replay_games_json = compact_replay_actual_games_json(replay_games)
    );
    fs::write(path, chunk).map_err(|error| format!("replay_chunk_write_failed={error}"))
}

fn compact_live_slot_to_owlive_from_replay_games(
    run_id: &str,
    generation: usize,
    replay_games: &[ReplayGame],
    active_player_count: usize,
) -> Result<Option<LiveReplayCompactionStats>, String> {
    let append_path_string = live_replay_file_path(run_id, generation);
    let append_path = Path::new(&append_path_string);
    if append_path.exists() {
        let compact_bytes = append_path
            .metadata()
            .map_err(|error| format!("live_replay_append_metadata_failed={error}"))?
            .len();
        let actual_games = replay_actual_games(replay_games);
        return Ok(Some(LiveReplayCompactionStats {
            game_count: actual_games.len(),
            compact_bytes,
            source: "append",
        }));
    }
    let slot_path_string = legacy_live_replay_slot_file_path(run_id, generation);
    let slot_path = Path::new(&slot_path_string);
    if !slot_path.exists() {
        return Ok(None);
    }
    let actual_games = replay_actual_games(replay_games);
    if actual_games.is_empty() {
        return Ok(None);
    }
    let compact_path_string = live_replay_file_path(run_id, generation);
    let compact_path = Path::new(&compact_path_string);
    if let Some(parent) = compact_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("live_replay_compact_dir_failed={error}"))?;
    }
    let tmp_path = compact_path.with_extension("owlive.tmp");
    let mut bytes = Vec::new();
    bytes.extend_from_slice(LIVE_REPLAY_BINARY_MAGIC);
    bytes.push(LIVE_REPLAY_RECORD_HEADER);
    push_usize_as_u32(&mut bytes, generation, "live_replay_compact_generation")?;
    push_usize_as_u32(
        &mut bytes,
        active_player_count,
        "live_replay_compact_active_player_count",
    )?;
    push_usize_as_u32(
        &mut bytes,
        actual_games.len(),
        "live_replay_compact_game_count",
    )?;
    for participants in &actual_games {
        let first = participants
            .first()
            .ok_or_else(|| "live_replay_compact_empty_game".to_string())?;
        push_usize_as_u32(
            &mut bytes,
            first.game_index,
            "live_replay_compact_game_index",
        )?;
        push_usize_as_u32(
            &mut bytes,
            participants.len(),
            "live_replay_compact_participant_count",
        )?;
        for participant in participants {
            push_usize_as_u32(
                &mut bytes,
                participant.model_index,
                "live_replay_compact_model_index",
            )?;
            push_string(
                &mut bytes,
                &participant.opponent_label,
                "live_replay_compact_opponent_label",
            )?;
        }
    }
    for participants in &actual_games {
        let first = participants
            .first()
            .ok_or_else(|| "live_replay_compact_empty_game".to_string())?;
        for frame in first.frames.iter() {
            bytes.push(LIVE_REPLAY_RECORD_FRAME);
            push_usize_as_u32(
                &mut bytes,
                first.game_index,
                "live_replay_compact_frame_game",
            )?;
            push_replay_frame_binary(&mut bytes, frame)?;
        }
    }
    for participants in &actual_games {
        let first = participants
            .first()
            .ok_or_else(|| "live_replay_compact_empty_game".to_string())?;
        bytes.push(LIVE_REPLAY_RECORD_RESULT);
        push_usize_as_u32(
            &mut bytes,
            first.game_index,
            "live_replay_compact_result_game",
        )?;
        push_usize_as_u32(
            &mut bytes,
            participants.len(),
            "live_replay_compact_reward_count",
        )?;
        for participant in participants {
            push_i32(&mut bytes, participant.reward);
        }
    }
    write_binary_with_retries(&tmp_path, &bytes, "live_replay_compact_write_failed")?;
    fs::rename(&tmp_path, compact_path)
        .map_err(|error| format!("live_replay_compact_rename_failed={error}"))?;
    fs::remove_file(slot_path)
        .map_err(|error| format!("live_replay_slot_remove_failed={error}"))?;
    Ok(Some(LiveReplayCompactionStats {
        game_count: actual_games.len(),
        compact_bytes: u64::try_from(bytes.len())
            .map_err(|_| "live_replay_compact_size_overflow".to_string())?,
        source: "owslot",
    }))
}

fn replay_actual_games(replay_games: &[ReplayGame]) -> Vec<Vec<&ReplayGame>> {
    let mut by_actual_game = BTreeMap::<(usize, usize), Vec<&ReplayGame>>::new();
    for game in replay_games {
        by_actual_game
            .entry((game.generation, game.game_index))
            .or_default()
            .push(game);
    }
    by_actual_game
        .into_values()
        .map(|mut participants| {
            participants.sort_by(|left, right| {
                left.opponent_label
                    .cmp(&right.opponent_label)
                    .then_with(|| left.model_index.cmp(&right.model_index))
            });
            participants
        })
        .collect()
}

fn replay_chunk_records_from_replay_games(replay_games: &[ReplayGame]) -> Vec<ReplayChunkRecord> {
    let mut generation_counts = BTreeMap::new();
    let mut actual_game_indices = BTreeMap::<usize, BTreeSet<usize>>::new();
    for game in replay_games {
        *generation_counts.entry(game.generation).or_insert(0usize) += 1;
        actual_game_indices
            .entry(game.generation)
            .or_default()
            .insert(game.game_index);
    }
    generation_counts
        .iter()
        .map(|(generation, game_count)| ReplayChunkRecord {
            generation: *generation,
            game_count: *game_count,
            actual_game_count: actual_game_indices
                .get(generation)
                .map(BTreeSet::len)
                .unwrap_or_default(),
        })
        .collect()
}

fn replay_chunk_record_game_count(replay_history: &[ReplayChunkRecord]) -> usize {
    replay_history
        .iter()
        .map(|record| record.game_count)
        .sum::<usize>()
}

fn replay_chunks_json(run_id: &str, replay_history: &[ReplayChunkRecord]) -> String {
    replay_history
        .iter()
        .map(|record| {
            format!(
                "{{\"generation\":{},\"path\":\"/telemetry/replays_{}_generation_{}.json\",\"gameCount\":{},\"actualGameCount\":{}}}",
                record.generation,
                run_id,
                record.generation,
                record.game_count,
                record.actual_game_count
            )
        })
        .collect::<Vec<_>>()
        .join(",")
}

fn replay_chunk_artifact_items(run_id: &str, excluded_generations: &[usize]) -> Vec<String> {
    let excluded = excluded_generations
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    replay_artifact_generations(run_id)
        .into_iter()
        .filter(|generation| !excluded.contains(generation))
        .map(|generation| {
            format!(
                "{{\"generation\":{},\"path\":\"/telemetry/replays_{}_generation_{}.json\",\"gameCount\":{},\"actualGameCount\":{}}}",
                generation,
                run_id,
                generation,
                DEFAULT_FULL_REPLAY_PARTICIPANT_VIEWS,
                DEFAULT_FULL_REPLAY_ACTUAL_GAMES
            )
        })
        .collect()
}

fn replay_artifact_generations(run_id: &str) -> BTreeSet<usize> {
    let mut generations = BTreeSet::new();
    let telemetry_dir = Path::new("dashboard/public/telemetry");
    let entries = match fs::read_dir(telemetry_dir) {
        Ok(entries) => entries,
        Err(_) => return generations,
    };
    let replay_prefix = format!("replays_{run_id}_generation_");
    let live_prefix = format!("live_{run_id}_generation_");
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if let Some(generation) = generation_from_named_artifact(&name, &replay_prefix, ".json")
            .or_else(|| generation_from_named_artifact(&name, &live_prefix, ".owlive"))
        {
            generations.insert(generation);
        }
    }
    generations
}

fn generation_from_named_artifact(name: &str, prefix: &str, suffix: &str) -> Option<usize> {
    name.strip_prefix(prefix)?
        .strip_suffix(suffix)?
        .parse()
        .ok()
}

fn write_generation_log_artifact(
    run_id: &str,
    generation: usize,
    metric: &MetricRecord,
    evaluated: &[EvaluatedModel],
    replay_games: &[ReplayGame],
    top_model_count: usize,
) -> Result<(), String> {
    let path_string = generation_log_path(run_id, generation);
    let path = Path::new(&path_string);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|error| format!("generation_log_dir_failed={error}"))?;
    }
    let replay_game_count = replay_games.len();
    let actual_game_count = replay_games
        .iter()
        .map(|game| game.game_index)
        .collect::<BTreeSet<_>>()
        .len();
    let top_models = evaluated
        .iter()
        .take(top_model_count)
        .map(|model| {
            format!(
                "{{\"modelIndex\":{},\"modelId\":\"M-{:03}\",\"reward\":{}}}",
                model.model_index, model.model_index, model.reward
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    let log = format!(
        "{{\n  \"runId\": \"{run_id}\",\n  \"generation\": {generation},\n  \"metric\": {metric},\n  \"replayChunkPath\": \"/telemetry/replays_{run_id}_generation_{generation}.json\",\n  \"replayGameCount\": {replay_game_count},\n  \"actualReplayGameCount\": {actual_game_count},\n  \"topModelsPath\": \"{top_models_path}\",\n  \"topModels\": [{top_models}]\n}}\n",
        metric = metric_json(metric),
        top_models_path = generation_top_models_public_path(run_id, generation),
    );
    write_text_with_retries(path, &log, "generation_log_write_failed")
}

fn generation_log_path(run_id: &str, generation: usize) -> String {
    format!("dashboard/public/telemetry/generation_logs/{run_id}/generation_{generation}.json")
}

fn replay_chunk_path(run_id: &str, generation: usize) -> String {
    format!("dashboard/public/telemetry/replays_{run_id}_generation_{generation}.json")
}

fn replay_game_file_path(run_id: &str, generation: usize, game_index: usize) -> String {
    format!(
        "dashboard/public/telemetry/replay_games/{run_id}/generation_{generation}_game_{game_index}.json"
    )
}

fn replay_game_public_path(run_id: &str, generation: usize, game_index: usize) -> String {
    format!("/telemetry/replay_games/{run_id}/generation_{generation}_game_{game_index}.json")
}

fn live_replay_file_path(run_id: &str, generation: usize) -> String {
    format!("dashboard/public/telemetry/live_{run_id}_generation_{generation}.owlive")
}

fn live_replay_public_path(run_id: &str, generation: usize) -> String {
    format!("/telemetry/live_{run_id}_generation_{generation}.owlive")
}

fn legacy_live_replay_slot_file_path(run_id: &str, generation: usize) -> String {
    format!("dashboard/public/telemetry/live_{run_id}_generation_{generation}.owslot")
}

fn model_selected(
    evaluated: &[EvaluatedModel],
    evolution_config: &EvolutionConfig,
    model_index: usize,
) -> bool {
    evaluated
        .iter()
        .take(evolution_config.elite_count)
        .any(|model| model.model_index == model_index)
}

fn compact_replay_actual_games_json(replay_games: &[ReplayGame]) -> String {
    let mut by_actual_game = BTreeMap::<(usize, usize), Vec<&ReplayGame>>::new();
    for game in replay_games {
        by_actual_game
            .entry((game.generation, game.game_index))
            .or_default()
            .push(game);
    }
    by_actual_game
        .values()
        .filter_map(|games| {
            games
                .first()
                .map(|first| compact_replay_actual_game_json(first, games))
        })
        .collect::<Vec<_>>()
        .join(",")
}

fn compact_replay_actual_game_json(first: &ReplayGame, participants: &[&ReplayGame]) -> String {
    format!(
        "[{},{},[{}],{}]",
        first.generation,
        first.game_index,
        participants
            .iter()
            .map(|game| compact_replay_participant_json(game))
            .collect::<Vec<_>>()
            .join(","),
        compact_replay_frames_json(first.frames.as_ref())
    )
}

fn indexed_replay_games_json(
    run_id: &str,
    generation: usize,
    actual_games: &[Vec<&ReplayGame>],
) -> String {
    actual_games
        .iter()
        .filter_map(|participants| {
            participants
                .first()
                .map(|first| indexed_replay_game_json(run_id, generation, first, participants))
        })
        .collect::<Vec<_>>()
        .join(",")
}

fn indexed_replay_game_json(
    run_id: &str,
    generation: usize,
    first: &ReplayGame,
    participants: &[&ReplayGame],
) -> String {
    format!(
        "{{\"generation\":{},\"gameIndex\":{},\"path\":\"{}\",\"frameCount\":{},\"participants\":[{}]}}",
        first.generation,
        first.game_index,
        replay_game_public_path(run_id, generation, first.game_index),
        first.frames.len(),
        participants
            .iter()
            .map(|game| indexed_replay_participant_json(game))
            .collect::<Vec<_>>()
            .join(",")
    )
}

fn indexed_replay_participant_json(game: &ReplayGame) -> String {
    format!(
        "{{\"modelId\":\"M-{model:03}\",\"opponentId\":\"{opponent}\",\"reward\":{}}}",
        game.reward,
        model = game.model_index,
        opponent = game.opponent_label
    )
}

fn compact_replay_participant_json(game: &ReplayGame) -> String {
    format!(
        "[\"M-{model:03}\",\"{opponent}\",{}]",
        game.reward,
        model = game.model_index,
        opponent = game.opponent_label
    )
}

fn compact_replay_frames_json(frames: &[ReplayFrame]) -> String {
    format!(
        "[{}]",
        frames
            .iter()
            .map(compact_replay_frame_json)
            .collect::<Vec<_>>()
            .join(",")
    )
}

fn compact_replay_frame_json(frame: &ReplayFrame) -> String {
    format!(
        "[{},[{}],[{}],[{}]]",
        frame.step,
        frame
            .planets
            .iter()
            .map(compact_replay_planet_json)
            .collect::<Vec<_>>()
            .join(","),
        frame
            .fleets
            .iter()
            .map(compact_replay_fleet_json)
            .collect::<Vec<_>>()
            .join(","),
        frame
            .comet_groups
            .iter()
            .map(compact_replay_comet_group_json)
            .collect::<Vec<_>>()
            .join(",")
    )
}

fn compact_replay_comet_group_json(group: &ReplayCometGroup) -> String {
    format!(
        "[[{}],[{}],{}]",
        group
            .planet_ids
            .iter()
            .map(|planet_id| planet_id.to_string())
            .collect::<Vec<_>>()
            .join(","),
        group
            .paths
            .iter()
            .map(|path| compact_replay_comet_path_json(path))
            .collect::<Vec<_>>()
            .join(","),
        group.path_index
    )
}

fn compact_replay_comet_path_json(path: &[(f32, f32)]) -> String {
    format!(
        "[{}]",
        path.iter()
            .map(|(x, y)| format!("[{x:.4},{y:.4}]"))
            .collect::<Vec<_>>()
            .join(",")
    )
}

fn compact_replay_planet_json(planet: &Planet) -> String {
    format!(
        "[{},{},{:.4},{:.4},{:.4},{:.0},{:.0}]",
        planet.id, planet.owner, planet.x, planet.y, planet.radius, planet.ships, planet.production
    )
}

fn compact_replay_fleet_json(fleet: &Fleet) -> String {
    format!(
        "[{},{},{:.4},{:.4},{:.6},{:.0}]",
        fleet.id, fleet.owner, fleet.x, fleet.y, fleet.angle, fleet.ships
    )
}

fn metric_json(metric: &MetricRecord) -> String {
    format!(
        "{{\"generation\":{},\"winRate\":{:.3},\"gamesPerSecond\":{:.3},\"turnsPerSecond\":{:.3},\"p95LatencyMs\":{:.3},\"gpuUtilization\":{},\"evaluatedGames\":{},\"sampledReplayGames\":{},\"modelActionCalls\":{},\"launchActions\":{},\"launchedShips\":{},\"captures\":{},\"fleetHits\":{},\"hitShips\":{},\"sunDestroyedFleets\":{},\"sunDestroyedShips\":{},\"avgFleetSize\":{:.6},\"avgLaunchActionsPerTurn\":{:.6},\"avgLaunchedShipsPerTurn\":{:.6},\"avgModelActionMs\":{:.6},\"inferenceBatchCalls\":{},\"maxInferenceBatchSize\":{},\"simultaneousGames\":{},\"modelActionSeconds\":{:.6},\"simulationStepSeconds\":{:.6},\"evaluationSeconds\":{:.6},\"replaySeconds\":{:.6},\"replayWriteSeconds\":{:.6},\"generationValidationGames\":{},\"generationValidationSeconds\":{:.6},\"backpropSamples\":{},\"backpropModels\":{},\"backpropSeconds\":{:.6},\"reproductionSeconds\":{:.6},\"generationSeconds\":{:.6}}}",
        metric.generation,
        metric.win_rate,
        metric.games_per_second,
        metric.turns_per_second,
        metric.p95_latency_ms,
        metric.gpu_utilization,
        metric.evaluated_games,
        metric.sampled_replay_games,
        metric.model_action_calls,
        metric.launch_actions,
        metric.launched_ships,
        metric.captures,
        metric.fleet_hits,
        metric.hit_ships,
        metric.sun_destroyed_fleets,
        metric.sun_destroyed_ships,
        metric.avg_fleet_size,
        metric.avg_launch_actions_per_turn,
        metric.avg_launched_ships_per_turn,
        metric.avg_model_action_ms,
        metric.inference_batch_calls,
        metric.max_inference_batch_size,
        metric.simultaneous_games,
        metric.model_action_seconds,
        metric.simulation_step_seconds,
        metric.evaluation_seconds,
        metric.replay_seconds,
        metric.replay_write_seconds,
        metric.generation_validation_games,
        metric.generation_validation_seconds,
        metric.backprop_samples,
        metric.backprop_models,
        metric.backprop_seconds,
        metric.reproduction_seconds,
        metric.generation_seconds
    )
}

fn generation_validation_json(record: &GenerationValidationRecord) -> String {
    format!(
        "{{\"validationGeneration\":{},\"evaluatedGeneration\":{},\"modelCount\":{},\"games\":{},\"wins\":{},\"draws\":{},\"losses\":{},\"winRate\":{:.6}}}",
        record.validation_generation,
        record.evaluated_generation,
        record.model_count,
        record.game_count,
        record.win_count,
        record.draw_count,
        record.loss_count,
        record.win_rate
    )
}

fn json_bool(value: bool) -> &'static str {
    if value {
        "true"
    } else {
        "false"
    }
}

fn replay_frame_json(frame: &ReplayFrame) -> String {
    format!(
        "{{\"step\":{},\"planets\":[{}],\"fleets\":[{}],\"comets\":[{}]}}",
        frame.step,
        frame
            .planets
            .iter()
            .map(replay_planet_json)
            .collect::<Vec<_>>()
            .join(","),
        frame
            .fleets
            .iter()
            .map(replay_fleet_json)
            .collect::<Vec<_>>()
            .join(","),
        frame
            .comet_groups
            .iter()
            .map(replay_comet_group_json)
            .collect::<Vec<_>>()
            .join(",")
    )
}

fn replay_comet_group_json(group: &ReplayCometGroup) -> String {
    format!(
        "{{\"planetIds\":[{}],\"paths\":[{}],\"pathIndex\":{}}}",
        group
            .planet_ids
            .iter()
            .map(|planet_id| planet_id.to_string())
            .collect::<Vec<_>>()
            .join(","),
        group
            .paths
            .iter()
            .map(|path| compact_replay_comet_path_json(path))
            .collect::<Vec<_>>()
            .join(","),
        group.path_index
    )
}

fn replay_planet_json(planet: &Planet) -> String {
    format!(
        "{{\"id\":{},\"owner\":{},\"x\":{:.4},\"y\":{:.4},\"radius\":{:.4},\"ships\":{:.0},\"production\":{:.0}}}",
        planet.id, planet.owner, planet.x, planet.y, planet.radius, planet.ships, planet.production
    )
}

fn replay_fleet_json(fleet: &Fleet) -> String {
    format!(
        "{{\"id\":{},\"owner\":{},\"x\":{:.4},\"y\":{:.4},\"angle\":{:.6},\"ships\":{:.0}}}",
        fleet.id, fleet.owner, fleet.x, fleet.y, fleet.angle, fleet.ships
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_training_player_count_is_four_player() {
        let config = AgentConfig::default();
        assert_eq!(config.training_players_per_game, PLAYER_COUNT_FOUR);
    }

    #[test]
    fn checkpoint_save_path_uses_resume_checkpoint_when_supplied() {
        let config = AgentConfig::default();
        let cli = TrainerCli {
            smoke: false,
            dashboard_strict: false,
            use_cuda: true,
            full_replays: false,
            player_count_override: None,
            generation_override: None,
            resume_checkpoint: Some("artifacts/custom_resume.checkpoint.bin".to_string()),
        };

        assert_eq!(
            trainer_checkpoint_save_path(&cli, &config),
            "artifacts/custom_resume.checkpoint.bin"
        );
    }

    #[test]
    fn checkpoint_save_path_uses_config_default_without_resume() {
        let config = AgentConfig::default();
        let cli = TrainerCli {
            smoke: false,
            dashboard_strict: false,
            use_cuda: true,
            full_replays: false,
            player_count_override: None,
            generation_override: None,
            resume_checkpoint: None,
        };

        assert_eq!(
            trainer_checkpoint_save_path(&cli, &config),
            config.trainer_checkpoint_path
        );
    }

    #[test]
    fn game_assignments_count_participations_not_focus_games() {
        let config = AgentConfig::default();
        let assignments = game_assignments(
            config.training_population_size,
            config.default_self_play_games,
            config.training_players_per_game,
            &config,
        )
        .unwrap();
        let expected_game_count = config.training_population_size * config.default_self_play_games
            / config.training_players_per_game;
        assert_eq!(assignments.len(), expected_game_count);

        let mut participation_counts = vec![0usize; config.training_population_size];
        for assignment in assignments {
            for player_slot in 0..config.training_players_per_game {
                let model_index = assignment.model_indices[player_slot];
                assert!(!assignment.model_indices[..player_slot].contains(&model_index));
                participation_counts[model_index] += 1;
            }
        }
        assert!(participation_counts
            .iter()
            .all(|count| *count == config.default_self_play_games));
    }

    #[test]
    fn seeded_state_uses_official_like_four_player_map() {
        let config = AgentConfig::default();
        let state = seeded_state(&config, PLAYER_COUNT_FOUR, FIRST_GENERATION, 0).unwrap();
        assert!(state.planets.len() >= config.map_min_planet_groups * MAP_GROUP_SIZE);
        assert!(state.planets.len() <= config.map_max_planet_groups * MAP_GROUP_SIZE);
        assert_eq!(state.planets.len() % MAP_GROUP_SIZE, 0);
        for player in PLAYER_IDS {
            let homes = state
                .planets
                .iter()
                .filter(|planet| planet.owner == player)
                .count();
            assert_eq!(homes, 1);
        }
    }

    #[test]
    fn replay_games_expose_every_participant_model() {
        let config = AgentConfig::default();
        let assignments = game_assignments(
            config.trainer_smoke_population_size,
            config.trainer_smoke_games_per_model,
            PLAYER_COUNT_FOUR,
            &config,
        )
        .unwrap();
        let games = assignments
            .iter()
            .map(|_| GameRunResult {
                rewards: [config.training_reward_win; MAX_PLAYER_SLOTS],
                frames: Vec::new(),
                stats: ComputeStats::default(),
                gameplay: [GameplayStats::default(); MAX_PLAYER_SLOTS],
            })
            .collect::<Vec<_>>();
        let replay_games = replay_games_from_evaluation(
            FIRST_GENERATION,
            &assignments,
            &games,
            PLAYER_COUNT_FOUR,
            assignments.len(),
        );
        assert_eq!(
            replay_games.len(),
            config.trainer_smoke_population_size * config.trainer_smoke_games_per_model
        );

        let mut replay_counts = vec![0usize; config.trainer_smoke_population_size];
        for replay_game in replay_games {
            let self_label = format!("M-{:03}", replay_game.model_index);
            assert!(!replay_game.opponent_label.contains(&self_label));
            replay_counts[replay_game.model_index] += 1;
        }
        assert!(replay_counts
            .iter()
            .all(|count| *count == config.trainer_smoke_games_per_model));
    }

    #[test]
    fn replay_games_limit_actual_games_per_generation() {
        const CAPTURED_ACTUAL_GAMES: usize = 2;
        let config = AgentConfig::default();
        let assignments = game_assignments(
            config.trainer_smoke_population_size,
            config.trainer_smoke_games_per_model,
            PLAYER_COUNT_FOUR,
            &config,
        )
        .unwrap();
        let games = assignments
            .iter()
            .map(|_| GameRunResult {
                rewards: [config.training_reward_win; MAX_PLAYER_SLOTS],
                frames: Vec::new(),
                stats: ComputeStats::default(),
                gameplay: [GameplayStats::default(); MAX_PLAYER_SLOTS],
            })
            .collect::<Vec<_>>();
        let replay_games = replay_games_from_evaluation(
            FIRST_GENERATION,
            &assignments,
            &games,
            PLAYER_COUNT_FOUR,
            CAPTURED_ACTUAL_GAMES,
        );
        assert_eq!(
            replay_games.len(),
            CAPTURED_ACTUAL_GAMES * PLAYER_COUNT_FOUR
        );
        let mut game_view_counts = BTreeMap::<usize, usize>::new();
        for replay_game in replay_games {
            *game_view_counts.entry(replay_game.game_index).or_default() += 1;
        }
        assert_eq!(game_view_counts.len(), CAPTURED_ACTUAL_GAMES);
        assert!(game_view_counts
            .values()
            .all(|count| *count == PLAYER_COUNT_FOUR));
    }

    #[test]
    fn replay_games_share_frames_for_same_actual_game() {
        let config = AgentConfig::default();
        let assignments = game_assignments(
            config.trainer_smoke_population_size,
            config.trainer_smoke_games_per_model,
            PLAYER_COUNT_FOUR,
            &config,
        )
        .unwrap();
        let state = seeded_state(&config, PLAYER_COUNT_FOUR, FIRST_GENERATION, 0).unwrap();
        let frame = replay_frame_from_state(&state);
        let games = assignments
            .iter()
            .map(|_| GameRunResult {
                rewards: [config.training_reward_win; MAX_PLAYER_SLOTS],
                frames: vec![frame.clone()],
                stats: ComputeStats::default(),
                gameplay: [GameplayStats::default(); MAX_PLAYER_SLOTS],
            })
            .collect::<Vec<_>>();
        let replay_games = replay_games_from_evaluation(
            FIRST_GENERATION,
            &assignments,
            &games,
            PLAYER_COUNT_FOUR,
            1,
        );

        assert_eq!(replay_games.len(), PLAYER_COUNT_FOUR);
        for replay_game in replay_games.iter().skip(1) {
            assert!(Arc::ptr_eq(&replay_games[0].frames, &replay_game.frames));
        }
    }

    #[test]
    fn replay_chunk_records_keep_counts_without_frame_payloads() {
        const CAPTURED_ACTUAL_GAMES: usize = 2;
        let config = AgentConfig::default();
        let assignments = game_assignments(
            config.trainer_smoke_population_size,
            config.trainer_smoke_games_per_model,
            PLAYER_COUNT_FOUR,
            &config,
        )
        .unwrap();
        let state = seeded_state(&config, PLAYER_COUNT_FOUR, FIRST_GENERATION, 0).unwrap();
        let frame = replay_frame_from_state(&state);
        let games = assignments
            .iter()
            .map(|_| GameRunResult {
                rewards: [config.training_reward_win; MAX_PLAYER_SLOTS],
                frames: vec![frame.clone()],
                stats: ComputeStats::default(),
                gameplay: [GameplayStats::default(); MAX_PLAYER_SLOTS],
            })
            .collect::<Vec<_>>();
        let replay_games = replay_games_from_evaluation(
            FIRST_GENERATION,
            &assignments,
            &games,
            PLAYER_COUNT_FOUR,
            CAPTURED_ACTUAL_GAMES,
        );
        let records = replay_chunk_records_from_replay_games(&replay_games);

        assert_eq!(
            records,
            vec![ReplayChunkRecord {
                generation: FIRST_GENERATION,
                game_count: CAPTURED_ACTUAL_GAMES * PLAYER_COUNT_FOUR,
                actual_game_count: CAPTURED_ACTUAL_GAMES,
            }]
        );
        assert_eq!(
            replay_chunks_json("test_run", &records),
            "{\"generation\":1,\"path\":\"/telemetry/replays_test_run_generation_1.json\",\"gameCount\":8,\"actualGameCount\":2}"
        );
        assert_eq!(
            replay_chunk_record_game_count(&records),
            CAPTURED_ACTUAL_GAMES * PLAYER_COUNT_FOUR
        );
        assert!(std::mem::size_of::<ReplayChunkRecord>() < std::mem::size_of::<ReplayGame>());
    }

    #[test]
    fn live_replay_storage_is_append_only_owlive() {
        const LIVE_APPEND_TEST_GENERATION: usize = 900_001;
        const LIVE_APPEND_TEST_RUN_ID: &str = "test_live_append_storage";
        let config = AgentConfig::default();
        let live_path_string =
            live_replay_file_path(LIVE_APPEND_TEST_RUN_ID, LIVE_APPEND_TEST_GENERATION);
        let slot_path_string =
            legacy_live_replay_slot_file_path(LIVE_APPEND_TEST_RUN_ID, LIVE_APPEND_TEST_GENERATION);
        let live_path = Path::new(&live_path_string);
        let slot_path = Path::new(&slot_path_string);
        let _ = fs::remove_file(live_path);
        let _ = fs::remove_file(slot_path);
        let state =
            seeded_state(&config, PLAYER_COUNT_FOUR, LIVE_APPEND_TEST_GENERATION, 0).unwrap();
        let frame = replay_frame_from_state(&state);
        let active_game = ActiveGame {
            assignment: GameAssignment {
                model_indices: [0, 1, 2, 3],
            },
            state,
            frames: Vec::new(),
            stats: ComputeStats::default(),
            gameplay: [GameplayStats::default(); MAX_PLAYER_SLOTS],
            rewards: [config.training_reward_draw; MAX_PLAYER_SLOTS],
            fleet_action_traces: BTreeMap::new(),
            completed: false,
            started_at: Instant::now(),
        };

        let mut storage = create_live_replay_storage(
            LIVE_APPEND_TEST_GENERATION,
            &[active_game],
            1,
            PLAYER_COUNT_FOUR,
            LIVE_APPEND_TEST_RUN_ID,
            &config,
        )
        .unwrap();
        assert!(storage.public_path.ends_with(".owlive"));
        assert!(!slot_path.exists());
        let header = fs::read(live_path).unwrap();
        assert!(header.starts_with(LIVE_REPLAY_BINARY_MAGIC));
        assert_eq!(
            header[LIVE_REPLAY_BINARY_MAGIC.len()],
            LIVE_REPLAY_RECORD_HEADER
        );
        let header_len = fs::metadata(live_path).unwrap().len();
        assert!(header_len < config.dashboard_live_replay_slot_bytes as u64);

        append_live_replay_frame(&mut storage, 0, &frame).unwrap();
        let frame_len = fs::metadata(live_path).unwrap().len();
        assert!(frame_len > header_len);
        assert!(frame_len < config.dashboard_live_replay_slot_bytes as u64);

        append_live_replay_result(
            &storage,
            0,
            &[config.training_reward_draw; MAX_PLAYER_SLOTS],
            PLAYER_COUNT_FOUR,
        )
        .unwrap();
        let result_len = fs::metadata(live_path).unwrap().len();
        assert!(result_len > frame_len);

        let _ = fs::remove_file(live_path);
    }

    #[test]
    fn collect_action_requests_skips_dead_players() {
        let config = AgentConfig::default();
        let mut state = seeded_state(&config, PLAYER_COUNT_FOUR, FIRST_GENERATION, 0).unwrap();
        for planet in state.planets.iter_mut() {
            if planet.owner == PLAYER_ONE {
                planet.owner = -1;
            }
        }
        state.fleets.retain(|fleet| fleet.owner != PLAYER_ONE);
        let active_game = ActiveGame {
            assignment: GameAssignment {
                model_indices: [0, 1, 2, 3],
            },
            state,
            frames: Vec::new(),
            stats: ComputeStats::default(),
            gameplay: [GameplayStats::default(); MAX_PLAYER_SLOTS],
            rewards: [config.training_reward_draw; MAX_PLAYER_SLOTS],
            fleet_action_traces: BTreeMap::new(),
            completed: false,
            started_at: Instant::now(),
        };
        let requests = collect_action_requests(&[active_game], &config, PLAYER_COUNT_FOUR).unwrap();
        assert_eq!(requests.len(), PLAYER_COUNT_FOUR - 1);
        assert!(requests
            .iter()
            .all(|request| request.player_id != PLAYER_ONE));
    }

    #[test]
    fn collect_action_requests_skips_players_with_only_fleets() {
        let config = AgentConfig::default();
        let mut state = seeded_state(&config, PLAYER_COUNT_FOUR, FIRST_GENERATION, 0).unwrap();
        let fleet_source_planet_id = state.planets[0].id;
        for planet in state.planets.iter_mut() {
            if planet.owner == PLAYER_ONE {
                planet.owner = -1;
            }
        }
        state.fleets.push(reference_fleet(
            state.next_fleet_id,
            PLAYER_ONE,
            config.board_center,
            config.board_center,
            0.0,
            fleet_source_planet_id,
            config.map_home_planet_ships,
        ));
        state.next_fleet_id += 1;
        assert!(state.alive_players().contains(&PLAYER_ONE));

        let active_game = ActiveGame {
            assignment: GameAssignment {
                model_indices: [0, 1, 2, 3],
            },
            state,
            frames: Vec::new(),
            stats: ComputeStats::default(),
            gameplay: [GameplayStats::default(); MAX_PLAYER_SLOTS],
            rewards: [config.training_reward_draw; MAX_PLAYER_SLOTS],
            fleet_action_traces: BTreeMap::new(),
            completed: false,
            started_at: Instant::now(),
        };
        let requests = collect_action_requests(&[active_game], &config, PLAYER_COUNT_FOUR).unwrap();
        assert_eq!(requests.len(), PLAYER_COUNT_FOUR - 1);
        assert!(requests
            .iter()
            .all(|request| request.player_id != PLAYER_ONE));
    }

    #[test]
    fn terminal_when_only_one_player_alive() {
        let config = AgentConfig::default();
        let mut state = seeded_state(&config, PLAYER_COUNT_FOUR, FIRST_GENERATION, 0).unwrap();
        for planet in state.planets.iter_mut() {
            if planet.owner != PLAYER_ZERO {
                planet.owner = -1;
            }
        }
        state.fleets.retain(|fleet| fleet.owner == PLAYER_ZERO);
        assert!(is_terminal(&state, &config));
    }

    #[test]
    fn trainer_checkpoint_roundtrip_preserves_population_and_archive() {
        let config = AgentConfig::default();
        let shape = True2DTransformerShape {
            layers: config.default_model_layers,
            row_count: config.true2d_internal_rows,
        };
        let evolution_config = EvolutionConfig {
            population_size: 1,
            elite_count: 1,
            games_per_model: 1,
            episode_steps: config.trainer_smoke_episode_steps,
            active_player_count: PLAYER_COUNT_TWO,
            profile_name: RUN_PROFILE_SMOKE,
            strict_game_rules: false,
        };
        let population =
            vec![True2DTransformer::seeded(shape, MODEL_SEED_OFFSET, &config).unwrap()];
        let champion_archive = vec![GenerationChampion {
            generation: FIRST_GENERATION,
            model: population[0].clone(),
        }];
        let path = std::env::temp_dir().join(format!(
            "orbit_wars_checkpoint_roundtrip_{}.bin",
            std::process::id()
        ));
        save_trainer_checkpoint(
            path.to_str().unwrap(),
            "test-run",
            FIRST_GENERATION + 1,
            &population,
            &champion_archive,
        )
        .unwrap();
        let loaded =
            load_trainer_checkpoint(path.to_str().unwrap(), shape, &config, &evolution_config)
                .unwrap();
        let _ = fs::remove_file(&path);
        assert_eq!(loaded.run_id, "test-run");
        assert_eq!(loaded.next_generation, FIRST_GENERATION + 1);
        assert_eq!(loaded.population.len(), 1);
        assert_eq!(loaded.population[0].weights, population[0].weights);
        assert_eq!(loaded.champion_archive.len(), 1);
        assert_eq!(loaded.champion_archive[0].generation, FIRST_GENERATION);
        assert_eq!(
            loaded.champion_archive[0].model.weights,
            population[0].weights
        );
    }

    #[test]
    fn training_samples_store_input_output_and_final_reward() {
        let config = AgentConfig::default();
        const TEST_GAME_INDEX: usize = 0;
        const TEST_PLAYER_SLOT: usize = 1;
        const TEST_MODEL_INDEX: usize = 2;
        const TEST_STEP: usize = 7;
        let request = ActionRequest {
            game_index: TEST_GAME_INDEX,
            player_slot: TEST_PLAYER_SLOT,
            player_id: PLAYER_ONE,
            model_index: TEST_MODEL_INDEX,
            step: TEST_STEP,
            planets: Vec::new(),
            initial_planets: Vec::new(),
            angular_velocity: 0.0,
            rows: vec![RowFeature::empty(&config); config.max_rows],
        };
        let outputs = vec![ActionOutput::repeated(0.25, 0.75); config.max_rows];
        let mut samples = TrainingSamples::with_capacity(1, &config).unwrap();
        record_training_samples(
            &mut samples,
            FIRST_GENERATION,
            std::iter::once(&request),
            &outputs,
            &config,
        )
        .unwrap();
        let mut rewards = [config.training_reward_loss; MAX_PLAYER_SLOTS];
        rewards[TEST_PLAYER_SLOT] = config.training_reward_win;
        let games = vec![GameRunResult {
            rewards,
            frames: Vec::new(),
            stats: ComputeStats::default(),
            gameplay: [GameplayStats::default(); MAX_PLAYER_SLOTS],
        }];
        samples.assign_rewards(&games).unwrap();

        assert_eq!(samples.sample_count(), 1);
        assert_eq!(
            samples.input_float_count(),
            config.max_rows * config.true2d_input_features
        );
        assert_eq!(
            samples.output_float_count(),
            config.max_rows * config.true2d_output_features
        );
        assert_eq!(samples.rewards, vec![config.training_reward_win]);
        assert_eq!(
            samples.sun_target_penalties.len(),
            config.max_rows * config.true2d_action_targets_per_source
        );
        assert!(samples
            .sun_target_penalties
            .iter()
            .all(|penalty| *penalty == 0.0));
        assert_eq!(
            samples.metadata[0],
            TrainingSampleMetadata {
                generation: FIRST_GENERATION,
                game_index: TEST_GAME_INDEX,
                player_slot: TEST_PLAYER_SLOT,
                player_id: PLAYER_ONE,
                model_index: TEST_MODEL_INDEX,
                step: TEST_STEP,
            }
        );
        assert_eq!(samples.output_rows[0], 0.25);
        assert_eq!(samples.output_rows[1], 0.75);
    }

    #[test]
    fn decode_sample_index_offsets_from_first_recorded_sample() {
        const FIRST_RECORDED_SAMPLE: usize = 10;
        const THIRD_REQUEST_INDEX: usize = 2;

        assert_eq!(
            decode_sample_index(Some(FIRST_RECORDED_SAMPLE), THIRD_REQUEST_INDEX),
            Some(FIRST_RECORDED_SAMPLE + THIRD_REQUEST_INDEX)
        );
        assert_eq!(decode_sample_index(None, THIRD_REQUEST_INDEX), None);
    }

    #[test]
    fn population_forward_scratch_reuses_packed_input_capacity() {
        let config = AgentConfig::default();
        let first_request = ActionRequest {
            game_index: 0,
            player_slot: 0,
            player_id: PLAYER_ZERO,
            model_index: 3,
            step: 0,
            planets: Vec::new(),
            initial_planets: Vec::new(),
            angular_velocity: 0.0,
            rows: vec![RowFeature::empty(&config); config.max_rows],
        };
        let second_request = ActionRequest {
            model_index: 5,
            ..first_request.clone()
        };
        let mut scratch = PopulationForwardScratch::default();

        scratch
            .pack_chunk(&[first_request.clone(), second_request], &config)
            .unwrap();
        let first_capacity = scratch.input_rows.capacity();
        assert_eq!(
            scratch.input_rows.len(),
            2 * config.max_rows * config.true2d_input_features
        );
        assert_eq!(scratch.model_indices, vec![3, 5]);

        scratch.pack_chunk(&[first_request], &config).unwrap();
        assert!(scratch.input_rows.capacity() >= first_capacity);
        assert_eq!(
            scratch.input_rows.len(),
            config.max_rows * config.true2d_input_features
        );
        assert_eq!(scratch.model_indices, vec![3]);
    }

    #[test]
    fn backprop_training_data_keeps_all_selected_model_samples() {
        let config = AgentConfig::default();
        const FIRST_TEST_MODEL: usize = 0;
        const SECOND_TEST_MODEL: usize = 1;
        const TEST_SOURCE_ROW: usize = 3;
        const TEST_TARGET_SLOT: usize = 4;
        let mut samples = TrainingSamples::with_capacity(2, &config).unwrap();
        for model_index in [FIRST_TEST_MODEL, SECOND_TEST_MODEL] {
            let request = ActionRequest {
                game_index: 0,
                player_slot: model_index,
                player_id: PLAYER_IDS[model_index],
                model_index,
                step: model_index,
                planets: Vec::new(),
                initial_planets: Vec::new(),
                angular_velocity: 0.0,
                rows: vec![RowFeature::empty(&config); config.max_rows],
            };
            let outputs = vec![ActionOutput::repeated(0.25, 0.75); config.max_rows];
            record_training_samples(
                &mut samples,
                FIRST_GENERATION,
                std::iter::once(&request),
                &outputs,
                &config,
            )
            .unwrap();
        }
        samples
            .add_sun_target_penalty(
                FIRST_TEST_MODEL,
                TEST_SOURCE_ROW,
                TEST_TARGET_SLOT,
                -config.training_sun_target_penalty_scale,
                &config,
            )
            .unwrap();
        let games = vec![GameRunResult {
            rewards: [
                config.training_reward_win,
                config.training_reward_loss,
                config.training_reward_draw,
                config.training_reward_draw,
            ],
            frames: Vec::new(),
            stats: ComputeStats::default(),
            gameplay: [GameplayStats::default(); MAX_PLAYER_SLOTS],
        }];
        samples.assign_rewards(&games).unwrap();
        let mut selected = vec![None; 2];
        selected[FIRST_TEST_MODEL] = Some(FIRST_TEST_MODEL);
        selected[SECOND_TEST_MODEL] = Some(SECOND_TEST_MODEL);

        let data = training_samples_for_selected_models(&samples, &selected, &config).unwrap();

        assert_eq!(data.rewards.len(), 2);
        assert_eq!(
            data.model_indices,
            vec![FIRST_TEST_MODEL, SECOND_TEST_MODEL]
        );
        assert_eq!(
            data.input_rows.len(),
            2 * config.max_rows * config.true2d_input_features
        );
        assert_eq!(
            data.output_rows.len(),
            2 * config.max_rows * config.true2d_output_features
        );
        assert_eq!(
            data.sun_target_penalties.len(),
            2 * config.max_rows * config.true2d_action_targets_per_source
        );
        let penalty_offset =
            TEST_SOURCE_ROW * config.true2d_action_targets_per_source + TEST_TARGET_SLOT;
        assert_eq!(
            data.sun_target_penalties[penalty_offset],
            -config.training_sun_target_penalty_scale
        );
        assert_eq!(data.model_count, 2);
    }

    #[test]
    fn sun_destroyed_fleet_adds_penalty_to_origin_target_slot() {
        let config = AgentConfig::default();
        const TEST_GAME_INDEX: usize = 0;
        const TEST_SOURCE_ROW: usize = 2;
        const TEST_TARGET_SLOT: usize = 5;
        const TEST_TARGET_ROW: usize = 8;
        const TEST_FLEET_ID: i32 = 42;
        const TEST_SHIP_COUNT: i32 = 7;
        let request = ActionRequest {
            game_index: TEST_GAME_INDEX,
            player_slot: 0,
            player_id: PLAYER_ZERO,
            model_index: 0,
            step: 0,
            planets: Vec::new(),
            initial_planets: Vec::new(),
            angular_velocity: 0.0,
            rows: vec![RowFeature::empty(&config); config.max_rows],
        };
        let outputs = vec![ActionOutput::repeated(0.25, 0.75); config.max_rows];
        let mut samples = TrainingSamples::with_capacity(1, &config).unwrap();
        record_training_samples(
            &mut samples,
            FIRST_GENERATION,
            std::iter::once(&request),
            &outputs,
            &config,
        )
        .unwrap();
        let mut game = ActiveGame {
            assignment: GameAssignment {
                model_indices: [0, 1, 2, 3],
            },
            state: seeded_state(
                &config,
                PLAYER_COUNT_FOUR,
                FIRST_GENERATION,
                TEST_GAME_INDEX,
            )
            .unwrap(),
            frames: Vec::new(),
            stats: ComputeStats::default(),
            gameplay: [GameplayStats::default(); MAX_PLAYER_SLOTS],
            rewards: [config.training_reward_draw; MAX_PLAYER_SLOTS],
            fleet_action_traces: BTreeMap::new(),
            completed: false,
            started_at: Instant::now(),
        };
        let events = SimulationStepEvents {
            launched_fleets: vec![orbit_wars_core::LaunchedFleetEvent {
                player: PLAYER_ZERO,
                fleet_id: TEST_FLEET_ID,
                ship_count: TEST_SHIP_COUNT,
            }],
            sun_destroyed_fleets: vec![orbit_wars_core::SunDestroyedFleetEvent {
                player: PLAYER_ZERO,
                fleet_id: TEST_FLEET_ID,
                ship_count: TEST_SHIP_COUNT,
            }],
            ..SimulationStepEvents::default()
        };
        let action_traces_by_player = vec![
            vec![ActionTrace {
                sample_index: 0,
                source_row: TEST_SOURCE_ROW,
                target_slot: TEST_TARGET_SLOT,
                target_row: TEST_TARGET_ROW,
            }],
            Vec::new(),
            Vec::new(),
            Vec::new(),
        ];

        apply_sun_target_penalty_events(
            &mut game,
            &events,
            &action_traces_by_player,
            PLAYER_COUNT_FOUR,
            &mut samples,
            &config,
        )
        .unwrap();

        let penalty_offset =
            TEST_SOURCE_ROW * config.true2d_action_targets_per_source + TEST_TARGET_SLOT;
        assert_eq!(
            samples.sun_target_penalties[penalty_offset],
            -config.training_sun_target_penalty_scale
        );
        assert!(game.fleet_action_traces.is_empty());
    }

    #[test]
    fn champion_archive_keeps_configured_generation_window() {
        let config = AgentConfig::default();
        let shape = True2DTransformerShape {
            layers: config.default_model_layers,
            row_count: config.true2d_internal_rows,
        };
        let population = (0..config.generation_archive_champions_per_generation)
            .map(|index| True2DTransformer::seeded(shape, index as u64, &config).unwrap())
            .collect::<Vec<_>>();
        let evaluated = (0..config.generation_archive_champions_per_generation)
            .map(|model_index| EvaluatedModel {
                model_index,
                reward: model_index as i32,
                gameplay: GameplayStats::default(),
            })
            .collect::<Vec<_>>();
        let mut archive = Vec::new();
        for generation in 1..=(config.generation_archive_max_generations + 1) {
            archive_generation_champions(
                &mut archive,
                generation,
                &evaluated,
                &population,
                &config,
            )
            .unwrap();
        }
        assert_eq!(archive.len(), config.generation_validation_max_models);
        assert_eq!(
            archive.first().unwrap().generation,
            config.generation_archive_max_generations + 1
                - config.generation_archive_max_generations
                + 1
        );
        assert_eq!(
            archive.last().unwrap().generation,
            config.generation_archive_max_generations + 1
        );
    }

    #[test]
    fn generation_tournament_uses_twenty_participations_per_champion_model() {
        const EXPECTED_TOURNAMENT_PARTICIPATIONS_PER_MODEL: usize = 20;

        let config = AgentConfig::default();
        assert_eq!(
            config.generation_validation_games_per_model,
            EXPECTED_TOURNAMENT_PARTICIPATIONS_PER_MODEL
        );
        let assignments = game_assignments(
            config.generation_validation_max_models,
            config.generation_validation_games_per_model,
            PLAYER_COUNT_FOUR,
            &config,
        )
        .unwrap();
        let expected_game_count = config.generation_validation_max_models
            * config.generation_validation_games_per_model
            / PLAYER_COUNT_FOUR;
        assert_eq!(assignments.len(), expected_game_count);
    }

    #[test]
    fn weighted_reproduction_slots_favor_stronger_elites() {
        let config = AgentConfig::default();
        let evolution_config = EvolutionConfig {
            population_size: config.training_population_size,
            elite_count: config.training_elite_count,
            games_per_model: config.default_self_play_games,
            episode_steps: config.episode_steps,
            active_player_count: PLAYER_COUNT_FOUR,
            profile_name: RUN_PROFILE_FULL,
            strict_game_rules: true,
        };
        let shape = True2DTransformerShape {
            layers: config.default_model_layers,
            row_count: config.true2d_internal_rows,
        };
        let model = True2DTransformer::seeded(shape, MODEL_SEED_OFFSET, &config).unwrap();
        let parents = (0..evolution_config.elite_count)
            .map(|index| {
                let win_count = config.default_self_play_games - index;
                let loss_count = config.default_self_play_games - win_count;
                ReproductionParent {
                    model: model.clone(),
                    score: EvaluatedModel {
                        model_index: index,
                        reward: (win_count as i32 * config.training_reward_win)
                            + (loss_count as i32 * config.training_reward_loss),
                        gameplay: GameplayStats {
                            game_count: config.default_self_play_games,
                            win_count,
                            loss_count,
                            ..GameplayStats::default()
                        },
                    },
                }
            })
            .collect::<Vec<_>>();
        let slots = reproduction_parent_slot_counts(&parents, &evolution_config, &config).unwrap();
        assert_eq!(slots.iter().sum::<usize>(), config.training_population_size);
        assert!(slots
            .iter()
            .all(|slot| *slot >= config.training_reproduction_min_slots_per_elite));
        assert!(slots
            .iter()
            .all(|slot| *slot <= config.training_reproduction_max_slots_per_elite));
        assert!(slots.first().unwrap() > slots.last().unwrap());
    }

    #[test]
    fn tournament_submit_candidate_prefers_winrate_before_reward() {
        const GAME_COUNT: usize = 10;
        const LOWER_WIN_COUNT: usize = 5;
        const HIGHER_WIN_COUNT: usize = 7;
        const LOWER_WINRATE_REWARD: i32 = 100;
        const HIGHER_WINRATE_REWARD: i32 = -10;

        let scores = vec![
            EvaluatedModel {
                model_index: 0,
                reward: LOWER_WINRATE_REWARD,
                gameplay: GameplayStats {
                    game_count: GAME_COUNT,
                    win_count: LOWER_WIN_COUNT,
                    loss_count: GAME_COUNT - LOWER_WIN_COUNT,
                    ..GameplayStats::default()
                },
            },
            EvaluatedModel {
                model_index: 1,
                reward: HIGHER_WINRATE_REWARD,
                gameplay: GameplayStats {
                    game_count: GAME_COUNT,
                    win_count: HIGHER_WIN_COUNT,
                    loss_count: GAME_COUNT - HIGHER_WIN_COUNT,
                    ..GameplayStats::default()
                },
            },
        ];

        let candidate = tournament_submit_candidate(&scores).unwrap();

        assert_eq!(candidate.model_index, 1);
    }

    #[test]
    fn weighted_reproduction_preserves_population_size() {
        const TEST_POPULATION_SIZE: usize = 4;
        const TEST_ELITE_COUNT: usize = 2;
        const TEST_MIN_SLOTS: usize = 1;
        const TEST_MAX_SLOTS: usize = 3;

        let mut config = AgentConfig::default();
        config.training_reproduction_min_slots_per_elite = TEST_MIN_SLOTS;
        config.training_reproduction_max_slots_per_elite = TEST_MAX_SLOTS;
        let evolution_config = EvolutionConfig {
            population_size: TEST_POPULATION_SIZE,
            elite_count: TEST_ELITE_COUNT,
            games_per_model: config.default_self_play_games,
            episode_steps: config.episode_steps,
            active_player_count: PLAYER_COUNT_TWO,
            profile_name: RUN_PROFILE_SMOKE,
            strict_game_rules: false,
        };
        let shape = True2DTransformerShape {
            layers: config.default_model_layers,
            row_count: config.true2d_internal_rows,
        };
        let model = True2DTransformer::seeded(shape, MODEL_SEED_OFFSET, &config).unwrap();
        let parents = (0..TEST_ELITE_COUNT)
            .map(|index| ReproductionParent {
                model: model.clone(),
                score: EvaluatedModel {
                    model_index: index,
                    reward: config.training_reward_win,
                    gameplay: GameplayStats {
                        game_count: 1,
                        win_count: 1,
                        ..GameplayStats::default()
                    },
                },
            })
            .collect::<Vec<_>>();
        let next_population =
            reproduce_from_weighted_parents(&parents, &evolution_config, &config).unwrap();
        assert_eq!(next_population.len(), TEST_POPULATION_SIZE);
    }

    #[test]
    fn checkpoint_population_shrinks_to_requested_size() {
        const CHECKPOINT_POPULATION_SIZE: usize = 2;
        const TARGET_POPULATION_SIZE: usize = 1;

        let config = AgentConfig::default();
        let evolution_config = EvolutionConfig {
            population_size: TARGET_POPULATION_SIZE,
            elite_count: TARGET_POPULATION_SIZE,
            games_per_model: config.default_self_play_games,
            episode_steps: config.episode_steps,
            active_player_count: PLAYER_COUNT_TWO,
            profile_name: RUN_PROFILE_SMOKE,
            strict_game_rules: false,
        };
        let shape = True2DTransformerShape {
            layers: config.default_model_layers,
            row_count: config.true2d_internal_rows,
        };
        let model = True2DTransformer::seeded(shape, MODEL_SEED_OFFSET, &config).unwrap();
        let population = vec![model; CHECKPOINT_POPULATION_SIZE];
        let adapted = adapt_checkpoint_population(population, &evolution_config).unwrap();
        assert_eq!(adapted.len(), TARGET_POPULATION_SIZE);
    }

    #[test]
    fn generation_validation_accumulator_counts_champion_models() {
        let config = AgentConfig::default();
        const FIRST_TEST_GENERATION: usize = 2;
        const SECOND_TEST_GENERATION: usize = 4;
        let shape = True2DTransformerShape {
            layers: config.default_model_layers,
            row_count: config.true2d_internal_rows,
        };
        let model = True2DTransformer::seeded(shape, MODEL_SEED_OFFSET, &config).unwrap();
        let archive = vec![
            GenerationChampion {
                generation: FIRST_TEST_GENERATION,
                model: model.clone(),
            },
            GenerationChampion {
                generation: FIRST_TEST_GENERATION,
                model: model.clone(),
            },
            GenerationChampion {
                generation: SECOND_TEST_GENERATION,
                model: model.clone(),
            },
            GenerationChampion {
                generation: SECOND_TEST_GENERATION,
                model,
            },
        ];
        let accumulators = generation_validation_accumulators(&archive);
        assert_eq!(
            accumulators
                .get(&FIRST_TEST_GENERATION)
                .unwrap()
                .model_count,
            PLAYER_COUNT_TWO
        );
        assert_eq!(
            accumulators
                .get(&SECOND_TEST_GENERATION)
                .unwrap()
                .model_count,
            PLAYER_COUNT_TWO
        );
    }

    #[test]
    fn generation_win_rate_json_dedups_by_validation_and_evaluated_generation() {
        const VALIDATION_GENERATION: usize = 32;
        const FIRST_EVALUATED_GENERATION: usize = 31;
        const SECOND_EVALUATED_GENERATION: usize = 32;
        const TEST_MODEL_COUNT: usize = 4;
        const TEST_GAME_COUNT: usize = 40;
        const OLD_WIN_COUNT: usize = 4;
        const NEW_WIN_COUNT: usize = 30;
        const DRAW_COUNT: usize = 0;
        const OLD_LOSS_COUNT: usize = 36;
        const NEW_LOSS_COUNT: usize = 10;
        const SECOND_WIN_COUNT: usize = 14;
        const SECOND_LOSS_COUNT: usize = 26;
        const UPDATED_WIN_RATE: f32 = 0.75;

        let existing_items = vec![
            format!(
                "{{\"validationGeneration\":{VALIDATION_GENERATION},\"evaluatedGeneration\":{FIRST_EVALUATED_GENERATION},\"modelCount\":{TEST_MODEL_COUNT},\"games\":{TEST_GAME_COUNT},\"wins\":{OLD_WIN_COUNT},\"draws\":{DRAW_COUNT},\"losses\":{OLD_LOSS_COUNT},\"winRate\":0.100000}}"
            ),
            format!(
                "{{\"validationGeneration\":{VALIDATION_GENERATION},\"evaluatedGeneration\":{FIRST_EVALUATED_GENERATION},\"modelCount\":{TEST_MODEL_COUNT},\"games\":{TEST_GAME_COUNT},\"wins\":{OLD_WIN_COUNT},\"draws\":{DRAW_COUNT},\"losses\":{OLD_LOSS_COUNT},\"winRate\":0.100000}}"
            ),
            format!(
                "{{\"validationGeneration\":{VALIDATION_GENERATION},\"evaluatedGeneration\":{SECOND_EVALUATED_GENERATION},\"modelCount\":{TEST_MODEL_COUNT},\"games\":{TEST_GAME_COUNT},\"wins\":{SECOND_WIN_COUNT},\"draws\":{DRAW_COUNT},\"losses\":{SECOND_LOSS_COUNT},\"winRate\":0.350000}}"
            ),
        ];
        let records = vec![GenerationValidationRecord {
            validation_generation: VALIDATION_GENERATION,
            evaluated_generation: FIRST_EVALUATED_GENERATION,
            model_count: TEST_MODEL_COUNT,
            game_count: TEST_GAME_COUNT,
            win_count: NEW_WIN_COUNT,
            draw_count: DRAW_COUNT,
            loss_count: NEW_LOSS_COUNT,
            win_rate: UPDATED_WIN_RATE,
        }];

        let deduped = dedup_generation_win_rate_items(existing_items, &records);
        let joined = deduped.join(",");

        assert_eq!(deduped.len(), 2);
        assert_eq!(
            joined
                .matches(&format!(
                    "\"evaluatedGeneration\":{FIRST_EVALUATED_GENERATION}"
                ))
                .count(),
            1
        );
        assert!(joined.contains("\"winRate\":0.750000"));
        assert!(!joined.contains("\"winRate\":0.100000"));
    }
}
