pub const MAX_ROWS: usize = 64;
pub const INITIAL_PLANET_SLOTS: usize = 40;
pub const COMET_SLOTS: usize = 20;
pub const PADDING_SLOTS: usize = 4;
pub const SHIP_LOG_PIVOT: f32 = 1_000.0;
pub const SHIP_LOG_CAP: f32 = 100_000.0;
pub const OWNER_CLASS_PLAYER_SLOTS: usize = 4;
pub const OWNER_CLASS_OWN: f32 = 1.0;
pub const OWNER_CLASS_ENEMY_1: f32 = 0.75;
pub const OWNER_CLASS_ENEMY_2: f32 = 0.5;
pub const OWNER_CLASS_ENEMY_3: f32 = 0.25;
pub const OWNER_CLASS_NEUTRAL: f32 = 0.0;
pub const OWNER_CLASS_EMPTY: f32 = -0.5;
pub const OWNER_CLASS_ABSENT_ON_MAP: f32 = -1.0;
pub const BOARD_SIZE: f32 = 100.0;
pub const BOARD_CENTER: f32 = 50.0;
pub const SUN_RADIUS: f32 = 10.0;
pub const ROTATION_RADIUS_LIMIT: f32 = 50.0;
pub const FLEET_SPEED_MAX: f32 = 6.0;
pub const FLEET_SPEED_REFERENCE_SHIPS: f32 = 1_000.0;
pub const FLEET_SPEED_CURVE_POWER: f32 = 1.5;
pub const INTERCEPT_ITERATIONS: usize = 128;
pub const MAX_PLAYERS: usize = 4;
pub const COMET_RADIUS: f32 = 1.0;
pub const COMET_PRODUCTION: f32 = 1.0;
pub const FLEET_SPAWN_OFFSET: f32 = 0.1;
pub const MAP_MIN_PLANET_GROUPS: usize = 5;
pub const MAP_MAX_PLANET_GROUPS: usize = 10;
pub const MAP_MIN_STATIC_GROUPS: usize = 3;
pub const MAP_PLANET_CLEARANCE: f32 = 7.0;
pub const MAP_HOME_PLANET_SHIPS: f32 = 10.0;
pub const DECODER_AIM_SEARCH_ANGLE_SAMPLES: usize = 128;
pub const DECODER_AIM_SEARCH_ANGLE_STEP_RADIANS: f32 = 0.01;
pub const DECODER_AIM_SEARCH_EXTRA_TICKS: usize = 8;
pub const DEFAULT_ANGULAR_VELOCITY: f32 = 0.03;
pub const DEFAULT_COMET_SPEED: f32 = 4.0;
pub const DEFAULT_EPISODE_STEPS: usize = 500;
pub const DEFAULT_SELF_PLAY_GAMES: usize = 20;
pub const DEFAULT_MUTATION_SIGMA: f32 = 0.02;
pub const DEFAULT_MODEL_LAYERS: usize = 61;
pub const DEFAULT_MODEL_D_MODEL: usize = 32;
pub const DEFAULT_MODEL_HEADS: usize = 4;
pub const TRUE2D_INTERNAL_ROWS: usize = MAX_ROWS;
pub const TRUE2D_INPUT_FEATURES: usize = 7;
pub const TRUE2D_ACTION_TARGETS_PER_SOURCE: usize = 12;
pub const TRUE2D_ACTION_TARGET_FEATURES: usize = 2;
pub const TRUE2D_OUTPUT_FEATURES: usize =
    TRUE2D_ACTION_TARGETS_PER_SOURCE * TRUE2D_ACTION_TARGET_FEATURES;
pub const TRUE2D_ATTENTION_HEADS: usize = 1;
pub const TRUE2D_ATTENTION_HEADS_MAX: usize = 16;
pub const TRUE2D_MATRIX_LAYER_PLANES: usize = 12;
pub const TRUE2D_MATRIX_LAYER_PLANES_MIN: usize = 1;
pub const TRUE2D_MATRIX_LAYER_PLANES_MAX: usize = 64;
pub const TRUE2D_TARGET_PARAMETER_COUNT: usize = 3_000_000;
pub const MODEL_D_MODEL_MIN: usize = 8;
pub const MODEL_D_MODEL_MAX: usize = 256;
pub const TRAINING_POPULATION_SIZE: usize = 64;
pub const TRAINING_ELITE_COUNT: usize = 6;
pub const TRAINING_PLAYERS_PER_GAME_TWO: usize = 2;
pub const TRAINING_PLAYERS_PER_GAME_FOUR: usize = 4;
pub const TRAINING_PLAYERS_PER_GAME: usize = TRAINING_PLAYERS_PER_GAME_FOUR;
pub const TRAINING_GENERATIONS_PER_RUN: usize = 1;
pub const TRAINING_DEMO_GENERATIONS: usize = 4;
pub const TRAINING_CAPTURE_TRAJECTORIES: bool = true;
pub const TRAINING_BACKPROP_ENABLED: bool = true;
pub const TRAINING_BACKPROP_LEARNING_RATE: f32 = 0.0005;
pub const TRAINING_BACKPROP_BATCH_SAMPLES: usize = 64;
pub const TRAINING_BACKPROP_SEND_THRESHOLD: f32 = 0.5;
pub const TRAINING_BACKPROP_REWARD_NORMALIZER: f32 = TRAINING_REWARD_WIN as f32;
pub const TRAINING_SUN_TARGET_PENALTY_SCALE: f32 = 4.0;
pub const TRAINING_REPRODUCTION_MIN_SLOTS_PER_ELITE: usize = 4;
pub const TRAINING_REPRODUCTION_MAX_SLOTS_PER_ELITE: usize = 18;
pub const TRAINING_REPRODUCTION_WINRATE_SMOOTHING_WINS: f32 = 1.0;
pub const TRAINING_REPRODUCTION_WINRATE_SMOOTHING_GAMES: f32 = 2.0;
pub const GENERATION_TOURNAMENT_KAGGLE_STAGE_FOR_HOST_SUBMIT: bool = true;
pub const KAGGLE_COMPETITION_SLUG: &str = "orbit-wars";
pub const KAGGLE_SUBMISSION_STAGE_DIR: &str = "artifacts/kaggle_auto_submit";
pub const KAGGLE_SUBMISSION_MAIN_PATH: &str = "kaggle_submission/main.py";
pub const KAGGLE_SUBMISSION_LIBRARY_PATH: &str = "target/release/liborbit_wars_ffi.so";
pub const KAGGLE_SUBMISSION_ARCHIVE_NAME: &str = "submission.tar.gz";
pub const GENERATION_ARCHIVE_CHAMPIONS_PER_GENERATION: usize = 4;
pub const GENERATION_ARCHIVE_MAX_GENERATIONS: usize = 32;
pub const GENERATION_VALIDATION_MAX_MODELS: usize = 128;
pub const GENERATION_VALIDATION_GAMES_PER_MODEL: usize = 20;
pub const GENERATION_VALIDATION_INTERVAL: usize = 32;
pub const GENERATION_REPLAY_INTERVAL: usize = 1;
pub const TRAINER_SMOKE_POPULATION_SIZE: usize = 8;
pub const TRAINER_SMOKE_ELITE_COUNT: usize = 2;
pub const TRAINER_SMOKE_GAMES_PER_MODEL: usize = 2;
pub const TRAINER_SMOKE_EPISODE_STEPS: usize = 64;
pub const DASHBOARD_STRICT_POPULATION_SIZE: usize = 4;
pub const DASHBOARD_STRICT_ELITE_COUNT: usize = 2;
pub const DASHBOARD_STRICT_GAMES_PER_MODEL: usize = 1;
pub const DASHBOARD_STRICT_REPLAYS_PER_MODEL: usize = 2;
pub const TRAINING_REWARD_WIN: i32 = 2;
pub const TRAINING_REWARD_DRAW: i32 = -1;
pub const TRAINING_REWARD_LOSS: i32 = -2;
pub const DASHBOARD_REPLAYS_PER_MODEL: usize = 10;
pub const DASHBOARD_GENERATION_REPLAY_GAME_COUNT: usize = 4;
pub const DASHBOARD_FULL_REPLAY_FRAME_STRIDE: usize = 1;
pub const DASHBOARD_REPLAY_FRAME_STRIDE: usize = 8;
pub const DASHBOARD_REPLAY_MAX_FRAMES: usize = 64;
pub const DASHBOARD_LIVE_REPLAY_ENABLED: bool = true;
pub const DASHBOARD_LIVE_REPLAY_GAME_COUNT: usize = 4;
pub const DASHBOARD_LIVE_REPLAY_FRAME_STRIDE: usize = 1;
pub const DASHBOARD_LIVE_REPLAY_UPDATE_INTERVAL_TURNS: usize = 1;
pub const DASHBOARD_LIVE_REPLAY_SLOT_BYTES: usize = 131_072;
pub const CUDA_TRUE2D_SMOKE_GAMES: usize = 2;
pub const CUDA_TRUE2D_FORWARD_TOLERANCE: f32 = 0.0005;
pub const CUDA_TRUE2D_MAX_REQUESTS_PER_FORWARD: usize = 8_192;

#[derive(Clone, Debug)]
pub struct AgentConfig {
    pub max_rows: usize,
    pub initial_planet_slots: usize,
    pub comet_slots: usize,
    pub padding_slots: usize,
    pub ship_log_pivot: f32,
    pub ship_log_cap: f32,
    pub owner_class_player_slots: usize,
    pub owner_class_own: f32,
    pub owner_class_enemy_1: f32,
    pub owner_class_enemy_2: f32,
    pub owner_class_enemy_3: f32,
    pub owner_class_neutral: f32,
    pub owner_class_empty: f32,
    pub owner_class_absent_on_map: f32,
    pub board_size: f32,
    pub board_center: f32,
    pub sun_radius: f32,
    pub rotation_radius_limit: f32,
    pub fleet_speed_max: f32,
    pub fleet_speed_reference_ships: f32,
    pub fleet_speed_curve_power: f32,
    pub intercept_iterations: usize,
    pub max_players: usize,
    pub comet_radius: f32,
    pub comet_production: f32,
    pub fleet_spawn_offset: f32,
    pub map_min_planet_groups: usize,
    pub map_max_planet_groups: usize,
    pub map_min_static_groups: usize,
    pub map_planet_clearance: f32,
    pub map_home_planet_ships: f32,
    pub decoder_aim_search_angle_samples: usize,
    pub decoder_aim_search_angle_step_radians: f32,
    pub decoder_aim_search_extra_ticks: usize,
    pub default_angular_velocity: f32,
    pub default_comet_speed: f32,
    pub episode_steps: usize,
    pub default_self_play_games: usize,
    pub default_mutation_sigma: f32,
    pub default_model_layers: usize,
    pub default_model_d_model: usize,
    pub default_model_heads: usize,
    pub true2d_internal_rows: usize,
    pub true2d_input_features: usize,
    pub true2d_output_features: usize,
    pub true2d_action_targets_per_source: usize,
    pub true2d_attention_heads: usize,
    pub true2d_matrix_layer_planes: usize,
    pub true2d_target_parameter_count: usize,
    pub training_population_size: usize,
    pub training_elite_count: usize,
    pub training_players_per_game: usize,
    pub training_generations_per_run: usize,
    pub training_demo_generations: usize,
    pub training_capture_trajectories: bool,
    pub training_backprop_enabled: bool,
    pub training_backprop_learning_rate: f32,
    pub training_backprop_batch_samples: usize,
    pub training_backprop_send_threshold: f32,
    pub training_backprop_reward_normalizer: f32,
    pub training_sun_target_penalty_scale: f32,
    pub training_reproduction_min_slots_per_elite: usize,
    pub training_reproduction_max_slots_per_elite: usize,
    pub training_reproduction_winrate_smoothing_wins: f32,
    pub training_reproduction_winrate_smoothing_games: f32,
    pub generation_tournament_kaggle_stage_for_host_submit: bool,
    pub kaggle_competition_slug: &'static str,
    pub kaggle_submission_stage_dir: &'static str,
    pub kaggle_submission_main_path: &'static str,
    pub kaggle_submission_library_path: &'static str,
    pub kaggle_submission_archive_name: &'static str,
    pub generation_archive_champions_per_generation: usize,
    pub generation_archive_max_generations: usize,
    pub generation_validation_max_models: usize,
    pub generation_validation_games_per_model: usize,
    pub generation_validation_interval: usize,
    pub generation_replay_interval: usize,
    pub trainer_smoke_population_size: usize,
    pub trainer_smoke_elite_count: usize,
    pub trainer_smoke_games_per_model: usize,
    pub trainer_smoke_episode_steps: usize,
    pub dashboard_strict_population_size: usize,
    pub dashboard_strict_elite_count: usize,
    pub dashboard_strict_games_per_model: usize,
    pub dashboard_strict_replays_per_model: usize,
    pub training_reward_win: i32,
    pub training_reward_draw: i32,
    pub training_reward_loss: i32,
    pub dashboard_replays_per_model: usize,
    pub dashboard_generation_replay_game_count: usize,
    pub dashboard_full_replay_frame_stride: usize,
    pub dashboard_replay_frame_stride: usize,
    pub dashboard_replay_max_frames: usize,
    pub dashboard_live_replay_enabled: bool,
    pub dashboard_live_replay_game_count: usize,
    pub dashboard_live_replay_frame_stride: usize,
    pub dashboard_live_replay_update_interval_turns: usize,
    pub dashboard_live_replay_slot_bytes: usize,
    pub cuda_true2d_smoke_games: usize,
    pub cuda_true2d_forward_tolerance: f32,
    pub cuda_true2d_max_requests_per_forward: usize,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_rows: MAX_ROWS,
            initial_planet_slots: INITIAL_PLANET_SLOTS,
            comet_slots: COMET_SLOTS,
            padding_slots: PADDING_SLOTS,
            ship_log_pivot: SHIP_LOG_PIVOT,
            ship_log_cap: SHIP_LOG_CAP,
            owner_class_player_slots: OWNER_CLASS_PLAYER_SLOTS,
            owner_class_own: OWNER_CLASS_OWN,
            owner_class_enemy_1: OWNER_CLASS_ENEMY_1,
            owner_class_enemy_2: OWNER_CLASS_ENEMY_2,
            owner_class_enemy_3: OWNER_CLASS_ENEMY_3,
            owner_class_neutral: OWNER_CLASS_NEUTRAL,
            owner_class_empty: OWNER_CLASS_EMPTY,
            owner_class_absent_on_map: OWNER_CLASS_ABSENT_ON_MAP,
            board_size: BOARD_SIZE,
            board_center: BOARD_CENTER,
            sun_radius: SUN_RADIUS,
            rotation_radius_limit: ROTATION_RADIUS_LIMIT,
            fleet_speed_max: FLEET_SPEED_MAX,
            fleet_speed_reference_ships: FLEET_SPEED_REFERENCE_SHIPS,
            fleet_speed_curve_power: FLEET_SPEED_CURVE_POWER,
            intercept_iterations: INTERCEPT_ITERATIONS,
            max_players: MAX_PLAYERS,
            comet_radius: COMET_RADIUS,
            comet_production: COMET_PRODUCTION,
            fleet_spawn_offset: FLEET_SPAWN_OFFSET,
            map_min_planet_groups: MAP_MIN_PLANET_GROUPS,
            map_max_planet_groups: MAP_MAX_PLANET_GROUPS,
            map_min_static_groups: MAP_MIN_STATIC_GROUPS,
            map_planet_clearance: MAP_PLANET_CLEARANCE,
            map_home_planet_ships: MAP_HOME_PLANET_SHIPS,
            decoder_aim_search_angle_samples: DECODER_AIM_SEARCH_ANGLE_SAMPLES,
            decoder_aim_search_angle_step_radians: DECODER_AIM_SEARCH_ANGLE_STEP_RADIANS,
            decoder_aim_search_extra_ticks: DECODER_AIM_SEARCH_EXTRA_TICKS,
            default_angular_velocity: DEFAULT_ANGULAR_VELOCITY,
            default_comet_speed: DEFAULT_COMET_SPEED,
            episode_steps: DEFAULT_EPISODE_STEPS,
            default_self_play_games: DEFAULT_SELF_PLAY_GAMES,
            default_mutation_sigma: DEFAULT_MUTATION_SIGMA,
            default_model_layers: DEFAULT_MODEL_LAYERS,
            default_model_d_model: DEFAULT_MODEL_D_MODEL,
            default_model_heads: DEFAULT_MODEL_HEADS,
            true2d_internal_rows: TRUE2D_INTERNAL_ROWS,
            true2d_input_features: TRUE2D_INPUT_FEATURES,
            true2d_output_features: TRUE2D_OUTPUT_FEATURES,
            true2d_action_targets_per_source: TRUE2D_ACTION_TARGETS_PER_SOURCE,
            true2d_attention_heads: TRUE2D_ATTENTION_HEADS,
            true2d_matrix_layer_planes: TRUE2D_MATRIX_LAYER_PLANES,
            true2d_target_parameter_count: TRUE2D_TARGET_PARAMETER_COUNT,
            training_population_size: TRAINING_POPULATION_SIZE,
            training_elite_count: TRAINING_ELITE_COUNT,
            training_players_per_game: TRAINING_PLAYERS_PER_GAME,
            training_generations_per_run: TRAINING_GENERATIONS_PER_RUN,
            training_demo_generations: TRAINING_DEMO_GENERATIONS,
            training_capture_trajectories: TRAINING_CAPTURE_TRAJECTORIES,
            training_backprop_enabled: TRAINING_BACKPROP_ENABLED,
            training_backprop_learning_rate: TRAINING_BACKPROP_LEARNING_RATE,
            training_backprop_batch_samples: TRAINING_BACKPROP_BATCH_SAMPLES,
            training_backprop_send_threshold: TRAINING_BACKPROP_SEND_THRESHOLD,
            training_backprop_reward_normalizer: TRAINING_BACKPROP_REWARD_NORMALIZER,
            training_sun_target_penalty_scale: TRAINING_SUN_TARGET_PENALTY_SCALE,
            training_reproduction_min_slots_per_elite: TRAINING_REPRODUCTION_MIN_SLOTS_PER_ELITE,
            training_reproduction_max_slots_per_elite: TRAINING_REPRODUCTION_MAX_SLOTS_PER_ELITE,
            training_reproduction_winrate_smoothing_wins:
                TRAINING_REPRODUCTION_WINRATE_SMOOTHING_WINS,
            training_reproduction_winrate_smoothing_games:
                TRAINING_REPRODUCTION_WINRATE_SMOOTHING_GAMES,
            generation_tournament_kaggle_stage_for_host_submit:
                GENERATION_TOURNAMENT_KAGGLE_STAGE_FOR_HOST_SUBMIT,
            kaggle_competition_slug: KAGGLE_COMPETITION_SLUG,
            kaggle_submission_stage_dir: KAGGLE_SUBMISSION_STAGE_DIR,
            kaggle_submission_main_path: KAGGLE_SUBMISSION_MAIN_PATH,
            kaggle_submission_library_path: KAGGLE_SUBMISSION_LIBRARY_PATH,
            kaggle_submission_archive_name: KAGGLE_SUBMISSION_ARCHIVE_NAME,
            generation_archive_champions_per_generation:
                GENERATION_ARCHIVE_CHAMPIONS_PER_GENERATION,
            generation_archive_max_generations: GENERATION_ARCHIVE_MAX_GENERATIONS,
            generation_validation_max_models: GENERATION_VALIDATION_MAX_MODELS,
            generation_validation_games_per_model: GENERATION_VALIDATION_GAMES_PER_MODEL,
            generation_validation_interval: GENERATION_VALIDATION_INTERVAL,
            generation_replay_interval: GENERATION_REPLAY_INTERVAL,
            trainer_smoke_population_size: TRAINER_SMOKE_POPULATION_SIZE,
            trainer_smoke_elite_count: TRAINER_SMOKE_ELITE_COUNT,
            trainer_smoke_games_per_model: TRAINER_SMOKE_GAMES_PER_MODEL,
            trainer_smoke_episode_steps: TRAINER_SMOKE_EPISODE_STEPS,
            dashboard_strict_population_size: DASHBOARD_STRICT_POPULATION_SIZE,
            dashboard_strict_elite_count: DASHBOARD_STRICT_ELITE_COUNT,
            dashboard_strict_games_per_model: DASHBOARD_STRICT_GAMES_PER_MODEL,
            dashboard_strict_replays_per_model: DASHBOARD_STRICT_REPLAYS_PER_MODEL,
            training_reward_win: TRAINING_REWARD_WIN,
            training_reward_draw: TRAINING_REWARD_DRAW,
            training_reward_loss: TRAINING_REWARD_LOSS,
            dashboard_replays_per_model: DASHBOARD_REPLAYS_PER_MODEL,
            dashboard_generation_replay_game_count: DASHBOARD_GENERATION_REPLAY_GAME_COUNT,
            dashboard_full_replay_frame_stride: DASHBOARD_FULL_REPLAY_FRAME_STRIDE,
            dashboard_replay_frame_stride: DASHBOARD_REPLAY_FRAME_STRIDE,
            dashboard_replay_max_frames: DASHBOARD_REPLAY_MAX_FRAMES,
            dashboard_live_replay_enabled: DASHBOARD_LIVE_REPLAY_ENABLED,
            dashboard_live_replay_game_count: DASHBOARD_LIVE_REPLAY_GAME_COUNT,
            dashboard_live_replay_frame_stride: DASHBOARD_LIVE_REPLAY_FRAME_STRIDE,
            dashboard_live_replay_update_interval_turns:
                DASHBOARD_LIVE_REPLAY_UPDATE_INTERVAL_TURNS,
            dashboard_live_replay_slot_bytes: DASHBOARD_LIVE_REPLAY_SLOT_BYTES,
            cuda_true2d_smoke_games: CUDA_TRUE2D_SMOKE_GAMES,
            cuda_true2d_forward_tolerance: CUDA_TRUE2D_FORWARD_TOLERANCE,
            cuda_true2d_max_requests_per_forward: CUDA_TRUE2D_MAX_REQUESTS_PER_FORWARD,
        }
    }
}

impl AgentConfig {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.max_rows != self.initial_planet_slots + self.comet_slots + self.padding_slots {
            return Err("row slots must sum to max_rows");
        }
        if self.max_rows == 0 || self.initial_planet_slots == 0 {
            return Err("row slots must be positive");
        }
        if !(self.ship_log_pivot > 0.0 && self.ship_log_cap > self.ship_log_pivot) {
            return Err("ship log scaling bounds are invalid");
        }
        if self.owner_class_player_slots != OWNER_CLASS_PLAYER_SLOTS
            || self.max_players != self.owner_class_player_slots
        {
            return Err("owner class player slots must match max_players");
        }
        let owner_classes = [
            self.owner_class_own,
            self.owner_class_enemy_1,
            self.owner_class_enemy_2,
            self.owner_class_enemy_3,
            self.owner_class_neutral,
            self.owner_class_empty,
            self.owner_class_absent_on_map,
        ];
        if owner_classes
            .iter()
            .any(|owner_class| !owner_class.is_finite())
        {
            return Err("owner class encoding is invalid");
        }
        for left_index in 0..owner_classes.len() {
            for right_index in left_index + 1..owner_classes.len() {
                if owner_classes[left_index] == owner_classes[right_index] {
                    return Err("owner class encoding is invalid");
                }
            }
        }
        if !(self.board_size.is_finite() && self.board_size > 0.0) {
            return Err("board size is invalid");
        }
        if !(self.fleet_speed_max >= 1.0 && self.fleet_speed_reference_ships > 1.0) {
            return Err("fleet speed bounds are invalid");
        }
        if self.max_players == 0 || self.episode_steps == 0 {
            return Err("simulation limits are invalid");
        }
        if self.map_min_planet_groups == 0
            || self.map_max_planet_groups < self.map_min_planet_groups
            || self.map_min_static_groups == 0
            || self.map_min_static_groups > self.map_max_planet_groups
            || !self.map_planet_clearance.is_finite()
            || self.map_planet_clearance < 0.0
            || !self.map_home_planet_ships.is_finite()
            || self.map_home_planet_ships <= 0.0
            || self.decoder_aim_search_angle_samples == 0
            || !self.decoder_aim_search_angle_step_radians.is_finite()
            || self.decoder_aim_search_angle_step_radians <= 0.0
            || self.decoder_aim_search_extra_ticks == 0
        {
            return Err("map generation parameters are invalid");
        }
        if self.default_model_layers == 0
            || self.default_model_heads == 0
            || self.default_model_d_model < MODEL_D_MODEL_MIN
            || self.default_model_d_model > MODEL_D_MODEL_MAX
            || self.default_model_d_model % self.default_model_heads != 0
            || self.true2d_internal_rows != self.max_rows
            || self.true2d_input_features != TRUE2D_INPUT_FEATURES
            || self.true2d_output_features != TRUE2D_OUTPUT_FEATURES
            || self.true2d_action_targets_per_source != TRUE2D_ACTION_TARGETS_PER_SOURCE
            || self.true2d_attention_heads == 0
            || self.true2d_attention_heads > TRUE2D_ATTENTION_HEADS_MAX
            || self.true2d_matrix_layer_planes != TRUE2D_MATRIX_LAYER_PLANES
            || self.true2d_matrix_layer_planes < TRUE2D_MATRIX_LAYER_PLANES_MIN
            || self.true2d_matrix_layer_planes > TRUE2D_MATRIX_LAYER_PLANES_MAX
            || self.true2d_target_parameter_count == 0
        {
            return Err("model shape is invalid");
        }
        if self.training_population_size == 0
            || self.training_elite_count == 0
            || self.training_elite_count > self.training_population_size
            || ![
                TRAINING_PLAYERS_PER_GAME_TWO,
                TRAINING_PLAYERS_PER_GAME_FOUR,
            ]
            .contains(&self.training_players_per_game)
            || self.training_generations_per_run == 0
            || self.training_demo_generations == 0
            || (self.training_backprop_enabled && !self.training_capture_trajectories)
            || !self.training_backprop_learning_rate.is_finite()
            || self.training_backprop_learning_rate <= 0.0
            || self.training_backprop_batch_samples == 0
            || !self.training_backprop_send_threshold.is_finite()
            || self.training_backprop_send_threshold < 0.0
            || self.training_backprop_send_threshold > 1.0
            || !self.training_backprop_reward_normalizer.is_finite()
            || self.training_backprop_reward_normalizer <= 0.0
            || !self.training_sun_target_penalty_scale.is_finite()
            || self.training_sun_target_penalty_scale < 0.0
            || self.training_reproduction_min_slots_per_elite == 0
            || self.training_reproduction_max_slots_per_elite
                < self.training_reproduction_min_slots_per_elite
            || self.training_reproduction_max_slots_per_elite > self.training_population_size
            || !self
                .training_reproduction_winrate_smoothing_wins
                .is_finite()
            || self.training_reproduction_winrate_smoothing_wins < 0.0
            || !self
                .training_reproduction_winrate_smoothing_games
                .is_finite()
            || self.training_reproduction_winrate_smoothing_games <= 0.0
            || self.kaggle_competition_slug.is_empty()
            || self.kaggle_submission_stage_dir.is_empty()
            || self.kaggle_submission_main_path.is_empty()
            || self.kaggle_submission_library_path.is_empty()
            || self.kaggle_submission_archive_name.is_empty()
            || self.generation_archive_champions_per_generation == 0
            || self.generation_archive_max_generations == 0
            || self.generation_validation_max_models == 0
            || self.generation_validation_games_per_model == 0
            || self.generation_validation_interval == 0
            || self.generation_replay_interval == 0
            || self.generation_archive_champions_per_generation
                > self.generation_validation_max_models
            || self.default_self_play_games == 0
            || !self.default_mutation_sigma.is_finite()
            || self.default_mutation_sigma < 0.0
            || self.trainer_smoke_population_size == 0
            || self.trainer_smoke_elite_count == 0
            || self.trainer_smoke_elite_count > self.trainer_smoke_population_size
            || self.trainer_smoke_games_per_model == 0
            || self.trainer_smoke_episode_steps == 0
            || self.dashboard_strict_population_size == 0
            || self.dashboard_strict_elite_count == 0
            || self.dashboard_strict_elite_count > self.dashboard_strict_population_size
            || self.dashboard_strict_games_per_model == 0
            || self.dashboard_strict_replays_per_model == 0
            || self.dashboard_replays_per_model == 0
            || self.dashboard_generation_replay_game_count == 0
            || self.dashboard_full_replay_frame_stride == 0
            || self.dashboard_replay_frame_stride == 0
            || self.dashboard_replay_max_frames == 0
            || self.dashboard_live_replay_game_count == 0
            || self.dashboard_live_replay_frame_stride == 0
            || self.dashboard_live_replay_update_interval_turns == 0
            || self.dashboard_live_replay_slot_bytes == 0
            || self.cuda_true2d_smoke_games == 0
            || !self.cuda_true2d_forward_tolerance.is_finite()
            || self.cuda_true2d_forward_tolerance <= 0.0
            || self.cuda_true2d_max_requests_per_forward == 0
        {
            return Err("training parameters are invalid");
        }
        if !(self.training_reward_win > 0
            && self.training_reward_draw < 0
            && self.training_reward_loss < self.training_reward_draw
            && self.training_reward_draw < self.training_reward_win)
        {
            return Err("training reward scale is invalid");
        }
        let validation_pool_size = self
            .generation_archive_champions_per_generation
            .checked_mul(self.generation_archive_max_generations)
            .ok_or("training parameters are invalid")?;
        if validation_pool_size > self.generation_validation_max_models
            || self.training_population_size > self.generation_validation_max_models
            || self.training_population_size < self.generation_archive_champions_per_generation
            || self.trainer_smoke_population_size < self.generation_archive_champions_per_generation
            || self.dashboard_strict_population_size
                < self.generation_archive_champions_per_generation
        {
            return Err("training parameters are invalid");
        }
        let min_reproduction_slots = self
            .training_reproduction_min_slots_per_elite
            .checked_mul(self.training_elite_count)
            .ok_or("training parameters are invalid")?;
        let max_reproduction_slots = self
            .training_reproduction_max_slots_per_elite
            .checked_mul(self.training_elite_count)
            .ok_or("training parameters are invalid")?;
        if min_reproduction_slots > self.training_population_size
            || max_reproduction_slots < self.training_population_size
        {
            return Err("training parameters are invalid");
        }
        Ok(())
    }
}
