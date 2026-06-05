pub const MAX_PLANETS: usize = 64;
pub const ACTION_SLOTS: usize = 8;
pub const AMOUNT_CLASS_COUNT: usize = 16;
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
pub const DEFAULT_EPISODE_STEPS: usize = 500;
pub const DECODER_AIM_SEARCH_ANGLE_SAMPLES: usize = 128;
pub const DECODER_AIM_SEARCH_ANGLE_STEP_RADIANS: f32 = 0.01;
pub const DECODER_AIM_SEARCH_EXTRA_TICKS: usize = 8;
pub const DEFAULT_ANGULAR_VELOCITY: f32 = 0.03;

#[derive(Clone, Copy, Debug)]
pub struct AgentConfig {
    pub max_planets: usize,
    pub action_slots: usize,
    pub amount_class_count: usize,
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
    pub episode_steps: usize,
    pub decoder_aim_search_angle_samples: usize,
    pub decoder_aim_search_angle_step_radians: f32,
    pub decoder_aim_search_extra_ticks: usize,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_planets: MAX_PLANETS,
            action_slots: ACTION_SLOTS,
            amount_class_count: AMOUNT_CLASS_COUNT,
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
            episode_steps: DEFAULT_EPISODE_STEPS,
            decoder_aim_search_angle_samples: DECODER_AIM_SEARCH_ANGLE_SAMPLES,
            decoder_aim_search_angle_step_radians: DECODER_AIM_SEARCH_ANGLE_STEP_RADIANS,
            decoder_aim_search_extra_ticks: DECODER_AIM_SEARCH_EXTRA_TICKS,
        }
    }
}

impl AgentConfig {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.max_planets == 0
            || self.action_slots == 0
            || self.amount_class_count == 0
            || self.max_players == 0
            || self.episode_steps == 0
        {
            return Err("non_positive_dimension");
        }
        if !(self.board_size.is_finite()
            && self.board_center.is_finite()
            && self.sun_radius.is_finite()
            && self.rotation_radius_limit.is_finite()
            && self.fleet_speed_max.is_finite()
            && self.fleet_speed_reference_ships.is_finite()
            && self.fleet_speed_curve_power.is_finite()
            && self.comet_radius.is_finite()
            && self.comet_production.is_finite()
            && self.fleet_spawn_offset.is_finite()
            && self.decoder_aim_search_angle_step_radians.is_finite())
        {
            return Err("non_finite_float");
        }
        if self.board_size <= 0.0
            || self.sun_radius < 0.0
            || self.rotation_radius_limit < 0.0
            || self.fleet_speed_max <= 0.0
            || self.fleet_speed_reference_ships <= 0.0
            || self.fleet_speed_curve_power <= 0.0
            || self.comet_radius <= 0.0
            || self.comet_production < 0.0
            || self.fleet_spawn_offset < 0.0
            || self.decoder_aim_search_angle_samples == 0
            || self.decoder_aim_search_angle_step_radians <= 0.0
        {
            return Err("invalid_value");
        }
        Ok(())
    }
}
