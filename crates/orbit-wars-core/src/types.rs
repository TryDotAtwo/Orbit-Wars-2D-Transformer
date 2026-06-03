use crate::config::{AgentConfig, TRUE2D_ACTION_TARGETS_PER_SOURCE};

pub const NO_PLANET_ID: i32 = -1;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Planet {
    pub id: i32,
    pub owner: i32,
    pub x: f32,
    pub y: f32,
    pub radius: f32,
    pub ships: f32,
    pub production: f32,
    pub velocity_x: f32,
    pub velocity_y: f32,
}

impl Planet {
    pub fn distance_squared_to(&self, other: &Planet) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        dx * dx + dy * dy
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fleet {
    pub id: i32,
    pub owner: i32,
    pub x: f32,
    pub y: f32,
    pub angle: f32,
    pub from_planet_id: i32,
    pub ships: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RowFeature {
    pub owner_class: f32,
    pub ship_log_percent: f32,
    pub x_position_normalized: f32,
    pub y_position_normalized: f32,
    pub production_normalized: f32,
    pub velocity_x_normalized: f32,
    pub velocity_y_normalized: f32,
}

impl RowFeature {
    pub fn empty(config: &AgentConfig) -> Self {
        Self {
            owner_class: config.owner_class_empty,
            ship_log_percent: 0.0,
            x_position_normalized: 0.0,
            y_position_normalized: 0.0,
            production_normalized: 0.0,
            velocity_x_normalized: 0.0,
            velocity_y_normalized: 0.0,
        }
    }

    pub fn absent_on_map(config: &AgentConfig) -> Self {
        Self {
            owner_class: config.owner_class_absent_on_map,
            ship_log_percent: 0.0,
            x_position_normalized: 0.0,
            y_position_normalized: 0.0,
            production_normalized: 0.0,
            velocity_x_normalized: 0.0,
            velocity_y_normalized: 0.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ActionTargetOutput {
    pub target_fraction: f32,
    pub send_fraction: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ActionOutput {
    pub targets: [ActionTargetOutput; TRUE2D_ACTION_TARGETS_PER_SOURCE],
}

impl ActionOutput {
    pub fn repeated(target_fraction: f32, send_fraction: f32) -> Self {
        Self {
            targets: [ActionTargetOutput {
                target_fraction,
                send_fraction,
            }; TRUE2D_ACTION_TARGETS_PER_SOURCE],
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MoveCommand {
    pub from_planet_id: i32,
    pub direction_angle: f32,
    pub ship_count: i32,
}
