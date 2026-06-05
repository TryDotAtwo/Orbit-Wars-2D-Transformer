use crate::config::AgentConfig;
use crate::geometry::{intercept_angle, intercept_angle_with_orbit_prediction, GeometryError, OrbitPrediction};
use crate::types::{ActionSlotOutput, AmountClass, MoveCommand, Planet};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecodeError {
    NonFiniteModelOutput,
    InvalidAmountClass,
    Geometry(GeometryError),
}

impl From<GeometryError> for DecodeError {
    fn from(value: GeometryError) -> Self {
        Self::Geometry(value)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DecodedMoveCommand {
    pub command: MoveCommand,
    pub slot_index: usize,
    pub source_row: usize,
    pub target_row: usize,
    pub amount_class: AmountClass,
}

pub fn decode_action_slots(
    planets_by_row: &[Planet],
    initial_planets: &[Planet],
    comet_planet_ids: &[i32],
    player: i32,
    angular_velocity: f32,
    slots: &[ActionSlotOutput],
    config: &AgentConfig,
) -> Result<Vec<MoveCommand>, DecodeError> {
    Ok(decode_action_slots_with_trace(
        planets_by_row,
        initial_planets,
        comet_planet_ids,
        player,
        angular_velocity,
        slots,
        config,
    )?
    .into_iter()
    .map(|decoded| decoded.command)
    .collect())
}

pub fn decode_action_slots_with_trace(
    planets_by_row: &[Planet],
    initial_planets: &[Planet],
    comet_planet_ids: &[i32],
    player: i32,
    angular_velocity: f32,
    slots: &[ActionSlotOutput],
    config: &AgentConfig,
) -> Result<Vec<DecodedMoveCommand>, DecodeError> {
    if !angular_velocity.is_finite() {
        return Err(DecodeError::NonFiniteModelOutput);
    }

    let mut decoded = Vec::new();
    let mut used_sources = [false; 64];
    let mut ranked_slots: Vec<(usize, f32)> = slots
        .iter()
        .take(config.action_slots)
        .enumerate()
        .map(|(slot_index, slot)| (slot_index, slot.fire_logit))
        .collect();
    ranked_slots.sort_by(|left, right| right.1.total_cmp(&left.1));

    for (slot_index, fire_logit) in ranked_slots {
        if !fire_logit.is_finite() || sigmoid(fire_logit) < 0.5 {
            continue;
        }
        let slot = &slots[slot_index];
        let source_row = argmax_finite(&slot.source_logits).ok_or(DecodeError::NonFiniteModelOutput)?;
        if source_row >= planets_by_row.len() || used_sources[source_row] {
            continue;
        }
        let source = planets_by_row[source_row];
        if source.owner != player || source.ships < 1.0 {
            continue;
        }

        let mut target_logits = slot.target_logits;
        target_logits[source_row] = f32::NEG_INFINITY;
        let target_row = argmax_finite(&target_logits).ok_or(DecodeError::NonFiniteModelOutput)?;
        if target_row >= planets_by_row.len() {
            continue;
        }
        let target = planets_by_row[target_row];
        let amount_index = argmax_finite(&slot.amount_logits).ok_or(DecodeError::NonFiniteModelOutput)?;
        let amount_class = AmountClass::from_index(amount_index).ok_or(DecodeError::InvalidAmountClass)?;
        let ship_count = amount_class.ship_count(source.ships);
        if ship_count < 1 {
            continue;
        }

        let direction_angle = if target_uses_orbit_prediction(&target, initial_planets, comet_planet_ids, config) {
            intercept_angle_with_orbit_prediction(
                &source,
                &target,
                ship_count as f32,
                config,
                OrbitPrediction { center_x: config.board_center, center_y: config.board_center, angular_velocity },
            )?
        } else {
            intercept_angle(&source, &target, ship_count as f32, config)?
        };

        used_sources[source_row] = true;
        decoded.push(DecodedMoveCommand {
            command: MoveCommand { from_planet_id: source.id, direction_angle, ship_count },
            slot_index,
            source_row,
            target_row,
            amount_class,
        });
    }

    Ok(decoded)
}

fn argmax_finite<const N: usize>(values: &[f32; N]) -> Option<usize> {
    let mut best_index = None;
    let mut best_value = f32::NEG_INFINITY;
    for (index, value) in values.iter().enumerate() {
        if !value.is_finite() && !value.is_infinite() {
            return None;
        }
        if *value > best_value {
            best_value = *value;
            best_index = Some(index);
        }
    }
    best_index
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

fn target_uses_orbit_prediction(
    target: &Planet,
    initial_planets: &[Planet],
    comet_planet_ids: &[i32],
    config: &AgentConfig,
) -> bool {
    !comet_planet_ids.contains(&target.id)
        && initial_planets
            .iter()
            .find(|planet| planet.id == target.id)
            .map(|initial| {
                let dx = initial.x - config.board_center;
                let dy = initial.y - config.board_center;
                let orbital_radius = (dx * dx + dy * dy).sqrt();
                orbital_radius + target.radius < config.rotation_radius_limit
            })
            .unwrap_or(false)
}
