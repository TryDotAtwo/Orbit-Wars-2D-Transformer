use crate::config::AgentConfig;
use crate::geometry::{
    fleet_speed, intercept_angle, intercept_angle_with_orbit_prediction, point_out_of_bounds,
    segment_intersects_sun, swept_pair_hit, GeometryError, OrbitPrediction,
};
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
    decode_action_slots_at_step(
        planets_by_row,
        initial_planets,
        comet_planet_ids,
        player,
        angular_velocity,
        0,
        slots,
        config,
    )
}

pub fn decode_action_slots_at_step(
    planets_by_row: &[Planet],
    initial_planets: &[Planet],
    comet_planet_ids: &[i32],
    player: i32,
    angular_velocity: f32,
    current_step: usize,
    slots: &[ActionSlotOutput],
    config: &AgentConfig,
) -> Result<Vec<MoveCommand>, DecodeError> {
    Ok(decode_action_slots_with_trace(
        planets_by_row,
        initial_planets,
        comet_planet_ids,
        player,
        angular_velocity,
        current_step,
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
    current_step: usize,
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

        let base_direction_angle = if target_uses_orbit_prediction(&target, initial_planets, comet_planet_ids, config) {
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
        let Some(direction_angle) = refine_valid_action_angle(
            &source,
            &target,
            planets_by_row,
            initial_planets,
            comet_planet_ids,
            current_step,
            ship_count,
            base_direction_angle,
            angular_velocity,
            config,
        ) else {
            continue;
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

fn refine_valid_action_angle(
    source: &Planet,
    target: &Planet,
    planets: &[Planet],
    initial_planets: &[Planet],
    comet_planet_ids: &[i32],
    current_step: usize,
    ship_count: i32,
    base_angle: f32,
    angular_velocity: f32,
    config: &AgentConfig,
) -> Option<f32> {
    if action_has_valid_first_contact(
        source,
        target,
        planets,
        initial_planets,
        comet_planet_ids,
        current_step,
        ship_count,
        base_angle,
        angular_velocity,
        config,
    ) {
        return Some(base_angle);
    }

    const SEARCH_SAMPLES: usize = 160;
    const SEARCH_STEP: f32 = 0.005;
    for sample in 1..=SEARCH_SAMPLES {
        let magnitude = ((sample + 1) / 2) as f32;
        let sign = if sample & 1 == 1 { 1.0 } else { -1.0 };
        let angle = base_angle + sign * magnitude * SEARCH_STEP;
        if action_has_valid_first_contact(
            source,
            target,
            planets,
            initial_planets,
            comet_planet_ids,
            current_step,
            ship_count,
            angle,
            angular_velocity,
            config,
        ) {
            return Some(angle);
        }
    }

    None
}

fn action_has_valid_first_contact(
    source: &Planet,
    target: &Planet,
    planets: &[Planet],
    initial_planets: &[Planet],
    comet_planet_ids: &[i32],
    current_step: usize,
    ship_count: i32,
    angle: f32,
    angular_velocity: f32,
    config: &AgentConfig,
) -> bool {
    if comet_planet_ids.contains(&target.id) {
        return false;
    }

    let direction_x = angle.cos();
    let direction_y = angle.sin();
    let spawn_distance = source.radius + config.fleet_spawn_offset;
    let mut fleet_x = source.x + direction_x * spawn_distance;
    let mut fleet_y = source.y + direction_y * spawn_distance;
    let speed = fleet_speed(ship_count as f32, config);
    let mut old_positions: Vec<(f32, f32)> = planets.iter().map(|planet| (planet.x, planet.y)).collect();

    for offset in 1..=config.episode_steps {
        let fleet_new_x = fleet_x + direction_x * speed;
        let fleet_new_y = fleet_y + direction_y * speed;
        let mut new_positions = Vec::with_capacity(planets.len());

        for (index, planet) in planets.iter().enumerate() {
            let (old_x, old_y) = old_positions[index];
            let (new_x, new_y) = next_planet_position_at_step(
                planet,
                initial_planets,
                current_step + offset,
                offset,
                angular_velocity,
                config,
            );
            new_positions.push((new_x, new_y));
            if swept_pair_hit(
                fleet_x,
                fleet_y,
                fleet_new_x,
                fleet_new_y,
                old_x,
                old_y,
                new_x,
                new_y,
                planet.radius,
            ) {
                return planet.id == target.id && !comet_planet_ids.contains(&planet.id);
            }
        }

        if point_out_of_bounds(fleet_new_x, fleet_new_y, config) {
            return false;
        }
        if segment_intersects_sun(fleet_x, fleet_y, fleet_new_x, fleet_new_y, config) {
            return false;
        }

        fleet_x = fleet_new_x;
        fleet_y = fleet_new_y;
        old_positions = new_positions;
    }

    false
}

fn next_planet_position_at_step(
    planet: &Planet,
    initial_planets: &[Planet],
    absolute_step: usize,
    offset: usize,
    angular_velocity: f32,
    config: &AgentConfig,
) -> (f32, f32) {
    if let Some(initial) = initial_planets.iter().find(|initial| initial.id == planet.id) {
        let dx = initial.x - config.board_center;
        let dy = initial.y - config.board_center;
        let orbital_radius = (dx * dx + dy * dy).sqrt();
        if orbital_radius + planet.radius < config.rotation_radius_limit {
            let initial_angle = dy.atan2(dx);
            let phase_step = absolute_step.saturating_sub(1);
            let angle = initial_angle + angular_velocity * phase_step as f32;
            return (
                config.board_center + orbital_radius * angle.cos(),
                config.board_center + orbital_radius * angle.sin(),
            );
        }
    }

    (
        planet.x + planet.velocity_x * offset as f32,
        planet.y + planet.velocity_y * offset as f32,
    )
}
