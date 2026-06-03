use crate::config::AgentConfig;
use crate::encoder::{encode_state, EncodeError};
use crate::geometry::{
    fleet_speed, intercept_angle, intercept_angle_with_orbit_prediction, swept_pair_hit,
    GeometryError, OrbitPrediction,
};
use crate::types::{ActionOutput, MoveCommand, Planet, NO_PLANET_ID};

const AIM_TIME_SAMPLE_MIDPOINT: f32 = 0.5;
const AIM_TIME_SAMPLE_FRACTIONS: [f32; 3] = [0.0, AIM_TIME_SAMPLE_MIDPOINT, 1.0];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecodeError {
    Encode(EncodeError),
    NonFiniteModelOutput,
    Geometry(GeometryError),
}

impl From<EncodeError> for DecodeError {
    fn from(value: EncodeError) -> Self {
        Self::Encode(value)
    }
}

impl From<GeometryError> for DecodeError {
    fn from(value: GeometryError) -> Self {
        Self::Geometry(value)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DecodedMoveCommand {
    pub command: MoveCommand,
    pub source_row: usize,
    pub target_slot: usize,
    pub target_row: usize,
}

pub fn decode_model_outputs(
    planets: &[Planet],
    initial_planets: &[Planet],
    comet_planet_ids: &[i32],
    player: i32,
    angular_velocity: f32,
    current_step: usize,
    outputs: &[ActionOutput],
    config: &AgentConfig,
) -> Result<Vec<MoveCommand>, DecodeError> {
    Ok(decode_model_outputs_with_trace(
        planets,
        initial_planets,
        comet_planet_ids,
        player,
        angular_velocity,
        current_step,
        outputs,
        config,
    )?
    .into_iter()
    .map(|decoded| decoded.command)
    .collect())
}

pub fn decode_model_outputs_with_trace(
    planets: &[Planet],
    initial_planets: &[Planet],
    comet_planet_ids: &[i32],
    player: i32,
    angular_velocity: f32,
    current_step: usize,
    outputs: &[ActionOutput],
    config: &AgentConfig,
) -> Result<Vec<DecodedMoveCommand>, DecodeError> {
    if !angular_velocity.is_finite() {
        return Err(DecodeError::NonFiniteModelOutput);
    }
    let encoded = encode_state(planets, initial_planets, comet_planet_ids, player, config)?;
    let active_rows = encoded.active_row_indices();
    if active_rows.is_empty() {
        return Ok(Vec::new());
    }

    let mut commands = Vec::new();
    for source_row in 0..config.max_rows {
        let Some(output) = outputs.get(source_row) else {
            continue;
        };
        let source_planet_id = encoded.row_planet_ids[source_row];
        if source_planet_id == NO_PLANET_ID {
            continue;
        }

        let Some(source_planet) = planet_by_id(planets, source_planet_id) else {
            continue;
        };
        if source_planet.owner != player {
            continue;
        }

        for (target_slot, target_output) in output.targets.into_iter().enumerate() {
            if !(target_output.target_fraction.is_finite()
                && target_output.send_fraction.is_finite())
            {
                return Err(DecodeError::NonFiniteModelOutput);
            }
            let target_row = target_row_from_fraction(target_output.target_fraction, &active_rows);
            if target_row == source_row {
                continue;
            }

            let target_planet_id = encoded.row_planet_ids[target_row];
            let Some(target_planet) = planet_by_id(planets, target_planet_id) else {
                continue;
            };

            let send_fraction = target_output.send_fraction.clamp(0.0, 1.0);
            if send_fraction <= 0.0 {
                continue;
            }

            let ship_count = (source_planet.ships * send_fraction).floor() as i32;
            if ship_count < 1 {
                continue;
            }

            let base_direction_angle = if target_uses_orbit_prediction(
                target_planet,
                initial_planets,
                comet_planet_ids,
                config,
            ) {
                intercept_angle_with_orbit_prediction(
                    source_planet,
                    target_planet,
                    ship_count as f32,
                    config,
                    OrbitPrediction {
                        center_x: config.board_center,
                        center_y: config.board_center,
                        angular_velocity,
                    },
                )?
            } else {
                intercept_angle(source_planet, target_planet, ship_count as f32, config)?
            };
            let direction_angle = refine_angle_to_hit_selected_target(
                source_planet,
                target_planet,
                initial_planets,
                comet_planet_ids,
                angular_velocity,
                current_step,
                ship_count as f32,
                base_direction_angle,
                config,
            )
            .unwrap_or(base_direction_angle);
            commands.push(DecodedMoveCommand {
                command: MoveCommand {
                    from_planet_id: source_planet.id,
                    direction_angle,
                    ship_count,
                },
                source_row,
                target_slot,
                target_row,
            });
        }
    }

    Ok(commands)
}

fn refine_angle_to_hit_selected_target(
    source: &Planet,
    target: &Planet,
    initial_planets: &[Planet],
    comet_planet_ids: &[i32],
    angular_velocity: f32,
    current_step: usize,
    ship_count: f32,
    base_angle: f32,
    config: &AgentConfig,
) -> Option<f32> {
    let speed = fleet_speed(ship_count, config);
    let max_ticks = max_hit_prediction_ticks(source, target, speed, config);
    let base_score = collision_ratio_for_angle(
        source,
        target,
        initial_planets,
        comet_planet_ids,
        angular_velocity,
        current_step,
        speed,
        base_angle,
        max_ticks,
        config,
    )?;
    if base_score < 1.0 {
        return Some(base_angle);
    }
    if let Some(angle) = time_search_angle_to_hit_selected_target(
        source,
        target,
        initial_planets,
        comet_planet_ids,
        angular_velocity,
        current_step,
        speed,
        max_ticks,
        config,
    ) {
        return Some(angle);
    }
    let mut best_angle = None;
    let mut best_ratio = base_score;
    for sample in 1..=config.decoder_aim_search_angle_samples {
        let offset = config.decoder_aim_search_angle_step_radians * sample as f32;
        for candidate in [base_angle - offset, base_angle + offset] {
            let Some(ratio) = collision_ratio_for_angle(
                source,
                target,
                initial_planets,
                comet_planet_ids,
                angular_velocity,
                current_step,
                speed,
                candidate,
                max_ticks,
                config,
            ) else {
                continue;
            };
            if ratio < 1.0 {
                return Some(candidate);
            }
            if ratio < best_ratio {
                best_ratio = ratio;
                best_angle = Some(candidate);
            }
        }
    }
    best_angle
}

fn time_search_angle_to_hit_selected_target(
    source: &Planet,
    target: &Planet,
    initial_planets: &[Planet],
    comet_planet_ids: &[i32],
    angular_velocity: f32,
    current_step: usize,
    speed: f32,
    max_ticks: usize,
    config: &AgentConfig,
) -> Option<f32> {
    for tick in 1..=max_ticks {
        let (old_target_x, old_target_y) = predict_planet_position_for_decoder(
            target,
            initial_planets,
            angular_velocity,
            current_step,
            tick - 1,
            config,
        )?;
        let (new_target_x, new_target_y) = predict_planet_position_for_decoder(
            target,
            initial_planets,
            angular_velocity,
            current_step,
            tick,
            config,
        )?;
        for fraction in AIM_TIME_SAMPLE_FRACTIONS {
            let aim_x = old_target_x + (new_target_x - old_target_x) * fraction;
            let aim_y = old_target_y + (new_target_y - old_target_y) * fraction;
            let candidate = (aim_y - source.y).atan2(aim_x - source.x);
            let Some(ratio) = collision_ratio_for_angle(
                source,
                target,
                initial_planets,
                comet_planet_ids,
                angular_velocity,
                current_step,
                speed,
                candidate,
                max_ticks,
                config,
            ) else {
                continue;
            };
            if ratio < 1.0 {
                return Some(candidate);
            }
        }
    }
    None
}

fn collision_ratio_for_angle(
    source: &Planet,
    target: &Planet,
    initial_planets: &[Planet],
    comet_planet_ids: &[i32],
    angular_velocity: f32,
    current_step: usize,
    speed: f32,
    angle: f32,
    max_ticks: usize,
    config: &AgentConfig,
) -> Option<f32> {
    if comet_planet_ids.contains(&target.id) {
        return None;
    }
    let direction_x = angle.cos();
    let direction_y = angle.sin();
    let start_x = source.x + direction_x * (source.radius + config.fleet_spawn_offset);
    let start_y = source.y + direction_y * (source.radius + config.fleet_spawn_offset);
    let mut best_ratio = f32::INFINITY;
    for tick in 1..=max_ticks {
        let old_fleet_x = start_x + direction_x * (tick - 1) as f32 * speed;
        let old_fleet_y = start_y + direction_y * (tick - 1) as f32 * speed;
        let new_fleet_x = start_x + direction_x * tick as f32 * speed;
        let new_fleet_y = start_y + direction_y * tick as f32 * speed;
        let (old_target_x, old_target_y) = predict_planet_position_for_decoder(
            target,
            initial_planets,
            angular_velocity,
            current_step,
            tick - 1,
            config,
        )?;
        let (new_target_x, new_target_y) = predict_planet_position_for_decoder(
            target,
            initial_planets,
            angular_velocity,
            current_step,
            tick,
            config,
        )?;
        let ratio = swept_collision_distance_ratio(
            old_fleet_x,
            old_fleet_y,
            new_fleet_x,
            new_fleet_y,
            old_target_x,
            old_target_y,
            new_target_x,
            new_target_y,
            target.radius,
        );
        best_ratio = best_ratio.min(ratio);
        if ratio < 1.0 {
            return Some(ratio);
        }
    }
    Some(best_ratio)
}

fn swept_collision_distance_ratio(
    old_fleet_x: f32,
    old_fleet_y: f32,
    new_fleet_x: f32,
    new_fleet_y: f32,
    old_target_x: f32,
    old_target_y: f32,
    new_target_x: f32,
    new_target_y: f32,
    radius: f32,
) -> f32 {
    if swept_pair_hit(
        old_fleet_x,
        old_fleet_y,
        new_fleet_x,
        new_fleet_y,
        old_target_x,
        old_target_y,
        new_target_x,
        new_target_y,
        radius,
    ) {
        0.0
    } else {
        relative_segment_distance(
            old_fleet_x - old_target_x,
            old_fleet_y - old_target_y,
            (new_fleet_x - old_fleet_x) - (new_target_x - old_target_x),
            (new_fleet_y - old_fleet_y) - (new_target_y - old_target_y),
        ) / radius
    }
}

fn relative_segment_distance(start_x: f32, start_y: f32, velocity_x: f32, velocity_y: f32) -> f32 {
    let segment_length_squared = velocity_x * velocity_x + velocity_y * velocity_y;
    let t = if segment_length_squared <= f32::EPSILON {
        0.0
    } else {
        (-(start_x * velocity_x + start_y * velocity_y) / segment_length_squared).clamp(0.0, 1.0)
    };
    let closest_x = start_x + velocity_x * t;
    let closest_y = start_y + velocity_y * t;
    (closest_x * closest_x + closest_y * closest_y).sqrt()
}

fn predict_planet_position_for_decoder(
    target: &Planet,
    initial_planets: &[Planet],
    angular_velocity: f32,
    current_step: usize,
    ticks_ahead: usize,
    config: &AgentConfig,
) -> Option<(f32, f32)> {
    if ticks_ahead == 0 {
        return Some((target.x, target.y));
    }
    let initial = initial_planets
        .iter()
        .find(|planet| planet.id == target.id)?;
    let initial_dx = initial.x - config.board_center;
    let initial_dy = initial.y - config.board_center;
    let initial_radius = (initial_dx * initial_dx + initial_dy * initial_dy).sqrt();
    if initial_radius + target.radius >= config.rotation_radius_limit {
        return Some((
            target.x + target.velocity_x * ticks_ahead as f32,
            target.y + target.velocity_y * ticks_ahead as f32,
        ));
    }
    let current_dx = target.x - config.board_center;
    let current_dy = target.y - config.board_center;
    let current_radius = (current_dx * current_dx + current_dy * current_dy).sqrt();
    if current_radius <= f32::EPSILON {
        return Some((target.x, target.y));
    }
    let future_step = current_step + ticks_ahead - 1;
    let rotation_step = future_step.max(1) as f32;
    let angle = initial_dy.atan2(initial_dx) + angular_velocity * rotation_step;
    Some((
        config.board_center + initial_radius * angle.cos(),
        config.board_center + initial_radius * angle.sin(),
    ))
}

fn max_hit_prediction_ticks(
    source: &Planet,
    target: &Planet,
    speed: f32,
    config: &AgentConfig,
) -> usize {
    let dx = target.x - source.x;
    let dy = target.y - source.y;
    let center_distance_ticks = ((dx * dx + dy * dy).sqrt() / speed).ceil() as usize;
    let board_diagonal = (config.board_size * config.board_size * 2.0).sqrt();
    let board_exit_ticks = (board_diagonal / speed).ceil() as usize;
    center_distance_ticks.max(board_exit_ticks) + config.decoder_aim_search_extra_ticks
}

fn target_row_from_fraction(target_fraction: f32, active_rows: &[usize]) -> usize {
    let scaled_index =
        (target_fraction.clamp(0.0, 1.0) * active_rows.len() as f32).floor() as usize;
    active_rows[scaled_index.min(active_rows.len() - 1)]
}

fn planet_by_id(planets: &[Planet], planet_id: i32) -> Option<&Planet> {
    planets.iter().find(|planet| planet.id == planet_id)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulator::SimulationState;

    fn planet(id: i32, owner: i32, x: f32, y: f32, ships: f32) -> Planet {
        Planet {
            id,
            owner,
            x,
            y,
            radius: 1.0,
            ships,
            production: 1.0,
            velocity_x: 0.0,
            velocity_y: 0.0,
        }
    }

    #[test]
    fn decoder_sends_owned_source_to_selected_target() {
        let config = AgentConfig::default();
        let planets = vec![
            planet(1, 0, 10.0, 10.0, 10.0),
            planet(2, -1, 20.0, 10.0, 5.0),
        ];
        let initial = planets.clone();
        let mut outputs = vec![ActionOutput::repeated(0.75, 0.5); config.max_rows];
        for target in outputs[0].targets.iter_mut().skip(1) {
            target.send_fraction = 0.0;
        }

        let commands =
            decode_model_outputs(&planets, &initial, &[], 0, 0.0, 0, &outputs, &config).unwrap();

        assert_eq!(commands.len(), 1);
        assert_eq!(commands[0].from_planet_id, 1);
        assert_eq!(commands[0].ship_count, 5);
        assert!(commands[0].direction_angle.abs() < 0.000_001);

        outputs[0].targets[0].target_fraction = 0.0;
        let hold_commands =
            decode_model_outputs(&planets, &initial, &[], 0, 0.0, 0, &outputs, &config).unwrap();
        assert!(hold_commands.is_empty());
    }

    #[test]
    fn decoder_ignores_fractional_ship_launches() {
        let config = AgentConfig::default();
        let planets = vec![
            planet(1, 0, 10.0, 10.0, 3.0),
            planet(2, -1, 20.0, 10.0, 5.0),
        ];
        let initial = planets.clone();
        let mut outputs = vec![ActionOutput::repeated(0.75, 0.2); config.max_rows];
        for target in outputs[0].targets.iter_mut().skip(1) {
            target.send_fraction = 0.0;
        }

        let commands =
            decode_model_outputs(&planets, &initial, &[], 0, 0.0, 0, &outputs, &config).unwrap();

        assert!(commands.is_empty());
    }

    #[test]
    fn decoder_predicts_orbiting_target_after_model_selection() {
        let config = AgentConfig::default();
        let planets = vec![
            planet(1, 0, 20.0, 80.0, 1_000.0),
            planet(2, -1, 80.0, 80.0, 10.0),
        ];
        let initial = planets.clone();
        let mut outputs = vec![ActionOutput::repeated(0.75, 1.0); config.max_rows];
        for target in outputs[0].targets.iter_mut().skip(1) {
            target.send_fraction = 0.0;
        }

        let commands =
            decode_model_outputs(&planets, &initial, &[], 0, 0.03, 0, &outputs, &config).unwrap();
        let command = commands.first().unwrap();

        assert!(command.direction_angle > 0.05);
        assert!(command.direction_angle < 0.35);
    }

    #[test]
    fn decoder_orbit_prediction_command_hits_selected_target_in_simulator() {
        let config = AgentConfig::default();
        let angular_velocity = 0.03;
        let planets = vec![
            planet(1, 0, 10.0, 90.0, 1_000.0),
            planet(2, -1, 80.0, 80.0, 10.0),
        ];
        let initial = planets.clone();
        let mut outputs = vec![ActionOutput::repeated(0.75, 1.0); config.max_rows];
        for target in outputs[0].targets.iter_mut().skip(1) {
            target.send_fraction = 0.0;
        }

        let commands = decode_model_outputs(
            &planets,
            &initial,
            &[],
            0,
            angular_velocity,
            0,
            &outputs,
            &config,
        )
        .unwrap();
        let mut state = SimulationState::new(planets, angular_velocity);
        let mut target_hit = false;
        for turn_index in 0..config.episode_steps {
            let actions = if turn_index == 0 {
                vec![commands.clone()]
            } else {
                vec![Vec::new()]
            };
            let events = state.step_turn_with_events(&actions, &config).unwrap();
            if events.player(0).hit_fleet_count > 0 {
                target_hit = true;
                break;
            }
        }

        assert!(target_hit);
        let target = state.planets.iter().find(|planet| planet.id == 2).unwrap();
        assert_eq!(target.owner, 0);
    }

    #[test]
    fn decoder_orbit_prediction_respects_step_one_phase_hold() {
        let config = AgentConfig::default();
        let angular_velocity = 0.18;
        let planets = vec![
            planet(1, 0, 5.0, 5.0, 1_000.0),
            planet(2, -1, 70.0, 50.0, 10.0),
        ];
        let mut state = SimulationState::new(planets, angular_velocity);
        state.step_turn(&[Vec::new()], &config).unwrap();
        assert_eq!(state.step, 1);

        let target = state.planets.iter().find(|planet| planet.id == 2).unwrap();
        let next_position = predict_planet_position_for_decoder(
            target,
            &state.initial_planets,
            angular_velocity,
            state.step,
            1,
            &config,
        )
        .unwrap();
        assert!((next_position.0 - target.x).abs() < f32::EPSILON);
        assert!((next_position.1 - target.y).abs() < f32::EPSILON);

        let mut outputs = vec![ActionOutput::repeated(0.75, 1.0); config.max_rows];
        for target in outputs[0].targets.iter_mut().skip(1) {
            target.send_fraction = 0.0;
        }
        let commands = decode_model_outputs(
            &state.planets,
            &state.initial_planets,
            &[],
            0,
            angular_velocity,
            state.step,
            &outputs,
            &config,
        )
        .unwrap();
        assert!(!commands.is_empty());

        let mut target_hit = false;
        for turn_index in 0..config.episode_steps {
            let actions = if turn_index == 0 {
                vec![commands.clone()]
            } else {
                vec![Vec::new()]
            };
            let events = state.step_turn_with_events(&actions, &config).unwrap();
            if events.player(0).hit_fleet_count > 0 {
                target_hit = true;
                break;
            }
        }

        assert!(target_hit);
        let target = state.planets.iter().find(|planet| planet.id == 2).unwrap();
        assert_eq!(target.owner, 0);
    }
}
