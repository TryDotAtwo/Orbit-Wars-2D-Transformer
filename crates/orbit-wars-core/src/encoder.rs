use std::cmp::Ordering;

use crate::config::{AgentConfig, MAX_ROWS};
use crate::scaler::ship_log_percent;
use crate::types::{Planet, RowFeature, NO_PLANET_ID};

#[derive(Clone, Debug)]
pub struct EncodedState {
    pub rows: [RowFeature; MAX_ROWS],
    pub row_planet_ids: [i32; MAX_ROWS],
}

impl EncodedState {
    pub fn active_row_indices(&self) -> Vec<usize> {
        self.row_planet_ids
            .iter()
            .enumerate()
            .filter_map(|(row_index, planet_id)| {
                if *planet_id == NO_PLANET_ID {
                    None
                } else {
                    Some(row_index)
                }
            })
            .collect()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EncodeError {
    InvalidConfig,
    InvalidOwnerClass,
    MissingHomePlanet,
    RowOverflow,
}

pub fn encode_state(
    planets: &[Planet],
    initial_planets: &[Planet],
    comet_planet_ids: &[i32],
    player: i32,
    config: &AgentConfig,
) -> Result<EncodedState, EncodeError> {
    config.validate().map_err(|_| EncodeError::InvalidConfig)?;

    let home_planet = planets
        .iter()
        .find(|planet| planet.owner == player)
        .ok_or(EncodeError::MissingHomePlanet)?;

    let mut rows = [RowFeature::empty(config); MAX_ROWS];
    let mut row_planet_ids = [NO_PLANET_ID; MAX_ROWS];
    let mut next_row = 0usize;

    fill_row(
        &mut rows,
        &mut row_planet_ids,
        &mut next_row,
        home_planet.id,
        planets,
        player,
        config,
    )?;

    let mut ordered_initial = initial_planets
        .iter()
        .copied()
        .filter(|planet| planet.id != home_planet.id)
        .collect::<Vec<_>>();
    ordered_initial.sort_by(|left, right| {
        let left_distance = left.distance_squared_to(home_planet);
        let right_distance = right.distance_squared_to(home_planet);
        left_distance.total_cmp(&right_distance)
    });

    for planet in ordered_initial
        .iter()
        .take(config.initial_planet_slots.saturating_sub(1))
    {
        fill_row(
            &mut rows,
            &mut row_planet_ids,
            &mut next_row,
            planet.id,
            planets,
            player,
            config,
        )?;
    }

    next_row = config.initial_planet_slots;

    let mut ordered_comets = planets
        .iter()
        .copied()
        .filter(|planet| comet_planet_ids.contains(&planet.id))
        .collect::<Vec<_>>();
    ordered_comets.sort_by(|left, right| {
        let left_order = comet_order(left.id, comet_planet_ids);
        let right_order = comet_order(right.id, comet_planet_ids);
        match left_order.cmp(&right_order) {
            Ordering::Equal => left
                .distance_squared_to(home_planet)
                .total_cmp(&right.distance_squared_to(home_planet)),
            other => other,
        }
    });

    for planet in ordered_comets.iter().take(config.comet_slots) {
        fill_row(
            &mut rows,
            &mut row_planet_ids,
            &mut next_row,
            planet.id,
            planets,
            player,
            config,
        )?;
    }

    Ok(EncodedState {
        rows,
        row_planet_ids,
    })
}

fn comet_order(planet_id: i32, comet_planet_ids: &[i32]) -> usize {
    comet_planet_ids
        .iter()
        .position(|candidate| *candidate == planet_id)
        .unwrap_or(comet_planet_ids.len())
}

fn fill_row(
    rows: &mut [RowFeature],
    row_planet_ids: &mut [i32],
    next_row: &mut usize,
    planet_id: i32,
    planets: &[Planet],
    player: i32,
    config: &AgentConfig,
) -> Result<(), EncodeError> {
    if *next_row >= config.max_rows {
        return Err(EncodeError::RowOverflow);
    }

    row_planet_ids[*next_row] = planet_id;
    rows[*next_row] =
        if let Some(current_planet) = planets.iter().find(|planet| planet.id == planet_id) {
            RowFeature {
                owner_class: owner_class_for_planet(current_planet.owner, player, config)?,
                ship_log_percent: ship_log_percent(current_planet.ships, config),
                x_position_normalized: position_normalized(current_planet.x, config),
                y_position_normalized: position_normalized(current_planet.y, config),
                production_normalized: production_normalized(current_planet.production),
                velocity_x_normalized: velocity_normalized(current_planet.velocity_x, config),
                velocity_y_normalized: velocity_normalized(current_planet.velocity_y, config),
            }
        } else {
            RowFeature::absent_on_map(config)
        };

    *next_row += 1;
    Ok(())
}

fn position_normalized(value: f32, config: &AgentConfig) -> f32 {
    value / config.board_size
}

fn production_normalized(value: f32) -> f32 {
    (value / 5.0).clamp(0.0, 1.0)
}

fn velocity_normalized(value: f32, config: &AgentConfig) -> f32 {
    (value / config.fleet_speed_max).clamp(-1.0, 1.0)
}

fn owner_class_for_planet(
    owner: i32,
    player: i32,
    config: &AgentConfig,
) -> Result<f32, EncodeError> {
    if owner < 0 {
        return Ok(config.owner_class_neutral);
    }
    if owner == player {
        return Ok(config.owner_class_own);
    }
    if player < 0 || owner as usize >= config.max_players || player as usize >= config.max_players {
        return Err(EncodeError::InvalidOwnerClass);
    }

    let opponent_slot = (0..config.max_players as i32)
        .filter(|candidate| *candidate != player)
        .position(|candidate| candidate == owner)
        .ok_or(EncodeError::InvalidOwnerClass)?;

    let enemy_classes = [
        config.owner_class_enemy_1,
        config.owner_class_enemy_2,
        config.owner_class_enemy_3,
    ];
    enemy_classes
        .get(opponent_slot)
        .copied()
        .ok_or(EncodeError::InvalidOwnerClass)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn planet(id: i32, owner: i32, x: f32, y: f32, ships: f32) -> Planet {
        Planet {
            id,
            owner,
            x,
            y,
            radius: 1.0,
            ships,
            production: 1.0,
            velocity_x: 3.0,
            velocity_y: -1.5,
        }
    }

    #[test]
    fn row_zero_is_home_planet_and_initial_rows_are_distance_sorted() {
        let config = AgentConfig::default();
        let initial = vec![
            planet(10, 1, 10.0, 10.0, 10.0),
            planet(20, -1, 20.0, 10.0, 10.0),
            planet(30, -1, 90.0, 90.0, 10.0),
        ];
        let current = initial.clone();

        let encoded = encode_state(&current, &initial, &[], 1, &config).unwrap();

        assert_eq!(encoded.row_planet_ids[0], 10);
        assert_eq!(encoded.row_planet_ids[1], 20);
        assert_eq!(encoded.row_planet_ids[2], 30);
        assert_eq!(encoded.rows[0].owner_class, config.owner_class_own);
        assert_eq!(encoded.rows[1].owner_class, config.owner_class_neutral);
        assert_eq!(encoded.rows[0].x_position_normalized, 0.1);
        assert_eq!(encoded.rows[0].y_position_normalized, 0.1);
        assert_eq!(encoded.rows[0].production_normalized, 0.2);
        assert_eq!(encoded.rows[0].velocity_x_normalized, 0.5);
        assert_eq!(encoded.rows[0].velocity_y_normalized, -0.25);
        assert_eq!(encoded.rows[1].x_position_normalized, 0.2);
        assert_eq!(encoded.rows[1].y_position_normalized, 0.1);
        assert_eq!(encoded.rows[1].production_normalized, 0.2);
        assert_eq!(encoded.rows[1].velocity_x_normalized, 0.5);
        assert_eq!(encoded.rows[1].velocity_y_normalized, -0.25);
    }

    #[test]
    fn owner_class_encodes_three_enemies_neutral_absent_and_empty_rows() {
        let config = AgentConfig::default();
        let initial = vec![
            planet(10, 0, 10.0, 10.0, 10.0),
            planet(20, 1, 20.0, 10.0, 10.0),
            planet(30, 2, 30.0, 10.0, 10.0),
            planet(40, 3, 40.0, 10.0, 10.0),
            planet(50, -1, 50.0, 10.0, 10.0),
            planet(60, -1, 60.0, 10.0, 10.0),
        ];
        let current = initial
            .iter()
            .copied()
            .filter(|planet| planet.id != 60)
            .collect::<Vec<_>>();
        let encoded = encode_state(&current, &initial, &[], 0, &config).unwrap();

        assert_eq!(encoded.rows[0].owner_class, config.owner_class_own);
        assert_eq!(encoded.rows[1].owner_class, config.owner_class_enemy_1);
        assert_eq!(encoded.rows[2].owner_class, config.owner_class_enemy_2);
        assert_eq!(encoded.rows[3].owner_class, config.owner_class_enemy_3);
        assert_eq!(encoded.rows[4].owner_class, config.owner_class_neutral);
        assert_eq!(
            encoded.rows[5].owner_class,
            config.owner_class_absent_on_map
        );
        assert_eq!(
            encoded.rows[config.initial_planet_slots].owner_class,
            config.owner_class_empty
        );
        assert_eq!(encoded.rows[5].x_position_normalized, 0.0);
        assert_eq!(encoded.rows[5].y_position_normalized, 0.0);
        assert_eq!(encoded.rows[5].production_normalized, 0.0);
        assert_eq!(encoded.rows[5].velocity_x_normalized, 0.0);
        assert_eq!(encoded.rows[5].velocity_y_normalized, 0.0);
    }

    #[test]
    fn owner_class_for_player_one_maps_single_lower_id_enemy_to_enemy_one() {
        let config = AgentConfig::default();
        let initial = vec![
            planet(10, 1, 10.0, 10.0, 10.0),
            planet(20, 0, 20.0, 10.0, 10.0),
        ];
        let encoded = encode_state(&initial, &initial, &[], 1, &config).unwrap();

        assert_eq!(encoded.rows[0].owner_class, config.owner_class_own);
        assert_eq!(encoded.rows[1].owner_class, config.owner_class_enemy_1);
    }

    #[test]
    fn official_initial_planets_can_have_neutral_home_owner() {
        let config = AgentConfig::default();
        let initial = vec![
            planet(10, -1, 10.0, 10.0, 10.0),
            planet(20, -1, 20.0, 10.0, 10.0),
        ];
        let current = vec![
            planet(10, 0, 10.0, 10.0, 10.0),
            planet(20, -1, 20.0, 10.0, 10.0),
        ];
        let encoded = encode_state(&current, &initial, &[], 0, &config).unwrap();

        assert_eq!(encoded.row_planet_ids[0], 10);
        assert_eq!(encoded.rows[0].owner_class, config.owner_class_own);
    }
}
