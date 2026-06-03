use std::collections::{BTreeMap, BTreeSet};

use crate::config::AgentConfig;
use crate::geometry::{fleet_speed, point_out_of_bounds, segment_intersects_sun, swept_pair_hit};
use crate::types::{Fleet, MoveCommand, Planet};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Arrival {
    pub owner: i32,
    pub ships: i32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CombatResult {
    pub owner: i32,
    pub ships: i32,
}

#[derive(Clone, Debug)]
pub struct CometGroup {
    pub spawn_step: usize,
    pub planet_ids: Vec<i32>,
    pub paths: Vec<Vec<(f32, f32)>>,
    pub starting_ships: f32,
    pub path_index: isize,
    pub active: bool,
}

#[derive(Clone, Debug)]
pub struct SimulationState {
    pub step: usize,
    pub angular_velocity: f32,
    pub planets: Vec<Planet>,
    pub fleets: Vec<Fleet>,
    pub initial_planets: Vec<Planet>,
    pub comet_groups: Vec<CometGroup>,
    pub next_fleet_id: i32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlayerScore {
    pub player: i32,
    pub ships: i32,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PlayerStepEvents {
    pub launched_fleet_count: usize,
    pub launched_ship_count: i64,
    pub hit_fleet_count: usize,
    pub hit_ship_count: i64,
    pub sun_destroyed_fleet_count: usize,
    pub sun_destroyed_ship_count: i64,
    pub out_of_bounds_destroyed_fleet_count: usize,
    pub out_of_bounds_destroyed_ship_count: i64,
    pub captured_planet_count: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LaunchedFleetEvent {
    pub player: i32,
    pub fleet_id: i32,
    pub ship_count: i32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SunDestroyedFleetEvent {
    pub player: i32,
    pub fleet_id: i32,
    pub ship_count: i32,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SimulationStepEvents {
    pub by_player: BTreeMap<i32, PlayerStepEvents>,
    pub launched_fleets: Vec<LaunchedFleetEvent>,
    pub sun_destroyed_fleets: Vec<SunDestroyedFleetEvent>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SimulationError {
    InvalidConfig,
    InvalidPlayerActionCount,
    LaunchFromMissingPlanet(i32),
    LaunchFromForeignPlanet(i32),
    LaunchNonPositiveShips(i32),
    LaunchOverdraft(i32),
    NonFiniteAction,
    CometPathPlanetMismatch,
}

#[derive(Clone, Copy, Debug)]
struct PlanetPath {
    old_x: f32,
    old_y: f32,
    new_x: f32,
    new_y: f32,
    check_collision: bool,
}

pub fn resolve_planet_combat(
    planet_owner: i32,
    planet_ships: i32,
    arrivals: &[Arrival],
) -> CombatResult {
    let mut attacker_totals = BTreeMap::<i32, i32>::new();
    for arrival in arrivals {
        *attacker_totals.entry(arrival.owner).or_insert(0) += arrival.ships;
    }

    let mut forces = attacker_totals.into_iter().collect::<Vec<_>>();
    forces.sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0)));

    let Some((largest_owner, largest_ships)) = forces.first().copied() else {
        return CombatResult {
            owner: planet_owner,
            ships: planet_ships,
        };
    };

    let second_ships = forces.get(1).map(|force| force.1).unwrap_or(0);
    if largest_ships == second_ships {
        return CombatResult {
            owner: planet_owner,
            ships: planet_ships,
        };
    }

    let surviving_attackers = largest_ships - second_ships;
    if largest_owner == planet_owner {
        return CombatResult {
            owner: planet_owner,
            ships: planet_ships + surviving_attackers,
        };
    }

    if surviving_attackers > planet_ships {
        CombatResult {
            owner: largest_owner,
            ships: surviving_attackers - planet_ships,
        }
    } else {
        CombatResult {
            owner: planet_owner,
            ships: planet_ships - surviving_attackers,
        }
    }
}

impl SimulationState {
    pub fn new(planets: Vec<Planet>, angular_velocity: f32) -> Self {
        let next_fleet_id = 0;
        Self {
            step: 0,
            angular_velocity,
            initial_planets: planets.clone(),
            planets,
            fleets: Vec::new(),
            comet_groups: Vec::new(),
            next_fleet_id,
        }
    }

    pub fn step_turn(
        &mut self,
        actions_by_player: &[Vec<MoveCommand>],
        config: &AgentConfig,
    ) -> Result<(), SimulationError> {
        self.step_turn_with_events(actions_by_player, config)
            .map(|_| ())
    }

    pub fn step_turn_with_events(
        &mut self,
        actions_by_player: &[Vec<MoveCommand>],
        config: &AgentConfig,
    ) -> Result<SimulationStepEvents, SimulationError> {
        config
            .validate()
            .map_err(|_| SimulationError::InvalidConfig)?;
        if actions_by_player.len() > config.max_players {
            return Err(SimulationError::InvalidPlayerActionCount);
        }

        let mut events = SimulationStepEvents::default();
        self.expire_comets();
        self.spawn_comets(config)?;
        self.launch_fleets(actions_by_player, config, &mut events)?;
        self.produce_ships();
        let planet_paths = self.compute_planet_paths(config);
        let arrivals = self.move_fleets(config, &planet_paths, &mut events);
        self.apply_planet_paths_and_expire_comets(&planet_paths);
        self.resolve_arrivals(arrivals, &mut events);
        self.step += 1;
        Ok(events)
    }

    pub fn scores(&self, config: &AgentConfig) -> Vec<PlayerScore> {
        let mut totals = vec![0i32; config.max_players];
        for planet in self.planets.iter().filter(|planet| planet.owner >= 0) {
            if let Some(total) = totals.get_mut(planet.owner as usize) {
                *total += planet.ships.floor() as i32;
            }
        }
        for fleet in self.fleets.iter().filter(|fleet| fleet.owner >= 0) {
            if let Some(total) = totals.get_mut(fleet.owner as usize) {
                *total += fleet.ships.floor() as i32;
            }
        }
        totals
            .into_iter()
            .enumerate()
            .map(|(player, ships)| PlayerScore {
                player: player as i32,
                ships,
            })
            .collect()
    }

    pub fn alive_players(&self) -> BTreeSet<i32> {
        let mut players = BTreeSet::new();
        for planet in self.planets.iter().filter(|planet| planet.owner >= 0) {
            players.insert(planet.owner);
        }
        for fleet in self.fleets.iter().filter(|fleet| fleet.owner >= 0) {
            players.insert(fleet.owner);
        }
        players
    }

    fn expire_comets(&mut self) {
        let expired_ids = self.expired_comet_ids();
        self.remove_comets_by_id(&expired_ids);
    }

    fn spawn_comets(&mut self, config: &AgentConfig) -> Result<(), SimulationError> {
        for group in self
            .comet_groups
            .iter_mut()
            .filter(|group| !group.active && group.spawn_step == self.step + 1)
        {
            if group.planet_ids.len() != group.paths.len() {
                return Err(SimulationError::CometPathPlanetMismatch);
            }
            group.active = true;
            group.path_index = -1;
            for planet_id in group.planet_ids.iter().copied() {
                let planet = Planet {
                    id: planet_id,
                    owner: -1,
                    x: -99.0,
                    y: -99.0,
                    radius: config.comet_radius,
                    ships: group.starting_ships,
                    production: config.comet_production,
                    velocity_x: 0.0,
                    velocity_y: 0.0,
                };
                self.planets.push(planet);
                self.initial_planets.push(planet);
            }
        }
        Ok(())
    }

    fn launch_fleets(
        &mut self,
        actions_by_player: &[Vec<MoveCommand>],
        config: &AgentConfig,
        events: &mut SimulationStepEvents,
    ) -> Result<(), SimulationError> {
        for (player, actions) in actions_by_player.iter().enumerate() {
            let player = player as i32;
            let mut requested_by_planet = BTreeMap::<i32, i32>::new();
            for action in actions {
                if !action.direction_angle.is_finite() {
                    return Err(SimulationError::NonFiniteAction);
                }
                if action.ship_count < 1 {
                    return Err(SimulationError::LaunchNonPositiveShips(
                        action.from_planet_id,
                    ));
                }
                let Some(source) = self
                    .planets
                    .iter()
                    .find(|planet| planet.id == action.from_planet_id)
                else {
                    return Err(SimulationError::LaunchFromMissingPlanet(
                        action.from_planet_id,
                    ));
                };
                if source.owner != player {
                    return Err(SimulationError::LaunchFromForeignPlanet(
                        action.from_planet_id,
                    ));
                }
                *requested_by_planet
                    .entry(action.from_planet_id)
                    .or_insert(0) += action.ship_count;
                if *requested_by_planet.get(&action.from_planet_id).unwrap()
                    > source.ships.floor() as i32
                {
                    return Err(SimulationError::LaunchOverdraft(action.from_planet_id));
                }
            }

            for action in actions {
                let source_index = self
                    .planets
                    .iter()
                    .position(|planet| planet.id == action.from_planet_id)
                    .ok_or(SimulationError::LaunchFromMissingPlanet(
                        action.from_planet_id,
                    ))?;
                let source = self.planets[source_index];
                self.planets[source_index].ships -= action.ship_count as f32;
                let spawn_distance = source.radius + config.fleet_spawn_offset;
                let fleet_id = self.next_fleet_id;
                self.fleets.push(Fleet {
                    id: fleet_id,
                    owner: player,
                    x: source.x + action.direction_angle.cos() * spawn_distance,
                    y: source.y + action.direction_angle.sin() * spawn_distance,
                    angle: action.direction_angle,
                    from_planet_id: action.from_planet_id,
                    ships: action.ship_count as f32,
                });
                events.player_mut(player).launched_fleet_count += 1;
                events.player_mut(player).launched_ship_count += i64::from(action.ship_count);
                events.launched_fleets.push(LaunchedFleetEvent {
                    player,
                    fleet_id,
                    ship_count: action.ship_count,
                });
                self.next_fleet_id += 1;
            }
        }
        Ok(())
    }

    fn produce_ships(&mut self) {
        for planet in self.planets.iter_mut().filter(|planet| planet.owner >= 0) {
            planet.ships += planet.production;
        }
    }

    fn compute_planet_paths(&mut self, config: &AgentConfig) -> BTreeMap<i32, PlanetPath> {
        let comet_ids = self
            .comet_groups
            .iter()
            .filter(|group| group.active)
            .flat_map(|group| group.planet_ids.iter().copied())
            .collect::<BTreeSet<_>>();
        let initial_by_id = self
            .initial_planets
            .iter()
            .map(|planet| (planet.id, *planet))
            .collect::<BTreeMap<_, _>>();
        let rotation_step = self.step.max(1) as f32;
        let mut planet_paths = BTreeMap::<i32, PlanetPath>::new();

        for planet in self
            .planets
            .iter()
            .filter(|planet| !comet_ids.contains(&planet.id))
        {
            let mut new_x = planet.x;
            let mut new_y = planet.y;
            if let Some(initial) = initial_by_id.get(&planet.id) {
                let dx = initial.x - config.board_center;
                let dy = initial.y - config.board_center;
                let orbital_radius = (dx * dx + dy * dy).sqrt();
                if orbital_radius + planet.radius < config.rotation_radius_limit {
                    let initial_angle = dy.atan2(dx);
                    let current_angle = initial_angle + self.angular_velocity * rotation_step;
                    new_x = config.board_center + orbital_radius * current_angle.cos();
                    new_y = config.board_center + orbital_radius * current_angle.sin();
                }
            }
            planet_paths.insert(
                planet.id,
                PlanetPath {
                    old_x: planet.x,
                    old_y: planet.y,
                    new_x,
                    new_y,
                    check_collision: true,
                },
            );
        }

        for group in self.comet_groups.iter_mut().filter(|group| group.active) {
            group.path_index += 1;
            for (index, planet_id) in group.planet_ids.iter().copied().enumerate() {
                let Some(planet) = self.planets.iter().find(|planet| planet.id == planet_id) else {
                    continue;
                };
                let Some(path) = group.paths.get(index) else {
                    continue;
                };
                let old_x = planet.x;
                let old_y = planet.y;
                let Some((new_x, new_y)) = path
                    .get(group.path_index.max(0) as usize)
                    .copied()
                    .filter(|_| group.path_index >= 0)
                else {
                    planet_paths.insert(
                        planet_id,
                        PlanetPath {
                            old_x,
                            old_y,
                            new_x: old_x,
                            new_y: old_y,
                            check_collision: true,
                        },
                    );
                    continue;
                };
                planet_paths.insert(
                    planet_id,
                    PlanetPath {
                        old_x,
                        old_y,
                        new_x,
                        new_y,
                        check_collision: old_x >= 0.0,
                    },
                );
            }
        }

        planet_paths
    }

    fn move_fleets(
        &mut self,
        config: &AgentConfig,
        planet_paths: &BTreeMap<i32, PlanetPath>,
        events: &mut SimulationStepEvents,
    ) -> BTreeMap<i32, Vec<Arrival>> {
        let mut arrivals = BTreeMap::<i32, Vec<Arrival>>::new();
        let mut moved_fleets = Vec::with_capacity(self.fleets.len());
        for fleet in self.fleets.iter().copied() {
            let speed = fleet_speed(fleet.ships, config);
            let new_x = fleet.x + fleet.angle.cos() * speed;
            let new_y = fleet.y + fleet.angle.sin() * speed;

            let hit_planet = self.planets.iter().find(|planet| {
                let Some(path) = planet_paths.get(&planet.id) else {
                    return false;
                };
                path.check_collision
                    && swept_pair_hit(
                        fleet.x,
                        fleet.y,
                        new_x,
                        new_y,
                        path.old_x,
                        path.old_y,
                        path.new_x,
                        path.new_y,
                        planet.radius,
                    )
            });

            if let Some(planet) = hit_planet {
                events.player_mut(fleet.owner).hit_fleet_count += 1;
                events.player_mut(fleet.owner).hit_ship_count += fleet.ships.floor() as i64;
                arrivals.entry(planet.id).or_default().push(Arrival {
                    owner: fleet.owner,
                    ships: fleet.ships.floor() as i32,
                });
            } else if point_out_of_bounds(new_x, new_y, config) {
                events
                    .player_mut(fleet.owner)
                    .out_of_bounds_destroyed_fleet_count += 1;
                events
                    .player_mut(fleet.owner)
                    .out_of_bounds_destroyed_ship_count += fleet.ships.floor() as i64;
                continue;
            } else if segment_intersects_sun(fleet.x, fleet.y, new_x, new_y, config) {
                events.player_mut(fleet.owner).sun_destroyed_fleet_count += 1;
                events.player_mut(fleet.owner).sun_destroyed_ship_count +=
                    fleet.ships.floor() as i64;
                events.sun_destroyed_fleets.push(SunDestroyedFleetEvent {
                    player: fleet.owner,
                    fleet_id: fleet.id,
                    ship_count: fleet.ships.floor() as i32,
                });
                continue;
            } else {
                moved_fleets.push(Fleet {
                    x: new_x,
                    y: new_y,
                    ..fleet
                });
            }
        }
        self.fleets = moved_fleets;
        arrivals
    }

    fn apply_planet_paths_and_expire_comets(&mut self, planet_paths: &BTreeMap<i32, PlanetPath>) {
        for planet in self.planets.iter_mut() {
            let Some(path) = planet_paths.get(&planet.id) else {
                continue;
            };
            planet.velocity_x = path.new_x - path.old_x;
            planet.velocity_y = path.new_y - path.old_y;
            planet.x = path.new_x;
            planet.y = path.new_y;
        }

        let expired_ids = self.expired_comet_ids_after_movement();
        self.remove_comets_by_id(&expired_ids);
    }

    fn expired_comet_ids(&self) -> BTreeSet<i32> {
        self.comet_groups
            .iter()
            .filter(|group| group.active)
            .flat_map(|group| {
                group
                    .planet_ids
                    .iter()
                    .copied()
                    .enumerate()
                    .filter_map(|(index, planet_id)| {
                        let expired = group.path_index >= 0
                            && group
                                .paths
                                .get(index)
                                .map(|path| group.path_index as usize >= path.len())
                                .unwrap_or(true);
                        expired.then_some(planet_id)
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    fn expired_comet_ids_after_movement(&self) -> BTreeSet<i32> {
        self.expired_comet_ids()
    }

    fn remove_comets_by_id(&mut self, expired_ids: &BTreeSet<i32>) {
        if expired_ids.is_empty() {
            return;
        }
        self.planets
            .retain(|planet| !expired_ids.contains(&planet.id));
        self.initial_planets
            .retain(|planet| !expired_ids.contains(&planet.id));
        for group in self.comet_groups.iter_mut() {
            group
                .planet_ids
                .retain(|planet_id| !expired_ids.contains(planet_id));
            if group.planet_ids.is_empty() {
                group.active = false;
            }
        }
    }

    fn resolve_arrivals(
        &mut self,
        arrivals: BTreeMap<i32, Vec<Arrival>>,
        events: &mut SimulationStepEvents,
    ) {
        for (planet_id, planet_arrivals) in arrivals {
            if let Some(planet) = self
                .planets
                .iter_mut()
                .find(|planet| planet.id == planet_id)
            {
                let previous_owner = planet.owner;
                let result = resolve_planet_combat(
                    planet.owner,
                    planet.ships.floor() as i32,
                    &planet_arrivals,
                );
                planet.owner = result.owner;
                planet.ships = result.ships as f32;
                if result.owner >= 0 && result.owner != previous_owner {
                    events.player_mut(result.owner).captured_planet_count += 1;
                }
            }
        }
    }
}

impl SimulationStepEvents {
    pub fn player(&self, player: i32) -> PlayerStepEvents {
        self.by_player.get(&player).copied().unwrap_or_default()
    }

    fn player_mut(&mut self, player: i32) -> &mut PlayerStepEvents {
        self.by_player.entry(player).or_default()
    }
}

pub fn is_terminal(state: &SimulationState, config: &AgentConfig) -> bool {
    state.step >= config.episode_steps || state.alive_players().len() <= 1
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> AgentConfig {
        AgentConfig::default()
    }

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
    fn combat_captures_planet_when_survivor_exceeds_garrison() {
        let result = resolve_planet_combat(0, 5, &[Arrival { owner: 1, ships: 8 }]);
        assert_eq!(result, CombatResult { owner: 1, ships: 3 });
    }

    #[test]
    fn combat_attacker_tie_destroys_attackers() {
        let result = resolve_planet_combat(
            0,
            5,
            &[
                Arrival { owner: 1, ships: 8 },
                Arrival { owner: 2, ships: 8 },
            ],
        );
        assert_eq!(result, CombatResult { owner: 0, ships: 5 });
    }

    #[test]
    fn step_launches_before_production() {
        let config = test_config();
        let mut state = SimulationState::new(vec![planet(1, 0, 10.0, 10.0, 10.0)], 0.0);
        state
            .step_turn(
                &[vec![MoveCommand {
                    from_planet_id: 1,
                    direction_angle: 0.0,
                    ship_count: 4,
                }]],
                &config,
            )
            .unwrap();

        assert_eq!(state.planets[0].ships, 7.0);
        assert_eq!(state.fleets.len(), 1);
    }

    #[test]
    fn fleet_collision_captures_target() {
        let config = test_config();
        let mut state = SimulationState::new(
            vec![
                planet(1, 0, 10.0, 10.0, 20.0),
                planet(2, -1, 14.0, 10.0, 2.0),
            ],
            0.0,
        );
        state
            .step_turn(
                &[vec![MoveCommand {
                    from_planet_id: 1,
                    direction_angle: 0.0,
                    ship_count: 8,
                }]],
                &config,
            )
            .unwrap();
        let events = state.step_turn_with_events(&[Vec::new()], &config).unwrap();

        let target = state.planets.iter().find(|planet| planet.id == 2).unwrap();
        assert_eq!(target.owner, 0);
        assert_eq!(target.ships, 6.0);
        assert!(state.fleets.is_empty());
        let player_events = events.player(0);
        assert_eq!(player_events.hit_fleet_count, 1);
        assert_eq!(player_events.hit_ship_count, 8);
        assert_eq!(player_events.captured_planet_count, 1);
    }

    #[test]
    fn fleet_crossing_sun_is_destroyed() {
        let config = test_config();
        let mut state = SimulationState::new(
            vec![
                planet(1, 0, 40.0, 50.0, 10.0),
                planet(2, -1, 80.0, 50.0, 10.0),
            ],
            0.0,
        );
        let events = state
            .step_turn_with_events(
                &[vec![MoveCommand {
                    from_planet_id: 1,
                    direction_angle: 0.0,
                    ship_count: 5,
                }]],
                &config,
            )
            .unwrap();

        assert!(state.fleets.is_empty());
        let player_events = events.player(0);
        assert_eq!(player_events.launched_fleet_count, 1);
        assert_eq!(player_events.launched_ship_count, 5);
        assert_eq!(player_events.sun_destroyed_fleet_count, 1);
        assert_eq!(player_events.sun_destroyed_ship_count, 5);
        assert_eq!(events.launched_fleets.len(), 1);
        assert_eq!(events.sun_destroyed_fleets.len(), 1);
        assert_eq!(events.launched_fleets[0].fleet_id, 0);
        assert_eq!(
            events.sun_destroyed_fleets[0].fleet_id,
            events.launched_fleets[0].fleet_id
        );
        assert_eq!(
            state
                .planets
                .iter()
                .find(|planet| planet.id == 2)
                .unwrap()
                .owner,
            -1
        );
    }

    #[test]
    fn orbiting_planet_rotates_after_fleet_movement() {
        let config = test_config();
        let mut state = SimulationState::new(vec![planet(1, 0, 60.0, 50.0, 10.0)], 0.1);
        state.step_turn(&[Vec::new()], &config).unwrap();
        assert!(state.planets[0].y > 50.0);
        assert!(state.planets[0].velocity_y > 0.0);
    }

    #[test]
    fn comet_spawns_and_expires_through_turn_loop() {
        let config = test_config();
        let mut state = SimulationState::new(vec![planet(1, 0, 10.0, 10.0, 10.0)], 0.0);
        state.comet_groups.push(CometGroup {
            spawn_step: 1,
            planet_ids: vec![100],
            paths: vec![vec![(20.0, 20.0), (30.0, 20.0)]],
            starting_ships: 4.0,
            path_index: 0,
            active: false,
        });

        state.step_turn(&[Vec::new()], &config).unwrap();
        assert!(state.planets.iter().any(|planet| planet.id == 100));
        state.step_turn(&[Vec::new()], &config).unwrap();
        state.step_turn(&[Vec::new()], &config).unwrap();
        assert!(!state.planets.iter().any(|planet| planet.id == 100));
    }

    #[test]
    fn comet_path_planet_mismatch_errors() {
        let config = test_config();
        let mut state = SimulationState::new(vec![planet(1, 0, 10.0, 10.0, 10.0)], 0.0);
        state.comet_groups.push(CometGroup {
            spawn_step: 1,
            planet_ids: vec![100],
            paths: vec![
                vec![(20.0, 20.0), (30.0, 20.0)],
                vec![(40.0, 40.0), (50.0, 40.0)],
            ],
            starting_ships: 4.0,
            path_index: 0,
            active: false,
        });

        assert_eq!(
            state.step_turn(&[Vec::new()], &config),
            Err(SimulationError::CometPathPlanetMismatch)
        );
    }
}
