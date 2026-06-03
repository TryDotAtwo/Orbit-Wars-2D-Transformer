# Kaggle Orbit Wars Competition Research

timestamp=2026-05-31T00:21:00+03:00
task_id=task_kaggle_orbit_wars_research_2026_05_31
source_status=official_kaggle_pages_and_official_kaggle_data_files_used

## Source Inventory

- official_overview=https://www.kaggle.com/competitions/orbit-wars/overview
- official_evaluation=https://www.kaggle.com/competitions/orbit-wars/overview/evaluation
- official_rules=https://www.kaggle.com/competitions/orbit-wars/rules
- official_data=https://www.kaggle.com/competitions/orbit-wars/data
- downloaded_files=`artifacts/orbit_wars_official_2026_05_31/README.md`, `agents.md`, `main.py`
- exported_rules=`artifacts/orbit_wars_official_2026_05_31/kaggle_pages_rules.csv`

## Competition Metadata

- title=Orbit Wars
- type=Kaggle Featured simulation competition
- host=Kaggle
- sponsor=Google LLC
- category=Featured
- tags=games, artificial intelligence, reinforcement learning, custom metric
- description=Conquer planets rotating around a sun in continuous 2D space; real-time strategy game for 2 or 4 players.
- start_date_utc=2026-04-16T19:01:30.503Z
- final_submission_deadline_utc=2026-06-23T23:59:00Z
- entry_deadline_utc=2026-06-16T23:59:00Z
- team_merger_deadline_utc=2026-06-16T23:59:00Z
- final_ladder_window=2026-06-24_to_approximately_2026-07-08_or_until_convergence
- total_prize_usd=50000
- prize_distribution=places_1_to_10_each_receive_5000_usd
- max_daily_submissions=5
- max_team_size=5
- final_submissions=2
- evaluation_metric=orbit_wars
- awards_points=true
- kernels_submissions_only=false
- team_count_at_query=3491

## Objective

Agent controls a player that starts with one home planet.
Agent sends fleets to capture neutral and enemy planets.
Winner has the largest total ship count on owned planets plus owned fleets at termination.

## Board Contract

- board_size=100x100 continuous space
- coordinate_origin=top_left
- sun_center=(50,50)
- sun_radius=10
- sun_rule=fleet path crossing sun destroys fleet
- symmetry=4-fold mirror symmetry around center: `(x,y)`, `(100-x,y)`, `(x,100-y)`, `(100-x,100-y)`
- planet_group_count=5_to_10_groups_of_4
- planet_count=20_to_40
- static_group_minimum=3
- orbiting_group_minimum=1

## Planet Contract

- planet_format=`[id, owner, x, y, radius, ships, production]`
- owner_values=`0..3` player id, `-1` neutral
- production_range=1_to_5
- radius_formula=`1 + ln(production)`
- initial_ships_range=5_to_99_skewed_low
- home_planet_ships=10
- orbiting_condition=`orbital_radius + planet_radius < 50`
- orbiting_angular_velocity_range=0.025_to_0.05_radians_per_turn
- static_condition=planet farther from center beyond rotation limit
- owned_planet_production=production ships per turn

## Fleet Contract

- fleet_format=`[id, owner, x, y, angle, from_planet_id, ships]`
- angle_unit=radians
- angle_zero=right
- angle_pi_over_2=down
- ships_constant_during_travel=true
- speed_formula=`1.0 + (maxSpeed - 1.0) * (log(ships) / log(1000)) ^ 1.5`
- one_ship_speed=1.0_units_per_turn
- maximum_speed_default=6.0_units_per_turn
- approximately_500_ship_speed=5_units_per_turn
- approximately_1000_ship_speed=maximum_speed
- movement=straight_line
- collision_detection=continuous_segment_from_old_position_to_new_position
- removal_conditions=out_of_bounds, sun_collision, planet_collision_after_combat_queue

## Launch Contract

- action_format=`[[from_planet_id, direction_angle, num_ships], ...]`
- no_action=`[]`
- from_planet_requirement=owned_planet_only
- maximum_launch_ships=current_planet_ships
- spawn_position=just_outside_planet_radius_in_launch_direction
- multiple_launches_per_turn=allowed
- multiple_launches_same_planet=allowed_if_total_ships_available

## Comet Contract

- comet_type=temporary_planets
- spawn_steps=50,150,250,350,450
- spawn_group_size=4
- spawn_symmetry=one_comet_per_quadrant
- radius=1.0
- production=1_ship_per_turn_when_owned
- starting_ships=random_skewed_low_minimum_of_4_rolls_from_1_to_99
- same_group_starting_ships=identical
- speed_default=4.0_units_per_turn
- observation_ids_field=`comet_planet_ids`
- observation_paths_field=`comets[].paths`
- observation_path_index_field=`comets[].path_index`
- comet_expiration=removed_when_leaves_board
- comet_expiration_order=before_fleet_launch
- departing_comet_launch_allowed=false

## Turn Order

1. comet_expiration
2. comet_spawning
3. fleet_launch
4. production
5. fleet_movement_and_collision_queue
6. planet_rotation_and_comet_movement_and_sweep_collision_queue
7. combat_resolution

## Combat Contract

- arriving_fleets_grouped_by_owner=true
- same_owner_arrivals_sum=true
- largest_attacker_fights_second_largest_attacker=true
- surviving_attacker_ships=largest_minus_second_largest
- attacker_tie_result=all_attacking_ships_destroyed
- surviving_same_owner_attacker=ships_added_to_garrison
- surviving_different_owner_attacker=fights_planet_garrison
- capture_condition=attacker_surplus_exceeds_garrison
- captured_garrison=attacker_surplus_after_garrison_subtraction

## Observation Contract

- `planets`: all planets including comets
- `fleets`: all active fleets
- `player`: current player id
- `angular_velocity`: planet rotation speed
- `initial_planets`: start positions for planet prediction
- `comets`: active comet group data
- `comet_planet_ids`: ids of planets that are comets
- `remainingOverageTime`: remaining overage time budget in seconds

## Termination And Score

- step_limit=500_turns
- elimination=only_one_or_zero_players_have_planets_or_fleets
- final_score=owned_planet_ships_plus_owned_fleet_ships
- winner=highest_final_score

## Local Development Workflow

- dependency=`kaggle-environments>=1.28.0`
- baseline_file=`main.py`
- required_agent_symbol=`agent`
- local_test_command_python=`env = make("orbit_wars", configuration={"seed": 42}, debug=True); env.run(["main.py", "random"])`
- notebook_render=`env.render(mode="ipython", width=800, height=600)`

## Submission Workflow

- required_join_step=accept competition rules through Kaggle website before first submission
- auth_methods=Kaggle API token file, OAuth browser flow, `KAGGLE_API_TOKEN`
- verify_cli=`kaggle competitions list -s "orbit wars"`
- download_data=`kaggle competitions download orbit-wars -p orbit-wars-data`
- single_file_submit=`kaggle competitions submit orbit-wars -f main.py -m "message"`
- multi_file_submit=`tar -czf submission.tar.gz main.py helper.py model_weights.pkl` then `kaggle competitions submit orbit-wars -f submission.tar.gz -m "message"`
- tarball_requirement=`main.py` at archive root
- notebook_submit=`kaggle competitions submit orbit-wars -k YOUR_USERNAME/orbit-wars-agent -f submission.tar.gz -v 1 -m "message"`
- status_command=`kaggle competitions submissions orbit-wars`
- episodes_command=`kaggle competitions episodes <SUBMISSION_ID>`
- replay_command=`kaggle competitions replay <EPISODE_ID> -p ./replays`
- logs_command=`kaggle competitions logs <EPISODE_ID> <AGENT_INDEX> -p ./logs`
- leaderboard_command=`kaggle competitions leaderboard orbit-wars -s`

## Evaluation Model

- validation_episode=self_play_against_copies_before_ladder_entry
- validation_failure_status=Error
- validation_failure_debug=download_agent_logs
- initial_rating_mu=600
- skill_model=Gaussian_N_mu_sigma_squared
- sigma=uncertainty_decreases_over_time
- matchmaking=similar_skill_rating_preferred
- new_submission_episode_rate=increased_for_faster_feedback
- win_effect=increase_winner_mu_and_decrease_loser_mu
- draw_effect=move_mu_values_toward_mean
- update_magnitude=depends_on_expected_result_deviation_and_sigma
- win_margin_effect=none
- leaderboard_display=best_scoring_bot_only
- submission_tracking=all_submissions_visible_on_submissions_page
- final_leaderboard=simulation_has_no_private_leaderboard

## Rules And Restrictions

- one_kaggle_account_only=true
- private_code_sharing_outside_team=forbidden
- public_code_sharing_allowed_only_on_competition_forum_or_notebooks
- public_code_license_requirement=OSI-approved_license_no_commercial_use_limit
- open_source_code_requirement=OSI-approved_license_no_commercial_use_limit
- external_data_and_models=allowed_if_publicly_available_equally_accessible_no_cost_or_reasonably_accessible_minimal_cost
- automated_ml_tools=allowed_with_appropriate_license
- runtime_ingress_egress=forbidden
- public_replays=episode_actions_may_be_publicly_available_and_downloadable
- data_access_license=Apache_2.0
- winner_license=CC-BY_4.0
- winner_obligation=detailed_reproducible_methodology_and_code_repository_or_procurement_details
- restricted_residency=Crimea_DNR_LNR_Cuba_Iran_North_Korea_and_sanctioned_or_export_controlled_persons

