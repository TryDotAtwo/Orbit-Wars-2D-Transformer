timestamp=2026-05-31T17:10:00+03:00
task_id=2026-05-31_game_length_and_send_amount_rules_question
branch=master
commit=none
environment=Windows_host
command=code_inspection_only
result=pass

## Findings

- dashboard_frame_limit_cause=`orbit-wars-trainer --smoke` uses `trainer_smoke_episode_steps=64`.
- full_replay_frame_count_formula=`episode_steps + 1` because frame step 0 is stored before first turn.
- official_episode_steps=500 in `DEFAULT_EPISODE_STEPS` and official research docs.
- decoder_ship_count=`floor(source_planet.ships * clamp(send_fraction,0,1))`.
- decoder_one_command_per_source_row=true.
- simulator_launch_guard=per-player per-planet overdraft check before fleet creation.
- current_dashboard_training_state=fixed smoke state, not official full game distribution.
- reference_compare_scope=5 deterministic scenarios against official Kaggle source, not exhaustive proof of every random map/comet case.

## Conclusion

- step_64_is_expected_for_smoke_run=true.
- full_500_turn_dashboard_run_not_yet_executed=true.
- excessive_sending_possible_from_untrained_model=true.
- invalid_overdraft_launch_from_decoder_not_expected=true.
