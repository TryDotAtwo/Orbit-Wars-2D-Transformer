timestamp=2026-05-31T18:20:00+03:00
task_id=2026-05-31_replay_play_chart_hover_sun_strategy
agent_id=codex
source=user
related_files=[dashboard/src/App.tsx,dashboard/src/ReplayCanvas.tsx,dashboard/src/Charts.tsx,dashboard/src/styles.css,project_config.yaml,PROJECT_MEMORY.md,index.md,docs/architecture/SUMMARY.md,docs/contracts/SUMMARY.md,docs/testing/SUMMARY.md,docs/performance/SUMMARY.md,docs/prompts/SUMMARY.md,test_results/2026-05-31_replay_play_chart_hover_sun_strategy.md]
intended_action=add_replay_play_button_add_chart_hover_values_and_analyze_sun_input_strategy
result_status=completed_with_sun_policy_recommendation_pending_user_approval
prompt_raw=Добавь кнопку плей, чтоб можно запустить просмотр било реплея. 

Так же графики не показивают при наводке че там за значение. 

Также смотри, кораблики в Солнце летят, потому что трансформер не получает тоже Солнце на вход и путается. У него информации этой нет. Давай подумаем как этого избежать не трогая архитектуру? Может планету, которая за солнцем, подавать как отсутсвующую на карте просто?
