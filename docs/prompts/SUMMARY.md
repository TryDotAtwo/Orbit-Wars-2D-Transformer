# docs/prompts/SUMMARY.md

timestamp=2026-06-03T16:45:00+03:00
section=prompts
active_task_id=2026-06-03_nvidia_perf_bug_review_no_code_changes

## Policy

- Save raw prompts in `prompt_history/` for tasks that change durable project state or may matter later.
- Quick status checks and tiny local lookups do not need prompt-history paperwork unless the agent decides the prompt is durable.
- This summary registers only recent or structurally important prompts.
- Do not copy the whole prompt archive into this file.

## Current Registration

- `prompt_history/20260603_163041_nvidia_perf_bug_review_no_code_changes.md`: user requested no-code-change project/code/performance review and explicit NVIDIA plugin use.
- `prompt_history/20260603_162321_simplify_agent_rules.md`: user requested simplifying `AGENTS.md` and related workflow docs because the protocol still felt obstructive.
- `prompt_history/20260603_161245_docs_cleanup_request.md`: user requested project review and documentation cleanup because the current docs were excessive and inconvenient.

## Recent Important Prompts

- `prompt_history/2026-06-03_population64_games20_top6_weighted_repro.md`: approval for population 64, 20 participations per model, top-6 weighted reproduction, and separate future meta tuning.
- `prompt_history/2026-06-03_increase_sun_target_penalty.md`: request to strengthen action-local Sun target penalty.
- `prompt_history/2026-06-02_fast_planet_miss_velocity_input.md`: request for velocity input, phase investigation, and checkpoint-boundary restart.
- `prompt_history/2026-06-02_no_submit_without_approval_validation_logs.md`: explicit no-submit-without-approval instruction after validation logs.
- `prompt_history/2026-06-02_git_init_with_bloat_guard_gitignore.md`: request to initialize Git and protect against staging heavy artifacts.

## Result Status

- latest_prompt_status=nvidia_perf_bug_review_completed_no_code_changes.
- full prompt history remains in `prompt_history/`.
