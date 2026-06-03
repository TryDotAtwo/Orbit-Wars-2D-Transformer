# AGENTS.md

source_protocol=lightweight_agent_protocol_v2
status=simplified_by_user_request_on_2026_06_03

## Operating Mode

- Be useful first: inspect only the context needed for the current task.
- Keep durable project knowledge in project files, but do not turn every small action into paperwork.
- Do not read `project_history_full/` without explicit user approval for that specific access.

## Startup Route

For non-trivial code, config, docs, training, checkpoint, artifact, submission, or external-service work:

1. Save the raw user prompt in `prompt_history/`.
2. Read `AGENTS.md` and `PROJECT_MEMORY.md`.
3. Read `project_config.yaml` only when parameters, paths, limits, tests, runtime, training, dashboard, CUDA, Kaggle, or external-service behavior may be affected.
4. Read `index.md` only when navigating unknown code, changing files, searching project structure, or registering durable results.
5. Read the relevant `docs/<section>/SUMMARY.md` only when that section is directly affected.

For quick status checks, tiny explanations, simple commands, or local file lookups, use judgment: do the smallest safe read set and avoid ritual context loading.

## Durable Memory

- Use `PROJECT_MEMORY.md` for current state, active contracts, known risks, and next actions.
- Use `test_results/` for verification records and investigations.
- Use `prompt_history/` for raw prompts when a task changes durable state or may matter later.
- Chat history is not durable memory.

## Registration

Register durable changes only:

- new or changed public functions/classes/APIs;
- tests and verification records;
- project contracts, decisions, hypotheses, conclusions, or experiments;
- new docs, tools, config parameters, scripts, artifacts intended for reuse.

Do not register every temporary file, generated build output, trivial wording change, or one-off inspection unless it affects future work.

## Coding Rules

- Keep external parameters in `project_config.yaml`.
- Avoid hidden fallbacks; errors should be visible.
- Avoid magic numbers except `0`, `1`, local formulas, explicit sentinels, and named test expectations.
- Do not make architectural behavior changes without explicit user approval.
- Do not submit to Kaggle or upload externally without explicit user approval for that action.

## Communication

- User-facing communication should be concise, explicit Russian when practical.
- State what changed, what was verified, and what was not run.
