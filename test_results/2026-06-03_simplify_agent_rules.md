# Simplify Agent Rules Verification

timestamp=2026-06-03T16:23:21+03:00
task_id=2026-06-03_simplify_agent_rules
status=passed_docs_only

## Scope

- Simplified `AGENTS.md` from a mandatory blanket route into a lightweight, task-relevant route.
- Kept durable safeguards:
  - no `project_history_full/` reads without explicit approval;
  - no Kaggle/external upload without explicit approval;
  - durable state remains in project files;
  - config parameters stay in `project_config.yaml`;
  - hidden fallbacks remain prohibited.
- Relaxed obstructive parts:
  - prompt persistence is required for durable tasks, not tiny status/lookups;
  - `project_config.yaml`, `index.md`, and `docs/*/SUMMARY.md` are read only when relevant;
  - registration is limited to durable changes.

## Files Changed

- `AGENTS.md`
- `PROJECT_MEMORY.md`
- `index.md`
- `docs/prompts/SUMMARY.md`
- `docs/testing/SUMMARY.md`
- `docs/roadmap/SUMMARY.md`
- `prompt_history/20260603_162321_simplify_agent_rules.md`
- `test_results/2026-06-03_simplify_agent_rules.md`

## Verification

- Checked for stale blanket wording across root docs and section summaries.
- Found and fixed the stale `prompt_history/` index note that still said it was required before task work.
- This was a docs-only workflow change; no Rust, CUDA, dashboard, config, checkpoint, or submission behavior changed.
