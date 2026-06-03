# 2026-06-03 Checkpoint Path Config Fix

task=trainer_checkpoint_path_config
code_changes=true

## Change

- Added `TRAINER_CHECKPOINT_PATH` and `AgentConfig::trainer_checkpoint_path`.
- Removed trainer-local hardcoded default checkpoint path.
- Added `trainer_checkpoint_save_path(cli, config)`:
  - returns `cli.resume_checkpoint` when the run explicitly resumes from a path;
  - otherwise returns `config.trainer_checkpoint_path`.

## Why

The deployment docs and `project_config.yaml` treat the checkpoint path as configured project state. The trainer previously loaded from `--resume-checkpoint` but saved to a hardcoded default path, which made non-default resume paths surprising.

## Verification

RED command:

```powershell
docker run --rm -v "<workspace>:/workspace" -w /workspace cmz-native-dev:2026-05-26 bash -lc "cargo test -p orbit-wars-trainer checkpoint_save_path -- --nocapture"
```

RED result:

- exit_code=1
- compile failed because `trainer_checkpoint_save_path` and `AgentConfig::trainer_checkpoint_path` were missing.

GREEN command:

```powershell
docker run --rm -v "<workspace>:/workspace" -w /workspace cmz-native-dev:2026-05-26 bash -lc "cargo test -p orbit-wars-trainer checkpoint_save_path -- --nocapture"
```

GREEN result:

- exit_code=0
- `checkpoint_save_path_uses_config_default_without_resume`: ok
- `checkpoint_save_path_uses_resume_checkpoint_when_supplied`: ok

Formatting:

```powershell
docker run --rm -v "<workspace>:/workspace" -w /workspace cmz-native-dev:2026-05-26 bash -lc "cargo fmt --all"
```

Result: exit_code=0.

## Notes

- No checkpoint binary format changed.
- No trainer selection, reproduction, CUDA, or replay behavior changed.
