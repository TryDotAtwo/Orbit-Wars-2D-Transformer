#!/usr/bin/env bash
set -euo pipefail

cd /workspace

WATCHER_LOG="artifacts/restart_after_current_generation_owslot.log"
CHECKPOINT="artifacts/full_training_2026-06-01_self_train_current_contract.checkpoint.bin"
TRAIN_LOG="artifacts/full_training_2026-06-01_self_train_current_contract.log"
PID_FILE="artifacts/full_training_2026-06-01_self_train_current_contract.pid"
TRAINER_LOG="artifacts/full_training_2026-06-01_self_train_current_contract.log"
NEXT_CUDA_LIB="${NEXT_CUDA_LIB:-}"

CURRENT_GEN="$(python3 -c 'import json; print(json.load(open("dashboard/public/telemetry/latest.json"))["activeGeneration"])')"
START_MTIME="$(stat -c %Y "$CHECKPOINT")"
OLD_PID="$(cat "$PID_FILE" 2>/dev/null || true)"

echo "watcher_start; current_gen=${CURRENT_GEN}; start_mtime=${START_MTIME}; old_pid=${OLD_PID}; at=$(date -Is)" >> "$WATCHER_LOG"

while true; do
  LATEST_GEN="$(python3 -c 'import json; print(json.load(open("dashboard/public/telemetry/latest.json"))["activeGeneration"])' 2>/dev/null || echo 0)"
  CHECKPOINT_MTIME="$(stat -c %Y "$CHECKPOINT" 2>/dev/null || echo 0)"

  if [[ "$LATEST_GEN" -gt "$CURRENT_GEN" && "$CHECKPOINT_MTIME" -gt "$START_MTIME" ]]; then
    echo "boundary_detected; latest_gen=${LATEST_GEN}; checkpoint_mtime=${CHECKPOINT_MTIME}; at=$(date -Is)" >> "$WATCHER_LOG"
    break
  fi

  if grep -q "event=generation_done; run_id=1780348585; generation=${CURRENT_GEN};" "$TRAIN_LOG" && [[ "$CHECKPOINT_MTIME" -gt "$START_MTIME" ]]; then
    echo "generation_done_detected; latest_gen=${LATEST_GEN}; checkpoint_mtime=${CHECKPOINT_MTIME}; at=$(date -Is)" >> "$WATCHER_LOG"
    break
  fi

  if [[ -n "$OLD_PID" ]] && ! kill -0 "$OLD_PID" 2>/dev/null; then
    echo "old_process_stopped_before_checkpoint_boundary; latest_gen=${LATEST_GEN}; checkpoint_mtime=${CHECKPOINT_MTIME}; start_mtime=${START_MTIME}; restart_blocked=true; at=$(date -Is)" >> "$WATCHER_LOG"
    exit 1
  fi

  sleep 10
done

OLD_PID="$(cat "$PID_FILE" 2>/dev/null || true)"
if [[ -n "$OLD_PID" ]] && kill -0 "$OLD_PID" 2>/dev/null; then
  echo "stopping_old; pid=${OLD_PID}; at=$(date -Is)" >> "$WATCHER_LOG"
  kill -TERM "$OLD_PID" 2>/dev/null || true
  for _ in $(seq 1 30); do
    kill -0 "$OLD_PID" 2>/dev/null || break
    sleep 1
  done
  if kill -0 "$OLD_PID" 2>/dev/null; then
    echo "force_kill_old; pid=${OLD_PID}; at=$(date -Is)" >> "$WATCHER_LOG"
    kill -KILL "$OLD_PID" 2>/dev/null || true
  fi
fi

python3 tools/restore_latest_history_from_trainer_log.py >> "$WATCHER_LOG" 2>&1 || {
  echo "history_restore_failed; at=$(date -Is)" >> "$WATCHER_LOG"
}
if [[ -n "$NEXT_CUDA_LIB" ]]; then
  cp "$NEXT_CUDA_LIB" target/liborbit_wars_cuda.so
  echo "cuda_library_replaced; source=${NEXT_CUDA_LIB}; target=target/liborbit_wars_cuda.so; at=$(date -Is)" >> "$WATCHER_LOG"
fi
echo "starting_new_release_resume; at=$(date -Is)" >> "$WATCHER_LOG"
nohup bash -lc \
  'cd /workspace && echo $$ > artifacts/full_training_2026-06-01_self_train_current_contract.pid && exec > >(tee -a artifacts/full_training_2026-06-01_self_train_current_contract.log > /proc/1/fd/1) 2>&1 && exec env ORBIT_WARS_CUDA_LIB_PATH=target/liborbit_wars_cuda.so target/release/orbit-wars-trainer --cuda --players 4 --generations 64 --resume-checkpoint artifacts/full_training_2026-06-01_self_train_current_contract.checkpoint.bin' \
  >/dev/null 2>&1 &
echo "new_launch_issued; at=$(date -Is)" >> "$WATCHER_LOG"
