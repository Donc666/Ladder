#!/usr/bin/env bash
# Default: relaunch run_task.py once per episode (fresh Isaac).
# In-process multi-episode reuse of the same scene makes later demos identical.
# Multiple terminals can share DATASET_ROOT (episode ids are flock-safe).
#
#   bash ./collect.sh 25
#   DATASET_ROOT=./collect_datas/stack_blocks_three bash ./collect.sh 25
#   RELAUNCH_EACH_EPISODE=0 bash ./collect.sh 25   # one Isaac boot, N episodes (not recommended)
#
# Logs:
#   ./run_logs/collect_<stamp>_<worker>.summary.log
#   ./run_logs/collect_<stamp>_<worker>_run001.log ...
#   ./run_logs/collect_<stamp>_<worker>_all.log
#
# Env: PYTHON PY_SCRIPT DATASET_ROOT LOG_DIR WORKER_ID CUDA_VISIBLE_DEVICES
#      RELAUNCH_EACH_EPISODE (default 1)  RELAUNCH_SLEEP (default 2)
#      TEE_LOG=0 to hide python stdout (still written to per-run logs)

set -u
ulimit -c 0

cd "$(dirname "${BASH_SOURCE[0]}")"

export PYTHONUNBUFFERED=1
PYTHON="${PYTHON:-python}"
PY_SCRIPT="${PY_SCRIPT:-./run_task.py}"
DATASET_ROOT="${DATASET_ROOT:-./collect_test2/pick_dual_bottles}"
LOG_DIR="${LOG_DIR:-./run_logs}"
WORKER_ID="${WORKER_ID:-w$$}"
NUM_RUNS=25
RELAUNCH_EACH_EPISODE="${RELAUNCH_EACH_EPISODE:-1}"
RELAUNCH_SLEEP="${RELAUNCH_SLEEP:-2}"
TEE_LOG="${TEE_LOG:-1}"
# Seconds to wait after "Finished this episode" before SIGKILL if Kit still hangs.
EPISODE_EXIT_GRACE="${EPISODE_EXIT_GRACE:-8}"

if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
  NUM_RUNS="${1}"
  shift
fi
if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
  echo "[INFO] ignoring legacy worker count '${1}' (use multiple terminals instead)"
  shift
fi

if [[ $# -gt 0 ]]; then
  EXTRA=("$@")
else
  EXTRA=(--enable_cameras --headless)
fi

if [[ ! -f "${PY_SCRIPT}" ]]; then
  echo "[ERROR] missing script: ${PY_SCRIPT}"
  exit 1
fi

mkdir -p "${DATASET_ROOT}/data" "${DATASET_ROOT}/_traj_data" "${DATASET_ROOT}/video" "${DATASET_ROOT}/meta" "${LOG_DIR}"
export DATASET_ROOT

stamp="$(date +%Y%m%d_%H%M%S)_${WORKER_ID}"
summary_log="${LOG_DIR}/collect_${stamp}.summary.log"
run_log="${LOG_DIR}/collect_${stamp}_all.log"
worker_meta="${DATASET_ROOT}/meta/workers/${WORKER_ID}.txt"
mkdir -p "${DATASET_ROOT}/meta/workers"
{
  echo "worker_id=${WORKER_ID}"
  echo "pid=$$"
  echo "start=$(date -Iseconds)"
  echo "num_runs=${NUM_RUNS}"
  echo "relaunch_each_episode=${RELAUNCH_EACH_EPISODE}"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"
  echo "cmd=${PYTHON} ${PY_SCRIPT} ${EXTRA[*]}"
} >"${worker_meta}"

echo "[INFO] worker_id=${WORKER_ID} pid=$$" | tee -a "${summary_log}"
echo "[INFO] runs=${NUM_RUNS}" | tee -a "${summary_log}"
echo "[INFO] PY_SCRIPT=${PY_SCRIPT}" | tee -a "${summary_log}"
echo "[INFO] DATASET_ROOT=${DATASET_ROOT}" | tee -a "${summary_log}"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}" | tee -a "${summary_log}"
echo "[INFO] python_log=${run_log}" | tee -a "${summary_log}"
echo "[INFO] summary_log=${summary_log}" | tee -a "${summary_log}"
echo "===== start $(date) =====" | tee -a "${summary_log}"

episode_saved() {
  grep -qE 'Finished this episode|skip kit close' "$1" 2>/dev/null
}

# Run one Isaac process. If Kit hangs after the episode is already written,
# SIGKILL so the next relaunch can start without waiting for Ctrl+\.
run_one_python() {
  local one_log="$1"
  : >"${one_log}"
  setsid "${PYTHON}" "${PY_SCRIPT}" "${EXTRA[@]}" >"${one_log}" 2>&1 &
  local py_pid=$!
  local tail_pid=""
  if [[ "${TEE_LOG}" == "1" ]]; then
    tail -n +1 -f "${one_log}" --pid="${py_pid}" &
    tail_pid=$!
  fi
  (
    while kill -0 "${py_pid}" 2>/dev/null; do
      if episode_saved "${one_log}"; then
        sleep "${EPISODE_EXIT_GRACE}"
        if kill -0 "${py_pid}" 2>/dev/null; then
          echo "[WARN] Isaac still alive ${EPISODE_EXIT_GRACE}s after episode end; SIGKILL pid=${py_pid}" | tee -a "${summary_log}"
          kill -TERM -"${py_pid}" 2>/dev/null || true
          sleep 2
          kill -KILL -"${py_pid}" 2>/dev/null || true
        fi
        break
      fi
      sleep 1
    done
  ) &
  local wd_pid=$!
  wait "${py_pid}"
  RUN_RC=$?
  kill "${wd_pid}" 2>/dev/null || true
  wait "${wd_pid}" 2>/dev/null || true
  if [[ -n "${tail_pid}" ]]; then
    kill "${tail_pid}" 2>/dev/null || true
    wait "${tail_pid}" 2>/dev/null || true
  fi
  # Reap leftover Kit children that survive python os._exit.
  kill -KILL -"${py_pid}" 2>/dev/null || true
}

count_run_result() {
  local rc="$1"
  local one_log="$2"
  local label="$3"
  if [[ ${rc} -eq 0 ]]; then
    ok_count=$((ok_count + 1))
    echo "[OK] [${WORKER_ID}] ${label} exit=0" | tee -a "${summary_log}"
  elif episode_saved "${one_log}" && [[ ${rc} -eq 131 || ${rc} -eq 137 || ${rc} -eq 143 ]]; then
    ok_count=$((ok_count + 1))
    echo "[OK] [${WORKER_ID}] ${label} episode saved (killed hung kit, exit=${rc})" | tee -a "${summary_log}"
  else
    fail_count=$((fail_count + 1))
    echo "[FAIL] [${WORKER_ID}] ${label} exit=${rc} (continue)" | tee -a "${summary_log}"
  fi
}

ok_count=0
fail_count=0

if [[ "${RELAUNCH_EACH_EPISODE}" == "1" ]]; then
  echo "[INFO] mode=relaunch-each-episode (one Isaac boot per episode)" | tee -a "${summary_log}"
  : >"${run_log}"
  for ((i = 1; i <= NUM_RUNS; i++)); do
    one_log="${LOG_DIR}/collect_${stamp}_run$(printf '%03d' "${i}").log"
    echo
    echo "===== [${WORKER_ID}] run ${i}/${NUM_RUNS}  log=${one_log} =====" | tee -a "${summary_log}"
    export NUM_EPISODES=1
    run_one_python "${one_log}"
    cat "${one_log}" >>"${run_log}"
    count_run_result "${RUN_RC}" "${one_log}" "run ${i}/${NUM_RUNS}"
    if [[ ${i} -lt ${NUM_RUNS} ]]; then
      sleep "${RELAUNCH_SLEEP}"
    fi
  done
else
  echo "[INFO] mode=single-boot NUM_EPISODES=${NUM_RUNS}" | tee -a "${summary_log}"
  export NUM_EPISODES="${NUM_RUNS}"
  export RECORD_STRIDE="${RECORD_STRIDE:-2}"
  run_one_python "${run_log}"
  echo "[INFO] python process finished exit=${RUN_RC}; summarizing files..." | tee -a "${summary_log}"
  count_run_result "${RUN_RC}" "${run_log}" "process"
fi

n_hdf5=$(find "${DATASET_ROOT}/data" -maxdepth 1 -name 'episode*.hdf5' 2>/dev/null | wc -l | tr -d ' ')
n_pkl=$(find "${DATASET_ROOT}/_traj_data" -maxdepth 1 -name 'episode*.pkl' 2>/dev/null | wc -l | tr -d ' ')
n_mp4=$(find "${DATASET_ROOT}/video" -maxdepth 1 -name 'episode*.mp4' 2>/dev/null | wc -l | tr -d ' ')

task_ok=$(grep -h 'skip_record=False' "${LOG_DIR}"/collect_"${stamp}"*.log 2>/dev/null | grep -c '\[SUCCESS\] task=' || true)
task_fail=$(grep -hE 'skip_record=True' "${LOG_DIR}"/collect_"${stamp}"*.log 2>/dev/null | grep -cE '\[(SUCCESS|FAIL)\] task=' || true)
task_total=$((task_ok + task_fail))
if [[ ${task_total} -gt 0 ]]; then
  task_sr=$(awk -v o="${task_ok}" -v t="${task_total}" 'BEGIN { printf "%.3f", o / t }')
else
  task_sr="n/a"
fi

echo
echo "===== done $(date)  worker=${WORKER_ID} process_ok=${ok_count} process_fail=${fail_count} =====" | tee -a "${summary_log}"
echo "[INFO] task success: ok=${task_ok} fail=${task_fail} total=${task_total} sr=${task_sr}" | tee -a "${summary_log}"
stage_json="${DATASET_ROOT}/meta/stage_stats.json"
if [[ -f "${stage_json}" ]]; then
  python3 - "${stage_json}" <<'PY' | tee -a "${summary_log}"
import json, sys
p = sys.argv[1]
with open(p, encoding="utf-8") as f:
    d = json.load(f)

def rate(ok, n):
    ok, n = int(ok or 0), int(n or 0)
    return "n/a" if n <= 0 else f"{ok}/{n} ({ok / n:.3f})"

a = int(d.get("episode_attempts", 0))
print(
    "[STAGE] "
    f"attempts={a} "
    f"plan={rate(d.get('plan_ok', 0), d.get('plan_calls', 0))} "
    f"stall={int(d.get('stall_fail', 0))} "
    f"eval={int(d.get('reached_eval', 0))} "
    f"check={rate(d.get('check_ok', 0), d.get('reached_eval', 0))} "
    f"dump={rate(d.get('dump_ok', 0), a)} "
    f"crash={int(d.get('crash', 0))} "
    f"overall={rate(d.get('dump_ok', 0), a)}"
)
metrics = d.get("check_metrics") or {}
if metrics:
    parts = []
    for name in sorted(metrics):
        val = metrics[name]
        if isinstance(val, dict):
            parts.append(f"{name}={rate(val.get('ok', 0), val.get('n', 0))}")
    if parts:
        print("[STAGE][check] " + " ".join(parts))
print(f"[INFO] stage_stats={p}")
PY
fi
echo "[INFO] recorded files under ${DATASET_ROOT}: hdf5=${n_hdf5} pkl=${n_pkl} mp4=${n_mp4}" | tee -a "${summary_log}"
echo "[INFO] logs: python=${run_log}  summary=${summary_log}" | tee -a "${summary_log}"
echo "end=$(date -Iseconds) process_ok=${ok_count} process_fail=${fail_count} task_ok=${task_ok} task_fail=${task_fail} sr=${task_sr}" >>"${worker_meta}"

if [[ ${ok_count} -eq 0 ]]; then
  exit 1
fi
exit 0
