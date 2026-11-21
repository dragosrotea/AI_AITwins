#!/usr/bin/env bash
# Start the broker and every DriveMate process: core, cv, voice, hr, maps (LEAD-T06). Ctrl+C stops them all.
#
#   scripts/run_all.sh                  # live mode where it exists
#   scripts/run_all.sh --fake           # every process on synthetic data
#   scripts/run_all.sh --fake=cv,hr     # only these on synthetic data
#
# A process whose module does not exist yet is skipped with a warning, so main always runs.
# Env: PYTHON (default .venv/bin/python, else python3), DRIVEMATE_PROFILE, DRIVEMATE_BUS_URL.
set -u
cd "$(dirname "$0")/.."

if [ -x .venv/bin/python ]; then DEFAULT_PYTHON=.venv/bin/python; else DEFAULT_PYTHON=python3; fi
PYTHON="${PYTHON:-$DEFAULT_PYTHON}"
FAKE_ALL=0
FAKE_LIST=","
for arg in "$@"; do
  case "$arg" in
    --fake) FAKE_ALL=1 ;;
    --fake=*) FAKE_LIST=",${arg#--fake=}," ;;
    -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

cleanup() {
  trap - INT TERM EXIT
  echo "[run_all] stopping"
  pkill -TERM -P $$ 2>/dev/null  # every pipeline member is a direct child of this shell
  wait 2>/dev/null
}
trap cleanup INT TERM EXIT

is_fake() { [ "$FAKE_ALL" = 1 ] || [[ "$FAKE_LIST" == *",$1,"* ]]; }

# start NAME MODULE FAKE_FLAG
start() {
  local name="$1" module="$2" flag="$3" path args=()
  path="${module//.//}"
  if [ ! -f "$path.py" ] && [ ! -f "$path/__main__.py" ]; then
    echo "[run_all] skip $name: $module not found"
    return
  fi
  if [ -n "$flag" ] && is_fake "$name"; then args=("$flag"); fi
  echo "[run_all] start $name: $PYTHON -m $module ${args[*]-}"
  "$PYTHON" -u -m "$module" ${args[@]+"${args[@]}"} 2>&1 | sed -u "s/^/[$name] /" &
}

start broker backend.bus.broker ""
for _ in $(seq 50); do
  curl -sf http://localhost:8765/stats >/dev/null 2>&1 && break
  sleep 0.1
done

start core  backend.core.run            --fake
start cv    backend.cv.run              --fake
start voice backend.voice.run           --fake
start hr    backend.integrations.hr.run   --sim
start maps  backend.integrations.maps.run --sim

echo "[run_all] running; dashboard: cd dashboard && npm run dev; check: python scripts/smoke_test.py"
wait
