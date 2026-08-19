#!/usr/bin/env bash
# Sourced by sensitivity/*.sh. Runs a list of values as independent runs,
# one after another. DRY=1 prints the plan without running.

run_sweep() {
  local axis="$1"; shift
  local values=("$@")
  echo "======================================================"
  echo "  sensitivity sweep: $axis"
  echo "  values: ${values[*]}"
  echo "  one independent training + evaluation per value; DRY=1 shows the plan"
  echo "======================================================"
  for v in "${values[@]}"; do
    local tag="${axis}_${v//[^A-Za-z0-9._-]/_}"
    echo "--- $axis=$v  (run_tag=$tag)"
    if [ "${DRY:-0}" = "1" ]; then
      # Apply the axis in a subshell so the values it exports can be shown
      # without leaking into the next iteration. Nothing here may fail the
      # sweep — a plan that stops at the first value looks like a finished plan.
      ( set_axis "$v"
        env | grep -E "^(GRPO_|SOLVER_|EDIT_|LABEL_|NUM_|PROPOSER_|SELF_PLAY|REWARD_JSON)=" \
          | sort | sed 's/^/      /' ) || true
      continue
    fi
    ( set_axis "$v"
      export MARKER="$tag" RUN_TAG="$tag"
      bash "$RUNNER" ) || { echo "[WARN] $axis=$v failed, moving on to the next value" >&2; continue; }
    MARKER="$tag" bash "$SWEEP_EVAL" || true
  done
}
