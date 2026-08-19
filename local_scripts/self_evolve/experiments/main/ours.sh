#!/usr/bin/env bash
# Main experiment: base + ours (full method).
# Full loop + default reward + SFT->GRPO. Evaluate with main/eval.sh
set -euo pipefail
export MARKER=ours
export RUN_TAG="${RUN_TAG:-ours}"
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib/run_self_evolve.sh"
