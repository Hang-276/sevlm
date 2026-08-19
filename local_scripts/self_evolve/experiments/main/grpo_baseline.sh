#!/usr/bin/env bash
# Main experiment: base + GRPO. Plain single-round GRPO post-training, no
# self-evolve loop. Trainer config matches the ours exp; only the reward and
# the presence of the loop differ.
set -euo pipefail
export MARKER=grpo_baseline
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib/run_grpo_baseline.sh"
