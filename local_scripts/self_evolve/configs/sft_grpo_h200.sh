#!/usr/bin/env bash
# SFT+GRPO: shared formal settings, with SFT enabled before every GRPO update.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/grpo_h200.sh"
EXP_NAME="sft_grpo_h200"
STAGES="sft,grpo"
