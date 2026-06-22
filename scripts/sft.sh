#!/bin/bash
# SFT launcher (LLaMA-Factory, env c1_sft — activated below).
# Usage: bash scripts/sft.sh [config.yaml]   # name relative to configs/
CONFIG=${1:-sft.yaml}
C1_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate c1_sft

API_KEYS_FILE="$C1_ROOT/api_keys.json"
if [ -f "$API_KEYS_FILE" ]; then
    export WANDB_API_KEY=$(python -c "import json; print(json.load(open('$API_KEYS_FILE')).get('wandb', {}).get('api_key', ''))")
    export WANDB_ENTITY=$(python -c "import json; print(json.load(open('$API_KEYS_FILE')).get('wandb', {}).get('entity', ''))")
fi

export HF_HUB_CACHE=${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}
export WANDB_PROJECT=${WANDB_PROJECT:-c1}
export WANDB_MODE=${WANDB_MODE:-online}
export DISABLE_VERSION_CHECK=1

# set -x only after the secrets are exported, so the key never hits the log
set -x
cd "$C1_ROOT"
llamafactory-cli train "configs/$CONFIG"
