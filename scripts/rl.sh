#!/bin/bash
# DAPO-C1 launcher (verl 0.8.0, recipe.dapo.main_dapo).
#
# Usage:
#   bash scripts/rl.sh <sft_model_dir> [hydra overrides...]
#
# Defaults: wandb project c1, run rl; 200 steps (best ckpt ~step 80); checkpoint
# every 20 steps to <output_dir>/global_step_N/actor/huggingface/ — an eval-ready
# HF model, no FSDP resume shards (if a run dies, delete that dir and
# relaunch; a full run is ~4h). Validation (900 puzzles, greedy) every 10 steps.
#
# Append hydra overrides to change anything, e.g. a full 1-epoch run:
#   bash scripts/rl.sh <sft_model_dir> trainer.total_training_steps=null
# Env knobs: GPUS (CUDA_VISIBLE_DEVICES, default all 8) + N_GPUS (must match; mind the
# verl 0.8.0 batch divisibility asserts), RL_OUTPUT_DIR, WANDB_PROJECT, WANDB_MODE.

set -e
MODEL_PATH=$1
[ -d "$MODEL_PATH" ] || { echo "usage: $0 <sft_model_dir> [hydra overrides...] (model dir must exist)"; exit 1; }
shift 1 || true

C1_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERL_ROOT=${VERL_ROOT:-$(cd "$C1_ROOT/.." && pwd)/verl}

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate c1-rl

API_KEYS_FILE="$C1_ROOT/api_keys.json"
if [ -f "$API_KEYS_FILE" ]; then
    export WANDB_API_KEY=$(jq -r '.wandb.api_key' "$API_KEYS_FILE")
    export WANDB_ENTITY=$(jq -r '.wandb.entity' "$API_KEYS_FILE")
fi

export HF_HUB_CACHE=${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}
export WANDB_PROJECT=${WANDB_PROJECT:-c1}
export WANDB_MODE=${WANDB_MODE:-online}
# keep this SHORT: ray socket paths under it must stay below the AF_UNIX 107-byte limit
export RAY_TMPDIR=${RAY_TMPDIR:-/tmp/ray_tmp}
export CUDA_VISIBLE_DEVICES=${GPUS:-0,1,2,3,4,5,6,7}

# set -x only after the secrets are exported, so the key never hits the log
set -x

# main_dapo is REQUIRED for dynamic sampling: verl.trainer.main_ppo silently
# ignores algorithm.filter_groups.*
python3 -m recipe.dapo.main_dapo \
    "hydra.searchpath=[file://$VERL_ROOT/verl/trainer/config]" \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.filter_groups.enable=True \
    algorithm.filter_groups.metric=score \
    algorithm.filter_groups.max_num_gen_batches=10 \
    data.train_files=$C1_ROOT/data/train_rl.parquet \
    data.val_files=$C1_ROOT/data/test.parquet \
    data.train_batch_size=32 \
    data.gen_batch_size=96 \
    data.max_prompt_length=1024 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_full_prompt=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.checkpoint.save_contents='["hf_model"]' \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=32 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=128 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=128 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward.reward_kwargs.max_resp_len=512 \
    custom_reward_function.path=$C1_ROOT/code/chess_reward_function.py \
    custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.log_val_generations=10 \
    trainer.project_name=$WANDB_PROJECT \
    trainer.experiment_name=rl \
    trainer.default_local_dir=${RL_OUTPUT_DIR:-$C1_ROOT/saves/rl} \
    trainer.n_gpus_per_node=${N_GPUS:-8} \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=10 \
    trainer.resume_mode=disable \
    trainer.total_training_steps=200 \
    trainer.total_epochs=1 \
    "$@"
