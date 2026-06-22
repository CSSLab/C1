# C1: Grounded Chess Reasoning in Language Models via Master Distillation

[![arXiv](https://img.shields.io/badge/arXiv-2603.20510-b31b1b?logo=arxiv)](https://arxiv.org/abs/2603.20510) [![Hugging Face](https://img.shields.io/badge/HuggingFace-Models-yellow?link=https://huggingface.co/UofTCSSLab)](https://huggingface.co/UofTCSSLab) [![Hugging Face](https://img.shields.io/badge/HuggingFace-Dataset-yellow?link=https://huggingface.co/datasets/UofTCSSLab/C1-data)](https://huggingface.co/datasets/UofTCSSLab/C1-data)

Official release and training/evaluation pipeline for **C1**: SFT on Stockfish-grounded
CoT distillation, then DAPO-C1 RL, on [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507).

## Results

900-puzzle test set, greedy pass@1, exact-match UCI.

| stage | accuracy | release artifact |
|---|---|---|
| SFT | **42.3%** | [`UofTCSSLab/C1-SFT-4B`](https://huggingface.co/UofTCSSLab/C1-SFT-4B) |
| RL  | **48.3%** | [`UofTCSSLab/C1-4B`](https://huggingface.co/UofTCSSLab/C1-4B) |

## Hardware / environments

Tested on 8x NVIDIA H100 80GB, AMD EPYC, driver 570 / CUDA 12.8.

Two conda envs (Python 3.12), built by the setup scripts:

```bash
conda create -n c1_sft python=3.12 -y && conda activate c1_sft && bash scripts/setup_c1_sft.sh   # LLaMA-Factory v0.9.5, torch 2.9.1+cu128 — SFT training
conda create -n c1-rl  python=3.12 -y && conda activate c1-rl  && bash scripts/setup_c1_rl.sh    # verl v0.8.0 + vllm 0.19.0, torch 2.10.0+cu128 — RL + all inference/eval
```

Both need [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) (v0.9.5) and
[verl](https://github.com/volcengine/verl) (v0.8.0) checked out as siblings
(`../LLaMA-Factory`, `../verl`).

`api_keys.json` in the repo root:

```json
{
  "openrouter": {"api_key": "..."},
  "wandb": {"api_key": "...", "entity": "..."}
}
```

## Data

Training and eval read four files from `data/` (repo root): `train_sft_cot.json` +
`dataset_info.json` (SFT, alpaca), `train_rl.parquet` (RL), and `test.parquet` (eval) —
39,601 SFT samples, 39,572 RL prompts, 900 test puzzles. Prepare them either way.

### Use the released data

Download [`UofTCSSLab/C1-data`](https://huggingface.co/datasets/UofTCSSLab/C1-data) into
the layout training expects. From the repo root, env `c1-rl`:

```python
import json
from datasets import load_dataset

REPO = "UofTCSSLab/C1-data"
# SFT: alpaca json + LLaMA-Factory dataset registration
json.dump(load_dataset(REPO, "sft", split="train").to_list(),
          open("data/train_sft_cot.json", "w"), ensure_ascii=False)
json.dump({"sft_data_cot": {"file_name": "train_sft_cot.json"}},
          open("data/dataset_info.json", "w"), indent=2)
# RL + test: verl parquet
load_dataset(REPO, "rl",   split="train").to_parquet("data/train_rl.parquet")
load_dataset(REPO, "test", split="test").to_parquet("data/test.parquet")
```

### Regenerate from scratch

Rebuilds the same files from the Lichess puzzle database (needs `api_keys.json` for the
teacher). Run from `code/`, env `c1-rl`:

```bash
python 0_data_selection.py      # Lichess puzzles → theme-balanced train_sft.csv / train_rl.csv / test.parquet
python 1_cot_generation.py      # distill CoT from google/gemini-3-flash-preview (OpenRouter) → train_sft_cot.csv
python 2_format_matching.py     # → train_sft_cot.json (alpaca) + train_rl.parquet (verl) + dataset_info.json
```

## SFT

### Train

```bash
bash scripts/sft.sh sft.yaml \
    > logs/sft_train.log 2>&1 &
```

Recipe:

- lr 1e-5, cosine schedule + 10% warmup, bf16, 10 epochs
- effective batch 256 (32/GPU x 8, accum 1 — accum > 1 quadruples DeepSpeed-z2 comm and ~2x step time)
- save per epoch (155 steps) to `output_dir` (`saves/sft`; point at a large disk — ~8GB/checkpoint)
- ~3h40m on 8x H100; wandb project `c1`, run `sft`

### Eval

Evaluate every epoch checkpoint and pick the best, env `c1-rl`:

```bash
for ck in saves/sft/checkpoint-*; do
  python -u code/full_eval.py \
      --model_path "$ck" --tokenizer_path Qwen/Qwen3-4B-Instruct-2507 \
      --test_data_path data/test.parquet --tensor_parallel_size 8 --max_model_len 4096
done
```

## RL

### Train

```bash
bash scripts/rl.sh <best_sft_checkpoint_dir> \
    > logs/rl_train.log 2>&1 &
```

Recipe:

- verl 0.8.0 `recipe.dapo.main_dapo` (dynamic sampling)
- train_batch 32 / gen_batch 96 / rollout n=32
- lr 1e-6, KL loss 0.001 (low_var_kl), clip 0.2/0.28, token-mean
- max_response 512, temp 1.0, binary reward (`code/chess_reward_function.py`)
- checkpoints every 20 steps to `saves/rl` (default) — eval-ready HF model at `global_step_N/actor/huggingface/` (hf_model only, no FSDP resume shards: ~16GB vs ~64GB; if a run dies, delete the dir and relaunch — full run ~4h)
- wandb project `c1`, run `rl`; any hydra key overridable by appending args

**200 steps is enough** (script default; append `trainer.total_training_steps=null` for a full epoch ≈ 412 steps): the curve plateaus at 45-48% around step 80.

### Eval

In-training validation runs every 10 steps (900 puzzles, greedy — same protocol as offline, reads ~0.5-1pp low). Eval saved checkpoints offline and pick the best:

```bash
python -u code/full_eval.py \
    --model_path saves/rl/global_step_80/actor/huggingface \
    --tokenizer_path Qwen/Qwen3-4B-Instruct-2507 \
    --test_data_path data/test.parquet --tensor_parallel_size 8 --max_model_len 4096
```

## Repo layout

```
code/
  0_data_selection.py     # puzzle selection (theme-balanced)
  1_cot_generation.py     # CoT distillation from google/gemini-3-flash-preview (OpenRouter)
  2_format_matching.py    # → alpaca json + verl parquet + dataset_info.json
  chess_reward_function.py# binary reward: FINAL_ANSWER exact-match UCI (RL + eval)
  full_eval.py            # offline eval of any full HF model dir (greedy pass@1)
  utils.py
configs/
  sft.yaml                # the SFT recipe
scripts/
  setup_c1_sft.sh / setup_c1_rl.sh   # environment builds
  sft.sh                  # SFT launcher (activates c1_sft, wandb env)
  rl.sh                   # DAPO-C1 launcher (recipe baked in)
data/                     # train/test artifacts
```

## Citation

```bibtex
@article{tang2026grounded,
  title={Grounded Chess Reasoning in Language Models via Master Distillation},
  author={Tang, Zhenwei and Wen, Qianfeng and Grief-Albert, Seth and Elgabra, Yahya and Yang, Blair and Dong, Honghua and Anderson, Ashton},
  journal={arXiv preprint arXiv:2603.20510},
  year={2026}
}
```
