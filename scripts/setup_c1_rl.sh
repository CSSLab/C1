#!/bin/bash
# c1-rl: verl v0.8.0 + vllm 0.19.0
# vllm 0.20.x pins torch 2.11 (cu130-only on PyPI -> needs driver >=580; ours is 570).
# vllm 0.19.0 pins torch 2.10.0 whose default PyPI build is cu128 - matches nvcc 12.8 exactly.
set -euxo pipefail

JT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # parent holding C1 / LLaMA-Factory / verl
PIP=pip                                                        # run with `conda activate c1-rl` first
PY=python

# 1) inference engine FIRST (official verl install order - it dictates torch)
$PIP install vllm==0.19.0

# 2) verl v0.8.0 editable without deps (its [vllm] extra caps at <=0.12.0 - stale metadata)
$PIP install --no-deps -e "$JT_ROOT/verl"

# 3) verl base deps, minus the stale numpy<2 pin (vllm's resolver decides numpy)
$PIP install accelerate codetiming datasets dill hydra-core pandas peft "pyarrow>=19.0.0" \
    pybind11 pylatexenc "ray[default]>=2.41.0" torchdata "tensordict>=0.8.0,<=0.10.0,!=0.9.0" \
    wandb "packaging>=20.0" tensorboard liger-kernel chess

# 4) flash-attn: no prebuilt wheel for torch 2.10 -> source build with nvcc 12.8
$PIP install ninja psutil
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export MAX_JOBS=32
$PIP install --no-build-isolation --no-cache-dir flash-attn==2.8.3

echo "=== VERIFY ==="
$PY -c "import torch; print('torch', torch.__version__, '| cuda', torch.version.cuda, '| available', torch.cuda.is_available(), '| ngpu', torch.cuda.device_count())"
$PY -c "import vllm; print('vllm', vllm.__version__)"
$PY -c "import verl; print('verl', verl.__version__)"
$PY -c "import flash_attn; print('flash_attn', flash_attn.__version__)"
$PY -c "import ray, tensordict, chess, numpy; print('ray', ray.__version__, '| tensordict', tensordict.__version__, '| numpy', numpy.__version__)"
echo "C1_RL_INSTALL_OK"
