#!/bin/bash
# c1_sft: LLaMA-Factory v0.9.5 training env (NO vllm)
# torch 2.9.1+cu128: newest torch with both cu128 wheels and a prebuilt flash-attn wheel
set -euxo pipefail

JT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # parent holding C1 / LLaMA-Factory / verl
PIP=pip                                                        # run with `conda activate c1_sft` first
PY=python

# torch first so LLaMA-Factory's resolver doesn't pick a default-index build
$PIP install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128

# LLaMA-Factory v0.9.5 editable (repo already checked out at tag)
$PIP install -e "$JT_ROOT/LLaMA-Factory"

# extras: metrics + deepspeed + liger-kernel (>=0.6.4 to skip the 0.6.3 flce bug)
$PIP install nltk jieba rouge-chinese "deepspeed==0.18.4" "liger-kernel>=0.6.4"

# flash-attn prebuilt wheel: cu12 / torch2.9 / cxx11abiTRUE / cp312 - no compile needed
$PIP install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

# project deps (C1 code + wandb logging)
$PIP install chess wandb

echo "=== VERIFY ==="
$PY -c "import torch; print('torch', torch.__version__, '| cuda', torch.version.cuda, '| available', torch.cuda.is_available(), '| ngpu', torch.cuda.device_count())"
$PY -c "import flash_attn; print('flash_attn', flash_attn.__version__)"
$PY -c "import deepspeed; print('deepspeed', deepspeed.__version__)"
$PY -c "import liger_kernel.transformers; print('liger ok')"
$PY -c "import transformers, peft, trl, datasets, accelerate; print('transformers', transformers.__version__, '| peft', peft.__version__, '| trl', trl.__version__, '| datasets', datasets.__version__, '| accelerate', accelerate.__version__)"
DISABLE_VERSION_CHECK=1 llamafactory-cli version
echo "C1_SFT_INSTALL_OK"
