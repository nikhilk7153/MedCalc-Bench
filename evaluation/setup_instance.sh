#!/bin/bash
# Runs on a fresh vast.ai GPU instance: installs vLLM into /workspace/venv and downloads the six open models
# (safetensors only where available) into /workspace/.hf_home. Expects the HF token at /workspace/.hf_home/token.
set -e
export HF_HOME=/workspace/.hf_home HF_HUB_ENABLE_HF_TRANSFER=1
cd /workspace
uv venv /workspace/venv --python 3.12
VIRTUAL_ENV=/workspace/venv uv pip install vllm pandas tqdm huggingface_hub hf_transfer
echo "$(date) install done"
/workspace/venv/bin/python - <<'EOF'
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import HfApi, snapshot_download
MODELS = ["meta-llama/Meta-Llama-3-8B-Instruct", "mistralai/Mistral-7B-Instruct-v0.2", "axiong/PMC_LLaMA_13B",
          "mistralai/Mixtral-8x7B-Instruct-v0.1", "meta-llama/Meta-Llama-3-70B-Instruct", "epfl-llm/meditron-70b"]
def download(model):
    files = [s.rfilename for s in HfApi().model_info(model).siblings]
    has_safetensors = any(f.endswith(".safetensors") and "/" not in f for f in files)
    ignore = ["original/*", "consolidated*", "*.pth", "*.pt", "*.msgpack", "*.h5"] + (["*.bin"] if has_safetensors else [])
    snapshot_download(model, ignore_patterns=ignore, max_workers=16)
    print("DOWNLOADED", model, flush=True)
with ThreadPoolExecutor(3) as ex:
    list(ex.map(download, MODELS))
EOF
echo "$(date) downloads done"
