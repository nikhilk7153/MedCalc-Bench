import sys
from huggingface_hub import snapshot_download, HfApi
models = ["meta-llama/Meta-Llama-3-8B-Instruct", "mistralai/Mistral-7B-Instruct-v0.2", "axiong/PMC_LLaMA_13B",
          "mistralai/Mixtral-8x7B-Instruct-v0.1", "meta-llama/Meta-Llama-3-70B-Instruct", "epfl-llm/meditron-70b"]
for m in models:
    files = [s.rfilename for s in HfApi().model_info(m).siblings]
    has_st = any(f.endswith(".safetensors") and "/" not in f for f in files)
    ignore = ["original/*", "consolidated*", "*.pth", "*.pt", "*.msgpack", "*.h5"] + (["*.bin"] if has_st else [])
    snapshot_download(m, ignore_patterns=ignore, max_workers=16)
    print("DOWNLOADED", m, flush=True)
