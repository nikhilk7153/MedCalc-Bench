import os, hashlib, fnmatch
from concurrent.futures import ProcessPoolExecutor
from huggingface_hub import HfApi, snapshot_download

models = ["meta-llama/Meta-Llama-3-8B-Instruct", "mistralai/Mistral-7B-Instruct-v0.2", "axiong/PMC_LLaMA_13B",
          "mistralai/Mixtral-8x7B-Instruct-v0.1", "meta-llama/Meta-Llama-3-70B-Instruct", "epfl-llm/meditron-70b"]

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()

jobs = []
for m in models:
    info = HfApi().model_info(m, files_metadata=True)
    files = [s.rfilename for s in info.siblings]
    has_st = any(f.endswith(".safetensors") and "/" not in f for f in files)
    ignore = ["original/*", "consolidated*", "*.pth", "*.pt", "*.msgpack", "*.h5"] + (["*.bin"] if has_st else [])
    snap = os.path.join(os.environ["HF_HOME"], "hub", "models--" + m.replace("/", "--"), "snapshots", info.sha)
    expected = [s for s in info.siblings if not any(fnmatch.fnmatch(s.rfilename, p) for p in ignore)]
    missing = [s.rfilename for s in expected if not os.path.exists(os.path.join(snap, s.rfilename))]
    size_bad = [s.rfilename for s in expected if s.rfilename not in missing and os.path.getsize(os.path.join(snap, s.rfilename)) != s.size]
    lfs = [(m, s.rfilename, os.path.join(snap, s.rfilename), s.lfs.sha256) for s in expected if s.lfs and s.rfilename not in missing]
    jobs += lfs
    print(f"{m}: {len(expected)} files expected, missing={missing}, size_mismatch={size_bad}, {len(lfs)} weight files to hash", flush=True)

with ProcessPoolExecutor(32) as ex:
    results = list(ex.map(sha256, [j[2] for j in jobs]))
bad = [(j[0], j[1]) for j, h in zip(jobs, results) if h != j[3]]
print(f"SHA-256 checked {len(jobs)} files ({sum(os.path.getsize(j[2]) for j in jobs)/1e9:.0f} GB): {len(bad)} mismatches {bad}")
