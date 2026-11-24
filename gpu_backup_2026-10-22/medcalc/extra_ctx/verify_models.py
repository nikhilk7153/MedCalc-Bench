import os, json, hashlib, fnmatch, struct, glob
from concurrent.futures import ProcessPoolExecutor
from huggingface_hub import HfApi
HF = "/workspace/.hf_home/hub"
MODELS = ["meta-llama/Meta-Llama-3-8B-Instruct", "mistralai/Mistral-7B-Instruct-v0.2", "axiong/PMC_LLaMA_13B",
          "mistralai/Mixtral-8x7B-Instruct-v0.1", "meta-llama/Meta-Llama-3-70B-Instruct", "epfl-llm/meditron-70b"]
def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 24), b""): h.update(c)
    return h.hexdigest()
def st_params(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]; hdr = json.loads(f.read(n))
    tot = 0
    for k, v in hdr.items():
        if k == "__metadata__": continue
        c = 1
        for d in v["shape"]: c *= d
        tot += c
    return tot, {v["dtype"] for k, v in hdr.items() if k != "__metadata__"}
jobs, report = [], {}
for m in MODELS:
    d = f"{HF}/models--{m.replace('/', '--')}"
    snaps = sorted(os.listdir(f"{d}/snapshots")); ref = open(f"{d}/refs/main").read().strip()
    info = HfApi().model_info(m, files_metadata=True)
    files = [s.rfilename for s in info.siblings]
    has_st = any(f.endswith(".safetensors") and "/" not in f for f in files)
    ignore = ["original/*", "consolidated*", "*.pth", "*.pt", "*.msgpack", "*.h5"] + (["*.bin"] if has_st else [])
    snap = f"{d}/snapshots/{ref}"
    exp = [s for s in info.siblings if not any(fnmatch.fnmatch(s.rfilename, p) for p in ignore)]
    missing = [s.rfilename for s in exp if not os.path.exists(f"{snap}/{s.rfilename}")]
    size_bad = [s.rfilename for s in exp if s.rfilename not in missing and os.path.getsize(f"{snap}/{s.rfilename}") != s.size]
    lfs = [(m, s.rfilename, f"{snap}/{s.rfilename}", s.lfs.sha256) for s in exp if s.lfs and s.rfilename not in missing]
    jobs += lfs
    cfg = json.load(open(f"{snap}/config.json"))
    st = sorted(glob.glob(f"{snap}/*.safetensors"))
    params, dtypes = (0, set())
    if st:
        for p in st:
            n, dt = st_params(p); params += n; dtypes |= dt
    bins = sorted(glob.glob(f"{snap}/*.bin"))
    report[m] = dict(local_revision=ref, hub_revision_now=info.sha, snapshots_on_disk=snaps, files_expected=len(exp), missing=missing, size_mismatch=size_bad,
                     weight_format="safetensors" if st else "pytorch .bin", weight_files=len(st) or len(bins),
                     params_from_safetensors_B=round(params / 1e9, 3) if st else None, tensor_dtypes=sorted(dtypes),
                     arch=cfg.get("architectures"), model_type=cfg.get("model_type"), hidden=cfg.get("hidden_size"), layers=cfg.get("num_hidden_layers"),
                     heads=cfg.get("num_attention_heads"), kv_heads=cfg.get("num_key_value_heads"), vocab=cfg.get("vocab_size"),
                     max_pos=cfg.get("max_position_embeddings"), rope_theta=cfg.get("rope_theta"), rope_scaling=cfg.get("rope_scaling"),
                     sliding_window=cfg.get("sliding_window"), experts=cfg.get("num_local_experts"), torch_dtype=cfg.get("torch_dtype"))
with ProcessPoolExecutor(32) as ex: hashes = list(ex.map(sha256, [j[2] for j in jobs]))
bad = [(j[0], j[1]) for j, h in zip(jobs, hashes) if h != j[3]]
for m, r in report.items(): print(json.dumps({m: r}))
print(json.dumps({"sha256_checked_files": len(jobs), "GB": round(sum(os.path.getsize(j[2]) for j in jobs) / 1e9), "mismatches": bad}))
