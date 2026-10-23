import glob, os, re, collections
import pandas as pd

ROOT = "/Users/nikhilkhandekar/Downloads/MedCalc-Bench-version-1.0 2"
HF = os.path.expanduser("~/.cache/huggingface/hub")
norm = lambda s: re.sub(r"\s+", " ", str(s)).strip()

VERSIONS = {
    "v1.0": [f"{ROOT}/dataset/test_data.csv"],
    "v1.1": [os.path.expanduser("~/Documents/MedCalc-Bench/calculator_implementations/test_data_10_6.csv")],
    "v1.2": sorted(glob.glob(f"{HF}/datasets--ncbi--MedCalc-Bench-v1.2/snapshots/*/test_data_11_18_final.csv")),
    "Verified": [f"{ROOT}/dataset/medcalc_bench_verified/test_data.csv", os.path.expanduser("~/Downloads/test_data_verified_v108.csv")]
                + sorted(glob.glob(f"{HF}/datasets--nsk7153--MedCalc-Bench-Verified/snapshots/*/data/test-*.parquet"))
                + sorted(glob.glob(f"{HF}/datasets--nsk7153--MedCalc-Bench-Verified-v2/snapshots/*/test_data.csv"))
                + sorted(glob.glob(f"{HF}/datasets--nsk7153--MedCalc-Bench-Verified-v2/snapshots/*/data/test-*.parquet")),
}
notes = {}   # version -> {(calc id, note id): set(normalized notes)}
for v, paths in VERSIONS.items():
    notes[v] = collections.defaultdict(set)
    for p in paths:
        if not os.path.exists(p):
            continue
        d = pd.read_parquet(p) if p.endswith(".parquet") else pd.read_csv(p, dtype=str, keep_default_na=False)
        for _, r in d.iterrows():
            notes[v][(str(r["Calculator ID"]).split(".")[0], str(r["Note ID"]).strip())].add(norm(r["Patient Note"]))

t = pd.read_csv(f"{ROOT}/dataset/test_data_10_22.csv", dtype=str, keep_default_na=False)
ext = t[t["Category"].isin(["risk", "severity", "diagnosis"]) & (t["Note Type"] == "Extracted")]
summary = collections.Counter(); unmatched = []
for _, r in ext.iterrows():
    k = (r["Calculator ID"], r["Note ID"].strip()); n = norm(r["Patient Note"])
    found = [v for v in VERSIONS if n in notes[v].get(k, set())]
    summary[" + ".join(found) if found else "none"] += 1
    if not found:
        unmatched.append((r["Row Number"], r["Calculator Name"][:28], r["Note ID"], [v for v in VERSIONS if k in notes[v]]))
print(f"extracted rule-based rows: {len(ext)}")
for k, n in summary.most_common():
    print(f"  {n:3d}  note found in: {k}")
print(f"\nnot found verbatim in any version ({len(unmatched)}):")
for u in unmatched:
    print("  row %-4s %-28s note %-15s (this Note ID exists in: %s)" % (u[0], u[1], u[2], ", ".join(u[3]) or "none"))
