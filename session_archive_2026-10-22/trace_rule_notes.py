import glob, re, collections, os
import pandas as pd

ROOT = "/Users/nikhilkhandekar/Downloads/MedCalc-Bench-version-1.0 2"
HF = os.path.expanduser("~/.cache/huggingface/hub")
norm = lambda s: re.sub(r"\s+", " ", str(s)).strip()

SOURCES = {  # label -> file; primary releases and working files only (nothing derived from the corrected set)
    "v1.0 (local)": f"{ROOT}/dataset/test_data.csv",
    "v1.0 (HF)": glob.glob(f"{HF}/datasets--ncbi--MedCalc-Bench-v1.0/snapshots/*/test_data.csv")[0],
    "10_15_2024": f"{ROOT}/final_files_2026-10-16/test_data_10_15_2024.csv",
    "11_14 (nsk7153 data)": glob.glob(f"{HF}/datasets--nsk7153--medcalc-bench-data/snapshots/*/test_data_11_14_final.csv")[0],
    "v1.2 11_18 (ncbi)": sorted(glob.glob(f"{HF}/datasets--ncbi--MedCalc-Bench-v1.2/snapshots/*/test_data_11_18_final.csv")),
    "1_17_2025": f"{ROOT}/dataset/test_data_1_17_2025.csv",
    "1_31": f"{ROOT}/dataset/test_set_1_31.csv",
    "2_6": f"{ROOT}/dataset/test_data_2_6.csv",
    "4_30": os.path.expanduser("~/Downloads/test_data_4_30.csv"),
    "Verified (local)": f"{ROOT}/dataset/medcalc_bench_verified/test_data.csv",
    "Verified v108": os.path.expanduser("~/Downloads/test_data_verified_v108.csv"),
    "Verified (HF snapshots)": sorted(glob.glob(f"{HF}/datasets--nsk7153--MedCalc-Bench-Verified/snapshots/*/data/test-*.parquet")),
    "Verified-v2 (HF)": sorted(glob.glob(f"{HF}/datasets--nsk7153--MedCalc-Bench-Verified-v2/snapshots/*/test_data.csv") +
                               glob.glob(f"{HF}/datasets--nsk7153--MedCalc-Bench-Verified-v2/snapshots/*/data/test-*.parquet")),
}

index = collections.defaultdict(set)  # (calc id, normalized note) -> source labels
for label, paths in SOURCES.items():
    for p in ([paths] if isinstance(paths, str) else paths):
        if not os.path.exists(p):
            continue
        d = pd.read_parquet(p) if p.endswith(".parquet") else pd.read_csv(p, dtype=str, keep_default_na=False)
        for _, r in d.iterrows():
            index[(str(r["Calculator ID"]).strip().split(".")[0], norm(r["Patient Note"]))].add(label)

t = pd.read_csv(f"{ROOT}/dataset/test_data_10_22.csv", dtype=str, keep_default_na=False)
v1 = {(r["Calculator ID"], r["Note ID"]): norm(r["Patient Note"]) for _, r in pd.read_csv(SOURCES["v1.0 (local)"], dtype=str, keep_default_na=False).iterrows()}
rule = t[t["Category"].isin(["risk", "severity", "diagnosis"])]
summary = collections.Counter(); rows = []
for _, r in rule.iterrows():
    n = norm(r["Patient Note"])
    if v1.get((r["Calculator ID"], r["Note ID"])) == n:
        continue
    found = sorted(index.get((r["Calculator ID"], n), set()))
    summary[(r["Calculator Name"][:30], r["Note Type"], ", ".join(found) or "NO SOURCE (edited here)")] += 1
    rows.append((r["Row Number"], r["Calculator Name"][:26], r["Note Type"], ", ".join(found) or "NO SOURCE"))
print("rule-based rows whose note is not v1.0's, grouped by where the exact note text appears:")
for (calc, nt, src), n in sorted(summary.items()):
    print(f"  {calc:30s} {nt:9s} {n:3d}  -> {src}")
print("\nrows with no source:", [x[0] for x in rows if x[3] == "NO SOURCE"])
