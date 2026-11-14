"""Collect the finalized dataset, model outputs and error analysis into one folder: <repo>/final_release_2026-11-08/.

Copies only (no recomputation, no model calls); source folders are not modified. Run after the scripts in README.md.
- dataset/test_data.csv: the revised test set every table was scored on (make_stats_table.load_release(): notes,
  questions, names and Time Point Instruction of dataset/test_data_10_22.csv; Relevant Entities, labels, limits and
  explanations of dataset/test_data_10_22_corrected.csv). Checked against the consolidated outputs' row numbers.
- dataset/one_shot_examples.json: the one-shot file of the MDRD/Sodium re-run. It differs from the main run's file only in
  the CKD-EPI example (ID 3, wording) and the Sodium Correction example (ID 26, "Hiller" -> "Hillier"), which is exactly
  what the spliced one-shot rows saw. Not final_files_2026-10-16/one_shot_10_17_2026_corrected.json: that file also has
  the corrected SOFA example (ID 43), which GPT-4 never ran and no consolidated output used.
- dataset/train_data.csv: dataset/train_data_10_22.csv (open items listed in the release README).
- outputs/, error_analysis/, tables/, figures/ (Appendix E.1): from this folder.
Writes MANIFEST.csv (rows and sha256 per file)."""
import hashlib, json, os, shutil, sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DEST = f"{ROOT}/final_release_2026-11-08"
sys.path.insert(0, HERE)
from make_stats_table import load_release

COPIES = {
    "dataset/one_shot_examples.json": "extra_runs/mdrd_sodium_wording_rerun_2026-10-22/input/one_shot_10_17_2026_corrected.json",
    "dataset/train_data.csv": "dataset/train_data_10_22.csv",
    "dataset/CHANGELOG.md": "dataset/CHANGELOG.md",
    "outputs/accuracy_by_category.csv": "final_runs/consolidated_2026-11-06/accuracy_by_category.csv",
    "error_analysis/error_classifications.csv": "final_runs/consolidated_2026-11-06/error_classifications.csv",
    "error_analysis/error_type_summary.csv": "final_runs/consolidated_2026-11-06/error_type_summary.csv",
    "error_analysis/error_type_table_values.csv": "final_runs/consolidated_2026-11-06/error_type_table_values.csv",
}
for f in sorted(os.listdir(f"{HERE}/outputs")):
    if f.endswith(".jsonl"):
        COPIES[f"outputs/{f}"] = f"final_runs/consolidated_2026-11-06/outputs/{f}"
for f in sorted(os.listdir(f"{HERE}/accuracy_vs_param_count")):
    if f.endswith((".png", ".csv")):
        COPIES[f"figures/{f}"] = f"final_runs/consolidated_2026-11-06/accuracy_vs_param_count/{f}"
for f in sorted(os.listdir(f"{HERE}/tables")):
    COPIES[f"tables/{f}"] = f"final_runs/consolidated_2026-11-06/tables/{f}"

test = load_release()
assert len(test) == 1048 and test["Row Number"].is_unique
# every consolidated output covers exactly these rows, with these notes
for f in [k for k in COPIES if k.startswith("outputs/") and k.endswith(".jsonl") and "code_exec" not in k]:
    out = [json.loads(l) for l in open(f"{ROOT}/{COPIES[f]}")]
    assert sorted(str(o["Row Number"]) for o in out) == sorted(test["Row Number"]), f

os.makedirs(f"{DEST}/dataset", exist_ok=True)
test.to_csv(f"{DEST}/dataset/test_data.csv", index=False)
for dst, src in COPIES.items():
    os.makedirs(os.path.dirname(f"{DEST}/{dst}"), exist_ok=True)
    shutil.copy2(f"{ROOT}/{src}", f"{DEST}/{dst}")


def rows(path):
    if path.endswith(".csv"): return len(pd.read_csv(path, dtype=str, keep_default_na=False))
    if path.endswith(".jsonl"): return sum(1 for _ in open(path))
    if path.endswith(".json"): return len(json.load(open(path)))
    return ""


manifest = []
for dirpath, _, files in sorted(os.walk(DEST)):
    for f in sorted(files):
        p = os.path.join(dirpath, f)
        rel = os.path.relpath(p, DEST)
        if rel in ("MANIFEST.csv", "README.md"): continue
        manifest.append({"File": rel, "Source": COPIES.get(rel, "built from dataset/test_data_10_22.csv + test_data_10_22_corrected.csv"),
                         "Rows": rows(p), "SHA256": hashlib.sha256(open(p, "rb").read()).hexdigest()})
pd.DataFrame(manifest).to_csv(f"{DEST}/MANIFEST.csv", index=False)
print(pd.DataFrame(manifest)[["File", "Rows"]].to_string(index=False))
