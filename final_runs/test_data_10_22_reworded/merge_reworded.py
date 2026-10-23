"""Build final_runs/test_data_10_22_reworded/: the main runs on test_data_10_22.csv with the 40 MDRD (calculator 9) and
Sodium Correction (calculator 26) rows replaced by the re-run on the reworded questions
(extra_runs/mdrd_sodium_wording_rerun_2026-10-22/). Zero-shot, one-shot and code-augmented only; the inputs are not
changed. Run from the project root: python3 final_runs/merge_reworded.py"""

import glob
import json
import os
import shutil
import sys

import pandas as pd

sys.path.insert(0, "evaluation")
from table_stats import compute_overall_accuracy

MAIN = "final_runs/test_data_10_22/outputs"
MAIN_CODE = "evaluation/outputs/test_data_10_22"
MAIN_EA = "evaluation/error_analysis_10_22"
RERUN = "extra_runs/mdrd_sodium_wording_rerun_2026-10-22"
OUT = "final_runs/test_data_10_22_reworded"
TAG = "test_data_10_22_reworded"
CALCS = {"9", "26"}

# main output file -> re-run output file
runs = {}
for style in ["zero_shot", "one_shot"]:
    for path in sorted(glob.glob(f"{MAIN}/*_{style}.jsonl")):
        name = os.path.basename(path)
        rerun_name = name.replace("OpenAI_gpt-4_", "OpenAI_gpt-4-0613_")
        matches = glob.glob(f"{RERUN}/*/outputs/{rerun_name}")
        assert len(matches) == 1, name
        runs[name] = (path, matches[0])
for model in ["gpt-3.5-turbo-0125", "gpt-4-0613"]:
    runs[f"code_exec_{model}.jsonl"] = (f"{MAIN_CODE}/code_exec_{model}.jsonl", f"{RERUN}/{model}/outputs/code_exec_{model}.jsonl")

if os.path.exists(OUT):
    raise SystemExit(f"{OUT} exists; remove it first")
os.makedirs(f"{OUT}/outputs")
os.makedirs(f"{OUT}/error_analysis")

summary = []
for name, (main_path, rerun_path) in runs.items():
    main_rows = [json.loads(line) for line in open(main_path)]
    rerun_rows = {int(r["Row Number"]): r for r in map(json.loads, open(rerun_path))}
    targets = {int(r["Row Number"]) for r in main_rows if str(r["Calculator ID"]) in CALCS}
    assert targets == set(rerun_rows), name
    merged = [rerun_rows.get(int(r["Row Number"]), r) for r in main_rows]
    with open(f"{OUT}/outputs/{name}", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in merged)

    if name.startswith("code_exec"):
        model, style = name[len("code_exec_"):-len(".jsonl")], "code_augmented"
    else:
        style = "zero_shot" if name.endswith("_zero_shot.jsonl") else "one_shot"
        model = name[:-len(f"_{style}.jsonl")].split("_", 1)[1]
    cwd = os.getcwd()
    os.chdir(OUT)
    stats = compute_overall_accuracy(name, model, f"{style}_{TAG}")
    os.chdir(cwd)
    acc = lambda rows: round(100 * sum(r["Result"] == "Correct" for r in rows) / len(rows), 2)
    summary.append({"Model": model, "Setting": style, "Rows": len(merged), "Rows replaced": len(targets),
                    "Main accuracy": acc(main_rows), "Merged accuracy": stats["overall"]["average"],
                    "MDRD main": acc([r for r in main_rows if str(r["Calculator ID"]) == "9"]),
                    "MDRD merged": acc([r for r in merged if str(r["Calculator ID"]) == "9"]),
                    "Sodium main": acc([r for r in main_rows if str(r["Calculator ID"]) == "26"]),
                    "Sodium merged": acc([r for r in merged if str(r["Calculator ID"]) == "26"])})
pd.DataFrame(summary).to_csv(f"{OUT}/accuracy_summary.csv", index=False)

# error analysis: main classifications with the calculator 9 and 26 rows replaced by the re-run's
for main_csv in sorted(glob.glob(f"{MAIN_EA}/*_shot_*.csv")):
    name = os.path.basename(main_csv)
    rerun_csv = f"{RERUN}/error_analysis/{name}"
    main_df, rerun_df = pd.read_csv(main_csv), pd.read_csv(rerun_csv)
    merged_df = pd.concat([main_df[~main_df["Calculator ID"].astype(str).isin(CALCS)], rerun_df]).sort_values("Row Number")
    merged_df.to_csv(f"{OUT}/error_analysis/{name}", index=False)
    main_json = json.load(open(main_csv + ".error_type.json"))
    rerun_json = json.load(open(rerun_csv + ".error_type.json"))
    merged_json = {k: v for k, v in main_json.items() if k.split("_", 1)[0] not in CALCS}
    merged_json.update(rerun_json)
    assert set(merged_json) == set(merged_df["Calculator ID"].astype(str) + "_" + merged_df["Note ID"]), name
    with open(f"{OUT}/error_analysis/{name}.error_type.json", "w") as f:
        json.dump(merged_json, f, indent=4)

shutil.copy2(__file__, f"{OUT}/merge_reworded.py")
print(pd.DataFrame(summary).to_string(index=False))
