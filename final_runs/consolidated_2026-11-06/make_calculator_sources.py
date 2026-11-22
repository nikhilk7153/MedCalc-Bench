"""One line per calculator ID: which changes made after the main run its rows use, which run each of its outputs comes
from, and any later change that is not used (with the reason). Writes calculator_sources.csv.

Reads outputs/ (run consolidate.py first) and ../../equation_audit/rerun_coverage_2026-11-01.csv (run_coverage.py).
MEDCALC_VARIANT=astra: see variant.py (the Astra corrections then count as used)."""
import glob
import json
import os
from collections import Counter

import pandas as pd
import variant

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

# when each run was made (output file times) and what it changed
RUNS = {
    "final_runs/test_data_10_22": "main run, 2026-10-22 04:51",
    "evaluation/outputs/test_data_10_22": "main code-prompt run, 2026-10-22 09:10",
    "corrected_rows_ckdepi_rerun_2026-10-22": "CKD-EPI / label-fix re-run, 2026-10-22 13:53",
    "mdrd_sodium_wording_rerun_2026-10-22": "MDRD / Sodium re-run, 2026-10-22 16:51",
    "astra_corrections_rerun_2026-10-22": "Astra re-run, 2026-10-22 14:24",
}
NOT_USED = {
    "Astra corrections (unreviewed)": "re-run 2026-10-22 14:24 on corrections not yet reviewed by a physician",
    "Corrected SOFA one-shot example": "re-run 2026-10-23/24; GPT-4 never ran it",
    "LDL Friedewald question wording": "re-run 2026-10-24; GPT-3.5 and GPT-4 never ran it",
}
if variant.ASTRA:
    del NOT_USED["Astra corrections (unreviewed)"]

test = pd.read_csv(f"{ROOT}/dataset/test_data_10_22_corrected.csv", dtype=str, keep_default_na=False)
calculators = test.groupby("Calculator ID", sort=False).agg(name=("Calculator Name", "first"), rows=("Row Number", "size"))

# which run each output (model, style, row) comes from, counted per calculator
sources = {}
for path in glob.glob(f"{variant.OUT}/outputs/*.jsonl"):
    for line in open(path):
        output = json.loads(line)
        sources.setdefault(output["Calculator ID"], Counter())[output["Source Run"]] += 1

coverage = pd.read_csv(f"{ROOT}/equation_audit/rerun_coverage_2026-11-01.csv", keep_default_na=False)
name_to_id = dict(zip(test["Calculator Name"], test["Calculator ID"]))
coverage["Calculator ID"] = coverage["Calculator"].map(name_to_id)

lines = []
for calc_id, calc in calculators.iterrows():
    changes = coverage[coverage["Calculator ID"] == calc_id].groupby("Change")["Row Number"].nunique()
    used = [f"{change} ({n} rows)" for change, n in changes.items() if change not in NOT_USED]
    not_used = [f"{change} ({n} rows): {NOT_USED[change]}" for change, n in changes.items() if change in NOT_USED]
    outputs = sources[calc_id]
    lines.append({
        "Calculator ID": int(calc_id),
        "Calculator Name": calc["name"],
        "Rows": calc["rows"],
        "Changes after the main run, used": "; ".join(used) or "none",
        "Outputs by run": "; ".join(f"{RUNS.get(run, run)}: {n}" for run, n in outputs.most_common()),
        "Later changes not used": "; ".join(not_used),
    })

table = pd.DataFrame(lines).sort_values("Calculator ID")
table.to_csv(f"{variant.OUT}/calculator_sources.csv", index=False)
print(table[["Calculator ID", "Rows", "Changes after the main run, used", "Later changes not used"]].to_string(index=False))
