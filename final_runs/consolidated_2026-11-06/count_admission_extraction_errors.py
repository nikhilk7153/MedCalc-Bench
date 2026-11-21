"""How many equation-based calculators had input extractions that were wrong for the admission (pre-treatment) time
point in v1.0, and what happened to those rows in the corrected test set.

Source: equation_audit/row_results_admission.csv (model audit of v1.0's 567 lab/physical rows under the admission rule;
rows with Label Error Type = input_extraction), equation_audit/flagged_rows.csv (manual review of the same rows).
A row counts as fixed if its note was replaced or its inputs were corrected in the revised set (notes and questions of
dataset/test_data_10_22.csv, inputs of dataset/test_data_10_22_corrected.csv). Date and dosage calculators are templates
and have no time point. Writes admission_input_extraction_rows.csv."""
import ast, json, os, re
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))  # repository root
norm = lambda s: re.sub(r"\s+", " ", str(s)).strip()


def inputs(s):
    try: d = json.loads(s)
    except Exception: d = ast.literal_eval(s)
    out = {}
    for k, v in d.items():  # ignore renamed keys, mEq vs mmol and float formatting
        if isinstance(v, (list, tuple)) and len(v) == 2 and isinstance(v[0], (int, float)):
            v = (round(float(v[0]), 4), str(v[1]).lower().replace("meq", "mmol"))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            v = round(float(v), 4)
        out[k.lower().strip()] = v
    return out


audit = pd.read_csv(f"{ROOT}/equation_audit/row_results_admission.csv", dtype=str, keep_default_na=False)
manual = pd.read_csv(f"{ROOT}/equation_audit/flagged_rows.csv", dtype=str, keep_default_na=False).set_index("Row Number")
v1 = pd.read_csv(f"{ROOT}/dataset/test_data.csv", dtype=str, keep_default_na=False).set_index("Row Number")
notes = pd.read_csv(f"{ROOT}/dataset/test_data_10_22.csv", dtype=str, keep_default_na=False)
labels = pd.read_csv(f"{ROOT}/dataset/test_data_10_22_corrected.csv", dtype=str, keep_default_na=False).set_index("Row Number")
final = {(r["Calculator ID"], r["Note ID"]): r for _, r in notes.iterrows()}

rows = audit[audit["Label Error Type"] == "input_extraction"].copy()  # audit rows use v1.0 row numbers


def fate(r):
    old = v1.loc[r["Row Number"]]
    new = final.get((r["Calculator ID"], old["Note ID"]))
    if new is None:
        return "note replaced"
    if inputs(old["Relevant Entities"]) != inputs(labels.at[new["Row Number"], "Relevant Entities"]):
        return "inputs corrected"
    return "note edited" if norm(old["Patient Note"]) != norm(new["Patient Note"]) else "unchanged"


rows["Manual Review"] = rows["Row Number"].map(lambda r: manual.at[r, "Manual Status"] if r in manual.index else "not reviewed")
rows["In Revised Set"] = rows.apply(fate, axis=1)
rows.to_csv(f"{HERE}/admission_input_extraction_rows.csv", index=False)

print(f"equation-based calculators audited: {audit['Calculator ID'].nunique()} (lab + physical)")
for label, mask in [("model audit flags an input-extraction error", rows.index == rows.index),
                    ("... and the manual review found it wrong or the note had no admission value",
                     rows["Manual Review"].isin(["incorrect", "insufficient information"])),
                    ("... and the manual review found it wrong", rows["Manual Review"] == "incorrect")]:
    s = rows[mask]
    print(f"{label}: {len(s)} rows in {s['Calculator ID'].nunique()} calculators")
print(pd.crosstab(rows["Calculator Name"], rows["In Revised Set"], margins=True).to_string())
