"""How many of v1.0's equation-based rows (dataset/test_data.csv: lab, physical, date, dosage) were wrong or replaced in
the corrected test set, and why.

Each v1.0 row is followed into the revised set (notes and questions of dataset/test_data_10_22.csv, inputs and labels of
dataset/test_data_10_22_corrected.csv) by its exact note text, else (patient notes only) by Calculator ID + Note ID:
- kept, label correct: the new label is within v1.0's own tolerance (exact for integers and dates);
- kept, inputs wrong / formula wrong: the label changed, with / without a change of inputs;
- replaced: the note is no longer in the set.
Reasons come from dataset/test_data_corrected_changes.csv and the admission-time audit
(equation_audit/row_results_admission.csv). Changes that were not v1.0 errors are listed separately. Writes
v10_equation_rows.csv."""
import ast, json, os, re
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))  # repository root
EQUATION = ["lab", "physical", "date", "dosage"]
norm = lambda s: re.sub(r"\s+", " ", str(s)).strip()


def num(s):
    try: return float(s)
    except ValueError: return None


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


v1 = pd.read_csv(f"{ROOT}/dataset/test_data.csv", dtype=str, keep_default_na=False)
new = pd.read_csv(f"{ROOT}/dataset/test_data_10_22.csv", dtype=str, keep_default_na=False).set_index("Row Number")
corrected = pd.read_csv(f"{ROOT}/dataset/test_data_10_22_corrected.csv", dtype=str, keep_default_na=False).set_index("Row Number")
for col in ["Relevant Entities", "Ground Truth Answer", "Lower Limit", "Upper Limit"]:
    new[col] = corrected[col]
new = new.reset_index()
by_text = {(r["Calculator ID"], norm(r["Patient Note"])): r for _, r in new.iterrows()}
by_id = {(r["Calculator ID"], r["Note ID"]): r for _, r in new.iterrows() if r["Note Type"] != "Template"}
reason = pd.read_csv(f"{ROOT}/dataset/test_data_corrected_changes.csv", dtype=str, keep_default_na=False).groupby("v1.0 Row Number")["Reason"].first()
audit = pd.read_csv(f"{ROOT}/equation_audit/row_results_admission.csv", dtype=str, keep_default_na=False).set_index("Row Number")["Label Error Type"]


def status(o):
    r = by_text.get((o["Calculator ID"], norm(o["Patient Note"])))
    if r is None and o["Note Type"] != "Template":
        r = by_id.get((o["Calculator ID"], o["Note ID"]))
    if r is None:
        return "replaced"
    a, b, lo, hi = num(o["Ground Truth Answer"]), num(r["Ground Truth Answer"]), num(o["Lower Limit"]), num(o["Upper Limit"])
    if a is not None and b is not None and lo is not None and lo != hi:
        changed = not lo <= b <= hi
    else:
        changed = norm(o["Ground Truth Answer"]) != norm(r["Ground Truth Answer"])
    if not changed:
        return "kept, label correct"
    return "kept, inputs wrong" if inputs(o["Relevant Entities"]) != inputs(r["Relevant Entities"]) else "kept, formula wrong"


def kind(o):
    why = o["Reason"].lower()
    if o["Status"] == "kept, label correct": return ""
    if o["Status"] == "kept, formula wrong": return "formula wrong"
    if "Conception" in o["Calculator Name"]: return "not an error: Verified template swap (Conception)"
    if o["Calculator Name"] == "Free Water Deficit" and o["Status"] == "kept, inputs wrong": return "not an error: note edited to high sodium"
    if "one-shot" in why: return "not an error: duplicate of one-shot example"
    if o["Audit"] == "input_extraction" or any(k in why for k in ["baseline", "admission", "pre-treatment", "after treatment"]):
        return "input extraction (admission / pre-treatment)"
    if "blood gas" in why or "user edit" in why: return "electrolytes from a blood gas / user sheet edit"
    if o["Status"] == "kept, inputs wrong": return "other input error (unit, misread value)"
    if "fentanyl" in why or "itself" in why: return "dosage template error (fentanyl mg / steroid to itself)"
    return "other note/label problem (flagged in audit)"


manual = pd.read_csv(f"{ROOT}/equation_audit/flagged_rows.csv", dtype=str, keep_default_na=False).set_index("Row Number")["Manual Status"]
NO_VALUE = ["no admission", "baseline", "no pre-treatment", "only after treatment", "not reported", "only a glucose-corrected",
            "no serum value", "dialysis"]
ELECTROLYTE_CALCULATORS = {"26", "38", "39", "63", "64", "65", "66", "67"}  # sodium correction, free water deficit, anion/delta gaps


def determinable(o):
    """Could the answer be computed from v1.0's note at admission, before treatment?"""
    why = o["Reason"].lower()
    if o["Kind"] == "" or o["Kind"].startswith("not an error"): return ""
    if o["Manual"] in ("insufficient information", "out of domain") or o["Audit"] == "note_missing_or_conflicting_input" \
            or any(k in why for k in NO_VALUE):
        return "not determinable: note lacks a usable value / out of population"
    # the review-sheet edits on the electrolyte calculators are the blood-gas rows (dataset/CHANGELOG.md, status 2026-10-20 (3))
    if "blood gas" in why or ("user edit" in why and o["Calculator ID"] in ELECTROLYTE_CALCULATORS):
        return "not determinable under the serum rule: electrolytes only from a blood gas"
    # several values for an input and v1.0's question named no time point (the admission instruction is new)
    if o["Manual"] == "ambiguous":
        return "not determinable: several values, v1.0 question gave no time point"
    return "determinable: v1.0 label or input wrong"


rows = v1[v1["Category"].isin(EQUATION)].copy()
rows["Status"] = rows.apply(status, axis=1)
rows["Reason"] = rows["Row Number"].map(reason).fillna("")
rows["Audit"] = rows["Row Number"].map(audit).fillna("")
rows["Manual"] = rows["Row Number"].map(manual).fillna("")
rows["Kind"] = rows.apply(kind, axis=1)
rows["Determinable"] = rows.apply(determinable, axis=1)
rows[["Row Number", "Calculator ID", "Calculator Name", "Category", "Note Type", "Status", "Kind", "Determinable", "Reason",
      "Audit", "Manual"]].to_csv(f"{HERE}/v10_equation_rows.csv", index=False)

changed = rows[rows["Kind"] != ""]
errors = changed[~changed["Kind"].str.startswith("not an error")]
print(f"v1.0 equation-based rows: {len(rows)} in {rows['Calculator ID'].nunique()} calculators")
print(f"changed: {len(changed)} rows in {changed['Calculator ID'].nunique()} calculators")
print(f"genuine v1.0 errors: {len(errors)} rows ({100 * len(errors) / len(rows):.1f}%) in {errors['Calculator ID'].nunique()} calculators\n")
print(changed.groupby("Kind").agg(rows=("Row Number", "size"), calculators=("Calculator ID", "nunique"))
      .sort_values("rows", ascending=False).to_string())
for label, subset in [("all genuine errors", errors), ("admission / pre-treatment input extraction",
                                                         errors[errors["Kind"].str.startswith("input extraction")])]:
    print(f"\n{label}: {len(subset)} rows")
    print(subset.groupby("Determinable").agg(rows=("Row Number", "size"), calculators=("Calculator ID", "nunique")).to_string())
