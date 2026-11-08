"""
Regenerates the labels, explanations and questions of the training set with calculator_implementations_verified/
generate_csv.py, run unchanged on a copy of that folder (the script reads and overwrites datasets/one_shot_data.csv).

Input: train_data_10_22_v1.0_labels.csv (from build_train_data_10_22.py). Before the run, v1.0's input names are mapped
to the names the verified code uses, and inputs that belong to no input of the calculator are dropped
(train_data_10_22_input_fixes.csv lists each change). "dosage conversion" becomes "dosage", so that generate_csv.py gives
Steroid Conversion its 5% limits. Two v1.0 errors that the verified code's literal unit conversion would turn into wrong
labels are corrected:
- SIRS white blood cell counts with a garbled unit or a count in thousands, from train_data_10_22_sirs_wbc_fixes.csv
  (sirs_wbc_fixes_10_22.py). The 9 SIRS rows whose note gives no blood count are left out, since the SIRS code needs one
  (train_data_10_22_excluded.csv);
- the 25 MME templates that dose the fentanyl patch in mg ("70 mg of FentANYL patch"): patches are dosed in µg, which is
  what the verified code expects, so the dose's unit becomes µg in the note and the input (v1.0's answer, which took the
  dose as µg, is unchanged);
- the 17 MME templates that write the buccal fentanyl dose in mg ("20 mg of FentaNYL buccal") while the input, and so
  the answer, has it in µg: the note's unit becomes µg.

After the run, to match test_data_10_22_corrected.csv: the time-point sentence generate_csv.py appends to the question
moves to a Time Point Instruction column, and "lab test" becomes "lab". generate_csv.py renumbers rows from 1;
train_data_10_22_row_map.csv maps each new Row Number to the v1.0 one.

Writes train_data_10_22.csv (and .zip), train_data_10_22_label_changes.csv (rows whose answer differs from v1.0's) and
logs/regenerate_train_data_10_22.log (generate_csv.py's output).
"""
import ast
import re
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(ROOT, "..", "calculator_implementations_verified")
TIME_POINT = "You should use the patient's medical values and health status when they were first admitted to the hospital prior to any treatment."

RENAME = {
    "QT interval": "QT Interval",
    "Congestive Heart Faliure": "Congestive Heart Failure",
    "Symptom onset ≤3 days": "Symptom onset <=3 days",
    "PaCO₂": "PaCO2",
    "Diabetes mellitus criteria for CCI rule": "Diabetes mellitus",
    "FentANYL patch Dose": "FentaNYL patch Dose",
    "FentANYL patch Dose Per Day": "FentaNYL patch Dose Per Day",
}
DROP = {
    33: {"Altered mental status for PECARN head injury criteria",
         "Occipital, parietal or temporal scalp hematoma; history of level of conciousness (LOC) ≥5 sec; not acting normally per parent or severe mechanism of injury?"},
    4: {"CHA2DS2-VASc"},
    68: {"cycle length"},
}

PATCH_MG = re.compile(r"(\d+(?:\.\d+)?) mg of FentANYL patch")
BUCCAL_MG = re.compile(r"(\d+(?:\.\d+)?) mg of FentaNYL buccal")

train = pd.read_csv(f"{ROOT}/train_data_10_22_v1.0_labels.csv")
wbc_fixes = pd.read_csv(f"{ROOT}/train_data_10_22_sirs_wbc_fixes.csv").set_index("v1.0 Row Number")

fixes, excluded = [], []
entities, notes = [], []
for _, r in train.iterrows():
    rn, cid, ents, note = r["Row Number"], int(r["Calculator ID"]), ast.literal_eval(r["Relevant Entities"]), r["Patient Note"]
    fixed = {}
    for key, value in ents.items():
        if key in DROP.get(cid, ()):
            fixes.append({"v1.0 Row Number": rn, "Calculator ID": cid, "Change": "dropped", "Input": key, "New Name": ""})
            continue
        if key in RENAME:
            fixes.append({"v1.0 Row Number": rn, "Calculator ID": cid, "Change": "renamed", "Input": key, "New Name": RENAME[key]})
        fixed[RENAME.get(key, key)] = value
    if cid == 51 and rn in wbc_fixes.index:
        new = wbc_fixes.loc[rn, "New Input"]
        fixes.append({"v1.0 Row Number": rn, "Calculator ID": cid, "Change": "white blood cell count: " + wbc_fixes.loc[rn, "Rule"],
                      "Input": str(fixed["White blood cell count"]), "New Name": new})
        if new == "(removed)":
            excluded.append({"v1.0 Row Number": rn, "Calculator ID": cid, "Calculator Name": r["Calculator Name"],
                             "Reason": "SIRS: " + wbc_fixes.loc[rn, "Rule"].split(": ", 1)[1] + "; the SIRS code needs a white blood cell count"})
        else:
            fixed["White blood cell count"] = ast.literal_eval(new)
    if cid == 49 and "FentaNYL patch Dose" in fixed and fixed["FentaNYL patch Dose"][1] == "mg":
        assert len(PATCH_MG.findall(note)) == 1 and float(PATCH_MG.search(note).group(1)) == fixed["FentaNYL patch Dose"][0]
        fixes.append({"v1.0 Row Number": rn, "Calculator ID": cid, "Change": "fentanyl patch dose mg -> µg (note and input)",
                      "Input": str(fixed["FentaNYL patch Dose"]), "New Name": str([fixed["FentaNYL patch Dose"][0], "µg"])})
        fixed["FentaNYL patch Dose"] = [fixed["FentaNYL patch Dose"][0], "µg"]
        note = PATCH_MG.sub(r"\1 µg of FentANYL patch", note)
    if cid == 49 and BUCCAL_MG.search(note):
        assert fixed["FentaNYL buccal Dose"][1] == "µg" and len(BUCCAL_MG.findall(note)) == 1
        assert float(BUCCAL_MG.search(note).group(1)) == fixed["FentaNYL buccal Dose"][0]
        fixes.append({"v1.0 Row Number": rn, "Calculator ID": cid, "Change": "fentanyl buccal dose mg -> µg (note; input already µg)",
                      "Input": str(fixed["FentaNYL buccal Dose"]), "New Name": ""})
        note = BUCCAL_MG.sub(r"\1 µg of FentaNYL buccal", note)
    entities.append(str(fixed))
    notes.append(note)
prepared = train.assign(**{"Relevant Entities": entities, "Patient Note": notes,
                           "Category": train["Category"].replace({"dosage conversion": "dosage"})})
pd.DataFrame(fixes).to_csv(f"{ROOT}/train_data_10_22_input_fixes.csv", index=False)
pd.DataFrame(excluded).to_csv(f"{ROOT}/train_data_10_22_excluded.csv", index=False)
keep = ~train["Row Number"].isin([x["v1.0 Row Number"] for x in excluded])
train, prepared = train[keep].reset_index(drop=True), prepared[keep].reset_index(drop=True)

os.makedirs(f"{ROOT}/logs", exist_ok=True)
with tempfile.TemporaryDirectory() as work:
    code = os.path.join(work, "calculator_implementations_verified")
    shutil.copytree(CODE, code, ignore=shutil.ignore_patterns("__pycache__", "datasets"))
    os.makedirs(os.path.join(work, "datasets"))
    prepared.to_csv(os.path.join(work, "datasets", "one_shot_data.csv"), index=False)
    with open(f"{ROOT}/logs/regenerate_train_data_10_22.log", "w") as log:
        done = subprocess.run([sys.executable, os.path.join(code, "generate_csv.py")], cwd=work, stdout=log, stderr=subprocess.STDOUT,
                              env={**os.environ, "PYTHONPATH": code})
    if done.returncode:
        sys.exit(f"generate_csv.py failed (exit {done.returncode}); see logs/regenerate_train_data_10_22.log")
    out = pd.read_csv(os.path.join(work, "datasets", "one_shot_data.csv"))

assert len(out) == len(train) and (out["Patient Note"].values == prepared["Patient Note"].values).all()
with_time_point = out["Question"].str.endswith(" " + TIME_POINT)
out["Question"] = out["Question"].where(~with_time_point, out["Question"].str[:-len(" " + TIME_POINT)])
out["Time Point Instruction"] = with_time_point.map({True: TIME_POINT, False: None})
out["Category"] = out["Category"].replace({"lab test": "lab"})

out.to_csv(f"{ROOT}/train_data_10_22.csv", index=False)
with zipfile.ZipFile(f"{ROOT}/train_data_10_22.csv.zip", "w", zipfile.ZIP_DEFLATED) as z:
    z.write(f"{ROOT}/train_data_10_22.csv", "train_data_10_22.csv")
pd.DataFrame({"Row Number": out["Row Number"], "v1.0 Row Number": train["Row Number"]}).to_csv(
    f"{ROOT}/train_data_10_22_row_map.csv", index=False)


def same(a, b):
    try:
        return abs(float(a) - float(b)) <= 1e-9
    except ValueError:
        return str(a).strip() == str(b).strip()


changed = [{"Row Number": n, "v1.0 Row Number": v, "Calculator ID": c, "Calculator Name": name,
            "v1.0 Answer": old, "New Answer": new}
           for n, v, c, name, old, new in zip(out["Row Number"], train["Row Number"], out["Calculator ID"], out["Calculator Name"],
                                              train["Ground Truth Answer"], out["Ground Truth Answer"]) if not same(old, new)]
pd.DataFrame(changed).to_csv(f"{ROOT}/train_data_10_22_label_changes.csv", index=False)
print(f"{len(out)} rows regenerated; {len(fixes)} input fixes; {int(with_time_point.sum())} rows with a time-point instruction; "
      f"{len(changed)} answers differ from v1.0")
