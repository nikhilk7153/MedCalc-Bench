"""
Writes train_data_10_22_sirs_wbc_fixes.csv: the corrected white blood cell count for each SIRS training row whose v1.0
input has a unit that does not fit a blood count ("m^3", "L", "mL", "dL") or a value outside 500-200,000 per µL.
- unit only: the value is the per-mm³/µL count the note gives; the unit becomes mm^3 (v1.0 "m^3") or µL.
- thousands: the note gives the count in 10^9/L or K/µL (e.g. "9.8 G/L", "14.2 K/UL", "WBC 12.7"); value × 1,000, in µL.
- drop: the note gives no blood count (none at all, or only a CSF count), so the input is removed and the criterion
  is not met, as for any criterion the note does not mention.
Each value was checked against the note (Note Text column).
"""
import ast
import os
import re
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
NOT_BLOOD = {6880: "CSF count", 6929: "CSF count", 6957: "CSF count", 7031: "CSF count (blood count only 'slightly elevated')",
             6808: "no white cell count in note", 6903: "no white cell count in note", 6945: "no white cell count in note",
             7013: "no white cell count in note", 7015: "no white cell count in note"}
EXACT = {7019: 12440.0}  # "12.44×109/L" stored as 12,440,000
WORDS = re.compile(r"white blood cells?|white cells?|white blood count|WBC|leukocyt|leucocyt|leukopenia|TLC|blood cell count", re.I)

train = pd.read_csv(f"{ROOT}/train_data_10_22_v1.0_labels.csv")
fixes = []
for _, r in train[train["Calculator ID"] == 51].iterrows():
    rn, note = r["Row Number"], r["Patient Note"]
    wbc = ast.literal_eval(r["Relevant Entities"]).get("White blood cell count")
    if not wbc or (wbc[1] in ("µL", "mm^3") and 500 <= wbc[0] <= 200000):
        continue
    value, unit = wbc
    if rn in NOT_BLOOD:
        rule, new = "drop: " + NOT_BLOOD[rn], None
    elif rn in EXACT:
        rule, new = "thousands", [EXACT[rn], "µL"]
    elif value < 500:
        rule, new = "thousands", [round(value * 1000, 6), "µL"]
    else:
        rule, new = "unit only", [value, "mm^3" if unit == "m^3" else "µL"]
    shown = [m.start() for m in WORDS.finditer(note)]
    found = []
    if new:
        k = new[0] / 1000
        forms = {f"{new[0]:g}", f"{new[0]:,.0f}", f"{new[0]:,.0f}".replace(",", " "), f"{new[0]:,.0f}".replace(",", "."),
                 f"{k:g}", f"{k:.1f}", f"{k:.2f}"}
        found = [m.start() for f in forms for m in re.finditer(rf"(?<![\d.,]){re.escape(f)}(?![\d])", note)]
    at = found[0] if found else (shown[0] if shown else None)
    fixes.append({"v1.0 Row Number": rn, "v1.0 Input": str(wbc), "New Input": str(new) if new else "(removed)", "Rule": rule,
                  "Value In Note": bool(found) if new else "", "Note Text": note[max(0, at - 70):at + 60].replace("\n", " ") if at is not None else ""})

out = pd.DataFrame(fixes)
out.to_csv(f"{ROOT}/train_data_10_22_sirs_wbc_fixes.csv", index=False)
print(out["Rule"].str.split(":").str[0].value_counts().to_string())
print("values not found in the note:", out[out["Value In Note"] == False][["v1.0 Row Number", "v1.0 Input", "New Input", "Note Text"]].to_string())
