"""
Flags training rows whose inputs (v1.0's model extractions) have a unit that does not fit the analyte or a value outside
a physiological range, since the verified code converts units literally. Writes train_data_10_22_implausible_inputs.csv.
"""
import ast
import os
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
CONC = {"mEq/L", "mmol/L"}
UNITS = {  # input -> units that fit it
    "sodium": CONC, "chloride": CONC, "bicarbonate": CONC, "potassium": CONC,
    "white blood cell count": {"µL", "mm^3", "L"},
    "creatinine": {"mg/dL", "µmol/L", "mg/L", "mmol/L"},
    "albumin": {"g/dL", "g/L"},
    "fentanyl patch dose": {"µg"},
}
RANGE = {  # input -> (unit, low, high)
    "sodium": (None, 100, 200), "chloride": (None, 60, 150), "bicarbonate": (None, 1, 60),
    "weight": ("kg", 0.3, 400), "height": ("cm", 30, 250), "heart rate or pulse": (None, 20, 300),
    "systolic blood pressure": (None, 40, 300), "diastolic blood pressure": (None, 10, 200),
    "temperature": ("degrees celsius", 25, 45), "respiratory rate": (None, 2, 80),
    "white blood cell count": ("µL", 100, 500000), "platelet count": ("µL", 1000, 2000000),
}
TO_CM = {"cm": 1, "m": 100, "in": 2.54, "ft": 30.48}
TO_KG = {"kg": 1, "g": 0.001, "lbs": 0.453592}

train = pd.read_csv(f"{ROOT}/train_data_10_22_v1.0_labels.csv")
flags = []
for _, r in train.iterrows():
    for key, value in ast.literal_eval(r["Relevant Entities"]).items():
        name = key.lower().replace("fentanyl", "fentanyl").replace("fentanyl", "fentanyl")
        name = "fentanyl patch dose" if key.lower() == "fentanyl patch dose" else name
        if not (isinstance(value, (list, tuple)) and len(value) == 2 and isinstance(value[0], (int, float)) and not isinstance(value[0], bool)):
            continue
        x, unit = value
        problem = None
        if name in UNITS and unit not in UNITS[name]:
            problem = f"unit {unit!r} does not fit"
        elif name in RANGE:
            want, lo, hi = RANGE[name]
            if name == "height":
                x = x * TO_CM.get(unit, float("nan"))
            elif name == "weight":
                x = x * TO_KG.get(unit, float("nan"))
            elif name == "temperature" and "fahr" in unit:
                x = (x - 32) * 5 / 9
            elif want == "µL" and unit == "L":
                x = x / 1e6
            if not lo <= x <= hi:
                problem = f"value {value[0]} {unit} outside {lo}-{hi}" + (f" {want}" if want else "")
        if problem:
            flags.append({"v1.0 Row Number": r["Row Number"], "Calculator ID": r["Calculator ID"],
                          "Calculator Name": r["Calculator Name"], "Input": key, "Value": value, "Problem": problem})

out = pd.DataFrame(flags)
out.to_csv(f"{ROOT}/train_data_10_22_implausible_inputs.csv", index=False)
print(f"{len(out)} implausible inputs in {out['v1.0 Row Number'].nunique()} of {len(train)} rows")
print(out.groupby("Calculator Name")["v1.0 Row Number"].nunique().sort_values(ascending=False).to_string())
