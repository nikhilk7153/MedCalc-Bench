"""Checks that the v1.0 equation-based errors are fixed in the corrected test set, the corrected training set and the
one-shot examples.

1. Calculator code: Estimated Due Date, Framingham, LDL, Calcium Correction, MELD Na and CKD-EPI labels are recomputed
   with an independent implementation of the published formulas and compared with the label (dates and integers exact,
   decimals within the label's own limits).
2. Dosage templates: MME notes that dose fentanyl patch/buccal in mg; steroid conversions of a drug to itself.
3. Misread inputs: Serum Osmolality notes that report urea (not urea nitrogen) in mg/dL next to a BUN input in mg/dL.
Revised test set: notes and questions of dataset/test_data_10_22.csv, inputs and labels of test_data_10_22_corrected.csv.
"""
import ast, json, math, os, re
from datetime import datetime, timedelta
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))  # repository root


# the one-shot examples use the calculator code's own input names
ALIASES = {"cycle_length": "cycle length", "menstrual_date": "Last menstrual date", "sys_bp": "Systolic Blood Pressure",
           "total_cholesterol": "Total cholesterol", "hdl_cholesterol": "high-density lipoprotein cholesterol",
           "bp_medicine": "Blood pressure being treated with medicines", "smoker": "Smoker", "triglycerides": "Triglycerides",
           "calcium": "Calcium", "albumin": "Albumin", "cvvhd": "Continuous veno-venous hemodialysis for ≥24 hours in the past week",
           "dialysis_twice": "Dialysis at least twice in the past week", "inr": "international normalized ratio",
           "sodium": "Sodium", "bilirubin": "Bilirubin"}


def ents(s):
    try: d = json.loads(s)
    except Exception: d = ast.literal_eval(s)
    return {ALIASES.get(k, k): v for k, v in d.items()}


# unit conversions to the units of the published formulas
TO_MG_DL = {  # quantity -> {unit: factor to mg/dL}
    "creatinine": {"mg/dl": 1, "µmol/l": 1 / 88.4, "umol/l": 1 / 88.4, "mmol/l": 1000 / 88.4, "mg/l": 0.1},
    "bilirubin": {"mg/dl": 1, "µmol/l": 1 / 17.1, "umol/l": 1 / 17.1},
    "cholesterol": {"mg/dl": 1, "mmol/l": 38.67},
    "triglycerides": {"mg/dl": 1, "mmol/l": 88.57},
    "calcium": {"mg/dl": 1, "mmol/l": 4.008, "meq/l": 2.004},
}


def conv(value, quantity):
    number, unit = value
    return number * TO_MG_DL[quantity][unit.lower()]


def due_date(e):
    lmp = datetime.strptime(e["Last menstrual date"], "%m/%d/%Y")
    return (lmp + timedelta(days=280 + (e["cycle length"] - 28))).strftime("%m/%d/%Y")


def framingham(e):
    age, male = e["age"][0], e["sex"].lower() == "male"
    tc, hdl = conv(e["Total cholesterol"], "cholesterol"), conv(e["high-density lipoprotein cholesterol"], "cholesterol")
    sbp, smoker, treated = e["Systolic Blood Pressure"][0], int(bool(e.get("Smoker"))), int(bool(e.get("Blood pressure being treated with medicines")))
    la, lc, lh, ls = math.log(age), math.log(tc), math.log(hdl), math.log(sbp)
    if male:
        las = math.log(min(age, 70))
        s = (52.00961 * la + 20.014077 * lc - 0.905964 * lh + 1.305784 * ls + 0.241549 * treated + 12.096316 * smoker
             - 4.605038 * la * lc - 2.84367 * las * smoker - 2.93323 * la * la - 172.300168)
        return 100 * (1 - 0.9402 ** math.exp(s))
    las = math.log(min(age, 78))
    s = (31.764001 * la + 22.465206 * lc - 1.187731 * lh + 2.552905 * ls + 0.420251 * treated + 13.07543 * smoker
         - 5.060998 * la * lc - 2.996945 * las * smoker - 146.5933061)
    return 100 * (1 - 0.98767 ** math.exp(s))


def ldl(e):
    return conv(e["Total cholesterol"], "cholesterol") - conv(e["high-density lipoprotein cholesterol"], "cholesterol") \
        - conv(e["Triglycerides"], "triglycerides") / 5


def calcium(e):
    number, unit = e["Albumin"]
    albumin = number / 10 if unit.lower() == "g/l" else number / 1000 if unit.lower() == "mg/dl" else number
    return conv(e["Calcium"], "calcium") + 0.8 * (4 - albumin)


def meld_na(e):
    dialysis = e.get("Dialysis at least twice in the past week") or e.get("Continuous veno-venous hemodialysis for ≥24 hours in the past week")
    cr = 4.0 if dialysis else min(max(conv(e["creatinine"], "creatinine"), 1), 4)
    bili, inr = max(conv(e["Bilirubin"], "bilirubin"), 1), max(e["international normalized ratio"], 1)
    na = min(max(e["Sodium"][0], 125), 137)
    meld = round(0.957 * math.log(cr) + 0.378 * math.log(bili) + 1.120 * math.log(inr) + 0.643, 1) * 10
    if meld > 11:
        meld = meld + 1.32 * (137 - na) - 0.033 * meld * (137 - na)
    return round(min(meld, 40))


def ckd_epi(e):
    female = e["sex"].lower() == "female"
    scr = conv(e["creatinine"], "creatinine")
    k, a = (0.7, -0.241) if female else (0.9, -0.302)
    return 142 * min(scr / k, 1) ** a * max(scr / k, 1) ** -1.2 * 0.9938 ** e["age"][0] * (1.012 if female else 1)


FORMULAS = {"13": due_date, "46": framingham, "44": ldl, "7": calcium, "23": meld_na, "3": ckd_epi}


def agrees(row, value):
    gt = row["Ground Truth Answer"]
    if isinstance(value, str):
        return value == datetime.strptime(gt, "%m/%d/%Y").strftime("%m/%d/%Y")
    if row["Calculator ID"] == "23":
        return round(value) == round(float(gt))
    return float(row["Lower Limit"]) <= value <= float(row["Upper Limit"])


def load_sets():
    test = pd.read_csv(f"{ROOT}/dataset/test_data_10_22.csv", dtype=str, keep_default_na=False).set_index("Row Number")
    corrected = pd.read_csv(f"{ROOT}/dataset/test_data_10_22_corrected.csv", dtype=str, keep_default_na=False).set_index("Row Number")
    for col in ["Relevant Entities", "Ground Truth Answer", "Lower Limit", "Upper Limit"]:
        test[col] = corrected[col]
    train = pd.read_csv(f"{ROOT}/dataset/train_data_10_22.csv", dtype=str, keep_default_na=False)
    shots = json.load(open(f"{ROOT}/final_files_2026-10-16/one_shot_10_17_2026.json"))
    one_shot = pd.DataFrame([{"Row Number": cid, "Calculator ID": cid, "Patient Note": x["Patient Note"],
                              "Relevant Entities": json.dumps(x["input_parameters"]), "Ground Truth Answer": str(x["Response"]["answer"]),
                              "Lower Limit": str(float(x["Response"]["answer"]) * 0.95) if cid not in ("13", "68", "69") else "",
                              "Upper Limit": str(float(x["Response"]["answer"]) * 1.05) if cid not in ("13", "68", "69") else ""}
                             for cid, x in shots.items()])
    return {"test": test.reset_index(), "train": train, "one-shot": one_shot}


lines = []
for name, data in load_sets().items():
    for cid, formula in FORMULAS.items():
        rows = data[data["Calculator ID"] == cid]
        bad, skipped = [], 0
        for _, r in rows.iterrows():
            try:
                value = formula(ents(r["Relevant Entities"]))
            except (KeyError, TypeError, ValueError):
                skipped += 1
                continue
            if not agrees(r, value):
                bad.append(f'{r["Row Number"]} (label {r["Ground Truth Answer"]}, formula {value if isinstance(value, str) else round(value, 3)})')
        lines.append({"Set": name, "Check": f"formula {cid}", "Rows": len(rows), "Disagree": len(bad), "Not checked (units)": skipped,
                      "Examples": "; ".join(bad[:4])})

    mme = data[data["Calculator ID"] == "49"]
    fentanyl_mg = mme[mme["Patient Note"].str.contains(r"\d\s*mg of fenta", case=False, regex=True)]
    lines.append({"Set": name, "Check": "MME fentanyl dosed in mg", "Rows": len(mme), "Disagree": len(fentanyl_mg),
                  "Examples": "; ".join(fentanyl_mg["Row Number"].head(4))})
    steroid = data[data["Calculator ID"] == "24"]
    same = [r["Row Number"] for _, r in steroid.iterrows()
            if ents(r["Relevant Entities"])["input steroid"][0].split()[0].lower() == ents(r["Relevant Entities"])["target steroid"].split()[0].lower()]
    lines.append({"Set": name, "Check": "steroid converted to itself", "Rows": len(steroid), "Disagree": len(same), "Examples": "; ".join(same[:4])})
    # urea in mmol/L equals BUN in mmol/L; urea in mg/dL is not BUN in mg/dL (v1.0's error)
    osm = data[data["Calculator ID"] == "30"]
    urea = [r["Row Number"] for _, r in osm.iterrows()
            if re.search(r"\burea\b", r["Patient Note"], re.I) and not re.search(r"urea nitrogen|\bBUN\b", r["Patient Note"], re.I)
            and str(ents(r["Relevant Entities"]).get("Blood Urea Nitrogen (BUN)", ["", ""])[1]).lower() in ("mg/dl", "g/l")]
    lines.append({"Set": name, "Check": "osmolality: urea in mg/dL taken as BUN", "Rows": len(osm), "Disagree": len(urea),
                  "Examples": "; ".join(urea[:6])})

out = pd.DataFrame(lines)
out.to_csv(f"{HERE}/verify_v10_fixes.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 140)
print(out.to_string(index=False))
