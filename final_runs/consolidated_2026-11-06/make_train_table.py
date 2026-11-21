"""Training-set table (paper Appendix C, Table 8): one row per calculator in dataset/train_data_10_22.csv with its type,
sub-type, number of attributes and number of notes. Writes tables/train_table.tex and prints the numbers for the
Section C.1 text.

Rows follow the published table's order and names. Number of Notes is counted from the data. Number of Attributes is a
property of the calculator, not of the notes, so it keeps the published values, with the same fixes as the test-set
calculator table: MAP 3 -> 2 (systolic, diastolic) and BMI 3 -> 2 (height, weight); Estimated Date of Conception 2 -> 1
(last menstrual date; the verified code takes no cycle length, and the training notes' inputs now have only that).
Two published rows did not match v1.0's training data: Wells' Criteria for DVT has no training notes (dropped), and
Centor (in the data) was missing (added; attributes 5, as in the test-set table). HOMA-IR's only training note was the
test note's case, so the corrected training set has none."""
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))  # repository root

# (calculator id, name as published, number of attributes)
ROWS = [
    ("2", "Creatinine Clearance (Cockcroft-Gault Equation)", 5), ("3", "CKD-EPI Equations for Glomerular Filtration Rate", 3),
    ("4", "CHA2DS2-VASc Score for Atrial Fibrillation Stroke Risk", 10), ("5", "Mean Arterial Pressure (MAP)", 2),
    ("6", "Body Mass Index (BMI)", 2), ("7", "Calcium Correction for Hypoalbuminemia", 2),
    ("8", "Wells' Criteria for Pulmonary Embolism", 9), ("9", "MDRD GFR Equation", 4), ("10", "Ideal Body Weight", 2),
    ("11", "QTc Bazett Calculator", 2), ("18", "HEART Score for Major Cardiac Events", 13),
    ("19", "Fibrosis-4 (FIB-4) Index for Liver Fibrosis", 4), ("20", "Centor Score (Modified/McIsaac) for Strep Pharyngitis", 5),
    ("22", "Maintenance Fluids Calculations", 1), ("26", "Sodium Correction for Hyperglycemia", 2), ("30", "Serum Osmolality", 3),
    ("33", "FeverPAIN Score for Strep Pharyngitis", 5), ("38", "Free Water Deficit", 4), ("39", "Anion Gap", 3),
    ("44", "LDL Calculated", 3), ("45", "CURB-65 Score for Pneumonia Severity", 6), ("48", "PERC Rule for Pulmonary Embolism", 9),
    ("51", "SIRS Criteria", 5), ("56", "QTc Fridericia Calculator", 2), ("57", "QTc Framingham Calculator", 2),
    ("58", "QTc Hodges Calculator", 2), ("59", "QTc Rautaharju Calculator", 2), ("60", "Body Surface Area Calculator", 2),
    ("61", "Target weight", 2), ("62", "Adjusted Body Weight", 3), ("63", "Delta Gap", 3), ("64", "Delta Ratio", 3),
    ("65", "Albumin Corrected Anion Gap", 4), ("66", "Albumin Corrected Delta Gap", 4), ("67", "Albumin Corrected Delta Ratio", 4),
    ("49", "MME Conversion", 2), ("24", "Steroid Conversion", 2), ("13", "Estimated Due Date", 2),
    ("68", "Estimated Date of Conception", 1), ("69", "Estimated Gestational Age", 2),
]
SUBTYPE = {"lab": ("Equation-Based", "Lab Test"), "physical": ("Equation-Based", "Physical"), "date": ("Equation-Based", "Date"),
           "dosage": ("Equation-Based", "Dosage"), "risk": ("Rule-Based", "Risk"), "severity": ("Rule-Based", "Severity"),
           "diagnosis": ("Rule-Based", "Diagnosis")}

train = pd.read_csv(f"{ROOT}/dataset/train_data_10_22.csv", dtype=str, keep_default_na=False)
counts = train.groupby("Calculator ID").size()
category = train.groupby("Calculator ID")["Category"].first()
assert set(counts.index) == {cid for cid, _, _ in ROWS}, set(counts.index) ^ {cid for cid, _, _ in ROWS}

lines = []
for cid, name, attributes in ROWS:
    kind, sub = SUBTYPE[category[cid]]
    lines.append(f"{name} & {kind} & {sub} & {attributes} & {counts[cid]} \\\\")
open(f"{HERE}/tables/train_table.tex", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))

types = train.groupby("Calculator ID")["Note Type"].agg(lambda s: "/".join(sorted(set(s))))
template = types[types == "Template"].index
print(f"\nC.1 text: {len(train)} instances, {train['Calculator ID'].nunique()} calculators; "
      f"{len(template)} template calculators with {counts[template].min()}-{counts[template].max()} notes each "
      f"({counts[template].sum()} in total); {counts.drop(template).sum()} notes from Open-Patients over "
      f"{len(counts) - len(template)} calculators; {55 - len(counts)} calculators without training notes.")
