"""
Samples 200 errors for the manual check of the GPT-4 error classifier (paper Section 4.2: "We manually evaluate the
annotations of 200 randomly sampled explanation errors, and find the accuracy of GPT-4 error classifier to be 89%").

Pool: every chain-of-thought error the classifier labelled in error_analysis_10_22 (8 models x zero-shot and one-shot),
keeping only rows that are still Incorrect under the corrected labels (final_runs/test_data_10_22_corrected) and whose
classifier output parsed to a list of error types. 200 are drawn uniformly at random (seed below), as in the paper.

Writes error_analysis_10_22/judge_agreement_sample_200.xlsx (Instructions, Sample, Summary) and a .csv of the sample.
"""
import glob
import json
import os
import re
import pandas as pd
from openpyxl import Workbook
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

HERE = os.path.dirname(os.path.abspath(__file__))
FOLDER = os.path.join(HERE, "error_analysis_10_22")
OUTPUTS = os.path.join(HERE, "..", "final_runs", "test_data_10_22_corrected", "outputs")
SEED = 20261029
N = 200
# The test set exactly as run (md5 e53d936a...). The output files store the bare question; the prompt had the row's
# Time Point Instruction after it (evaluation/run.py; confirmed by extra_runs/time_point_replay_check_2026-10-24).
AS_RUN = os.path.join(HERE, "..", "extra_runs", "gpt-4o-2024-08-06_benchmark_2026-10-24", "input", "test_data_10_22_as_run.csv")
STEM = {"gpt-3.5-turbo": "OpenAI_gpt-3.5-turbo-0125", "gpt-4": "OpenAI_gpt-4",
        "llama-3-70b": "meta-llama_Meta-Llama-3-70B-Instruct", "llama-3-8b": "meta-llama_Meta-Llama-3-8B-Instruct",
        "meditron-70b": "epfl-llm_meditron-70b", "mistral-7b": "mistralai_Mistral-7B-Instruct-v0.2",
        "mixtral-8x7b": "mistralai_Mixtral-8x7B-Instruct-v0.1", "pmc-llama-13b": "axiong_PMC_LLaMA_13B"}
TYPES = {"A": "Wrong calculator / rules used, or wrong knowledge about the calculation",
         "B": "Extracted different parameters than the ground truth",
         "C": "Performed the arithmetic incorrectly",
         "D": "No clear error in the prediction"}

clean = lambda s: ILLEGAL_CHARACTERS_RE.sub("", str(s)) if s is not None else ""


def letters(types):
    """The classifier sometimes writes 'C. performed the arithmetic incorrectly'; keep the letter."""
    out = sorted({m.group(1) for t in types if (m := re.match(r"\s*([ABCD])\b", str(t)))})
    return ", ".join(out) or "(none given)"


as_run = pd.read_csv(AS_RUN).set_index("Row Number")
pool = []
for path in sorted(glob.glob(os.path.join(FOLDER, "zero_shot_*.csv")) + glob.glob(os.path.join(FOLDER, "one_shot_*.csv"))):
    name = os.path.basename(path)[:-4]
    setting = "one_shot" if name.startswith("one_shot") else "zero_shot"
    model = name[len(setting) + 1:]
    outputs = {int(r["Row Number"]): r for r in map(json.loads, open(os.path.join(OUTPUTS, f"{STEM[model]}_{setting}.jsonl")))}
    judged = json.load(open(path + ".error_type.json"))
    for _, r in pd.read_csv(path).iterrows():
        out = outputs[int(r["Row Number"])]
        verdict = judged.get(f"{r['Calculator ID']}_{r['Note ID']}")
        if out["Result"] == "Correct" or not isinstance(verdict, dict) or not isinstance(verdict.get("error_type"), list):
            continue
        run_row = as_run.loc[int(r["Row Number"])]
        assert out["Question"] == run_row["Question"]
        time_point = run_row["Time Point Instruction"]
        question = run_row["Question"].rstrip() + (" " + time_point.strip() if isinstance(time_point, str) and time_point.strip() else "")
        pool.append({"Setting": setting.replace("_", "-"), "Model": model, "Row Number": int(r["Row Number"]),
                     "Calculator": out["Calculator Name"], "Category": out["Category"], "Patient Note": out["Patient Note"],
                     "Question": question, "Ground Truth Answer": out["Ground Truth Answer"],
                     "Ground Truth Explanation": r["explanation"], "Model Output": r["raw_answer"],
                     "Model Answer (extracted)": out["LLM Answer"], "Judge Error Type(s)": letters(verdict["error_type"]),
                     "Judge Analysis": verdict.get("analysis", "")})

sample = pd.DataFrame(pool).sample(n=N, random_state=SEED).reset_index(drop=True)
sample.insert(0, "Sample #", range(1, N + 1))
sample.to_csv(os.path.join(FOLDER, "judge_agreement_sample_200.csv"), index=False)

FONT, BOLD = Font(name="Arial", size=10), Font(name="Arial", size=10, bold=True)
HEAD_FILL, INPUT_FILL = PatternFill("solid", fgColor="D9E1F2"), PatternFill("solid", fgColor="FFFF00")
WRAP = Alignment(wrap_text=True, vertical="top")
wb = Workbook()

# Instructions
ws = wb.active
ws.title = "Instructions"
lines = [
    ("Manual check of the GPT-4 error classifier", BOLD),
    ("", FONT),
    ("Paper (Section 4.2): \"We manually evaluate the annotations of 200 randomly sampled explanation errors, and find the "
     "accuracy of GPT-4 error classifier to be 89%.\" This sheet repeats that check on the corrected benchmark.", FONT),
    (f"Sample: {N} errors drawn uniformly at random (seed {SEED}) from {len(pool):,} chain-of-thought errors (8 models, "
     "zero-shot and one-shot) that GPT-4 (gpt-4-0613, the paper's prompt, temperature 0) classified, excluding answers "
     "that are correct under the corrected labels. Built by evaluation/sample_judge_agreement.py.", FONT),
    ("The Question column is the task text the model was given: the question, followed by the time-point instruction "
     "(\"You should use the patient's medical values and health status when they were first admitted to the hospital prior "
     "to any treatment.\") on the rows that had one (all non-template rows except APACHE II, Caprini and SOFA).", FONT),
    ("", FONT),
    ("Error types (the classifier's prompt; it may give more than one):", BOLD),
    *[(f"{k}. {v}", FONT) for k, v in TYPES.items()],
    ("The paper's text describes D as \"all other cases of errors\", but the classifier was told D means \"no clear error in "
     "the prediction\". Judge D by the classifier's definition.", FONT),
    ("", FONT),
    ("What to fill in (yellow columns on the Sample sheet):", BOLD),
    ("Judge Correct?  Yes if the judge's error type(s) are right for the model output, No otherwise (pick from the list).", FONT),
    ("Your Error Type(s)  The letters you would assign, comma-separated (e.g. A, B). Optional when Judge Correct? is Yes.", FONT),
    ("Comments  Anything worth noting.", FONT),
    ("The Summary sheet counts the agreement as you go.", FONT),
    ("", FONT),
    ("Example of a filled-in row (not part of the sample):", BOLD),
    ("Judge Error Type(s): B   |   Judge Correct?: No   |   Your Error Type(s): A, B   |   Comments: used the 2009 CKD-EPI "
     "equation and also took the creatinine from day 3", FONT),
]
for i, (text, font) in enumerate(lines, start=1):
    c = ws.cell(row=i, column=1, value=text)
    c.font, c.alignment = font, Alignment(wrap_text=True, vertical="top")
ws.column_dimensions["A"].width = 120

# Sample
ws = wb.create_sheet("Sample")
inputs = ["Judge Correct?", "Your Error Type(s)", "Comments"]
columns = list(sample.columns) + inputs
widths = {"Sample #": 8, "Setting": 10, "Model": 14, "Row Number": 9, "Calculator": 24, "Category": 10,
          "Patient Note": 70, "Question": 40, "Ground Truth Answer": 12, "Ground Truth Explanation": 60,
          "Model Output": 60, "Model Answer (extracted)": 12, "Judge Error Type(s)": 10, "Judge Analysis": 50,
          "Judge Correct?": 11, "Your Error Type(s)": 12, "Comments": 40}
for j, name in enumerate(columns, start=1):
    c = ws.cell(row=1, column=j, value=name)
    c.font, c.alignment = BOLD, WRAP
    c.fill = INPUT_FILL if name in inputs else HEAD_FILL
    ws.column_dimensions[c.column_letter].width = widths[name]
for i, row in sample.iterrows():
    for j, name in enumerate(columns, start=1):
        value = clean(row[name]) if name in sample.columns else None
        if name in ("Sample #", "Row Number"):
            value = int(row[name])
        c = ws.cell(row=i + 2, column=j, value=value)
        c.font, c.alignment = FONT, WRAP
        if name in inputs:
            c.fill = INPUT_FILL
    ws.row_dimensions[i + 2].height = 150
judge_col = ws.cell(row=1, column=columns.index("Judge Correct?") + 1).column_letter
dv = DataValidation(type="list", formula1='"Yes,No"', allow_blank=True)
dv.add(f"{judge_col}2:{judge_col}{N + 1}")
ws.add_data_validation(dv)
ws.freeze_panes = "B2"
ws.auto_filter.ref = f"A1:{ws.cell(row=1, column=len(columns)).column_letter}{N + 1}"

# Summary
ws = wb.create_sheet("Summary")
type_col = ws.cell(row=1, column=columns.index("Judge Error Type(s)") + 1).column_letter
rng = lambda col: f"Sample!${col}$2:${col}${N + 1}"
rows = [
    ("Agreement with the GPT-4 error classifier", None, None),
    ("Sampled errors", f"=COUNTA({rng('A')})", None),
    ("Reviewed (Judge Correct? filled)", f"=COUNTA({rng(judge_col)})", None),
    ("Judge correct (Yes)", f'=COUNTIF({rng(judge_col)},"Yes")', None),
    ("Judge wrong (No)", f'=COUNTIF({rng(judge_col)},"No")', None),
    ("Classifier accuracy (Yes / reviewed)", "=IF(B3=0,\"\",B4/B3)", "0.0%"),
    ("Paper's reported accuracy", 0.89, "0.0%"),
    ("", None, None),
    ("By the judge's label", "Sampled", None),
]
for i, (label, value, fmt) in enumerate(rows, start=1):
    a, b = ws.cell(row=i, column=1, value=label), ws.cell(row=i, column=2, value=value)
    a.font = BOLD if i in (1, 9) else FONT
    b.font = BOLD if i == 9 else FONT
    if fmt:
        b.number_format = fmt
ws.cell(row=9, column=3, value="Judge correct (Yes)").font = BOLD
ws.cell(row=9, column=4, value="Accuracy").font = BOLD
ws.cell(row=7, column=3, value="Source: MedCalc-Bench (NeurIPS 2024), Section 4.2").font = FONT
for k, label in enumerate(sorted(sample["Judge Error Type(s)"].unique()), start=10):
    ws.cell(row=k, column=1, value=label).font = FONT
    ws.cell(row=k, column=2, value=f'=COUNTIF({rng(type_col)},A{k})').font = FONT
    ws.cell(row=k, column=3, value=f'=COUNTIFS({rng(type_col)},A{k},{rng(judge_col)},"Yes")').font = FONT
    acc = ws.cell(row=k, column=4, value=f'=IF(COUNTIFS({rng(type_col)},A{k},{rng(judge_col)},"<>")=0,"",C{k}/COUNTIFS({rng(type_col)},A{k},{rng(judge_col)},"<>"))')
    acc.font, acc.number_format = FONT, "0.0%"
ws.column_dimensions["A"].width = 38
for col in "BCD":
    ws.column_dimensions[col].width = 18

wb.calculation.fullCalcOnLoad = True  # formulas are saved without values; Excel computes them on opening
out = os.path.join(FOLDER, "judge_agreement_sample_200.xlsx")
wb.save(out)
print(f"pool {len(pool)}; sampled {N} -> {out}")
print(sample.groupby(["Setting", "Model"]).size().to_string())
print(sample["Judge Error Type(s)"].value_counts().to_string())
