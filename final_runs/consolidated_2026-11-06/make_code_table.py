"""Code-interpreter table: GPT-3.5 and GPT-4, zero-shot CoT vs code prompt, accuracy by category.
Splices the code-execution runs by the same rule as consolidate.py, the latest run of each row: main run (10-22 09:10-09:31)
< CKD-EPI re-run (10-22 13:53: the 13 label-fix rows) < MDRD/Sodium re-run (10-22 16:51: reworded rows). Writes
outputs/code_exec_*.jsonl and tables/code_interpreter_table.tex.
Zero-shot CoT columns come from outputs/ (run consolidate.py first). MEDCALC_VARIANT=astra: see variant.py."""
import json, os, re, sys
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction
import pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, f"{ROOT}/evaluation")
from evaluate import check_correctness
import variant

MAIN = f"{ROOT}/evaluation/outputs/test_data_10_22"
CKDEPI = f"{ROOT}/extra_runs/corrected_rows_ckdepi_rerun_2026-10-22"
MDRD = f"{ROOT}/extra_runs/mdrd_sodium_wording_rerun_2026-10-22"
MODELS = [("GPT-3.5-turbo", "gpt-3.5-turbo-0125", "OpenAI_gpt-3.5-turbo-0125"), ("GPT-4", "gpt-4-0613", "OpenAI_gpt-4")]
MDRD_ROWS = {str(r) for r in list(range(141, 161)) + list(range(366, 386))}
CATS = [("Equation", [("lab", "Lab Test"), ("physical", "Physical"), ("dosage", "Dosage"), ("date", "Date")]),
        ("Rule", [("severity", "Severity"), ("diagnosis", "Diagnosis"), ("risk", "Risk")])]

norm = lambda s: re.sub(r"\s+", " ", str(s)).strip()
as_run = pd.read_csv(f"{ROOT}/extra_runs/gpt-4o-2024-08-06_benchmark_2026-10-24/input/test_data_10_22_as_run.csv", dtype=str, keep_default_na=False).set_index("Row Number")
labels = variant.labels()


def load(path):
    return {str(x["Row Number"]): x for x in map(json.loads, open(path))}


def score(x, r):
    # as in code_exec_timed.py: no answer is wrong; otherwise check the extracted answer
    if not x["LLM Raw Answer"]:
        return "Incorrect"
    try:
        ok = check_correctness(x["LLM Answer"], labels.at[r, "Ground Truth Answer"], x["Calculator ID"],
                               labels.at[r, "Upper Limit"], labels.at[r, "Lower Limit"])
    except Exception:
        ok = 0
    return "Correct" if ok else "Incorrect"


def pct(v):
    acc = Fraction(sum(v), len(v)) * 100
    return f"{(Decimal(acc.numerator) / Decimal(acc.denominator)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)}\\%"


problems, rescore_diff, acc = [], [], {}
for label, version, zprefix in MODELS:
    rows = load(f"{MAIN}/code_exec_{version}.jsonl")
    source = {r: "evaluation/outputs/test_data_10_22" for r in rows}
    rerun = load(f"{CKDEPI}/{version}/outputs/code_exec_{version}.jsonl")
    for r in rerun:
        rows[r], source[r] = rerun[r], "corrected_rows_ckdepi_rerun_2026-10-22"
    if variant.ASTRA:
        rerun = load(f"{variant.ASTRA_RUN}/{version}/outputs/code_exec_{version}.jsonl")
        for r in variant.ASTRA_ROWS:
            rows[r], source[r] = rerun[r], "astra_corrections_rerun_2026-10-22"
    rerun = load(f"{MDRD}/{version}/outputs/code_exec_{version}.jsonl")
    for r in MDRD_ROWS:
        rows[r], source[r] = rerun[r], "mdrd_sodium_wording_rerun_2026-10-22"
    assert len(rows) == 1048
    by = {}
    with open(f"{variant.OUT}/outputs/code_exec_{version}.jsonl", "w") as f:
        for r in sorted(rows, key=int):
            x = dict(rows[r])
            ref = labels if r in MDRD_ROWS or r in variant.ASTRA_ROWS else as_run
            if norm(x["Question"]) != norm(ref.at[r, "Question"]) or norm(x["Patient Note"]) != norm(ref.at[r, "Patient Note"]):
                problems.append((label, r))
            new = score(x, r)
            if new != x["Result"] and labels.at[r, "Ground Truth Answer"] == x["Ground Truth Answer"]:
                rescore_diff.append((label, r, x["Result"], new))
            x.update({"Ground Truth Answer": labels.at[r, "Ground Truth Answer"],
                      "Ground Truth Explanation": labels.at[r, "Ground Truth Explanation"],
                      "Upper Limit": labels.at[r, "Upper Limit"], "Lower Limit": labels.at[r, "Lower Limit"],
                      "Result": new, "Source Run": source[r]})
            f.write(json.dumps(x) + "\n")
            by.setdefault(x["Category"], []).append(new == "Correct")
    acc[label, "code"] = by
    zero = {}
    for l in open(f"{variant.OUT}/outputs/{zprefix}_zero_shot.jsonl"):
        x = json.loads(l); zero.setdefault(x["Category"], []).append(x["Result"] == "Correct")
    acc[label, "zero"] = zero

print("prompt mismatches:", len(problems), problems[:10])
print("re-score differs from stored verdict on unchanged labels:", len(rescore_diff), rescore_diff[:10])
cols = [(m, s) for m, _, _ in MODELS for s in ("zero", "code")]
tex = [r"""\begin{table}[ht]
    \centering
    \caption{Comparison of GPT-3.5-turbo and GPT-4 in zero-shot CoT before and after adding code interpreter}
    \begin{tabularx}{\textwidth}{l>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}X}
    \toprule
    & \multicolumn{2}{c}{\textbf{GPT-3.5-turbo}} & \multicolumn{2}{c}{\textbf{GPT-4}} \\
    \cmidrule(lr){2-3} \cmidrule(lr){4-5}
    & \textbf{Zero-Shot CoT} & \textbf{Code Prompt} & \textbf{Zero-Shot CoT} & \textbf{Code Prompt}\\
    \midrule"""]
for group, cats in CATS:
    tex.append(rf"    \textbf{{{group}}} &  &  &  &  \\ \midrule")
    for i, (cat, name) in enumerate(cats):
        tail = r" \midrule" if i == len(cats) - 1 else ""
        tex.append(rf"    \quad {name} & " + " & ".join(pct(acc[c][cat]) for c in cols) + r" \\" + tail)
tex.append(r"    \textbf{Overall} & " + " & ".join(pct(sum(acc[c].values(), [])) for c in cols) + r" \\ ")
tex += [r"    \bottomrule", r"    \end{tabularx}", r"\end{table}"]
open(f"{variant.OUT}/tables/code_interpreter_table.tex", "w").write("\n".join(tex) + "\n")
print("\n".join(tex))
