"""Splice the runs into one output per (model, style, row), re-score against the corrected labels, and print accuracy by
category. Each output is the latest run of that row (by timestamp), among the runs every model took part in:
  main run (final_runs/test_data_10_22, 10-22 04:51-06:06)
  < CKD-EPI re-run (10-22 13:53: 13 label-fix rows and CKD-EPI rows 21-40, all styles)
  < MDRD/Sodium re-run (10-22 16:51: rows 141-160 and 366-385, zero- and one-shot).
Kept on the main run: the Astra rows (re-run 10-22 14:24 on unreviewed corrections; physician review pending), and the
LDL Friedewald and SOFA one-shot re-runs (10-23/10-24; GPT-4, and for LDL GPT-3.5, never ran them).
With MEDCALC_VARIANT=astra (see variant.py) the Astra rows come from the Astra re-run instead, scored against its labels."""
import json, os, re, sys
import numpy as np
import pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, f"{ROOT}/evaluation")
from evaluate import check_correctness
import variant
OUT = f"{variant.OUT}/outputs"

MODELS = [  # table name, main-run file prefix, re-run folder, re-run file prefix
    ("PMC-LLaMA", "axiong_PMC_LLaMA_13B", "PMC_LLaMA_13B", "axiong_PMC_LLaMA_13B"),
    ("MediTron", "epfl-llm_meditron-70b", "meditron-70b", "epfl-llm_meditron-70b"),
    ("Mistral", "mistralai_Mistral-7B-Instruct-v0.2", "Mistral-7B-Instruct-v0.2", "mistralai_Mistral-7B-Instruct-v0.2"),
    ("Mixtral", "mistralai_Mixtral-8x7B-Instruct-v0.1", "Mixtral-8x7B-Instruct-v0.1", "mistralai_Mixtral-8x7B-Instruct-v0.1"),
    ("Llama 3 8B", "meta-llama_Meta-Llama-3-8B-Instruct", "Meta-Llama-3-8B-Instruct", "meta-llama_Meta-Llama-3-8B-Instruct"),
    ("Llama 3 70B", "meta-llama_Meta-Llama-3-70B-Instruct", "Meta-Llama-3-70B-Instruct", "meta-llama_Meta-Llama-3-70B-Instruct"),
    ("GPT-3.5", "OpenAI_gpt-3.5-turbo-0125", "gpt-3.5-turbo-0125", "OpenAI_gpt-3.5-turbo-0125"),
    ("GPT-4", "OpenAI_gpt-4", "gpt-4-0613", "OpenAI_gpt-4-0613"),
]
STYLES = ["direct_answer", "zero_shot", "one_shot"]
CATS = ["lab", "physical", "date", "dosage", "risk", "severity", "diagnosis"]
MAIN = f"{ROOT}/final_runs/test_data_10_22/outputs"
CKDEPI = f"{ROOT}/extra_runs/corrected_rows_ckdepi_rerun_2026-10-22"
MDRD = f"{ROOT}/extra_runs/mdrd_sodium_wording_rerun_2026-10-22"

norm = lambda s: re.sub(r"\s+", " ", str(s)).strip()
as_run = pd.read_csv(f"{ROOT}/extra_runs/gpt-4o-2024-08-06_benchmark_2026-10-24/input/test_data_10_22_as_run.csv", dtype=str, keep_default_na=False).set_index("Row Number")
labels = variant.labels()
MDRD_ROWS = {str(r) for r in list(range(141, 161)) + list(range(366, 386))}

def load(path):
    return {str(x["Row Number"]): x for x in map(json.loads, open(path))}

def score(x, r):
    try:
        ok = check_correctness(x["LLM Answer"], labels.at[r, "Ground Truth Answer"], x["Calculator ID"],
                               labels.at[r, "Upper Limit"], labels.at[r, "Lower Limit"])
    except Exception:
        ok = 0
    return "Correct" if ok else "Incorrect"

os.makedirs(OUT, exist_ok=True)
table, problems, rescore_diff = {}, [], []
for name, prefix, folder, rprefix in MODELS:
    for style in STYLES:
        rows = load(f"{MAIN}/{prefix}_{style}.jsonl")
        source = {r: "final_runs/test_data_10_22" for r in rows}
        rerun = load(f"{CKDEPI}/{folder}/outputs/{rprefix}_{style}.jsonl")
        for r in rerun:
            rows[r], source[r] = rerun[r], "corrected_rows_ckdepi_rerun_2026-10-22"
        if variant.ASTRA:
            rerun = load(f"{variant.ASTRA_RUN}/{folder}/outputs/{rprefix}_{style}.jsonl")
            for r in variant.ASTRA_ROWS:
                rows[r], source[r] = rerun[r], "astra_corrections_rerun_2026-10-22"
        if style != "direct_answer":
            rerun = load(f"{MDRD}/{folder}/outputs/{rprefix}_{style}.jsonl")
            for r in MDRD_ROWS:
                rows[r], source[r] = rerun[r], "mdrd_sodium_wording_rerun_2026-10-22"
        assert len(rows) == 1048
        with open(f"{OUT}/{prefix}_{style}.jsonl", "w") as f:
            for r in sorted(rows, key=int):
                x = dict(rows[r])
                # prompt check: question and note must match the test row this output stands for
                ref = labels if (r in MDRD_ROWS and style != "direct_answer") or r in variant.ASTRA_ROWS else as_run
                if norm(x["Question"]) != norm(ref.at[r, "Question"]):
                    problems.append((name, style, r, "question"))
                note = norm(x["Patient Note"])
                if not (note == norm(ref.at[r, "Patient Note"]) or ("PMC" in prefix and norm(ref.at[r, "Patient Note"]).startswith(note[:200]))):
                    problems.append((name, style, r, "note"))
                new = score(x, r)
                if new != x["Result"] and labels.at[r, "Ground Truth Answer"] == x["Ground Truth Answer"]:
                    rescore_diff.append((name, style, r, x["Result"], new))
                x.update({"Ground Truth Answer": labels.at[r, "Ground Truth Answer"],
                          "Ground Truth Explanation": labels.at[r, "Ground Truth Explanation"],
                          "Upper Limit": labels.at[r, "Upper Limit"], "Lower Limit": labels.at[r, "Lower Limit"],
                          "Result": new, "Source Run": source[r]})
                f.write(json.dumps(x) + "\n")
                table.setdefault((name, style), {}).setdefault(x["Category"], []).append(new == "Correct")

print("prompt mismatches:", len(problems), problems[:10])
print("re-score differs from stored verdict on unchanged labels:", len(rescore_diff), rescore_diff[:10])
stats = []
for (name, style), cats in table.items():
    allr = sum(cats.values(), [])
    line = {"Model": name, "Style": style}
    for c in CATS + ["overall"]:
        v = np.array(allr if c == "overall" else cats[c], float); m = v.mean()
        line[c] = round(m * 100, 2); line[c + "_std"] = round(np.sqrt(m * (1 - m) / len(v)), 2); line[c + "_n"] = len(v)
    stats.append(line)
pd.DataFrame(stats).to_csv(f"{variant.OUT}/accuracy_by_category.csv", index=False)
print(pd.DataFrame(stats)[["Model", "Style"] + CATS + ["overall"]].to_string(index=False))
