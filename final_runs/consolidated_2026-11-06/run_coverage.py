"""One line per (re-run, row): which models ran which prompt styles, and whether that change can go in the final set.
Writes equation_audit/rerun_coverage_2026-11-01.csv."""
import collections, glob, json, os
import pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root
os.chdir(ROOT)
MODELS = [("meditron", "meditron-70b"), ("PMC_LLaMA", "PMC_LLaMA_13B"), ("Llama-3-70B", "Llama-3-70B"), ("Llama-3-8B", "Llama-3-8B"),
          ("Mixtral", "Mixtral-8x7B"), ("Mistral-7B", "Mistral-7B"), ("gpt-3.5", "GPT-3.5"), ("gpt-4", "GPT-4")]
ALL = [m for _, m in MODELS]
STYLES = ["direct_answer", "zero_shot", "one_shot", "code_exec"]
RUNS = {
    "corrected_rows_ckdepi_rerun_2026-10-22": ("13 label fixes + CKD-EPI one-shot wording", "Include"),
    "mdrd_sodium_wording_rerun_2026-10-22": ("MDRD / Sodium question wording", "Include (direct answer: old wording, footnote or leave out)"),
    "astra_corrections_rerun_2026-10-22": ("Astra corrections (unreviewed)", "Pending physician review"),
    "oss_rerun_2026-10-24/ldl_friedewald": ("LDL Friedewald question wording", "Leave out (GPT not run); changelog"),
    "oss_rerun_2026-10-24/sofa_oneshot": ("Corrected SOFA one-shot example", "Leave out (GPT-4 not run); changelog"),
    "sofa_oneshot_rerun_2026-10-23": ("Corrected SOFA one-shot example", "Leave out (GPT-4 not run); changelog"),
}
# changes that only alter the one-shot prompt; code execution uses no one-shot example
ONE_SHOT_ONLY = {"CKD-EPI one-shot example wording": ["one_shot"], "Corrected SOFA one-shot example": ["one_shot"]}


def model_of(f):
    b = os.path.basename(f)
    for k, m in MODELS:
        if k in b: return m


def style_of(f):
    b = os.path.basename(f)
    return next(s for s in STYLES if s in b)


cov = collections.defaultdict(lambda: collections.defaultdict(set))  # (change, row) -> style -> models
calc, why = {}, {}
for run, (change, decision) in RUNS.items():
    for f in glob.glob(f"extra_runs/{run}/**/*.jsonl", recursive=True):
        if "outputs" not in f: continue
        m, s = model_of(f), style_of(f)
        for l in open(f):
            x = json.loads(l)
            r = int(x["Row Number"]); calc[r] = x.get("Calculator Name", "")
            ch = change
            if run.startswith("corrected_rows_ckdepi"):
                ch = "CKD-EPI one-shot example wording" if 21 <= r <= 40 else "Label fix (prompt unchanged)"
            key = (ch, r); why[key] = decision
            cov[key][s].add(m)

out = []
for (change, r), st in sorted(cov.items(), key=lambda kv: (kv[0][1], kv[0][0])):
    line = {"Row Number": r, "Calculator": calc[r], "Change": change}
    missing = []
    affected = ONE_SHOT_ONLY.get(change, STYLES)
    for s in STYLES:
        ms = st.get(s, set())
        if s not in affected:
            line[s] = "n/a"; continue
        expected = ["GPT-3.5", "GPT-4"] if s == "code_exec" else ALL
        line[s] = f"{len(ms & set(expected))}/{len(expected)}"
        miss = [m for m in expected if m not in ms]
        if miss and (s != "code_exec" or ms): missing.append(f"{s}: {', '.join(miss)}")
        elif s == "code_exec" and not ms: missing.append("code_exec: not run")
    line["Missing"] = "; ".join(missing)
    line["Decision"] = why[(change, r)]
    out.append(line)
d = pd.DataFrame(out)
d.to_csv("equation_audit/rerun_coverage_2026-11-01.csv", index=False)
print(len(d), "lines;", d["Row Number"].nunique(), "rows")
print(d.groupby(["Change"] + STYLES).size().to_string())
