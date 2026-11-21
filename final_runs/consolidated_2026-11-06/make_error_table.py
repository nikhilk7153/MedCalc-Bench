"""Error-type table (paper Table 4) from the consolidated outputs and their classifications.
As in the paper: each wrong answer counts once, as its earliest error type (A > B > C > D); the number is the type's share
of all notes, the parentheses its share of the errors, and Error Rate = wrong answers / notes. Shares are taken over the
classified errors (empty or blocked classifications excluded) and scaled to the full error rate."""
import os
from decimal import Decimal, ROUND_HALF_UP
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
N = 1048
d = pd.read_csv(f"{HERE}/error_classifications.csv", dtype={"Error Types": str}, keep_default_na=False)
used = d[d.Status == "classified"]
NAMES = [("PMC-LLaMA", "PMC-LLaMA-13B"), ("MediTron", r"\textsc{MediTron}-70B"), ("Mistral", "Mistral-7B"), ("Mixtral", "Mixtral-8x7B"),
         ("Llama 3 8B", "Llama 3-8B"), ("Llama 3 70B", "Llama 3-70B"), ("GPT-3.5", "GPT-3.5"), ("GPT-4", "GPT-4")]
r2 = lambda x: Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
r0 = lambda x: int(Decimal(str(x)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))

stats = {}
for model, _ in NAMES:
    for style in ["zero_shot", "one_shot"]:
        wrong = int((d.Model.eq(model) & d.Style.eq(style)).sum())
        g = used[used.Model.eq(model) & used.Style.eq(style)]
        first = g["Error Types"].str[0].value_counts()
        share = {t: first.get(t, 0) / len(g) for t in "ABCD"}
        stats[model, style] = {"rate": wrong / N, "share": share, "wrong": wrong, "used": len(g)}

out = [r"""\begin{table}[h!] \small
    \centering
    \caption{Error type distribution of LLMs on \textsc{MedCalc-Bench}. Numbers in parentheses denote the relative proportions. Arrows indicate the proportion changes from zero-shot to one-shot learning.}
    \begin{tabular}{lccccc}
        \toprule
        \bf Model & \textbf{Type A Error} & \textbf{Type B Error} & \textbf{Type C Error} & \textbf{Type D Error} & \bf \textbf{Error Rate} \\
        \midrule"""]
rows_csv = []
for style, title in [("zero_shot", "Zero-shot CoT Prompting"), ("one_shot", "One-shot CoT Prompting")]:
    if style == "one_shot":
        out.append(r"        \midrule")
    out += [r"        \rowcolor[RGB]{234, 238, 234}", rf"        \multicolumn{{6}}{{l}}{{\it {title}}} \\", r"        % \midrule"]
    for model, label in NAMES:
        s = stats[model, style]; cells = []
        for t in "ABCD":
            pct = r0(100 * s["share"][t]); arrow = ""
            if style == "one_shot":
                before = r0(100 * stats[model, "zero_shot"]["share"][t])
                arrow = r"\(\uparrow\)" if pct > before else (r"\(\downarrow\)" if pct < before else "--")
            cells.append(f"{r2(s['rate'] * s['share'][t])} ({pct}\\%{arrow})")
            rows_csv.append({"Model": model, "Style": style, "Type": t, "Share of notes": s["rate"] * s["share"][t], "Share of errors": s["share"][t]})
        out.append(f"{label} & " + " & ".join(cells) + f" & {r2(s['rate'])} \\\\")
out += [r"        \bottomrule", r"    \end{tabular}", r"    \label{tab:error_analysis_results}", r"\end{table}"]
open(f"{HERE}/tables/error_analysis_table.tex", "w").write("\n".join(out) + "\n")
pd.DataFrame(rows_csv).to_csv(f"{HERE}/error_type_table_values.csv", index=False)
print("\n".join(out))
print({k: (v["wrong"], v["used"]) for k, v in stats.items()})
