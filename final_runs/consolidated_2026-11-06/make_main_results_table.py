"""Write the main-results LaTeX table from the consolidated outputs (same layout as the paper's Table)."""
import json, os
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction
import numpy as np
import variant  # MEDCALC_VARIANT=astra: see variant.py
D = f"{variant.OUT}/outputs"
CATS = ["lab", "physical", "date", "dosage", "risk", "severity", "diagnosis"]
MODELS = [
    (r"\includegraphics[width=0.4cm]{icons/pmc.png} PMC-LLaMA \cite{wu2024pmc}", "13B", "axiong_PMC_LLaMA_13B"),
    (r"\includegraphics[width=0.4cm]{icons/meditron.png} \textsc{MediTron} \cite{chen2023meditron}", "70B", "epfl-llm_meditron-70b"),
    (r"\includegraphics[width=0.4cm]{icons/mistral.png} Mistral \cite{jiang2023mistral}", "7B", "mistralai_Mistral-7B-Instruct-v0.2"),
    (r"\includegraphics[width=0.4cm]{icons/mistral.png} Mixtral \cite{jiang2024mixtral}", "8x7B", "mistralai_Mixtral-8x7B-Instruct-v0.1"),
    (r"\includegraphics[width=0.4cm]{icons/meta.png} Llama 3 \cite{touvron2023llama2}", "8B", "meta-llama_Meta-Llama-3-8B-Instruct"),
    (r"\includegraphics[width=0.4cm]{icons/meta.png} Llama 3 \cite{touvron2023llama2}", "70B", "meta-llama_Meta-Llama-3-70B-Instruct"),
    (r"\includegraphics[width=0.4cm]{icons/openai.png} GPT-3.5 \cite{ouyang2022training}", "N/A", "OpenAI_gpt-3.5-turbo-0125"),
    (r"\includegraphics[width=0.4cm]{icons/openai.png} GPT-4 \cite{openai2023gpt4}", "N/A", "OpenAI_gpt-4"),
]
SECTIONS = [("Zero-shot Direct Prompting", "direct_answer"), ("Zero-shot CoT Prompting", "zero_shot"), ("One-shot CoT Prompting", "one_shot")]

def half_up(x, places):
    return Decimal(str(x)).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)

def cell(v):
    acc = Fraction(sum(v), len(v)); m = float(acc); s = np.sqrt(m * (1 - m) / len(v))
    return rf"\makecell{{{half_up(Decimal(acc.numerator) * 100 / Decimal(acc.denominator), 1)}\\\textcolor{{gray}}{{\scriptsize$\pm${half_up(s, 2)}}}}}"

out = [r"""\begin{table}[h!] \small
    \centering
    \caption{\textsc{MedCalc-Bench} accuracy of different systems. All numbers are in percentages. Phys.: Physical; Sev.: Severity; Diag.: Diagnosis; Avg.: Average; $\pm$ std for all results are shown. Each std. is computed with the following formula: $\sqrt{(\text{accuracy} * (1- \text{accuracy})/(n)}$, where $n$ is the total number of patient notes for a given subcategory for rule or equation-based calculators.}
    \begin{tabular}{lc|cccc|ccc|c}
        \toprule
        \multirow{3}{*}{\bf Model} & \multirow{3}{*}{\bf Size} & \multicolumn{4}{c|}{\bf Equation} & \multicolumn{3}{c|}{\bf Rule-based} & \multirow{3}{*}{\bf Avg.} \\
        \cmidrule(lr){3-9}
         &  & Lab & Phys. & Date & Dosage & Risk & Sev. & Diag. & \\"""]
ns = {}
for title, style in SECTIONS:
    out += [r"        \midrule", r"        \rowcolor[RGB]{234, 238, 234}", rf"        \multicolumn{{10}}{{c}}{{\bf {title}}} \\", r"        \midrule"]
    for name, size, prefix in MODELS:
        by = {c: [] for c in CATS}
        for l in open(f"{D}/{prefix}_{style}.jsonl"):
            x = json.loads(l); by[x["Category"]].append(x["Result"] == "Correct")
        ns = {c: len(v) for c, v in by.items()}
        cells = [cell(by[c]) for c in CATS] + [cell(sum(by.values(), []))]
        out.append(f"{name} & {size} & " + "  & ".join(cells) + "  \\\\")
out += [r"        \bottomrule", r"    \end{tabular}", r"    \label{tab:main_results}", r"\end{table}"]
open(f"{variant.OUT}/tables/main_results_table.tex", "w").write("\n".join(out) + "\n")
print("n per category:", ns, "total", sum(ns.values()))
