"""Dataset statistics table (paper Table 1) for the corrected test set, computed as for v1.0: lengths in GPT-4 tokens
(tiktoken cl100k_base), attributes = number of Relevant Entities. The test set is the one the consolidated results use:
dataset/test_data_10_22.csv (notes and questions as run; MDRD/Sodium reworded, LDL not) with the inputs and labels of
dataset/test_data_10_22_corrected.csv."""
import ast, json, os, re
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction
import pandas as pd
import tiktoken
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root
HERE = os.path.dirname(os.path.abspath(__file__))
enc = tiktoken.get_encoding("cl100k_base")
ntok = lambda s: len(enc.encode(s))
ents = lambda s: len(json.loads(s) if s.strip().startswith('{"') else ast.literal_eval(s))

def load_release():
    t = pd.read_csv(f"{ROOT}/dataset/test_data_10_22.csv", dtype=str, keep_default_na=False).set_index("Row Number")
    c = pd.read_csv(f"{ROOT}/dataset/test_data_10_22_corrected.csv", dtype=str, keep_default_na=False).set_index("Row Number")
    for col in ["Relevant Entities", "Ground Truth Answer", "Lower Limit", "Upper Limit", "Ground Truth Explanation"]:
        t[col] = c[col]
    return t.reset_index()

def stats(d, question):
    d = d.assign(nl=d["Patient Note"].map(ntok), ql=question.map(ntok), na=d["Relevant Entities"].map(ents))
    rows = {}
    for cat, g in list(d.groupby("Category")) + [("overall", d)]:
        mean = lambda col: Fraction(int(col.sum()), len(col))  # exact, so .x5 rounds half up like the paper
        rows[cat] = (g["Calculator ID"].nunique(), len(g), mean(g.nl), mean(g.ql), g.na.min(), g.na.max(), mean(g.na))
    return rows

if __name__ == "__main__":
    v1 = pd.read_csv(f"{ROOT}/dataset/test_data.csv", dtype=str, keep_default_na=False)
    rel = load_release()
    with_tp = (rel["Question"] + rel["Time Point Instruction"].map(lambda s: " " + s if s else "")).str.strip()
    out = {"v1.0": stats(v1, v1["Question"]), "corrected": stats(rel, rel["Question"]), "corrected, question + time point": stats(rel, with_tp)}
    for k, s in out.items():
        print(k); print(pd.DataFrame(s, index=["tasks", "inst", "note", "q", "min", "max", "avg"]).T.map(lambda x: round(float(x), 2)).to_string())
    s = out["corrected"]
    f1 = lambda x: str((Decimal(x.numerator) / Decimal(x.denominator)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))
    ROWS = [("Equation-based Calculation Tasks", [("lab", "Lab", "LDL Concentration"), ("physical", "Physical", "QTc (Bazett Formula)"),
             ("date", "Date", "Estimated Due Date"), ("dosage", "Dosage", "Morphine Equivalents")]),
            ("Rule-based Calculation Tasks", [("risk", "Risk", "Caprini Score for VTE"), ("severity", "Severity", "Pneumonia Severity Idx"),
             ("diagnosis", "Diagnosis", "PERC Rule for PE")])]
    tex = [r"""\begin{table}[h!] \small
    \centering
    \caption{Statistics of \textsc{MedCalc-Bench}. Inst.: instance; Avg.: average; Attr.: attribute; Q.: question; L: length.}
    \begin{tabular}{lcccccccc}
    \toprule
     & \bf \makecell{\#Tasks} & \bf \makecell{\#Inst.} & \bf \makecell{Avg. L\\of Note} & \bf \makecell{Avg. L\\of Q.} & \bf \makecell{Min\\Attr.} & \bf \makecell{Max\\Attr.} & \bf \makecell{Avg.\\Attr.} & \bf \makecell{Example Calculation} \\
    \midrule"""]
    for i, (title, cats) in enumerate(ROWS):
        if i: tex.append(r"    \midrule")
        tex += [rf"    \multicolumn{{9}}{{l}}{{\textbf{{{title}}}}}\\", r"    \midrule"]
        for cat, label, ex in cats:
            t, n, nl, ql, mn, mx, av = s[cat]
            tex.append(rf"    \includegraphics[width=0.4cm]{{icons/{cat}.png}} {label} & {t} & {n} & {f1(nl)} & {f1(ql)} & {mn} & {mx} & {f1(av)} & {ex} \\")
    t, n, nl, ql, mn, mx, av = s["overall"]
    tex += [r"    \midrule", rf"    \textbf{{Overall}} & {t} & {n} & {f1(nl)} & {f1(ql)} & {mn} & {mx} & {f1(av)} & -- \\",
            r"    \bottomrule", r"    \end{tabular}", r"    \label{tab:dataset_statistics}", r"\end{table}"]
    open(f"{HERE}/tables/dataset_statistics_table.tex", "w").write("\n".join(tex) + "\n")
    print("\n".join(tex))
