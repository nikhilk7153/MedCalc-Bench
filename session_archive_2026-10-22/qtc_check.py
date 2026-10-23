import sys, ast
import pandas as pd

ROOT = "/Users/nikhilkhandekar/Downloads/MedCalc-Bench-version-1.0 2"
sys.path.insert(0, f"{ROOT}/calculator_implementations_verified")
from qt_calculator_bazett import bazett_calculator_explanation
from qt_calculator_fredericia import fredericia_calculator_explanation
from qt_calculator_framingham import framingham_calculator_explanation
from qt_calculator_hodges import hodges_calculator_explanation
from qt_calculator_rautaharju import rautaharju_calculator_explanation

CALC = {11: bazett_calculator_explanation, 56: fredericia_calculator_explanation, 57: framingham_calculator_explanation,
        58: hodges_calculator_explanation, 59: rautaharju_calculator_explanation}


def label(calc_id, heart_rate, qt):
    out = CALC[calc_id]({"heart_rate": [heart_rate, "beats per minute"], "qt_interval": [qt, "msec"]})
    answer = out["Answer"]
    return out["Explanation"], answer, round(answer * 0.95, 5), round(answer * 1.05, 5)


if __name__ == "__main__":
    d = pd.read_csv(f"{ROOT}/dataset/test_data_10_22.csv")
    q = d[d["Calculator ID"].isin(CALC)]
    print(q.groupby("Calculator Name").size().to_dict())
    bad = []
    for _, r in q.iterrows():
        e = ast.literal_eval(r["Relevant Entities"])
        expl, ans, lo, hi = label(r["Calculator ID"], e["Heart Rate or Pulse"][0], e["QT Interval"][0])
        ok = (expl.strip() == str(r["Ground Truth Explanation"]).strip(), abs(ans - float(r["Ground Truth Answer"])) < 1e-9,
              abs(lo - float(r["Lower Limit"])) < 1e-9, abs(hi - float(r["Upper Limit"])) < 1e-9)
        if not all(ok):
            bad.append((r["Row Number"], ok))
    print(f"code reproduces stored label for {len(q) - len(bad)}/{len(q)} QTc rows; mismatches (expl, answer, lower, upper):", bad[:10])
