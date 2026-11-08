"""Ambiguity audit, part A: earlier annotation rounds of the identical note.

For every row of the test set, each earlier version (v1.0, 1/17/2025, 1_31, Verified) that has the same calculator and
the identical note text is treated as an independent reading. Its input values replace ours key by key (only for keys
it records; a key it lacks keeps our value), the label is recomputed with our calculator code, and the row is flagged
when the alternative label falls outside our label's accepted range. Recomputing with our code removes differences
caused by the earlier versions' code bugs, so a flag means the readers disagreed on the note itself.

  python3 cross_version.py [--data ../../dataset/test_data_10_22_corrected.csv]
Writes cross_version_flags.csv (one row per flagged row and version) here.
"""
import argparse, ast, os, sys
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "session_archive_2026-10-22"))
from regen_labels import compute, labels, info

VERSIONS = {"v1.0": "dataset/test_data.csv", "1_17": "dataset/test_data_1_17_2025.csv",
            "1_31": "dataset/test_set_1_31.csv", "Verified": "dataset/medcalc_bench_verified/test_data.csv"}
META = {"file path", "explanation function", "calculator name", "type", "question"}
# earlier key spellings -> ours (keys whose meaning changed between versions are left out)
RENAME = {("25", "hypertension history"): "Hypertension", ("29", "renal disease"): "Renal disease history",
          ("32", "diabetes mellitus criteria for cci rule"): "Diabetes mellitus",
          ("18", "diabetes mellitus criteria for cci rule"): "Diabetes mellitus",
          ("32", "chronic obstructive pulmonary disease"): "Chronic Pulmonary Disease",
          ("28", "partial pressure of oxygen (pao2) for apache ii"): "PaO2",
          ("43", "partial pressure of oxygen (pao2) for apache ii"): "PaO2"}


def norm(k):
    return (k.replace("≤", "<=").replace("₂", "2").replace("Faliure", "Failure").replace("Continous", "Continuous")
            .strip().lower())


# v1.0's Charlson value forms -> the forms cci.py matches (v1.0's code scored its own forms 0)
CCI_FORMS = {"moderate": "moderate to severe", True: "localized", "True": "localized", False: "none", "False": "none"}


def same_reading(cid, old, val):
    """True when two input values record the same reading of the note: the same string up to case or Charlson's
    v1.0 spelling, or a measurement in a different unit encoding (unit typos such as 'm^3', or another unit)."""
    if isinstance(old, list) and isinstance(val, list) and len(old) == 2 and len(val) == 2 and old[1] != val[1]:
        return True
    if cid == "32" and isinstance(val, (str, bool)):
        val = CCI_FORMS.get(val, val)
    if isinstance(old, str) and isinstance(val, str):
        return old.strip().lower() == val.strip().lower()
    return False


def parse(s):
    try:
        return ast.literal_eval(s)
    except Exception:
        return None


def within(ans, lo, hi, alt):
    try:
        a, l, h = float(alt), float(lo), float(hi)
        return min(l, h) - 1e-9 <= a <= max(l, h) + 1e-9
    except ValueError:
        return str(alt).strip() == str(ans).strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=os.path.join(ROOT, "dataset/test_data_10_22_corrected.csv"))
    args = p.parse_args()
    cur = pd.read_csv(args.data, dtype=str)
    others = {k: pd.read_csv(os.path.join(ROOT, f), dtype=str) for k, f in VERSIONS.items()}
    for d in others.values():
        d["key"] = d["Calculator ID"] + "|" + d["Patient Note"].str.strip()
    out, compared = [], 0
    for _, r in cur.iterrows():
        cid = r["Calculator ID"]
        if cid == "49":  # MME inputs are one free-text field
            continue
        ours = parse(r["Relevant Entities"])
        our_keys = {norm(k): k for k in set(info[cid]) - META}
        seen = False
        for v, d in others.items():
            m = d[d["key"] == cid + "|" + r["Patient Note"].strip()]
            if not len(m):
                continue
            theirs = parse(m.iloc[0]["Relevant Entities"])
            if theirs is None:
                continue
            seen = True
            alt, changed = dict(ours), []
            for k, val in theirs.items():
                mine = RENAME.get((cid, norm(k))) or our_keys.get(norm(k))
                if mine is None:
                    continue
                old = ours.get(mine)
                if old in (False, "False", None) and val in (False, "False", None):
                    continue
                if same_reading(cid, old, val):
                    continue
                if old != val:
                    alt[mine] = val
                    changed.append(f"{mine}: {old} -> {val}")
            if not changed:
                continue
            try:
                ans = labels(r["Category"], compute(cid, alt))[0]
            except Exception as e:
                ans = f"error: {str(e)[:60]}"
            if not within(r["Ground Truth Answer"], r["Lower Limit"], r["Upper Limit"], ans):
                out.append({"Row Number": int(r["Row Number"]), "Calculator ID": cid, "Calculator Name": r["Calculator Name"],
                            "Category": r["Category"], "Note ID": r["Note ID"], "Version": v,
                            "Our Label": r["Ground Truth Answer"], "Alternative Label": ans, "Inputs Read Differently": " | ".join(changed)})
        compared += seen
    o = pd.DataFrame(out).sort_values(["Row Number", "Version"])
    o.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cross_version_flags.csv"), index=False)
    print(f"rows with an earlier reading of the identical note: {compared}; flagged rows: {o['Row Number'].nunique()}")
    print(o.drop_duplicates("Row Number").groupby("Category").size().to_string())


if __name__ == "__main__":
    main()
