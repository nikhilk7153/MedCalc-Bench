"""Recompute the label columns of a test set from each row's Relevant Entities with the verified calculator code,
using generate_csv.py's input mapping and answer/limit rules. Dry run unless --write is given."""
import ast, csv, importlib.util, io, json, os, shutil, sys, collections
import pandas as pd

ROOT = "/Users/nikhilkhandekar/Downloads/MedCalc-Bench-version-1.0 2"
SCRATCH = "/private/tmp/claude-501/-Users-nikhilkhandekar-Downloads-MedCalc-Bench-version-1-0-2/a77746bf-773a-4b91-94a5-3a6c7b678492/scratchpad"
CODE = f"{ROOT}/calculator_implementations_verified"
sys.path.insert(0, CODE)
from rounding import round_number
info = json.load(open(f"{CODE}/name_to_python.json"))
NUMERIC = {"lab", "lab test", "physical", "dosage"}
_modules = {}


def compute(cid, ents):
    if cid == "49":
        params = dict(ents)
    else:
        params = {info[cid][k]: (False if v == "False" else True if v == "True" else v) for k, v in ents.items()}
    fp = info[cid]["file path"]
    if fp not in _modules:
        spec = importlib.util.spec_from_file_location(os.path.splitext(os.path.basename(fp))[0], f"{CODE}/{fp}")
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); _modules[fp] = m
    return getattr(_modules[fp], info[cid]["explanation function"])(params)


def labels(category, out):
    if category in NUMERIC:
        a = float(out["Answer"])
        lo, hi = (a * 1.05, a * 0.95) if a < 0 else (a * 0.95, a * 1.05)
        return str(float(round_number(a))), str(float(round_number(lo))), str(float(round_number(hi)))
    return str(out["Answer"]), str(out["Answer"]), str(out["Answer"])


def same(a, b):
    try:
        return abs(float(a) - float(b)) < 1e-9
    except ValueError:
        return str(a).strip() == str(b).strip()


if __name__ == "__main__":
    rel = sys.argv[1]; write = "--write" in sys.argv
    path = f"{ROOT}/{rel}"
    raw = open(path, encoding="utf-8", newline="").read()
    term = "\r\n" if "\r\n" in raw else "\n"

    def serialize(rows):
        b = io.StringIO(); csv.writer(b, lineterminator=term).writerows(rows); s = b.getvalue()
        return s if raw.endswith(term) else s[:-len(term)]

    rows = list(csv.reader(io.StringIO(raw)))
    assert serialize(rows) == raw, "csv round trip would change the file"
    col = {c: i for i, c in enumerate(rows[0])}
    diffs = []
    for r in rows[1:]:
        cid = r[col["Calculator ID"]].strip()
        ents = ast.literal_eval(r[col["Relevant Entities"]])
        try:
            out = compute(cid, ents)
        except Exception as e:
            diffs.append((r[col["Row Number"]], r[col["Calculator Name"]], "code error", str(e)[:80])); continue
        ans, lo, hi = labels(r[col["Category"]], out)
        expl = out["Explanation"].strip()
        what = [name for name, new, old in [("answer", ans, r[col["Ground Truth Answer"]]), ("lower", lo, r[col["Lower Limit"]]),
                                            ("upper", hi, r[col["Upper Limit"]])] if not same(new, old)]
        if expl != r[col["Ground Truth Explanation"]].strip():
            what.append("explanation")
        if what:
            diffs.append((r[col["Row Number"]], r[col["Calculator Name"]], ",".join(what), f"{r[col['Ground Truth Answer']]} -> {ans}"))
            if write:
                r[col["Ground Truth Answer"]], r[col["Lower Limit"]], r[col["Upper Limit"]], r[col["Ground Truth Explanation"]] = ans, lo, hi, expl
    by_calc = collections.Counter((d[1], d[2]) for d in diffs)
    print(f"{rel}: {len(rows) - 1} rows; rows whose recomputed labels differ: {len(diffs)}")
    for (calc, what), n in sorted(by_calc.items()):
        print(f"  {calc[:45]:45s} {what:32s} {n}")
    answer_changes = [d for d in diffs if "answer" in d[2] or d[2] == "code error"]
    print(f"\nrows whose ANSWER would change ({len(answer_changes)}):")
    for d in answer_changes:
        print(f"  row {d[0]:>4} {d[1][:40]:40s} {d[3]}")
    if write and diffs:
        shutil.copy(path, f"{SCRATCH}/{os.path.basename(rel)}.before_regen")
        open(path, "w", encoding="utf-8", newline="").write(serialize(rows))
        print("written; backup at", f"{SCRATCH}/{os.path.basename(rel)}.before_regen")
