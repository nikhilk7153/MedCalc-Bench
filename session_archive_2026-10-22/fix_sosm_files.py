import ast, csv, importlib.util, io, json, os, shutil, sys
import pandas as pd

ROOT = "/Users/nikhilkhandekar/Downloads/MedCalc-Bench-version-1.0 2"
SCRATCH = "/private/tmp/claude-501/-Users-nikhilkhandekar-Downloads-MedCalc-Bench-version-1-0-2/a77746bf-773a-4b91-94a5-3a6c7b678492/scratchpad"
CODE = f"{ROOT}/calculator_implementations_verified"
sys.path.insert(0, CODE)
info = json.load(open(f"{CODE}/name_to_python.json"))["30"]
spec = importlib.util.spec_from_file_location("sOsm", f"{CODE}/sOsm.py")
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
explain = getattr(module, info["explanation function"])

for rel in sys.argv[1:]:
    path = f"{ROOT}/{rel}"
    raw = open(path, encoding="utf-8", newline="").read()
    terminator = "\r\n" if "\r\n" in raw else "\n"

    def serialize(rows):
        buf = io.StringIO(); csv.writer(buf, lineterminator=terminator).writerows(rows); s = buf.getvalue()
        return s if raw.endswith(terminator) else s[:-len(terminator)]

    rows = list(csv.reader(io.StringIO(raw)))
    assert serialize(rows) == raw, f"{rel}: csv round trip would change the file"
    col = {c: i for i, c in enumerate(rows[0])}
    fixed = 0
    for r in rows[1:]:
        if r[col["Calculator ID"]].strip() != "30":
            continue
        ents = ast.literal_eval(r[col["Relevant Entities"]])
        result = explain({info[k]: v for k, v in ents.items()})
        old, new = r[col["Ground Truth Explanation"]], result["Explanation"].strip()
        if old.strip() == new:
            continue
        assert abs(float(result["Answer"]) - float(r[col["Ground Truth Answer"]])) < 1e-9, (rel, r[col["Row Number"]])
        old_lines, new_lines = old.strip().split("\n"), new.split("\n")
        assert len(old_lines) == len(new_lines) and all(a == b or a.startswith("Plugging") for a, b in zip(old_lines, new_lines)), (rel, r[col["Row Number"]])
        r[col["Ground Truth Explanation"]] = old.replace(old.strip(), new)  # keep any surrounding whitespace as it was
        fixed += 1
    backup = f"{SCRATCH}/{os.path.basename(rel)}.before_sosm"
    shutil.copy(path, backup)
    open(path, "w", encoding="utf-8", newline="").write(serialize(rows))
    a = pd.read_csv(backup, dtype=str, keep_default_na=False); b = pd.read_csv(path, dtype=str, keep_default_na=False)
    diff = a != b
    print(f"{rel}: fixed {fixed} explanations | rows changed {int(diff.any(axis=1).sum())} | columns changed {list(diff.columns[diff.any()])} | backup {backup}")
