import csv, io, ast, json, random, re, shutil, sys
sys.path.insert(0, "/private/tmp/claude-501/-Users-nikhilkhandekar-Downloads-MedCalc-Bench-version-1-0-2/a77746bf-773a-4b91-94a5-3a6c7b678492/scratchpad")
from qtc_check import ROOT, CALC, label

P = f"{ROOT}/dataset/test_data_10_22.csv"
BACKUP = "/private/tmp/claude-501/-Users-nikhilkhandekar-Downloads-MedCalc-Bench-version-1-0-2/a77746bf-773a-4b91-94a5-3a6c7b678492/scratchpad/test_data_10_22.before_qtc_dedupe.csv"
GROUPS = [[184, 191, 197], [190, 193], [616, 620], [625, 626], [643, 647], [674, 683], [675, 680]]

raw = open(P, encoding="utf-8", newline="").read()
rows = list(csv.reader(io.StringIO(raw)))


def serialize(rows):
    # the file separates records with \r\n and has no terminator after the last one
    buf = io.StringIO(); csv.writer(buf, lineterminator="\r\n").writerows(rows)
    s = buf.getvalue()
    return s[:-2] if not raw.endswith("\r\n") else s


assert serialize(rows) == raw, "csv round trip would change the file; stopping"
header = rows[0]; col = {c: i for i, c in enumerate(header)}
by_row = {int(r[col["Row Number"]]): r for r in rows[1:]}

def hr_of(r):
    return ast.literal_eval(r[col["Relevant Entities"]])["Heart Rate or Pulse"][0]

qtc = [r for r in rows[1:] if int(r[col["Calculator ID"]]) in CALC]
used = {hr_of(r) for r in qtc}
one_shot = json.load(open(f"{ROOT}/final_files_2026-10-16/one_shot_10_17_2026.json"))
for cid in CALC:
    m = re.search(r"heart rate of (\d+) bpm", one_shot[str(cid)]["Patient Note"])
    if m:
        used.add(int(m.group(1)))
lo_hr, hi_hr = min(hr_of(r) for r in qtc), max(hr_of(r) for r in qtc)
rng = random.Random(20261022)

changes = []
for group in GROUPS:
    first = by_row[group[0]]
    for rn in group[1:]:
        r = by_row[rn]
        assert r[col["Patient Note"]] == first[col["Patient Note"]] and r[col["Question"]] == first[col["Question"]]
        cid, old_hr = int(r[col["Calculator ID"]]), hr_of(r)
        qt = ast.literal_eval(r[col["Relevant Entities"]])["QT Interval"][0]
        new_hr = rng.choice([h for h in range(lo_hr, hi_hr + 1) if h not in used])
        used.add(new_hr)
        expl, ans, lo, hi = label(cid, new_hr, qt)
        old_note = r[col["Patient Note"]]
        assert old_note.count(f"heart rate of {old_hr} bpm") == 1
        r[col["Patient Note"]] = old_note.replace(f"heart rate of {old_hr} bpm", f"heart rate of {new_hr} bpm")
        assert r[col["Relevant Entities"]].count(f"[{old_hr}, 'beats per minute']") == 1
        r[col["Relevant Entities"]] = r[col["Relevant Entities"]].replace(f"[{old_hr}, 'beats per minute']", f"[{new_hr}, 'beats per minute']")
        r[col["Ground Truth Answer"]], r[col["Lower Limit"]], r[col["Upper Limit"]] = str(ans), str(lo), str(hi)
        r[col["Ground Truth Explanation"]] = expl.strip()
        changes.append((rn, r[col["Calculator Name"]], old_hr, new_hr, ans))

shutil.copy(P, BACKUP)
open(P, "w", encoding="utf-8", newline="").write(serialize(rows))
for c in changes:
    print(f"row {c[0]} (line {c[0] + 1}) {c[1]}: heart rate {c[2]} -> {c[3]} bpm, new answer {c[4]}")
