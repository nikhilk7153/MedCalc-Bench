"""
For every non-template test row, finds the earlier copies of its note (same Note ID in v1.0, 10/15/2024, 1/17/2025, 1_31,
Verified and the training set) and, when our text matches none of them, lists what differs from the closest copy.
Writes note_provenance.csv (one line per row) and note_edits.csv (one line per differing passage).
"""
import difflib, re, zipfile
import pandas as pd

ROOT = "/Users/nikhilkhandekar/Downloads/MedCalc-Bench-version-1.0 2"
OUT = f"{ROOT}/label_audit/frozen_inputs_check_2026-10-26"
SOURCES = {"v1.0": "dataset/test_data.csv", "10_15_2024": "final_files_2026-10-16/test_data_10_15_2024.csv",
           "1_17": "dataset/test_data_1_17_2025.csv", "1_31": "dataset/test_set_1_31.csv",
           "verified": "dataset/test_verified.csv", "train_verified": "dataset/train_verified.csv"}

data = pd.read_csv(f"{ROOT}/dataset/test_data_10_22_corrected.csv")
copies = {}
for name, path in SOURCES.items():
    for nid, note in pd.read_csv(f"{ROOT}/{path}", usecols=["Note ID", "Patient Note"]).itertuples(index=False):
        copies.setdefault(str(nid), {}).setdefault(note, name)
with zipfile.ZipFile(f"{ROOT}/dataset/train_data.csv.zip") as z:
    train = pd.read_csv(z.open(z.namelist()[0]), usecols=["Note ID", "Patient Note"])
for nid, note in train.itertuples(index=False):
    copies.setdefault(str(nid), {}).setdefault(note, "train")

provenance, edits = [], []
for _, r in data[data["Note Type"] != "Template"].iterrows():
    note, found = r["Patient Note"], copies.get(str(r["Note ID"]), {})
    line = {"Row Number": r["Row Number"], "Calculator": r["Calculator Name"], "Note ID": r["Note ID"]}
    exact = sorted({src for text, src in found.items() if text.strip() == note.strip()})
    if exact:
        provenance.append({**line, "Status": "identical", "Sources": ";".join(exact)})
        continue
    if not found:
        provenance.append({**line, "Status": "no earlier copy", "Sources": ""})
        continue
    text, src = max(found.items(), key=lambda t: difflib.SequenceMatcher(None, t[0], note, autojunk=False).ratio())
    sm = difflib.SequenceMatcher(None, text, note, autojunk=False)
    passages = [(text[i1:i2], note[j1:j2], note[max(0, j1 - 80):j2 + 40]) for op, i1, i2, j1, j2 in sm.get_opcodes() if op != "equal"]
    numeric = any(re.search(r"\d", a + b) for a, b, _ in passages)
    provenance.append({**line, "Status": "differs (numbers)" if numeric else "differs (text only)", "Sources": src,
                       "Similarity": round(sm.ratio(), 4), "Passages": len(passages)})
    for a, b, context in passages:
        edits.append({**line, "Closest Copy": src, "Earlier Text": a, "Our Text": b, "Context In Our Note": context,
                      "Has Number": bool(re.search(r"\d", a + b))})

pd.DataFrame(provenance).to_csv(f"{OUT}/note_provenance.csv", index=False)
pd.DataFrame(edits).to_csv(f"{OUT}/note_edits.csv", index=False)
print(pd.DataFrame(provenance)["Status"].value_counts().to_string())
