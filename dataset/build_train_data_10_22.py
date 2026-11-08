"""
Builds the training set for the corrected benchmark from v1.0's training set (train_data.csv.zip).

A training row is removed when its calculator has a held-out note (a row of test_data_10_22_corrected.csv or the
calculator's one-shot example in one_shot_10_17_2026_corrected.json) that
- is the same text (ignoring whitespace), or
- is a near copy: an Extracted note with at least 90% of its 8-word shingles in the held-out note (the same case report
  with small edits, e.g. a changed unit or sodium value).
Notes that are held out for a different calculator stay, as do template notes that only share wording. Rows keep their
v1.0 Row Number and labels.

Questions are then set to the final test set's question for the same calculator. Steroid Conversion (24) names a
different drug pair in each row, so its questions keep their drug pair and only lose v1.0's trailing space, which is
the test set's wording.

Writes train_data_10_22_v1.0_labels.csv (v1.0 labels; regenerate_train_data_10_22.py turns it into train_data_10_22.csv),
train_data_10_22_removed.csv (each removed row, its reason and the held-out row)
and train_data_10_22_questions.csv (each calculator's v1.0 and new question).
"""
import collections
import json
import os
import re
import zipfile
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
NEAR = 0.9

norm = lambda s: re.sub(r"\s+", " ", str(s)).strip()


def shingles(text, k=8):
    words = text.lower().split()
    return {" ".join(words[i:i + k]) for i in range(max(1, len(words) - k + 1))}


with zipfile.ZipFile(f"{ROOT}/train_data.csv.zip") as z:
    train = pd.read_csv(z.open("train_data.csv"))
test = pd.read_csv(f"{ROOT}/test_data_10_22_corrected.csv")
one_shot = json.load(open(f"{ROOT}/../final_files_2026-10-16/one_shot_10_17_2026_corrected.json"))

held = collections.defaultdict(list)  # calculator ID -> [(source, note, shingles)]
for _, r in test.iterrows():
    note = norm(r["Patient Note"])
    held[int(r["Calculator ID"])].append((f"test row {r['Row Number']}", note, shingles(note)))
for cid, example in one_shot.items():
    note = norm(example["Patient Note"])
    held[int(cid)].append(("one-shot example", note, shingles(note)))

removed = []
for i, r in train.iterrows():
    note = norm(r["Patient Note"])
    candidates = held.get(int(r["Calculator ID"]), [])
    exact = [src for src, text, _ in candidates if text == note]
    if exact:
        removed.append({"index": i, "Reason": "same note", "Held-Out Match": "; ".join(exact), "Containment": 1.0})
        continue
    if r["Note Type"] != "Extracted":
        continue
    s = shingles(note)
    best = max(((len(s & sh) / max(1, min(len(s), len(sh))), src) for src, _, sh in candidates), default=(0, ""))
    if best[0] >= NEAR:
        removed.append({"index": i, "Reason": "near copy (same case report, small edits)", "Held-Out Match": best[1],
                        "Containment": round(best[0], 3)})

drop = [x["index"] for x in removed]
out = train.drop(index=drop).copy()

STEROID = re.compile(r"^Based on the patient's dose of .+, what is the equivalent dosage in mg of .+\?$")
test_question = test.groupby("Calculator ID")["Question"].agg(lambda s: sorted(set(s)))
questions = []
for cid, rows in out.groupby("Calculator ID"):
    old = sorted(set(rows["Question"]))
    if cid == 24:
        new = rows["Question"].str.strip()
        assert new.str.match(STEROID).all() and test["Question"][test["Calculator ID"] == 24].str.match(STEROID).all()
    else:
        assert len(test_question[cid]) == 1, f"calculator {cid} has several test questions"
        new = test_question[cid][0]
    out.loc[rows.index, "Question"] = new
    questions.append({"Calculator ID": cid, "Calculator Name": rows["Calculator Name"].iloc[0], "Rows": len(rows),
                      "Changed": sorted(set(out.loc[rows.index, "Question"])) != old,
                      "v1.0 Question": " || ".join(old) if cid != 24 else f"{len(old)} drug pairs, with a trailing space",
                      "New Question": new if cid != 24 else "same drug pairs, trailing space removed"})
pd.DataFrame(questions).to_csv(f"{ROOT}/train_data_10_22_questions.csv", index=False)

out.to_csv(f"{ROOT}/train_data_10_22_v1.0_labels.csv", index=False)

log = pd.DataFrame(removed)
log = pd.concat([train.loc[log["index"], ["Row Number", "Calculator ID", "Calculator Name", "Note ID", "Note Type"]].reset_index(drop=True),
                 log.drop(columns="index")], axis=1)
log.to_csv(f"{ROOT}/train_data_10_22_removed.csv", index=False)

print(f"v1.0 training rows: {len(train)}; removed {len(drop)} "
      f"({(log['Reason'] == 'same note').sum()} same note, {(log['Reason'] != 'same note').sum()} near copies); kept {len(out)}")
print(f"kept rows whose note is held out for a different calculator: "
      f"{out['Patient Note'].map(norm).isin({t for v in held.values() for _, t, _ in v}).sum()}")
