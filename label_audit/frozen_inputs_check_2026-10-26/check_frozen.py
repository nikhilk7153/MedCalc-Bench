"""
Checks the frozen parts of the corrected test set (notes, questions, time-point instructions, one-shot examples) for
mistakes. Writes one CSV per check next to this script; nothing else is changed.
"""
import ast, collections, json, re, sys
import pandas as pd

ROOT = "/Users/nikhilkhandekar/Downloads/MedCalc-Bench-version-1.0 2"
sys.path.insert(0, f"{ROOT}/session_archive_2026-10-22")
from regen_labels import compute, info


def compute_params(cid, params):
    """compute() for inputs already keyed by the function's parameter names, as in the one-shot file"""
    if set(params) <= set(info[cid]) - {"file path", "explanation function", "calculator name", "type", "question"}:
        return compute(cid, params)
    by_param = {v: k for k, v in info[cid].items()}
    return compute(cid, {by_param[k]: v for k, v in params.items()})

OUT = f"{ROOT}/label_audit/frozen_inputs_check_2026-10-26"
data = pd.read_csv(f"{ROOT}/dataset/test_data_10_22_corrected.csv")
one_shot = json.load(open(f"{ROOT}/final_files_2026-10-16/one_shot_10_17_2026_corrected.json"))
v10 = pd.read_csv(f"{ROOT}/dataset/test_data.csv")

# 1. note hygiene
MOJIBAKE = re.compile("Ã|â€|Â|�|&[a-z]+;|&#\\d+;")
hygiene = []
for _, r in data.iterrows():
    note = str(r["Patient Note"])
    issues = []
    if len(note) < 200:
        issues.append(f"short note ({len(note)} chars)")
    if MOJIBAKE.search(note):
        issues.append(f"encoding residue {sorted(set(MOJIBAKE.findall(note)))}")
    if not re.search(r"[.!?)\]\"'%]\s*$", note):
        issues.append(f"ends mid-sentence: ...{note[-60:]!r}")
    if issues:
        hygiene.append({"Row Number": r["Row Number"], "Calculator": r["Calculator Name"], "Issues": "; ".join(issues)})
by_id = data.groupby("Note ID")["Patient Note"].nunique()
for note_id in by_id[by_id > 1].index:
    hygiene.append({"Row Number": "", "Calculator": "", "Issues": f"Note ID {note_id} has {by_id[note_id]} different texts"})
pd.DataFrame(hygiene).to_csv(f"{OUT}/note_hygiene.csv", index=False)

# 2. questions and time-point instructions, one line per calculator
v10_q = v10.groupby("Calculator ID")["Question"].agg(lambda s: sorted(set(s)))
questions = []
for cid, g in data.groupby("Calculator ID"):
    qs = sorted(set(g["Question"]))
    questions.append({"Calculator ID": cid, "Calculator": g["Calculator Name"].iloc[0], "Rows": len(g),
                      "Distinct Questions": len(qs), "Question": " || ".join(qs),
                      "Same As v1.0": qs == v10_q.get(cid, []),
                      "Rows With Time Point": int(g["Time Point Instruction"].notna().sum())})
pd.DataFrame(questions).to_csv(f"{OUT}/questions.csv", index=False)

# 3. every number in a row's inputs should be written in its note
def number_forms(x):
    x = float(x)
    forms = {f"{x:g}", f"{x:.1f}", f"{x:.2f}", str(x)}
    if x == int(x):
        forms.add(str(int(x)))
        forms.add(f"{int(x):,}")
    return forms

def in_note(x, note):
    return any(re.search(rf"(?<![\d.]){re.escape(f)}(?![\d])", note) for f in number_forms(x))

grounding = []
for _, r in data.iterrows():
    if r["Category"] not in {"lab", "physical", "dosage", "risk", "severity", "diagnosis"}:
        continue
    note = str(r["Patient Note"]).replace(" ", " ")
    for key, value in ast.literal_eval(r["Relevant Entities"]).items():
        if isinstance(value, list) and len(value) == 2 and isinstance(value[0], (int, float)) and not isinstance(value[0], bool):
            if not in_note(value[0], note):
                grounding.append({"Row Number": r["Row Number"], "Calculator": r["Calculator Name"], "Input": key,
                                  "Value": value[0], "Unit": value[1]})
pd.DataFrame(grounding).to_csv(f"{OUT}/inputs_not_in_note.csv", index=False)

# 4. one-shot examples: answer reproduces from its inputs; example note is not a test note
test_notes = set(data["Patient Note"].str.strip())
examples = []
for cid, ex in one_shot.items():
    row = {"Calculator ID": cid, "Calculator": info[cid]["calculator name"],
           "Example Answer": ex["Response"]["answer"], "In Test Set": ex["Patient Note"].strip() in test_notes,
           "Test Rows With This Note": ";".join(f"{n} ({c})" for n, c in zip(data.loc[data["Patient Note"].str.strip() == ex["Patient Note"].strip(), "Row Number"], data.loc[data["Patient Note"].str.strip() == ex["Patient Note"].strip(), "Calculator ID"]))}
    try:
        row["Recomputed"] = compute_params(cid, ex["input_parameters"])["Answer"]
    except Exception as e:
        row["Recomputed"] = f"ERROR {type(e).__name__}: {e}"
    try:
        row["Matches"] = abs(float(row["Recomputed"]) - float(row["Example Answer"])) <= 0.05 * abs(float(row["Example Answer"])) + 1e-9
    except (TypeError, ValueError):
        row["Matches"] = str(row["Recomputed"]) == str(row["Example Answer"])
    examples.append(row)
pd.DataFrame(examples).to_csv(f"{OUT}/one_shot_examples.csv", index=False)

print(f"note hygiene: {len(hygiene)} flags; questions: {len(questions)} calculators; "
      f"inputs not in note: {len(grounding)} values in {len({g['Row Number'] for g in grounding})} rows; "
      f"one-shot: {sum(not e['Matches'] for e in examples)} answer mismatches, {sum(e['In Test Set'] for e in examples)} in test set")
