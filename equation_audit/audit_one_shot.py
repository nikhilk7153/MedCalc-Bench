"""Check whether the one-shot examples' ground-truth answers are correct with gpt-6-astra (one call per example).

Run: OPENAI_API_KEY=... python3 equation_audit/audit_one_shot.py [current|v1.0]
Reads ~/Documents/MedCalc-Bench-Verified/datasets/one_shot_data.csv (current, default) or one_shot_data_v1.0.csv
(the original v1.0 examples); neither is modified. Covers all 55 examples.
The model sees the note, question, label, accepted range and label explanation, and judges the label directly.
Writes one_shot{tag}_verify.jsonl and one_shot{tag}_verify.csv (tag is '' or '_v1.0'). Reruns resume from the JSONL.
"""
import sys
from pathlib import Path

import pandas as pd

import run_gpt6_astra as audit

ROOT = Path(__file__).resolve().parent
SOURCES = {'current': ('one_shot_data.csv', ''), 'v1.0': ('one_shot_data_v1.0.csv', '_v1.0')}
DATASETS = Path.home() / 'Documents/MedCalc-Bench-Verified/datasets'

VERIFY_PROMPT = """You are checking one example from a medical calculation benchmark. You get the patient note, the question,
the benchmark's ground-truth answer with its accepted range, and the benchmark's step-by-step explanation.
Decide whether the ground-truth answer is correct. Check that every input was taken correctly from the note, using the
patient's values at the time of admission or initial presentation before any treatment; that units were converted
correctly; that the standard published formula or scoring rule was used; and that the arithmetic and rounding are right.
The answer counts as correct if the correct value falls inside the accepted range.
Respond with JSON only:
{"verdict": "correct" | "incorrect" | "ambiguous",
 "error_type": "wrong_equation" | "input_extraction" | "unit_conversion" | "arithmetic_or_rounding" | "note_missing_or_conflicting_input" | "none",
 "corrected_answer": "...",
 "explanation": "..."}
Use "ambiguous" when the note does not allow a single defensible answer."""


def verify(row, path):
    user = (f"Patient note:\n{row['Patient Note']}\n\nQuestion:\n{row['Question']}\n\n"
            f"Ground-truth answer: {row['Ground Truth Answer']} "
            f"(accepted range {row['Lower Limit']} to {row['Upper Limit']})\n\n"
            f"Ground-truth explanation:\n{row['Ground Truth Explanation']}")
    try:
        out, tokens = audit.chat(VERIFY_PROMPT, user)
    except Exception as e:
        print(f"row {row['Row Number']}: {e}", file=sys.stderr)
        return
    audit.append(path, {'Row Number': int(row['Row Number']), **out, 'Tokens': tokens})


def main():
    name, tag = SOURCES[sys.argv[1] if len(sys.argv) > 1 else 'current']
    df = pd.read_csv(DATASETS / name, dtype=str)
    rows = df.to_dict('records')
    by_row = {int(r['Row Number']): r for r in rows}
    print(f'{len(rows)} rows from {name} to check')

    checked = audit.run(rows, verify, ROOT / f'one_shot{tag}_verify.jsonl')
    out = []
    for n, v in sorted(checked.items()):
        r = by_row[n]
        out.append({'Row Number': n, 'Calculator ID': r['Calculator ID'], 'Calculator Name': r['Calculator Name'],
                    'Note ID': r['Note ID'], 'Ground Truth Answer': r['Ground Truth Answer'],
                    'Lower Limit': r['Lower Limit'], 'Upper Limit': r['Upper Limit'],
                    'Verdict': v.get('verdict', ''), 'Error Type': v.get('error_type', ''),
                    'Corrected Answer': v.get('corrected_answer', ''), 'Explanation': v.get('explanation', ''),
                    'Question': r['Question'], 'Patient Note': r['Patient Note'],
                    'Ground Truth Explanation': r['Ground Truth Explanation']})
    out = pd.DataFrame(out)
    out = out.sort_values(['Verdict', 'Calculator Name'], key=lambda s: s.map({'incorrect': 0, 'ambiguous': 1, 'correct': 2})
                          if s.name == 'Verdict' else s)
    out.to_csv(ROOT / f'one_shot{tag}_verify.csv', index=False, encoding='utf-8-sig')
    print(out['Verdict'].value_counts().to_string())
    missing = len(rows) - len(checked)
    if missing:
        print(f'{missing} rows failed to run; rerun the script to retry them.')


if __name__ == '__main__':
    main()
