"""Final label check: ask gpt-6-astra whether each row's ground-truth answer is correct (one call per row).

Run: OPENAI_API_KEY=... python3 equation_audit/verify_dataset.py <csv> <tag>
The time point comes from each question: admission values when the question says so (as in MedCalc-Bench Verified),
otherwise whatever the note clearly documents. Writes final_<tag>_verify.jsonl and final_<tag>_verify.csv in
equation_audit/. Reruns resume from the JSONL. AUDIT_WORKERS sets concurrency (default 3).
"""
import sys
from pathlib import Path

import pandas as pd

import run_gpt6_astra as audit

ROOT = Path(__file__).resolve().parent

PROMPT = """You are checking one example from a medical calculation benchmark. You get the patient note, the question,
the benchmark's ground-truth answer with its accepted range, and the benchmark's step-by-step explanation.
Decide whether the ground-truth answer is correct. Check that every input was taken correctly from the note, that units
were converted correctly, that the standard published formula or scoring rule was used, and that the arithmetic and
rounding are right. Use the time point the question specifies; if it specifies none, accept values the note clearly
documents for the encounter being scored. Criteria the note does not mention are treated as absent.
The answer counts as correct if the correct value falls inside the accepted range.
Respond with JSON only:
{"verdict": "correct" | "incorrect" | "ambiguous",
 "error_type": "wrong_equation" | "input_extraction" | "unit_conversion" | "arithmetic_or_rounding" | "note_missing_or_conflicting_input" | "out_of_domain" | "none",
 "corrected_answer": "...",
 "explanation": "..."}
Use "ambiguous" only when the note does not allow a single defensible answer (a required value is missing, values
conflict, or the patient is outside the calculator's intended population)."""


def verify(row, path):
    user = (f"Patient note:\n{row['Patient Note']}\n\nQuestion:\n{row['Question']}\n\n"
            f"Ground-truth answer: {row['Ground Truth Answer']} "
            f"(accepted range {row['Lower Limit']} to {row['Upper Limit']})\n\n"
            f"Ground-truth explanation:\n{row['Ground Truth Explanation']}")
    try:
        out, tokens = audit.chat(PROMPT, user)
    except Exception as e:
        print(f"row {row['Row Number']}: {e}", file=sys.stderr)
        return
    audit.append(path, {'Row Number': int(row['Row Number']), **out, 'Tokens': tokens})


def main():
    source, tag = Path(sys.argv[1]), sys.argv[2]
    df = pd.read_csv(source, dtype=str, keep_default_na=False)
    rows = df.to_dict('records')
    by_row = {int(r['Row Number']): r for r in rows}
    print(f'{len(rows)} rows from {source.name} to check')

    checked = audit.run(rows, verify, ROOT / f'final_{tag}_verify.jsonl')
    out = pd.DataFrame([{**{k: by_row[n][k] for k in ('Row Number', 'Calculator ID', 'Calculator Name', 'Note ID', 'Note Type',
                                                          'Ground Truth Answer', 'Lower Limit', 'Upper Limit')},
                         'Verdict': v.get('verdict', ''), 'Error Type': v.get('error_type', ''),
                         'Corrected Answer': v.get('corrected_answer', ''), 'Explanation': v.get('explanation', ''),
                         'Question': by_row[n]['Question'], 'Patient Note': by_row[n]['Patient Note'],
                         'Ground Truth Explanation': by_row[n]['Ground Truth Explanation']}
                        for n, v in sorted(checked.items())])
    order = {'incorrect': 0, 'ambiguous': 1, 'correct': 2}
    out = out.sort_values(['Verdict', 'Calculator Name'], key=lambda s: s.map(order) if s.name == 'Verdict' else s)
    out.to_csv(ROOT / f'final_{tag}_verify.csv', index=False, encoding='utf-8-sig')
    print(out['Verdict'].value_counts().to_string())
    missing = len(rows) - len(checked)
    if missing:
        print(f'{missing} rows failed to run; rerun the script to retry them.')


if __name__ == '__main__':
    main()
