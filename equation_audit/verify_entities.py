"""Check that each row's Relevant Entities match its patient note, with gpt-6-astra (one call per row).

Run: OPENAI_API_KEY=... python3 equation_audit/verify_entities.py <csv> <tag>
Labels are computed from the entities by the calculator code, so this checks extraction only, not the label.
The model gets the note, question, extracted entities and the calculator's full entity list (name_to_python.json,
$MEDCALC_IMPL or medcalc_verified_code/calculator_implementations). Template rows are skipped.
Writes entities_<tag>.jsonl and entities_<tag>.csv in equation_audit/. Reruns resume from the JSONL.
AUDIT_ADMISSION=1 enforces admission/initial-presentation values before any treatment, whatever the question says.
"""
import json
import os
import sys
from pathlib import Path

import pandas as pd

import run_gpt6_astra as audit

ROOT = Path(__file__).resolve().parent
IMPLEMENTATIONS = Path(os.environ.get('MEDCALC_IMPL', Path(__file__).resolve().parent.parent / 'medcalc_verified_code/calculator_implementations'))
META_KEYS = {'file path', 'explanation function', 'calculator name', 'type', 'question'}

PROMPT = """You are checking the extracted inputs for one example of a medical calculation benchmark.
You get the patient note, the question, the calculator's input names, and the inputs extracted from the note.
Check every extracted input against the note: the value, the unit, and that it comes from the time point the question
specifies (if it specifies none, any value the note clearly documents for the encounter being scored is acceptable).
For true/false inputs, check the finding is documented as stated. Also check the calculator's other inputs: a finding
the note documents as present but that is missing from the extraction or extracted as false is an omission.
Inputs the note does not mention are correctly absent or false. Do not compute or judge the final answer.
Respond with JSON only:
{"all_match": true | false,
 "problems": [{"input": "...", "extracted": "...", "note_says": "...",
               "issue": "wrong_value" | "wrong_unit" | "wrong_time_point" | "not_in_note" | "omitted" | "conflicting_values_in_note"}]}
Return an empty "problems" list when everything matches."""

# AUDIT_ADMISSION=1: always use admission values before treatment, even when the question names no time point
ADMISSION_RULE = ("\nTime point: use the patient's values at the time of admission or initial presentation, before any "
                  "treatment, regardless of the question's wording; values documented only later, at follow-up or "
                  "after treatment started are not admission values.")
if os.environ.get('AUDIT_ADMISSION') == '1':
    PROMPT += ADMISSION_RULE


def main():
    source, tag = Path(sys.argv[1]), sys.argv[2]
    info = json.load(open(IMPLEMENTATIONS / 'name_to_python.json'))
    df = pd.read_csv(source, dtype=str, keep_default_na=False)
    df = df[df['Note Type'] != 'Template']
    rows = df.to_dict('records')
    by_row = {int(r['Row Number']): r for r in rows}

    def check(row, path):
        inputs = [k for k in info.get(row['Calculator ID'], {}) if k not in META_KEYS]
        user = (f"Calculator: {row['Calculator Name']}\nCalculator inputs: {inputs}\n\n"
                f"Question:\n{row['Question']}\n\nPatient note:\n{row['Patient Note']}\n\n"
                f"Extracted inputs:\n{row['Relevant Entities']}")
        try:
            out, tokens = audit.chat(PROMPT, user)
        except Exception as e:
            print(f"row {row['Row Number']}: {e}", file=sys.stderr)
            return
        audit.append(path, {'Row Number': int(row['Row Number']), **out, 'Tokens': tokens})

    print(f'{len(rows)} non-template rows from {source.name} to check')
    checked = audit.run(rows, check, ROOT / f'entities_{tag}.jsonl')
    out = []
    for n, c in sorted(checked.items()):
        r, problems = by_row[n], c.get('problems') or []
        out.append({'Row Number': n, 'Calculator ID': r['Calculator ID'], 'Calculator Name': r['Calculator Name'],
                    'Note ID': r['Note ID'], 'Note Type': r['Note Type'],
                    'Entities Match': bool(c.get('all_match')) and not problems, 'Problem Count': len(problems),
                    'Issues': '; '.join(sorted({p.get('issue', '') for p in problems})),
                    'Problems': json.dumps(problems, ensure_ascii=False),
                    'Relevant Entities': r['Relevant Entities'], 'Question': r['Question'], 'Patient Note': r['Patient Note']})
    out = pd.DataFrame(out).sort_values(['Entities Match', 'Calculator Name'])
    out.to_csv(ROOT / f'entities_{tag}.csv', index=False, encoding='utf-8-sig')
    print(out['Entities Match'].value_counts().to_string())
    missing = len(rows) - len(checked)
    if missing:
        print(f'{missing} rows failed to run; rerun the script to retry them.')


if __name__ == '__main__':
    main()
