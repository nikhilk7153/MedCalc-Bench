"""Audit the Verified notes added to test_data_corrected.csv with the same two-pass gpt-6-astra check.

Run: OPENAI_API_KEY=... python3 equation_audit/audit_new_verified_rows.py (after build_corrected_test.py).
Covers rows replaced with a new Verified note or added as a top-up; template-based notes are skipped.
Writes new_rows_pass1.jsonl, new_rows_pass2.jsonl and new_rows_audit.csv. Reruns resume from the JSONL caches.
"""
from pathlib import Path

import pandas as pd

import run_gpt6_astra as audit

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / 'dataset'
NEW_NOTE_ACTIONS = {'Replaced with new Verified note (same calculator)', 'Added new Verified note (calculator top-up)'}


def main():
    rows = pd.read_csv(DATA / 'test_data_corrected.csv', dtype=str)
    log = pd.read_csv(DATA / 'test_data_corrected_changes.csv', dtype=str)
    targets = set(log.loc[log['Action'].isin(NEW_NOTE_ACTIONS), 'Row Number'])
    rows = rows[rows['Row Number'].isin(targets) & rows['Note Type'].ne('Template')]
    rows = rows.to_dict('records')
    by_row = {int(r['Row Number']): r for r in rows}
    print(f'{len(rows)} new non-template Verified rows to audit')

    solved = audit.run(rows, audit.solve, ROOT / 'new_rows_pass1.jsonl')
    failed = [by_row[n] for n, s in solved.items() if s['Correct'] != 1]
    judged = audit.run(failed, audit.adjudicate, ROOT / 'new_rows_pass2.jsonl', solved)

    src = log.set_index('Row Number')
    out = []
    for n, s in sorted(solved.items()):
        j = judged.get(n, {})
        verdict = j.get('verdict', 'agrees_with_label' if s['Correct'] == 1 else '')
        out.append({'Row Number': n, 'Verified Row Number': src.loc[str(n), 'Verified Row Number'],
                    'Calculator Name': s['Calculator Name'], 'Note ID': by_row[n]['Note ID'],
                    'Ground Truth Answer': s['Ground Truth Answer'], 'Model Answer': s['Model Answer'],
                    'Verdict': verdict, 'Label Error Type': j.get('label_error_type', ''),
                    'Corrected Answer': j.get('corrected_answer', ''), 'Adjudication': j.get('explanation', ''),
                    'Problem': verdict in ('label_wrong', 'both_wrong', 'ambiguous')})
    out = pd.DataFrame(out).sort_values(['Problem', 'Calculator Name'], ascending=[False, True])
    out.to_csv(ROOT / 'new_rows_audit.csv', index=False, encoding='utf-8-sig')
    print(out['Verdict'].value_counts().to_string())
    missing = len(rows) - len(solved)
    if missing:
        print(f'{missing} rows failed to run; rerun the script to retry them.')


if __name__ == '__main__':
    main()
