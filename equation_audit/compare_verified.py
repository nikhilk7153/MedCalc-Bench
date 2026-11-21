"""Report how MedCalc-Bench Verified handles the lab/physical rows flagged in v1.0.

Run: python3 equation_audit/compare_verified.py (after build_flagged_report.py).
Needs dataset/test_verified.csv (v1.0.8) from huggingface.co/datasets/nsk7153/MedCalc-Bench-Verified.
Rows are matched on Calculator ID + Note ID; regenerated template notes count as replaced.
Writes verified_changes.csv (flagged rows only) and verified_comparison.csv (all 567 rows).
"""
import difflib
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
KEY = ['Calculator ID', 'Note ID']

# Manual reading of the note diffs (see the Edit column). Everything else is classified automatically.
NOTE_EDIT_STILL_OPEN = {824: 'Edit only removes whitespace; albumin is still written as 3.5 mg/dL.'}
INPUTS_ADDED = {561: 'Total cholesterol added to the note; recomputed with corrected equation.',
                562: 'Total cholesterol added to the note; recomputed with corrected equation.',
                565: 'HDL added to the note; recomputed with corrected equation.',
                566: 'Note replaced with a different case; recomputed with corrected equation.',
                509: 'Pre-dialysis serum values removed and urine sodium/creatinine added.'}
QUESTIONABLE = {780: 'New label -7.2 is a negative albumin-corrected anion gap (Na 133, Cl 122, HCO3 20.2); physiologically implausible, review.'}
NOT_AN_ERROR = {367: 'Model preferred Katz over the label\'s Hillier correction; manual review found no error.'}


def num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def close(a, b):
    a, b = num(a), num(b)
    return a is not None and b is not None and abs(a - b) <= max(0.05 * abs(b), 0.01)


def first_num(s):
    m = re.search(r'-?\d+(?:\.\d+)?', str(s))
    return m.group() if m else None


def note_diff(a, b):
    if not isinstance(b, str) or a == b:
        return ''
    ops = difflib.SequenceMatcher(None, a, b).get_opcodes()
    return ' | '.join(f"'{a[i1:i2][:80]}' -> '{b[j1:j2][:80]}'" for op, i1, i2, j1, j2 in ops if op != 'equal')[:500]


def outcome(x):
    n = x['Row Number']
    if x['Flag'] == 'Clean':
        return 'Clean; label changed in Verified' if x['Label Change'] == 'Material' else 'Clean'
    if x['Verified Status'] == 'Removed or replaced':
        return 'Resolved: row removed or replaced'
    if n in QUESTIONABLE:
        return 'Changed, but new label questionable'
    if n in NOT_AN_ERROR:
        return 'Not an error (model-only flag)'
    if n in INPUTS_ADDED:
        return 'Resolved: missing inputs added and relabelled'
    if x['Label Change'] == 'Material' and x['Matches Our Correction']:
        return 'Resolved: label corrected to our value'
    if x['Verified Status'] == 'Note edited' and n not in NOTE_EDIT_STILL_OPEN:
        return 'Resolved: note edited'
    return 'Still open'


def main():
    v10 = pd.read_csv(ROOT.parent / 'dataset/test_data.csv', dtype=str)
    ver = pd.read_csv(ROOT.parent / 'dataset/test_verified.csv', dtype=str)
    df = pd.read_csv(ROOT / 'review_all_rows.csv', encoding='utf-8-sig', dtype=str)
    df = df.merge(v10[['Row Number', 'Note ID', 'Note Type']], on='Row Number')
    df = df.merge(ver[KEY + ['Row Number', 'Patient Note', 'Ground Truth Answer', 'Ground Truth Explanation']]
                  .rename(columns=lambda c: c if c in KEY else f'Verified {c}'), on=KEY, how='left')
    df['Row Number'] = df['Row Number'].astype(int)

    same = df['Patient Note'] == df['Verified Patient Note']
    found = df['Verified Row Number'].notna()
    df['Verified Status'] = 'Same note'
    df.loc[found & ~same, 'Verified Status'] = 'Note edited'
    df.loc[~found | (df['Note Type'].eq('Template') & ~same), 'Verified Status'] = 'Removed or replaced'
    kept = df['Verified Status'] != 'Removed or replaced'
    df['Label Change'] = ''
    df.loc[kept, 'Label Change'] = df[kept].apply(
        lambda x: 'Material' if not close(x['Ground Truth Answer'], x['Verified Ground Truth Answer'])
        else ('Precision only' if num(x['Ground Truth Answer']) != num(x['Verified Ground Truth Answer']) else 'None'), axis=1)
    df['Matches Our Correction'] = kept & df['Flag'].ne('Clean') & df.apply(
        lambda x: close(first_num(x['Model Corrected Answer']), x['Verified Ground Truth Answer'])
        or close(first_num(x['Manual Corrected Answer']), x['Verified Ground Truth Answer']), axis=1)
    df['Note Edit'] = df.apply(lambda x: note_diff(x['Patient Note'], x['Verified Patient Note'])
                               if x['Verified Status'] == 'Note edited' else '', axis=1)
    df['Outcome'] = df.apply(outcome, axis=1)
    comments = {**NOTE_EDIT_STILL_OPEN, **INPUTS_ADDED, **QUESTIONABLE, **NOT_AN_ERROR}
    df['Comment'] = df['Row Number'].map(comments).fillna('')

    cols = ['Row Number', 'Verified Row Number', 'Calculator ID', 'Calculator Name', 'Flag', 'Problem', 'Outcome',
            'Verified Status', 'Ground Truth Answer', 'Verified Ground Truth Answer', 'Label Change',
            'Model Corrected Answer', 'Manual Corrected Answer', 'Manual Reason', 'Model Explanation', 'Comment',
            'Note Edit', 'Question', 'Patient Note', 'Verified Patient Note', 'Ground Truth Explanation',
            'Verified Ground Truth Explanation']
    df = df[cols].rename(columns={'Row Number': 'v1.0 Row Number', 'Ground Truth Answer': 'v1.0 Ground Truth Answer',
                                  'Patient Note': 'v1.0 Patient Note', 'Ground Truth Explanation': 'v1.0 Ground Truth Explanation'})
    order = ['Still open', 'Changed, but new label questionable', 'Not an error (model-only flag)',
             'Resolved: label corrected to our value', 'Resolved: missing inputs added and relabelled',
             'Resolved: note edited', 'Resolved: row removed or replaced', 'Clean; label changed in Verified', 'Clean']
    df = df.sort_values(['Outcome', 'Calculator Name', 'v1.0 Row Number'],
                        key=lambda s: s.map(order.index) if s.name == 'Outcome' else s)
    df.to_csv(ROOT / 'verified_comparison.csv', index=False, encoding='utf-8-sig')
    df[df['Flag'] != 'Clean'].to_csv(ROOT / 'verified_changes.csv', index=False, encoding='utf-8-sig')

    print(df['Verified Status'].value_counts().to_string(), '\n')
    print(pd.crosstab(df['Outcome'], df['Flag'], margins=True).reindex(order + ['All']).dropna(how='all').to_string())
    print('\nwrote verified_changes.csv and verified_comparison.csv')


if __name__ == '__main__':
    main()
