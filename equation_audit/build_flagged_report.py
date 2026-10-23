"""Merge gpt-6-astra verdicts with the manual label review into one highlighted spreadsheet.

Run: python3 equation_audit/build_flagged_report.py (after run_gpt6_astra.py).
Writes flagged_rows.xlsx (all lab/physical rows, colour-coded) and flagged_rows.csv (flagged rows only).
"""
from pathlib import Path

import pandas as pd
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).resolve().parent
BAD_EQUATION = {46: 'Framingham: mmol/L substituted into mg/dL equation',
                23: 'MELD Na: score rounded before multiplying by 10'}
# Labels consistently use delta gap = AG - 12; the model's disagreement is a convention difference, not a label error.
CONVENTION_ONLY = {63, 66}
MODEL_WRONG = {'label_wrong', 'both_wrong'}
MANUAL_WRONG = {'incorrect', 'invalid numeric target'}
MANUAL_UNUSABLE = {'ambiguous', 'insufficient information', 'out of domain'}

FILLS = {'Remove calculator (bad equation)': 'BFBFBF', 'Both': 'F4A6A6',
         'Model only': 'F9D29D', 'Manual only': 'FFF2A8', 'Clean': None}


def load_model():
    admission = ROOT / 'row_results_admission.csv'
    path = admission if admission.exists() else ROOT / 'row_results.csv'
    print(f'model verdicts from {path.name}')
    return pd.read_csv(path)


def classify(r):
    if r['Calculator ID'] in BAD_EQUATION:
        return 'Remove calculator (bad equation)'
    model = r['Calculator ID'] not in CONVENTION_ONLY and (r['Verdict'] in MODEL_WRONG or r['Verdict'] == 'ambiguous')
    manual = r['Manual Status'] in MANUAL_WRONG | MANUAL_UNUSABLE
    return {(True, True): 'Both', (True, False): 'Model only', (False, True): 'Manual only'}.get((model, manual), 'Clean')


def problem(r):
    if r['Calculator ID'] in BAD_EQUATION:
        return 'wrong'
    wrong = r['Verdict'] in MODEL_WRONG and r['Calculator ID'] not in CONVENTION_ONLY or r['Manual Status'] in MANUAL_WRONG
    return 'wrong' if wrong else ('unusable' if r['Flag'] != 'Clean' else '')


def main():
    model = load_model()
    manual = pd.read_csv(ROOT.parent / 'label_audit/individual_note_review.csv', encoding='utf-8-sig',
                         usecols=['Row Number', 'Status', 'Corrected Answer or Qualification', 'Reason'])
    manual.columns = ['Row Number', 'Manual Status', 'Manual Corrected Answer', 'Manual Reason']
    df = model.merge(manual, on='Row Number', how='left')
    df['Flag'] = df.apply(classify, axis=1)
    df['Problem'] = df.apply(problem, axis=1)
    df['Note'] = df['Calculator ID'].map(BAD_EQUATION).fillna('')
    df.loc[df['Calculator ID'].isin(CONVENTION_ONLY) & df['Verdict'].ne('agrees_with_label'), 'Note'] = \
        'Model used delta-delta; label uses AG - 12 consistently (usable)'
    cols = ['Row Number', 'Calculator ID', 'Calculator Name', 'Flag', 'Problem', 'Ground Truth Answer',
            'Model Answer', 'Verdict', 'Label Error Type', 'Corrected Answer', 'Adjudication',
            'Manual Status', 'Manual Corrected Answer', 'Manual Reason', 'Note']
    df = df[cols].rename(columns={'Verdict': 'Model Verdict', 'Label Error Type': 'Model Error Type',
                                  'Corrected Answer': 'Model Corrected Answer', 'Adjudication': 'Model Explanation'})

    summary = df.groupby(['Calculator ID', 'Calculator Name']).agg(
        Rows=('Flag', 'size'), Wrong=('Problem', lambda s: s.eq('wrong').sum()),
        Unusable=('Problem', lambda s: s.eq('unusable').sum()),
        Both=('Flag', lambda s: s.eq('Both').sum()), Model_only=('Flag', lambda s: s.eq('Model only').sum()),
        Manual_only=('Flag', lambda s: s.eq('Manual only').sum())).reset_index()
    summary['Keep'] = summary['Rows'] - summary['Wrong'] - summary['Unusable']
    summary['Action'] = summary['Calculator ID'].map(lambda c: 'Remove calculator' if c in BAD_EQUATION else 'Keep, drop flagged rows')
    summary = summary.sort_values(['Action', 'Wrong'], ascending=[False, False])

    flagged = df[df['Flag'] != 'Clean']
    flagged.to_csv(ROOT / 'flagged_rows.csv', index=False)
    out = ROOT / 'flagged_rows.xlsx'
    with pd.ExcelWriter(out, engine='openpyxl') as xw:
        summary.to_excel(xw, sheet_name='By calculator', index=False)
        df.to_excel(xw, sheet_name='All rows', index=False)
        pd.DataFrame([{'Colour': k, 'Meaning': v} for k, v in {
            'Remove calculator (bad equation)': 'Whole calculator has a systematically wrong equation',
            'Both': 'Flagged wrong/unusable by gpt-6-astra and by the manual review (highest confidence)',
            'Model only': 'Flagged by gpt-6-astra only; needs human check',
            'Manual only': 'Flagged by the manual review only',
            'Clean': 'No flag from either source'}.items()]).to_excel(xw, sheet_name='Legend', index=False)
        for name, ws in xw.sheets.items():
            ws.freeze_panes = 'A2'
            ws.auto_filter.ref = ws.dimensions
            for cell in ws[1]:
                cell.font = Font(bold=True)
            for i, col in enumerate(ws.columns, 1):
                width = max(len(str(c.value or '')) for c in list(col)[:200])
                ws.column_dimensions[get_column_letter(i)].width = min(max(width, 10), 60)
        for ws, col in ((xw.sheets['All rows'], cols.index('Flag')), (xw.sheets['Legend'], 0)):
            for row in ws.iter_rows(min_row=2):
                colour = FILLS.get(row[col].value)
                if colour:
                    for cell in row:
                        cell.fill = PatternFill('solid', fgColor=colour)

    print(df['Flag'].value_counts().to_string())
    print(f"\nwrong: {df['Problem'].eq('wrong').sum()}  unusable: {df['Problem'].eq('unusable').sum()}  "
          f"total flagged: {len(flagged)} of {len(df)}  |  clean rows kept: {len(df) - len(flagged)}")
    print(f'wrote {out.name} and flagged_rows.csv')


if __name__ == '__main__':
    main()
