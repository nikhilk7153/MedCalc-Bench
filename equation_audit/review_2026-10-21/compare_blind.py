"""Compare the blind extractions (blind_result_<id>.json) with the stored inputs/answers of the 200 handwritten rows.

Run: python3 equation_audit/review_2026-10-21/compare_blind.py
For each row: the blind reviewer's own MDCalc score, the code's score on the blind extraction, the stored answer,
field-by-field differences between the blind extraction and the stored Relevant Entities, the first-pass verdict
(findings_calc_<id>.json) and the blind note issues. Writes blind_comparison.csv and adjudication_queue.json
(every row where the views disagree or a reviewer raised an issue). Reads test_data_10_20_2026.csv; writes nothing else.
"""
from pathlib import Path

import ast
import json
import shutil
import subprocess
import sys
import tempfile

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / 'equation_audit'))
from run_generate_csv import patch_sofa  # noqa: E402

CALCS = ['15', '17', '21', '25', '27', '28', '29', '32', '36', '43']
UNIT_FACTORS = {('mg/dL', 'µmol/L'): None}  # unit differences are reported, not converted


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def code_scores(test, blind_entities):
    """Run the calculator code on the blind extractions (same harness as run_generate_csv.py)."""
    rows = test.copy()
    rows['Relevant Entities'] = rows['Row Number'].map(lambda r: str(blind_entities.get(r, {})))
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        shutil.copytree(ROOT / 'calculator_implementations_verified', tmp / 'calculator_implementations',
                        ignore=shutil.ignore_patterns('__pycache__'))
        patch_sofa(tmp / 'calculator_implementations/sofa.py')
        (tmp / 'datasets').mkdir()
        out = {}
        for _, r in rows.iterrows():  # one row at a time so a crash on one extraction does not lose the rest
            pd.DataFrame([r]).to_csv(tmp / 'datasets/one_shot_data.csv', index=False)
            run = subprocess.run([sys.executable, 'calculator_implementations/generate_csv.py'], cwd=tmp,
                                 capture_output=True, text=True)
            if run.returncode:
                out[r['Row Number']] = 'ERROR: ' + run.stderr.strip().splitlines()[-1][:120]
            else:
                out[r['Row Number']] = pd.read_csv(tmp / 'datasets/one_shot_data.csv', dtype=str)['Ground Truth Answer'].iloc[0]
        return out


def field_diffs(stored, blind):
    diffs = []
    for k in sorted(set(stored) | set(blind)):
        s, b = stored.get(k), blind.get(k)
        if s == b:
            continue
        if b is None and s is False or s is None and b is False:
            continue  # silent note and an explicit False score the same
        if isinstance(s, list) and isinstance(b, list) and len(s) == 2 and len(b) == 2:
            if num(s[0]) == num(b[0]) and str(s[1]) == str(b[1]):
                continue
        diffs.append({'field': k, 'stored': s, 'blind': b})
    return diffs


def main():
    test = pd.read_csv(ROOT / 'dataset/test_data_10_20_2026.csv', dtype=str, keep_default_na=False)
    records, queue = [], []
    for c in CALCS:
        blind = json.load(open(HERE / f'blind_result_{c}.json'))
        first = {str(i['row']): i for i in json.load(open(HERE / f'findings_calc_{c}.json'))['items']}
        t = test[test['Calculator ID'] == c]
        b_rows = {str(r['row']): r for r in blind['rows']}
        b_ent = {r: {k: v['value'] for k, v in br.get('fields', {}).items()} for r, br in b_rows.items()}
        code = code_scores(t, b_ent)
        for _, r in t.iterrows():
            rn = r['Row Number']; br = b_rows.get(rn, {})
            stored = ast.literal_eval(r['Relevant Entities'])
            diffs = field_diffs(stored, b_ent.get(rn, {}))
            fp = first.get(rn, {})
            rec = {'calculator_id': c, 'calculator': r['Calculator Name'], 'row': rn, 'stored_answer': r['Ground Truth Answer'],
                   'blind_self_score': br.get('score'), 'code_on_blind': code.get(rn), 'first_pass_status': fp.get('status'),
                   'first_pass_answer': fp.get('mdcalc_answer'), 'n_field_diffs': len(diffs),
                   'field_diffs': json.dumps(diffs, ensure_ascii=False, default=str),
                   'blind_note_issues': json.dumps(br.get('note_issues', []), ensure_ascii=False)}
            s = num(r['Ground Truth Answer'])
            disagree = (num(br.get('score')) != s or num(code.get(rn)) != s or fp.get('status') not in (None, 'OK')
                        or bool(br.get('note_issues')) or bool(diffs))
            rec['needs_adjudication'] = disagree
            records.append(rec)
            if disagree:
                queue.append({**rec, 'patient_note': r['Patient Note'], 'question': r['Question'],
                              'stored_relevant_entities': r['Relevant Entities'], 'first_pass_details': fp.get('details'),
                              'first_pass_quote': fp.get('evidence_quote'), 'blind_fields': br.get('fields'),
                              'blind_working': br.get('score_working')})
    df = pd.DataFrame(records)
    df.to_csv(HERE / 'blind_comparison.csv', index=False)
    json.dump(queue, open(HERE / 'adjudication_queue.json', 'w'), indent=1, ensure_ascii=False, default=str)
    agree = df[~df['needs_adjudication']]
    print(f'{len(df)} rows; all views agree on {len(agree)}; {len(queue)} queued for adjudication')
    print(df.groupby('calculator_id')['needs_adjudication'].sum().to_string())


if __name__ == '__main__':
    main()
