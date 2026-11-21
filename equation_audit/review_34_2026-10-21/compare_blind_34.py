"""Compare the blind extractions of the 34 v1.0 rule-based notes (dropped by v1.1/test_set_1_31) with their stored labels.

Run: python3 equation_audit/review_34_2026-10-21/compare_blind_34.py
Per row: stored answer, blind reviewer's MDCalc score, the code's score on the blind extraction, field differences,
completeness flag and note issues. Writes blind_comparison_34.csv and adjudication_queue_34.json.
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

GROUPS = ['curb65', 'pharyngitis', 'heart_perc_sirs']


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def code_score(row, entities, tmp):
    r = row.copy()
    r['Relevant Entities'] = str(entities)
    pd.DataFrame([r]).to_csv(tmp / 'datasets/one_shot_data.csv', index=False)
    run = subprocess.run([sys.executable, 'calculator_implementations/generate_csv.py'], cwd=tmp, capture_output=True, text=True)
    if run.returncode:
        return 'ERROR: ' + run.stderr.strip().splitlines()[-1][:120]
    return pd.read_csv(tmp / 'datasets/one_shot_data.csv', dtype=str)['Ground Truth Answer'].iloc[0]


def main():
    test = pd.read_csv(ROOT / 'dataset/test_data_10_20_2026.csv', dtype=str, keep_default_na=False).set_index('Row Number', drop=False)
    records, queue = [], []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        shutil.copytree(ROOT / 'calculator_implementations_verified', tmp / 'calculator_implementations',
                        ignore=shutil.ignore_patterns('__pycache__'))
        patch_sofa(tmp / 'calculator_implementations/sofa.py')
        (tmp / 'datasets').mkdir()
        for g in GROUPS:
            for br in json.load(open(HERE / f'blind_result_{g}.json'))['rows']:
                rn = str(br['row']); r = test.loc[rn]
                stored = ast.literal_eval(r['Relevant Entities'])
                blind = {k: v['value'] for k, v in br.get('fields', {}).items()}
                diffs = [{'field': k, 'stored': stored.get(k), 'blind': blind.get(k)} for k in sorted(set(stored) | set(blind))
                         if stored.get(k) != blind.get(k) and not (stored.get(k) in (None, False) and blind.get(k) in (None, False))]
                cs = code_score(r, blind, tmp)
                rec = {'calculator_id': r['Calculator ID'], 'calculator': r['Calculator Name'], 'row': rn, 'note_id': r['Note ID'],
                       'stored_answer': r['Ground Truth Answer'], 'blind_score': br.get('score'), 'code_on_blind': cs,
                       'completeness': br.get('completeness'), 'completeness_reason': br.get('completeness_reason'),
                       'eligibility_note': br.get('eligibility_note'), 'n_field_diffs': len(diffs),
                       'field_diffs': json.dumps(diffs, ensure_ascii=False, default=str),
                       'note_issues': json.dumps(br.get('note_issues', []), ensure_ascii=False)}
                s = num(r['Ground Truth Answer'])
                flag = (num(br.get('score')) != s or br.get('completeness') != 'COMPLETE'
                        or any(num(i.get('alternative_score')) not in (None, s) for i in br.get('note_issues', [])))
                rec['queued'] = flag
                records.append(rec)
                if flag:
                    queue.append({**rec, 'question': r['Question'], 'patient_note': r['Patient Note'],
                                  'stored_relevant_entities': r['Relevant Entities'], 'stored_explanation': r['Ground Truth Explanation'],
                                  'blind_fields': br.get('fields'), 'blind_working': br.get('score_working')})
    df = pd.DataFrame(records)
    df.to_csv(HERE / 'blind_comparison_34.csv', index=False)
    json.dump(queue, open(HERE / 'adjudication_queue_34.json', 'w'), indent=1, ensure_ascii=False, default=str)
    print(f'{len(df)} rows; blind score = stored on {(df.blind_score.map(num) == df.stored_answer.map(num)).sum()}; '
          f'INCOMPLETE {(df.completeness != "COMPLETE").sum()}; queued for adjudication {len(queue)}')
    print(df[['row', 'calculator_id', 'note_id', 'stored_answer', 'blind_score', 'code_on_blind', 'completeness', 'queued']].to_string(index=False))


if __name__ == '__main__':
    main()
