"""Migrate 33 rule-based rows of dataset/test_data_10_20_2026.csv to the latest physician-approved version of their v1.0 note.

Run: python3 equation_audit/rulebased_v10_check/migrate_AB.py
A (7 rows): the slot goes back to v1.0's note, taken from its latest approved version (Verified v1.0.8, else v1.1).
B (26 rows): v1.0 notes Verified dropped; Patient Note and Relevant Entities from v1.1 (same note ID).
Inputs are mapped to the calculator code's names with the build's rename_entities; labels (answer, limits, explanation)
are recomputed with calculator_implementations_verified. Question, Category and the other columns are kept.
The file keeps its CRLF format. Writes migration_AB_changes.csv next to this script.
"""
from pathlib import Path

import ast
import csv
import io
import os
import shutil
import subprocess
import sys
import tempfile

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / 'equation_audit'))
from build_corrected_test import rename_entities  # noqa: E402
from run_generate_csv import patch_sofa  # noqa: E402

TEST = ROOT / 'dataset/test_data_10_20_2026.csv'
A = {'133': 'pmc-2803837-1', '137': 'usmle-9051', '269': 'pmc-8053007-1', '273': 'usmle-9492', '276': 'pmc-4793133-1',
     '573': 'pmc-4614570-1', '576': 'pmc-8088400-1'}
B = ['127', '128', '136', '221', '224', '227', '230', '236', '237', '238', '240', '261', '264', '265', '266', '267', '268',
     '274', '278', '279', '280', '301', '315', '320', '574', '578']
UNIT_FIXES = {'degrees fahreneheit': 'degrees fahrenheit'}  # misspelling in v1.1's inputs (Centor usmle-12605)
NOTE_FIXES = {'degrees celsisus': 'degrees celsius'}


def load(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False).drop_duplicates(['Calculator ID', 'Note ID']).set_index(['Calculator ID', 'Note ID'])


def map_inputs(cid, text):
    ent = ast.literal_eval(rename_entities(cid, text))
    return str({k: ([v[0], UNIT_FIXES.get(v[1], v[1])] if isinstance(v, list) and len(v) == 2 else v) for k, v in ent.items()})


def fix_note(note):
    for old, new in NOTE_FIXES.items():
        note = note.replace(old, new)
    return note


def relabel(rows):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        shutil.copytree(ROOT / 'calculator_implementations_verified', tmp / 'calculator_implementations', ignore=shutil.ignore_patterns('__pycache__'))
        patch_sofa(tmp / 'calculator_implementations/sofa.py')
        (tmp / 'datasets').mkdir()
        d = pd.DataFrame(rows)
        d.assign(Category=d['Category'].replace({'lab': 'lab test'})).to_csv(tmp / 'datasets/one_shot_data.csv', index=False)
        run = subprocess.run([sys.executable, 'calculator_implementations/generate_csv.py'], cwd=tmp, capture_output=True, text=True)
        if run.returncode:
            sys.exit(run.stderr[-2000:])
        return pd.read_csv(tmp / 'datasets/one_shot_data.csv', dtype=str, keep_default_na=False)


def fmt(value, output_type):
    if output_type == 'integer' and value.replace('.', '', 1).replace('-', '', 1).isdigit() and float(value).is_integer():
        return str(int(float(value)))
    return value


def main():
    ver = load(ROOT / 'dataset/test_verified.csv')
    v11 = load(Path.home() / 'Downloads/MedCalc-Bench-version-1.1/dataset/test_data.csv')
    raw = TEST.read_bytes()
    rows = list(csv.reader(io.StringIO(raw.decode('utf-8'), newline='')))
    header = rows[0]; col = {c: header.index(c) for c in header}
    by_row = {r[col['Row Number']]: r for r in rows[1:]}

    new_rows, log = [], []
    for rn in list(A) + B:
        r = by_row[rn]; cid = r[col['Calculator ID']]
        note_id = A.get(rn, r[col['Note ID']])
        if rn in A and (cid, note_id) in ver.index:
            src_name, src = 'Verified v1.0.8', ver.loc[(cid, note_id)]
        else:
            src_name, src = 'v1.1', v11.loc[(cid, note_id)]
        rec = dict(zip(header, r))
        old = {k: rec[k] for k in ('Note ID', 'Patient Note', 'Relevant Entities', 'Ground Truth Answer')}
        rec['Note ID'] = note_id
        rec['Patient Note'] = fix_note(src['Patient Note'])
        rec['Relevant Entities'] = map_inputs(cid, src['Relevant Entities'])
        new_rows.append(rec)
        log.append({'Row Number': rn, 'Group': 'A' if rn in A else 'B', 'Calculator ID': cid, 'Calculator Name': rec['Calculator Name'],
                    'Source': src_name, 'Old Note ID': old['Note ID'], 'New Note ID': note_id,
                    'Note Text Changed': old['Patient Note'] != rec['Patient Note'],
                    'Old Relevant Entities': old['Relevant Entities'], 'New Relevant Entities': rec['Relevant Entities'],
                    'Old Answer': old['Ground Truth Answer'], 'Source Answer': src['Ground Truth Answer']})
    out = relabel(new_rows)
    for i, rec in enumerate(new_rows):
        r = by_row[rec['Row Number']]; ot = rec['Output Type']
        for c in ('Note ID', 'Patient Note', 'Relevant Entities'):
            r[col[c]] = rec[c]
        r[col['Ground Truth Answer']] = fmt(out.at[i, 'Ground Truth Answer'], ot)
        r[col['Lower Limit']] = fmt(out.at[i, 'Lower Limit'], ot)
        r[col['Upper Limit']] = fmt(out.at[i, 'Upper Limit'], ot)
        r[col['Ground Truth Explanation']] = out.at[i, 'Ground Truth Explanation']
        log[i]['New Answer'] = r[col['Ground Truth Answer']]

    buf = io.StringIO(); csv.writer(buf, lineterminator='\r\n').writerows(rows)
    TEST.write_bytes(buf.getvalue().removesuffix('\r\n').encode('utf-8'))
    pd.DataFrame(log).to_csv(HERE / 'migration_AB_changes.csv', index=False)
    print(f'{len(log)} rows migrated; changes in {HERE / "migration_AB_changes.csv"}')


if __name__ == '__main__':
    main()
