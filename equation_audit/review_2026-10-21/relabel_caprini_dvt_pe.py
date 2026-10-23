"""Relabel the Caprini rows of dataset/test_data_10_20_2026.csv after the History of DVT/PE fix (2026-10-21).

Run: python3 equation_audit/review_2026-10-21/relabel_caprini_dvt_pe.py
calculator_implementations_verified/caprini_score.py now counts prior DVT/PE once (+3), as MDCalc does, instead of +3 each.
Inputs are unchanged; only Ground Truth Answer, Lower/Upper Limit and Ground Truth Explanation are recomputed, and only
rows whose labels change are written. The file keeps its CRLF format. Writes caprini_dvt_pe_changes.csv next to this script.
"""
from pathlib import Path

import csv
import io
import shutil
import subprocess
import sys
import tempfile

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / 'equation_audit'))
from run_generate_csv import patch_sofa  # noqa: E402

TEST = ROOT / 'dataset/test_data_10_20_2026.csv'
CAPRINI = '36'
LABELS = ('Ground Truth Answer', 'Lower Limit', 'Upper Limit', 'Ground Truth Explanation')


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
    raw = TEST.read_bytes()
    rows = list(csv.reader(io.StringIO(raw.decode('utf-8'), newline='')))
    header = rows[0]; col = {c: header.index(c) for c in header}
    caprini = [r for r in rows[1:] if r[col['Calculator ID']] == CAPRINI]
    out = relabel([dict(zip(header, r)) for r in caprini])
    log = []
    for i, r in enumerate(caprini):
        new = {c: fmt(out.at[i, c], r[col['Output Type']]) if c != 'Ground Truth Explanation' else out.at[i, c] for c in LABELS}
        if new['Ground Truth Answer'] == r[col['Ground Truth Answer']]:
            continue  # rows without both prior DVT and PE score the same; leave them untouched
        log.append({'Row Number': r[col['Row Number']], 'Old Answer': r[col['Ground Truth Answer']], 'New Answer': new['Ground Truth Answer']})
        for c in LABELS:
            r[col[c]] = new[c]
    buf = io.StringIO(); csv.writer(buf, lineterminator='\r\n').writerows(rows)
    TEST.write_bytes(buf.getvalue().removesuffix('\r\n').encode('utf-8'))
    pd.DataFrame(log).to_csv(HERE / 'caprini_dvt_pe_changes.csv', index=False)
    print(f'{len(log)} of {len(caprini)} Caprini rows relabelled')


if __name__ == '__main__':
    main()
