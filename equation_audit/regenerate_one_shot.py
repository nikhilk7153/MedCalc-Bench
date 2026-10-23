"""Regenerate the one-shot examples in final_files_2026-10-16/one_shot_10_17_2026.csv with the calculator code.

Run: python3 equation_audit/regenerate_one_shot.py
For each example, the explanation and answer are recomputed from its Input Parameters with
calculator_implementations_verified (SOFA patched as in run_generate_csv.py); lab/physical/dosage answers are rounded
as generate_csv.py does, and whole numbers are written without ".0". The time-point sentence ("You should use the
patient's medical values ... prior to any treatment.") is removed from every Question. Patient Note and Input
Parameters are not changed. The file keeps its format; the previous version is backed up to
dataset/superseded/snapshot_2026-10-21_before_review/. Writes one_shot_regeneration_changes.csv next to this script.
"""
from pathlib import Path

import csv
import importlib.util
import io
import json
import shutil
import sys
import tempfile

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from run_generate_csv import patch_sofa  # noqa: E402

SRC = ROOT / 'final_files_2026-10-16/one_shot_10_17_2026.csv'
BACKUP = ROOT / 'dataset/superseded/snapshot_2026-10-21_before_review/one_shot_10_17_2026.before_regeneration.csv'
TEST = ROOT / 'dataset/test_data_10_20_2026.csv'
TIME_POINT = " You should use the patient's medical values and health status when they were first admitted to the hospital prior to any treatment."
ROUNDED = {'lab', 'lab test', 'physical', 'dosage'}


def load_function(impl, calc_info, cid):
    path = impl / calc_info[cid]['file path']
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, calc_info[cid]['explanation function'])


def fmt(value, rounded, round_number):
    if isinstance(value, (tuple, list)):
        return str(list(value))
    if rounded:
        value = round_number(float(value))
    if isinstance(value, (int, float)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def main():
    raw = SRC.read_bytes()
    text = raw.decode('utf-8')
    terminator = '\r\n' if text.split('\n', 1)[0].endswith('\r') else '\n'
    trailing = text.endswith(terminator)
    rows = list(csv.reader(io.StringIO(text, newline='')))
    buf = io.StringIO(); csv.writer(buf, lineterminator=terminator).writerows(rows)
    assert (buf.getvalue() if trailing else buf.getvalue().removesuffix(terminator)) == text, 'CSV does not round-trip'
    header = rows[0]; col = {c: header.index(c) for c in header}
    categories = pd.read_csv(TEST, dtype=str, keep_default_na=False).groupby('Calculator ID')['Category'].first()

    log = []
    with tempfile.TemporaryDirectory() as tmp:
        impl = Path(tmp) / 'calculator_implementations'
        shutil.copytree(ROOT / 'calculator_implementations_verified', impl, ignore=shutil.ignore_patterns('__pycache__'))
        patch_sofa(impl / 'sofa.py')
        sys.path.insert(0, str(impl))
        from rounding import round_number
        calc_info = json.load(open(impl / 'name_to_python.json'))
        for r in rows[1:]:
            cid = r[col['Calculator ID']]
            result = load_function(impl, calc_info, cid)(json.loads(r[col['Input Parameters']]))
            new = {'Question': r[col['Question']].replace(TIME_POINT, ''),
                   'Step-by-Step Thinking': result['Explanation'],
                   'Answer': fmt(result['Answer'], categories[cid] in ROUNDED, round_number)}
            entry = {'Calculator ID': cid, 'Calculator Name': r[col['Calculator Name']]}
            for c, v in new.items():
                entry[f'{c} changed'] = r[col[c]] != v
                if c == 'Answer':
                    entry['Old Answer'], entry['New Answer'] = r[col[c]], v
                r[col[c]] = v
            log.append(entry)

    shutil.copy2(SRC, BACKUP)
    buf = io.StringIO(); csv.writer(buf, lineterminator=terminator).writerows(rows)
    out = buf.getvalue() if trailing else buf.getvalue().removesuffix(terminator)
    SRC.write_bytes(out.encode('utf-8'))
    pd.DataFrame(log).to_csv(HERE / 'one_shot_regeneration_changes.csv', index=False)
    d = pd.DataFrame(log)
    print(f"{len(d)} examples; questions changed {d['Question changed'].sum()}, explanations changed "
          f"{d['Step-by-Step Thinking changed'].sum()}, answers changed {d['Answer changed'].sum()}")


if __name__ == '__main__':
    main()
