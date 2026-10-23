"""Regenerate answers, limits, explanations and questions for the corrected test set with generate_csv.py.

Run: python3 equation_audit/run_generate_csv.py (after build_corrected_test.py)
     python3 equation_audit/run_generate_csv.py <input.csv> <output.csv>   (any dataset in the same format, e.g. Verified)
Uses calculator_implementations_verified (a copy of medcalc_verified_code/calculator_implementations, branch
fix/anion-gap-and-sofa, commit 810287c, with the criteria lists removed from the explanations as in v1.0), or $MEDCALC_IMPL
if set (code + name_to_python.json),
without modifying it:
the folder is copied to a temporary directory, because generate_csv.py always reads and overwrites
datasets/one_shot_data.csv relative to the working directory.
Writes dataset/test_data_corrected_generated.csv by default.
"""
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / 'dataset'
TIME_POINT = "You should use the patient's medical values and health status when they were first admitted to the hospital prior to any treatment."
IMPLEMENTATIONS = Path(os.environ.get('MEDCALC_IMPL', Path(__file__).resolve().parent.parent / 'calculator_implementations_verified'))

# sofa.py raises UnboundLocalError when a row has urine output but no creatinine; guard the comparisons
# in the temporary copy until the fix lands in the repo.
SOFA_GUARD = [("    if 'creatinine' in input_parameters:\n        creatinine_exp, creatinine =",
               "    creatinine = None\n    if 'creatinine' in input_parameters:\n        creatinine_exp, creatinine ="),
              ("if creatinine > 5.0 or", "if (creatinine is not None and creatinine > 5.0) or"),
              ("elif 3.5 <= creatinine <= 5.0 or", "elif (creatinine is not None and 3.5 <= creatinine <= 5.0) or"),
              ("elif 2.0 <= creatinine < 3.5:", "elif creatinine is not None and 2.0 <= creatinine < 3.5:"),
              ("elif 1.2 <= creatinine < 2.0:", "elif creatinine is not None and 1.2 <= creatinine < 2.0:"),
              ("elif creatinine < 1.2:", "elif creatinine is not None and creatinine < 1.2:")]


def patch_sofa(path):
    s = path.read_text()
    if 'creatinine = None' in s:
        return
    for old, new in SOFA_GUARD:
        assert s.count(old) == 1, f'sofa.py changed; update SOFA_GUARD ({old!r})'
        s = s.replace(old, new)
    path.write_text(s)


def main():
    source = Path(sys.argv[1]) if len(sys.argv) > 2 else DATA / 'test_data_corrected.csv'
    target = Path(sys.argv[2]) if len(sys.argv) > 2 else DATA / 'test_data_corrected_generated.csv'
    corrected = pd.read_csv(source, dtype=str)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        shutil.copytree(IMPLEMENTATIONS, tmp / 'calculator_implementations', ignore=shutil.ignore_patterns('__pycache__'))
        patch_sofa(tmp / 'calculator_implementations/sofa.py')
        (tmp / 'datasets').mkdir()
        # generate_csv.py only rounds and sets the +/-5% limits for Category "lab test"; v1.0 calls it "lab"
        corrected.assign(Category=corrected['Category'].replace({'lab': 'lab test'})).to_csv(
            tmp / 'datasets/one_shot_data.csv', index=False)
        run = subprocess.run([sys.executable, 'calculator_implementations/generate_csv.py'], cwd=tmp,
                             capture_output=True, text=True)
        (ROOT / 'generate_csv.log').write_text(run.stdout + run.stderr)
        if run.returncode:
            sys.exit(f'generate_csv.py failed; see equation_audit/generate_csv.log\n{run.stderr[-1500:]}')
        out = pd.read_csv(tmp / 'datasets/one_shot_data.csv', dtype=str)

    out['Category'] = corrected['Category'].values  # keep the input's own category names
    if target.name == 'test_data_corrected_generated.csv':
        # keep the build's questions (generate_csv.py rewrites them with the time-point sentence) and put the time-point
        # instruction in its own column, for run.py to add to the system prompt: every non-template row except the
        # calculators whose Verified question has no time-point sentence (APACHE II 28, Caprini 36, SOFA 43)
        out['Question'] = corrected['Question'].values
        tp = (out['Note Type'] != 'Template') & ~out['Calculator ID'].astype(str).isin({'28', '36', '43'})
        out['Time Point Instruction'] = tp.map({True: TIME_POINT, False: ''})
    assert len(out) == len(corrected)
    for col in ('Row Number', 'Calculator ID', 'Note ID', 'Patient Note'):
        assert (out[col] == corrected[col]).all(), col
    out.to_csv(target, index=False)

    changed = {c: int((out[c].astype(str) != corrected[c].astype(str)).sum()) for c in
               ('Question', 'Relevant Entities', 'Ground Truth Answer', 'Lower Limit', 'Upper Limit', 'Ground Truth Explanation')}
    print(f'{len(out)} rows written to {target}; cells changed: {changed}')


if __name__ == '__main__':
    main()
