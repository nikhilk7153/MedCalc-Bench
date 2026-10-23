"""Convert final_files_2026-10-16/one_shot_10_17_2026.csv to the one-shot JSON that evaluation/run.py reads.

Run: python3 equation_audit/build_one_shot_json.py
Same format as dataset/one_shot_finalized_explanation.json: {calculator_id: {"Response": {"step_by_step_thinking",
"answer"}, "Patient Note", "input_parameters"}}, in the CSV's order, indent 4. Answers are numbers, dates as strings,
gestational age as a list; where the previous JSON has the same answer value, its type is kept (e.g. 40.0, not 40).
The questions are not stored: run.py uses the test question as the example task (Steroid Conversion: the example's own).
Checks that each one-shot question equals the test question of its calculator in dataset/test_data_10_20_2026.csv
without the time-point sentence (removed from the one-shot questions by regenerate_one_shot.py).
"""
from pathlib import Path

import ast
import json
import re

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / 'final_files_2026-10-16/one_shot_10_17_2026.csv'
OUT = ROOT / 'final_files_2026-10-16/one_shot_10_17_2026.json'
PREVIOUS = ROOT / 'dataset/one_shot_finalized_explanation.json'
TEST = ROOT / 'dataset/test_data_10_20_2026.csv'
OWN_QUESTION = {'24'}  # Steroid Conversion: test questions name other drug pairs
TIME_POINT = " You should use the patient's medical values and health status when they were first admitted to the hospital prior to any treatment."


def parse_answer(text, previous):
    if text.startswith('['):
        value = ast.literal_eval(text)
    elif re.fullmatch(r'-?\d+', text):
        value = int(text)
    elif re.fullmatch(r'-?\d*\.\d+', text):
        value = float(text)
    else:
        value = text
    if isinstance(previous, (int, float)) and isinstance(value, (int, float)) and previous == value:
        return previous
    return value


def main():
    shots = pd.read_csv(SRC, dtype=str, keep_default_na=False)
    previous = json.load(open(PREVIOUS))
    test = pd.read_csv(TEST, dtype=str, keep_default_na=False)
    questions = test.groupby('Calculator ID')['Question'].agg(lambda s: {q.replace(TIME_POINT, '') for q in s})
    assert len(shots) == 55 and shots['Calculator ID'].is_unique
    out = {}
    for _, r in shots.iterrows():
        cid = r['Calculator ID']
        if cid not in OWN_QUESTION:
            assert r['Question'] in questions[cid], f'one-shot question for {cid} differs from the test question'
        prev = previous.get(cid, {}).get('Response', {}).get('answer')
        out[cid] = {'Response': {'step_by_step_thinking': r['Step-by-Step Thinking'], 'answer': parse_answer(r['Answer'], prev)},
                    'Patient Note': r['Patient Note'],
                    'input_parameters': json.loads(r['Input Parameters'])}
    with open(OUT, 'w') as f:
        json.dump(out, f, indent=4)
    print(f'{len(out)} examples written to {OUT}')


if __name__ == '__main__':
    main()
