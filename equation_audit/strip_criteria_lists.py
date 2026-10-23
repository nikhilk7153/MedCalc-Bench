"""Remove the scoring-criteria / conversion-table preamble from explanations, as in v1.0 (which has none).

Run: python3 equation_audit/strip_criteria_lists.py (after run_generate_csv.py)
Verified's code starts the explanations of 21 calculators with the full criteria list (e.g. "The criteria for the
CHA2DS2-VASc score are listed below: 1. ... The CHA2DS2-VASc score is calculated by summing the points for each
criterion."). That preamble is removed; the step-by-step scoring that follows is kept unchanged. The GCS rule for
components not mentioned/not testable (a labelling convention, not part of the list) is kept.
Reads final_files_2026-10-16/ and writes copies to final_files_2026-10-20_no_criteria_lists/ with a record of every
changed explanation; the input files are not modified.
"""
from pathlib import Path

import hashlib
import json
import re

import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC = ROOT.parent / 'final_files_2026-10-16'
OUT = ROOT.parent / 'final_files_2026-10-20_no_criteria_lists'
HEADER = re.compile(r'\s*[^\n]*(?:listed below|shown below)[^\n]*:', re.IGNORECASE)
# closing sentence of the preamble, e.g. "The total score is calculated by summing the points for each criterion."
SUMMING = re.compile(r'[^\n.]*(?:calculated|taken) by summing the (?:points|score) for each (?:criterion|criteria|of the three components)\.')
LIST_LINE = re.compile(r'\s*(?:\d+\.\s|\(?Grade \d)')
KEEP = re.compile(r"For each criteria, if a patient's value is not mentioned/not testable in the note, we assume that it gets the full score for that attribute\.")


def strip(text):
    m = HEADER.match(text)
    if not m:
        return text
    rest = text[m.end():]
    s = SUMMING.search(rest)
    if s:
        kept = KEEP.search(rest[:s.start()])
        body = rest[s.end():].lstrip()
        return (kept.group(0) + '\n' + body) if kept else body
    # no closing sentence (MME, Steroid Conversion): drop the numbered list lines
    lines = rest.lstrip('\n').split('\n')
    while lines and (LIST_LINE.match(lines[0]) or not lines[0].strip()):
        lines.pop(0)
    return '\n'.join(lines).lstrip()


def main():
    OUT.mkdir(exist_ok=True)
    record = []

    test = pd.read_csv(SRC / 'test_data_final.csv', dtype=str, keep_default_na=False)
    new = test['Ground Truth Explanation'].map(strip)
    for i in test.index[new != test['Ground Truth Explanation']]:
        record.append({'File': 'test_data_final.csv', 'Row Number': test.at[i, 'Row Number'],
                       'Calculator ID': test.at[i, 'Calculator ID'], 'Calculator Name': test.at[i, 'Calculator Name'],
                       'Removed': test.at[i, 'Ground Truth Explanation'][:len(test.at[i, 'Ground Truth Explanation']) - len(new[i])].strip()[:200],
                       'New start': new[i][:120]})
    test['Ground Truth Explanation'] = new
    test.to_csv(OUT / 'test_data_final.csv', index=False)

    shots = json.load(open(SRC / 'one_shot_finalized_explanation.json', encoding='utf-8'))
    for cid, shot in shots.items():
        old = shot['Response']['step_by_step_thinking']
        shot['Response']['step_by_step_thinking'] = strip(old)
        if shot['Response']['step_by_step_thinking'] != old:
            record.append({'File': 'one_shot_finalized_explanation.json', 'Row Number': '', 'Calculator ID': cid,
                           'Calculator Name': '', 'Removed': old[:len(old) - len(shot['Response']['step_by_step_thinking'])].strip()[:200],
                           'New start': shot['Response']['step_by_step_thinking'][:120]})
    with open(OUT / 'one_shot_finalized_explanation.json', 'w', encoding='utf-8') as f:
        json.dump(shots, f, indent=4)

    one_shot = pd.read_csv(SRC / 'one_shot_finalized.csv', dtype=str, keep_default_na=False)
    one_shot['Step-by-Step Thinking'] = one_shot['Step-by-Step Thinking'].map(strip)
    one_shot.to_csv(OUT / 'one_shot_finalized.csv', index=False)

    pd.DataFrame(record).to_csv(OUT / 'criteria_lists_removed.csv', index=False)
    files = ['test_data_final.csv', 'one_shot_finalized_explanation.json', 'one_shot_finalized.csv']
    (OUT / 'CHECKSUMS.sha256').write_text(''.join(f'{hashlib.sha256((OUT / n).read_bytes()).hexdigest()}  {n}\n' for n in files))
    print(f'{len(record)} explanations changed; written to {OUT}')


if __name__ == '__main__':
    main()
