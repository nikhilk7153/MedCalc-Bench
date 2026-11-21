"""Find v1.0 equation-based rows whose note reports more than one value for a numeric input.

v1.0's questions named no time point, so such a row was not determinable in v1.0 (the admission, pre-treatment
instruction is new in the correction). Covers the v1.0 rows that were kept in the corrected set
(final_runs/consolidated_2026-11-06/v10_equation_rows.csv: Status "kept, label correct", Extracted notes); the changed
rows are already counted there, and template notes state each input once.

Run: python3 equation_audit/check_multiple_values_v10.py [--limit N]
Uses gpt-6-astra (as run_gpt6_astra.py), key from OPENAI_API_KEY or the repository's .env (read at runtime, never
written). Results are cached per row in multiple_values_v10.jsonl, so reruns resume; the summary goes to
multiple_values_v10.csv.
"""
import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from openai import OpenAI, RateLimitError

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
MODEL = 'gpt-6-astra'
CACHE = ROOT / 'multiple_values_v10.jsonl'
SUMMARY = ROOT / 'multiple_values_v10.csv'
WORKERS = int(os.environ.get('AUDIT_WORKERS', 4))

PROMPT = """You are auditing a medical calculation benchmark. The question below did not say at which time point the patient's values should be taken.
For each numeric input listed, find every value the patient note reports for that quantity for this patient: at presentation, admission, before or after treatment, at follow-up, at discharge, or as a range.
Count as one value: the same measurement repeated or restated in another unit. Do not count reference ranges, normal limits, thresholds, values of other people, changes or differences (e.g. "lost 4 kg"), or doses.
An input has several values when the note reports two or more different values of that quantity for the patient, or gives it as a range.
Respond with JSON only:
{"inputs": [{"input": "...", "values": [{"value": "...", "when": "..."}], "several_values": true|false}],
 "any_several_values": true|false,
 "admission_value_used": true|false|null,
 "explanation": "..."}
"admission_value_used": whether the benchmark's extracted inputs are the values at admission or initial presentation, before treatment (null if the note gives no time information)."""

lock = threading.Lock()


def api_key():
    if os.environ.get('OPENAI_API_KEY'):
        return os.environ['OPENAI_API_KEY']
    for line in open(REPO / '.env'):
        m = re.match(r'\s*(?:export\s+)?OPENAI_API_KEY\s*=\s*["\']?([^"\'\s]+)', line)
        if m:
            return m.group(1)
    raise SystemExit('OPENAI_API_KEY not found in the environment or .env')


client = OpenAI(api_key=api_key(), max_retries=6, timeout=300)


def chat(user):
    for attempt in range(10):
        try:
            r = client.chat.completions.create(model=MODEL, response_format={'type': 'json_object'},
                                               messages=[{'role': 'system', 'content': PROMPT}, {'role': 'user', 'content': user}])
            return json.loads(r.choices[0].message.content)
        except RateLimitError as e:
            if 'insufficient_quota' not in str(e) or attempt == 9:
                raise
            print('credits exhausted; waiting 60s for auto-reload', file=sys.stderr)
            time.sleep(60)


def numeric_inputs(entities):
    import ast
    try:
        d = json.loads(entities)
    except Exception:
        d = ast.literal_eval(entities)
    return {k: v for k, v in d.items() if not isinstance(v, bool) and (isinstance(v, (int, float)) or
            (isinstance(v, (list, tuple)) and v and isinstance(v[0], (int, float))))}


def check(row):
    inputs = numeric_inputs(row['Relevant Entities'])
    user = (f"Patient note:\n{row['Patient Note']}\n\nQuestion:\n{row['Question']}\n\n"
            f"Numeric inputs and the values the benchmark extracted:\n{json.dumps(inputs, ensure_ascii=False)}")
    out = chat(user)
    out['Row Number'] = row['Row Number']
    with lock, open(CACHE, 'a') as f:
        f.write(json.dumps(out, ensure_ascii=False) + '\n')
    return out


GAS = re.compile(r'blood gas|arterial|\babg\b|\bvbg\b|venous gas|gas analysis', re.I)


def counts(inp):
    """Whether the model's 'several values' stands after removing what is not a second value of the input at another
    time: ages at past events (history), birth weights, and blood-gas electrolytes next to the serum value
    (electrolytes come from the serum panel)."""
    name, values = inp['input'].lower(), inp['values']
    if name == 'age':
        return False
    values = [v for v in values if not re.search(r'\bbirth\b', v['when'], re.I)]
    if name in ('sodium', 'chloride', 'bicarbonate', 'potassium', 'glucose'):
        values = [v for v in values if not GAS.search(v['when'] + ' ' + str(v['value']))]
    numbers = {re.sub(r'[^\d.]', '', re.split(r'\s|\(', str(v['value']).strip())[0]) for v in values}
    return len(values) >= 2 and len(numbers) >= 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()

    fate = pd.read_csv(REPO / 'final_runs/consolidated_2026-11-06/v10_equation_rows.csv', dtype=str, keep_default_na=False)
    keep = fate[fate['Status'].str.startswith('kept, label correct') & (fate['Note Type'] == 'Extracted')]['Row Number']
    v1 = pd.read_csv(REPO / 'dataset/test_data.csv', dtype=str, keep_default_na=False).set_index('Row Number')
    rows = v1.loc[keep].reset_index()
    done = {json.loads(l)['Row Number'] for l in open(CACHE)} if CACHE.exists() else set()
    todo = [r for _, r in rows.iterrows() if r['Row Number'] not in done][:args.limit]
    print(f'{len(rows)} rows in scope, {len(done)} cached, {len(todo)} to check', flush=True)
    with ThreadPoolExecutor(WORKERS) as pool:
        for i, out in enumerate(pool.map(check, todo), 1):
            if i % 25 == 0:
                print(f'{i}/{len(todo)}', flush=True)

    results = {}
    for line in open(CACHE):
        x = json.loads(line)
        results[x['Row Number']] = x
    summary = []
    for _, r in rows.iterrows():
        x = results.get(r['Row Number'])
        if x is None:
            continue
        several = [i['input'] for i in x.get('inputs', []) if i.get('several_values')]
        kept = [i['input'] for i in x.get('inputs', []) if i.get('several_values') and counts(i)]
        summary.append({'Row Number': r['Row Number'], 'Calculator ID': r['Calculator ID'], 'Calculator Name': r['Calculator Name'],
                        'Category': r['Category'], 'Several Values': bool(x.get('any_several_values')),
                        'Inputs With Several Values': '; '.join(several),
                        'Not Determinable In v1.0': bool(kept), 'Inputs Counted': '; '.join(kept),
                        'Values': json.dumps([i for i in x.get('inputs', []) if i.get('several_values')], ensure_ascii=False),
                        'Admission Value Used': x.get('admission_value_used'), 'Explanation': x.get('explanation', '')})
    s = pd.DataFrame(summary)
    s.to_csv(SUMMARY, index=False)
    hit, real = s[s['Several Values']], s[s['Not Determinable In v1.0']]
    print(f'checked {len(s)} of {len(rows)}; model flags several values: {len(hit)} rows in {hit["Calculator ID"].nunique()} '
          f'calculators; after removing past ages, birth weights and blood-gas pairs: {len(real)} rows in '
          f'{real["Calculator ID"].nunique()} calculators')
    print(real.groupby('Calculator Name').size().sort_values(ascending=False).to_string())


if __name__ == '__main__':
    main()
