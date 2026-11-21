"""Run gpt-6-astra over the lab and physical (equation-based) rows of test_data.csv; does not modify the source dataset.

Run: OPENAI_API_KEY=... python3 equation_audit/run_gpt6_astra.py
Pass 1 solves each row blind to the label and grades it with the benchmark's rules.
Pass 2 shows failed rows the label explanation and asks whether the label or the model is wrong.
Results are cached per row in JSONL, so reruns resume.
"""
import json
import os
import re
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import pandas as pd
from openai import OpenAI, RateLimitError

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / 'dataset/test_data.csv'
MODEL = 'gpt-6-astra'
TAG = '_admission'  # inputs restricted to admission, pre-treatment values
EQUATION_CATEGORIES = {'lab', 'physical'}  # date and dosage calculators are out of scope
WORKERS = int(os.environ.get('AUDIT_WORKERS', 3))  # keep low so credit auto-reload can keep up

# long notes with many inputs (APACHE II, SOFA, Caprini) can need more than 5 minutes per request
client = OpenAI(max_retries=int(os.environ.get('AUDIT_RETRIES', 6)), timeout=float(os.environ.get('AUDIT_TIMEOUT', 300)))
lock = threading.Lock()

SOLVE_PROMPT = """You are a careful clinical calculation assistant. Read the patient note and answer the question.
Use the patient's values at the time of admission or initial presentation, before any treatment; ignore later, follow-up or post-treatment measurements.
Extract every input from the note, convert units explicitly, state the standard published formula you use, and compute step by step.
Respond with JSON only: {"inputs": {...}, "formula": "...", "reasoning": "...", "answer": "..."}
The "answer" field must contain only the final value in the format the question requests (a bare number, a M/D/Y date, or a (weeks, days) tuple)."""

ADJUDICATE_PROMPT = """You are auditing a medical calculation benchmark label. You get a patient note, the question, the benchmark's ground-truth answer and explanation, and an independent model answer that disagrees with it.
Decide which is correct by recomputing carefully from the note using the standard published formula.
Use the patient's values at the time of admission or initial presentation, before any treatment; ignore later, follow-up or post-treatment measurements. A label is not wrong merely because it uses admission values.
Respond with JSON only:
{"verdict": "label_wrong" | "model_wrong" | "both_wrong" | "ambiguous",
 "label_error_type": "wrong_equation" | "input_extraction" | "unit_conversion" | "arithmetic_or_rounding" | "note_missing_or_conflicting_input" | "none",
 "corrected_answer": "...",
 "explanation": "..."}
Use "wrong_equation" only when the label explanation applies an incorrect or non-standard formula, constant or rule."""


def chat(system, user):
    for attempt in range(10):
        try:
            r = client.chat.completions.create(
                model=MODEL,
                messages=[{'role': 'system', 'content': system}, {'role': 'user', 'content': user}],
                response_format={'type': 'json_object'},
            )
            return json.loads(r.choices[0].message.content), r.usage.total_tokens
        except RateLimitError as e:
            # an exhausted balance needs time for auto-reload; other rate limits are retried by the client
            if 'insufficient_quota' not in str(e) or attempt == 9:
                raise
            print('credits exhausted; waiting 60s for auto-reload', file=sys.stderr)
            time.sleep(60)


def grade(answer, row):
    """Mirror evaluation/evaluate.py, but return None instead of raising on unparseable answers."""
    cid = int(row['Calculator ID'])
    try:
        if cid in (13, 68):
            fmt = lambda s: datetime.strptime(s.strip(), '%m/%d/%Y').date()
            return int(fmt(answer) == fmt(row['Ground Truth Answer']))
        if cid == 69:
            pat = r"(\d+)\s*(?:weeks?)?[\"']?\s*,\s*[\"']?(\d+)"
            return int(re.search(pat, answer).groups() == re.search(pat, row['Ground Truth Answer']).groups())
        value = float(re.search(r'-?\d+(?:\.\d+)?', answer.replace(',', '')).group())
        return int(float(row['Lower Limit']) <= value <= float(row['Upper Limit']))
    except (AttributeError, ValueError, TypeError):
        return None


def load_cache(path):
    if not path.exists():
        return {}
    return {r['Row Number']: r for r in map(json.loads, path.open())}


def append(path, record):
    with lock, path.open('a') as f:
        f.write(json.dumps(record) + '\n')


def solve(row, path):
    user = f"Patient note:\n{row['Patient Note']}\n\nQuestion:\n{row['Question']}"
    try:
        out, tokens = chat(SOLVE_PROMPT, user)
        answer = str(out.get('answer', ''))
    except Exception as e:  # keep going; failed rows are retried on the next run
        print(f"row {row['Row Number']}: {e}", file=sys.stderr)
        return
    append(path, {'Row Number': int(row['Row Number']), 'Calculator ID': int(row['Calculator ID']),
                  'Calculator Name': row['Calculator Name'], 'Category': row['Category'],
                  'Ground Truth Answer': row['Ground Truth Answer'], 'Model Answer': answer,
                  'Correct': grade(answer, row), 'Model Inputs': out.get('inputs'),
                  'Model Formula': out.get('formula'), 'Model Reasoning': out.get('reasoning'),
                  'Tokens': tokens})


def adjudicate(row, solved, path):
    user = (f"Patient note:\n{row['Patient Note']}\n\nQuestion:\n{row['Question']}\n\n"
            f"Benchmark ground-truth answer: {row['Ground Truth Answer']} "
            f"(accepted range {row['Lower Limit']} to {row['Upper Limit']})\n"
            f"Benchmark explanation:\n{row['Ground Truth Explanation']}\n\n"
            f"Independent model answer: {solved['Model Answer']}\n"
            f"Independent model formula: {solved['Model Formula']}\n"
            f"Independent model reasoning:\n{solved['Model Reasoning']}")
    try:
        out, tokens = chat(ADJUDICATE_PROMPT, user)
    except Exception as e:
        print(f"row {row['Row Number']}: {e}", file=sys.stderr)
        return
    append(path, {'Row Number': int(row['Row Number']), **out, 'Tokens': tokens})


def run(rows, fn, path, *extra):
    keep = {int(r['Row Number']) for r in rows}
    done = {n: r for n, r in load_cache(path).items() if n in keep}
    todo = [r for r in rows if int(r['Row Number']) not in done]
    print(f'{path.name}: {len(done)} cached, {len(todo)} to run')
    with ThreadPoolExecutor(WORKERS) as pool:
        for i, _ in enumerate(pool.map(lambda r: fn(r, *[e[int(r['Row Number'])] for e in extra], path), todo), 1):
            if i % 50 == 0:
                print(f'  {i}/{len(todo)}')
    return {n: r for n, r in load_cache(path).items() if n in keep}


def main():
    df = pd.read_csv(DATA, dtype=str)
    df = df[df['Category'].isin(EQUATION_CATEGORIES)]
    rows = df.to_dict('records')
    by_row = {int(r['Row Number']): r for r in rows}
    print(f"{len(rows)} equation-based rows across {df['Calculator ID'].nunique()} calculators")

    solved = run(rows, solve, ROOT / f'pass1_solve{TAG}.jsonl')
    failed = [by_row[n] for n, s in solved.items() if s['Correct'] != 1]
    judged = run(failed, adjudicate, ROOT / f'pass2_adjudicate{TAG}.jsonl', solved)

    detail = []
    for n, s in sorted(solved.items()):
        j = judged.get(n, {})
        detail.append({**{k: v for k, v in s.items() if k not in ('Model Inputs', 'Model Reasoning', 'Tokens')},
                       'Verdict': j.get('verdict', 'agrees_with_label' if s['Correct'] == 1 else ''),
                       'Label Error Type': j.get('label_error_type', ''),
                       'Corrected Answer': j.get('corrected_answer', ''),
                       'Adjudication': j.get('explanation', '')})
    detail = pd.DataFrame(detail)
    detail.to_csv(ROOT / f'row_results{TAG}.csv', index=False)

    stats = defaultdict(dict)
    for (cid, name, cat), g in detail.groupby(['Calculator ID', 'Calculator Name', 'Category']):
        label_wrong = g['Verdict'].isin(['label_wrong', 'both_wrong'])
        stats[cid] = {'Calculator ID': cid, 'Calculator Name': name, 'Category': cat, 'Rows': len(g),
                      'Model Accuracy': round(g['Correct'].eq(1).mean(), 3),
                      'Label Wrong': int(label_wrong.sum()),
                      'Label Wrong (wrong_equation)': int((label_wrong & g['Label Error Type'].eq('wrong_equation')).sum()),
                      'Ambiguous': int(g['Verdict'].eq('ambiguous').sum()),
                      'Model Wrong': int(g['Verdict'].eq('model_wrong').sum())}
    summary = pd.DataFrame(stats.values())
    summary['Label Wrong Rate'] = (summary['Label Wrong'] / summary['Rows']).round(3)
    summary = summary.sort_values(['Label Wrong (wrong_equation)', 'Label Wrong Rate'], ascending=False)
    summary.to_csv(ROOT / f'calculator_summary{TAG}.csv', index=False)

    tokens = sum(s['Tokens'] for s in solved.values()) + sum(j['Tokens'] for j in judged.values())
    print(summary.to_string(index=False))
    print(f"\nOverall model accuracy: {detail['Correct'].eq(1).mean():.3f}  |  total tokens: {tokens:,}")
    missing = len(rows) - len(solved)
    if missing:
        print(f'{missing} rows failed to run; rerun the script to retry them.')


if __name__ == '__main__':
    main()
