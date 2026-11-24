"""Ask gpt-6-astra to propose corrected Relevant Entities (and minimal note fixes) for flagged test rows.

Run: OPENAI_API_KEY=... python3 equation_audit/propose_fixes.py <todo.csv> <tag>
<todo.csv> needs: Row Number, Calculator ID, Calculator Name, Note ID, Patient Note, Question, Relevant Entities, detail
(the problem already identified). Proposals are reviewed by hand before being copied into equation_audit/overrides/.
Writes proposals_<tag>.jsonl and proposals_<tag>.csv in equation_audit/. Reruns resume from the JSONL.
"""
import ast
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

import run_gpt6_astra as audit

ROOT = Path(__file__).resolve().parent
IMPLEMENTATIONS = Path(os.environ.get('MEDCALC_IMPL', ROOT.parent / 'medcalc_verified_code_snapshot/calculator_implementations'))
META_KEYS = {'file path', 'explanation function', 'calculator name', 'type', 'question'}

PROMPT = """You are correcting the extracted inputs of one example in a medical calculation benchmark. The final answer is
computed by code from the extracted inputs, so the inputs must match the patient note exactly.
You get the calculator, its input names with the values each input uses elsewhere in the dataset, the question, the note,
the current extracted inputs, and a reviewer's finding about what is wrong.
Rules:
- Use the time point the question specifies (admission before treatment when stated); otherwise the encounter the note describes.
- A criterion the note does not mention is absent/false; do not invent values.
- Keep the same key names and value formats as the current inputs (and the listed example values).
- Change only what the note supports; give the exact note text as evidence for every change.
- Report note problems: "contradiction" (the note states conflicting values), "unit_typo" (an obviously wrong unit),
  "missing_required_input" (a value the calculator needs is genuinely absent and has no defined default), or "none".
- Only for contradiction or unit_typo, propose the smallest note edit as exact [old_text, new_text] pairs, where old_text
  appears verbatim once in the note. Never add new clinical facts.
Respond with JSON only:
{"note_problem": "none" | "contradiction" | "unit_typo" | "missing_required_input",
 "note_edits": [["old", "new"]],
 "entities": {...full corrected inputs...},
 "changes": [{"input": "...", "old": "...", "new": "...", "evidence": "..."}],
 "finding_valid": true | false,
 "explanation": "..."}"""


def input_catalog(info, ver):
    """Calculator input names plus up to 6 example values seen in Verified's test set."""
    seen = defaultdict(lambda: defaultdict(set))
    for cid, ent in zip(ver['Calculator ID'], ver['Relevant Entities']):
        try:
            for k, v in ast.literal_eval(ent).items():
                seen[cid][k].add(json.dumps(v, ensure_ascii=False)[:60])
        except (ValueError, SyntaxError):
            pass
    out = {}
    for cid, meta in info.items():
        names = [k for k in meta if k not in META_KEYS]
        out[cid] = {k: sorted(seen[cid].get(k, set()))[:6] for k in names}
    return out


def main():
    source, tag = Path(sys.argv[1]), sys.argv[2]
    info = json.load(open(IMPLEMENTATIONS / 'name_to_python.json'))
    ver = pd.read_csv(ROOT.parent / 'dataset/test_verified.csv', dtype=str, keep_default_na=False)
    catalog = input_catalog(info, ver)
    df = pd.read_csv(source, dtype=str, keep_default_na=False)
    rows = df.to_dict('records')
    by_row = {int(r['Row Number']): r for r in rows}

    def propose(row, path):
        user = (f"Calculator: {row['Calculator Name']}\nInputs and example values: {json.dumps(catalog.get(row['Calculator ID'], {}), ensure_ascii=False)}\n\n"
                f"Question:\n{row['Question']}\n\nPatient note:\n{row['Patient Note']}\n\n"
                f"Current extracted inputs:\n{row['Relevant Entities']}\n\nReviewer finding:\n{row['detail']}")
        try:
            out, tokens = audit.chat(PROMPT, user)
        except Exception as e:
            print(f"row {row['Row Number']}: {e}", file=sys.stderr)
            return
        audit.append(path, {'Row Number': int(row['Row Number']), **out, 'Tokens': tokens})

    print(f'{len(rows)} rows from {source.name}')
    done = audit.run(rows, propose, ROOT / f'proposals_{tag}.jsonl')
    out = []
    for n, p in sorted(done.items()):
        r = by_row[n]
        out.append({'Row Number': n, 'Calculator ID': r['Calculator ID'], 'Calculator Name': r['Calculator Name'], 'Note ID': r['Note ID'],
                    'Note Problem': p.get('note_problem', ''), 'Finding Valid': p.get('finding_valid', ''),
                    'Changes': json.dumps(p.get('changes', []), ensure_ascii=False), 'Note Edits': json.dumps(p.get('note_edits', []), ensure_ascii=False),
                    'Proposed Entities': json.dumps(p.get('entities', {}), ensure_ascii=False), 'Explanation': p.get('explanation', ''),
                    'Current Entities': r['Relevant Entities'], 'Finding': r['detail']})
    pd.DataFrame(out).to_csv(ROOT / f'proposals_{tag}.csv', index=False, encoding='utf-8-sig')
    print(pd.DataFrame(out)['Note Problem'].value_counts().to_string())
    if len(done) < len(rows):
        print(f'{len(rows) - len(done)} rows failed; rerun to retry.')


if __name__ == '__main__':
    main()
