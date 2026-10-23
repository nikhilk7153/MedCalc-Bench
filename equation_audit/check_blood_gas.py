"""Find extracted inputs taken from a blood gas (ABG/VBG) instead of the serum chemistry panel, with gpt-6-astra.

Run: OPENAI_API_KEY=... python3 equation_audit/check_blood_gas.py <csv> <tag>
Checks non-template rows whose note mentions a blood gas. Inputs that only a blood gas provides (pH, PaO2, PaCO2, FiO2,
A-a gradient, oxygen saturation) are out of scope. For each other input taken from a blood gas, the model reports whether
the note gives a serum/plasma chemistry value for the same analyte at the time point the question asks for.
Writes blood_gas_<tag>.jsonl and blood_gas_<tag>.csv in equation_audit/. Reruns resume from the JSONL.
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

import run_gpt6_astra as audit

ROOT = Path(__file__).resolve().parent
BLOOD_GAS = re.compile(r'blood gas|\bABG\b|\bVBG\b|gasometr|arterial gas|arterial (blood|pH|sample)|venous (pH|blood gas)|'
                       r'\bpa?CO2\b|pCO₂|PaCO₂|\bpa?O2\b|PaO₂|base (excess|deficit)|gas analys', re.I)

PROMPT = """You are checking where the extracted inputs of one medical-calculator example come from.
Rule being checked: laboratory inputs such as sodium, potassium, chloride, bicarbonate/total CO2, glucose, BUN, creatinine,
calcium, albumin or hematocrit must come from the serum/plasma chemistry panel (or CBC), not from a blood gas (ABG/VBG)
analyzer. Inputs that only a blood gas provides (pH, PaO2, PaCO2, FiO2, A-a gradient, oxygen saturation) are allowed and
out of scope.
For each in-scope extracted input, decide from the note whether its value was reported as part of a blood gas result.
If it was, say whether the note also gives a serum/plasma chemistry value for the same analyte at the time point the
question asks for (quote it), or none.
Respond with JSON only:
{"inputs_from_blood_gas": [{"input": "...", "extracted": "...", "blood_gas_text": "exact note text",
                            "serum_alternative": "exact note text or null", "serum_value": "value with unit or null"}],
 "explanation": "..."}
Return an empty list when no in-scope input comes from a blood gas, or when it cannot be told apart from the chemistry panel."""


def main():
    source, tag = Path(sys.argv[1]), sys.argv[2]
    df = pd.read_csv(source, dtype=str, keep_default_na=False)
    df = df[(df['Note Type'] != 'Template') & df['Patient Note'].str.contains(BLOOD_GAS)]
    rows = df.to_dict('records')
    by_row = {int(r['Row Number']): r for r in rows}

    def check(row, path):
        user = (f"Calculator: {row['Calculator Name']}\n\nQuestion:\n{row['Question']}\n\nPatient note:\n{row['Patient Note']}\n\n"
                f"Extracted inputs:\n{row['Relevant Entities']}")
        try:
            out, tokens = audit.chat(PROMPT, user)
        except Exception as e:
            print(f"row {row['Row Number']}: {e}", file=sys.stderr)
            return
        audit.append(path, {'Row Number': int(row['Row Number']), **out, 'Tokens': tokens})

    print(f'{len(rows)} rows mention a blood gas in {source.name}')
    done = audit.run(rows, check, ROOT / f'blood_gas_{tag}.jsonl')
    out = []
    for n, c in sorted(done.items()):
        r, hits = by_row[n], c.get('inputs_from_blood_gas') or []
        out.append({'Row Number': n, 'Calculator ID': r['Calculator ID'], 'Calculator Name': r['Calculator Name'], 'Note ID': r['Note ID'],
                    'Uses Blood Gas Value': bool(hits), 'Inputs From Blood Gas': ', '.join(h.get('input', '') for h in hits),
                    'Serum Alternative Available': ', '.join(f"{h.get('input')}: {h.get('serum_value')}" for h in hits if h.get('serum_value')),
                    'Details': json.dumps(hits, ensure_ascii=False), 'Explanation': c.get('explanation', ''),
                    'Relevant Entities': r['Relevant Entities'], 'Ground Truth Answer': r['Ground Truth Answer']})
    out = pd.DataFrame(out)
    out.to_csv(ROOT / f'blood_gas_{tag}.csv', index=False, encoding='utf-8-sig')
    print(out['Uses Blood Gas Value'].value_counts().to_string())
    if len(done) < len(rows):
        print(f'{len(rows) - len(done)} rows failed; rerun to retry.')


if __name__ == '__main__':
    main()
