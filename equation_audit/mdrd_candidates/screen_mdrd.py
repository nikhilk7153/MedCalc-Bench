"""Screen MDRD candidate notes (Verified training set) for eligibility with gpt-6-astra, one call per note.

Run: OPENAI_API_KEY=... python3 equation_audit/mdrd_candidates/screen_mdrd.py
Eligible = chronic kidney disease present at admission, not on dialysis (or transplant) at admission, no acute kidney
injury at admission, and a serum creatinine measured at admission before treatment. Writes screen_mdrd.jsonl/.csv here.
"""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import run_gpt6_astra as audit  # noqa: E402

PROMPT = """You are screening patient notes for a benchmark question that asks for the MDRD GFR at the time of admission,
before any treatment. A note is ELIGIBLE only if all hold at admission: (1) the patient has chronic kidney disease
(documented CKD, chronic renal failure/insufficiency, or diabetic nephropathy); (2) the patient is NOT on dialysis
(hemodialysis or peritoneal) and has no kidney transplant; (3) there is no acute kidney injury or acute-on-chronic
worsening at admission; (4) a serum creatinine value measured at admission, before treatment, is documented; age and sex
are documented. Dialysis started only after admission does not disqualify (1)-(2), but then use only pre-dialysis values.
Respond with JSON only:
{"eligible": true|false, "reasons_failed": ["..."], "ckd_quote": "...", "dialysis_quote": "...",
 "admission_creatinine": "value and unit as written, or null", "creatinine_quote": "...", "age": "...", "sex": "..."}"""


def main():
    df = pd.read_csv(HERE / 'keyword_candidates_45.csv', dtype=str, keep_default_na=False)
    rows = [dict(r, **{'Row Number': str(i + 1)}) for i, r in enumerate(df.to_dict('records'))]

    def check(row, path):
        try:
            out, tokens = audit.chat(PROMPT, f"Patient note:\n{row['Patient Note']}")
        except Exception as e:
            print(f"{row['Note ID']}: {e}", file=sys.stderr)
            return
        audit.append(path, {'Row Number': int(row['Row Number']), 'Note ID': row['Note ID'], **out, 'Tokens': tokens})

    done = audit.run(rows, check, HERE / 'screen_mdrd.jsonl')
    out = pd.DataFrame([{**v, 'Relevant Entities': rows[n - 1]['Relevant Entities'],
                         'Stored Answer': rows[n - 1]['Ground Truth Answer']} for n, v in sorted(done.items())])
    out.to_csv(HERE / 'screen_mdrd.csv', index=False)
    print(out['eligible'].value_counts().to_string())


if __name__ == '__main__':
    main()
