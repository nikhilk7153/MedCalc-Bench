"""Build the final one-shot set: the original v1.0 examples, with only the wrong ones replaced by Verified's.

Run: python3 equation_audit/build_one_shot_final.py
Inputs: v1.0 one-shot CSV and JSON (this project), Verified one-shot CSV and JSON with the Anion Gap fix
(dataset/one_shot_final_verified_base.csv / .json). Decisions come from equation_audit/final_one_shot_v10_final_verify.csv
and final_one_shot_verify.csv, reviewed by hand (see REPLACE).
Writes dataset/one_shot_final.csv, dataset/one_shot_final_explanation.json (what evaluation/run.py reads) and
dataset/one_shot_final_changes.csv.
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
V10_CSV = Path.home() / 'Documents/MedCalc-Bench-Verified/datasets/one_shot_data_v1.0.csv'
V10_JSON = ROOT / 'evaluation/one_shot_finalized_explanation.json'
VER_CSV = ROOT / 'dataset/one_shot_final_verified_base.csv'
VER_JSON = ROOT / 'dataset/one_shot_final_verified_base.json'

# calculator ID -> why the v1.0 example is replaced by Verified's
REPLACE = {
    '28': 'APACHE II: v1.0 label 32 does not follow from the documented values',
    '36': 'Caprini: v1.0 label 1 misses stroke, central line, thrombophilia and other factors',
    '46': 'Framingham: cholesterol substituted in mmol/L into the mg/dL equation (label 0.0)',
    '31': 'HOMA-IR: insulin multiplied by 6 instead of divided (label 30.271)',
    '29': 'PSI: age 25 used instead of the documented 72',
    '49': 'MME: fentanyl buccal dosed in mg',
    '4': 'CHA2DS2-VASc: heart failure counted but not documented (Verified relabels the same note)',
    '25': 'HAS-BLED: TIA counted as stroke (Verified relabels the same note)',
    '17': 'RCRI: intraperitoneal surgery not counted (Verified relabels the same note)',
    '15': 'Child-Pugh: v1.0 score 4 is impossible (minimum 5); Verified gives 6 for the same note',
    '38': 'Free Water Deficit: a 22 kg weight loss was used as the body weight',
    '43': 'SOFA: creatinine and bilirubin entered with wrong units',
    '68': 'Estimated Date of Conception: use Verified\'s note (no cycle length), as in the final test set',
}


def main():
    v10 = pd.read_csv(V10_CSV, dtype=str, keep_default_na=False)
    ver = pd.read_csv(VER_CSV, dtype=str, keep_default_na=False).set_index('Calculator ID')
    v10_json, ver_json = json.load(open(V10_JSON)), json.load(open(VER_JSON))
    assert len(v10) == 55 and set(v10_json) == set(ver_json) == set(v10['Calculator ID'])

    out, log, out_json = [], [], {}
    for _, row in v10.iterrows():
        cid = row['Calculator ID']
        if cid in REPLACE:
            new = ver.loc[cid].copy()
            new['Calculator ID'] = cid
            new['Row Number'], new['Category'], new['Output Type'] = row['Row Number'], row['Category'], row['Output Type']
            out.append(new[v10.columns])
            out_json[cid] = ver_json[cid]
            log.append({'Calculator ID': cid, 'Calculator Name': row['Calculator Name'], 'Reason': REPLACE[cid],
                        'Old Answer': row['Ground Truth Answer'], 'New Answer': new['Ground Truth Answer'],
                        'Same Note': row['Patient Note'].strip() == new['Patient Note'].strip()})
        else:
            out.append(row)
            out_json[cid] = v10_json[cid]
    out = pd.DataFrame(out)[v10.columns]
    for cid, entry in out_json.items():  # JSON and CSV must describe the same example
        assert entry['Patient Note'].strip() == out.set_index('Calculator ID').loc[cid, 'Patient Note'].strip(), cid
    out.to_csv(ROOT / 'dataset/one_shot_final.csv', index=False)
    with open(ROOT / 'dataset/one_shot_final_explanation.json', 'w') as f:
        json.dump({k: out_json[k] for k in v10_json}, f, indent=4)
    pd.DataFrame(log).to_csv(ROOT / 'dataset/one_shot_final_changes.csv', index=False)
    print(f'{len(out)} examples: {len(log)} replaced with Verified, {len(out) - len(log)} kept from v1.0')


if __name__ == '__main__':
    main()
