"""Build the physician review sheet for the corrected test set: every correction to a scored (rule-based) row,
and the judgment calls that were kept unchanged.

Run: python3 equation_audit/build_physician_review.py <labels-before-corrections.csv>
<labels-before-corrections.csv> is the generated test set from before the per-row overrides (same row numbers).
Writes dataset/physician_review.csv. Groups, in review order:
  1 Rule: one decision covers several rows      2 Case: individual clinical reading
  3 Mechanical: extraction contradicts the note (confirm)      4 Kept: judgment calls left unchanged
"Label If Rejected" re-scores the row with that row's changes reverted (calculator code in medcalc_verified_code_snapshot).
"""
import ast
import glob
import importlib.util
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / 'dataset'
IMPL = ROOT.parent / 'medcalc_verified_code_snapshot/calculator_implementations'
sys.path.insert(0, str(IMPL))
from rounding import round_number  # noqa: E402

CALC = json.load(open(IMPL / 'name_to_python.json'))
RULE_BASED = {'diagnosis', 'risk', 'severity'}

RULES = {
    'R1': ('Centor/FeverPAIN: when the note does not mention cough (or coryza), is the "absence of cough/coryza" criterion met (+1)?',
           ['301', '313', '315', '320', '448', '451', '452', '465']),
    'R2': ('Caprini: does the operation of the current admission also count as "major surgery in the last month", and does bed rest '
           'without a stated duration count as "confined to bed >72 hours"?', ['909', '910', '913', '914', '918', '921']),
    'R3': ('APACHE II: count acute renal failure (doubles the creatinine points) only when it is explicitly documented, not inferred '
           'from creatinine or low urine output?', ['870', '871', '877']),
    'R4': ('HAS-BLED: does anemia (hemoglobin about 8 g/dL) count as "predisposition to bleeding"?', ['346', '362']),
    'R5': ("Handwritten note missing a value the score needs: restore the value v1.0's own inputs used (note edit), or replace the note?",
           ['869', '875', '890']),
}
CASES = {
    '849': 'Verbal response: is "I am in the Lego aisle of Target" confused (V4) or inappropriate words (V3)?',
    '861': 'Tracheostomy prevents speech: verbal "not testable" (the code gives full points) or "no verbal response" (1)?',
    '264': 'ECG with atrial fibrillation and rapid ventricular response, no ST changes: normal (0) or non-specific repolarization disturbance (1)?',
    '274': 'ECG with atrial fibrillation and abnormal R-wave progression: normal (0) or non-specific repolarization disturbance (1)? (Also: obesity from BMI 30.7.)',
    '266': 'Which ECG is the presenting one: the normal pre-exercise baseline, or the ST-deviation ECG at transfer (after office medications)?',
    '245': 'Do mild bibasilar crackles meet the RCRI heart-failure criterion?',
    '428': 'Age at the first admission is inferred as 61 (59 at presentation + about 2 years): use 61 or 59?',
    '430': 'CKD documented without stage or labs: moderate-to-severe CKD?',
    '434': 'Abdominal-wall lesion present at admission, later confirmed as acinic cell cancer from a prior parotid primary: localized or metastatic solid tumor?',
    '446': 'Stage IIIb CKD with creatinine 1.92 mg/dL at the first evaluation: moderate-to-severe CKD? Age at that evaluation (55) or later (65)?',
    '458': 'Tonsillar purulence with surrounding erythema: severe tonsil inflammation?',
    '461': 'Illness began a week before admission: symptom onset <=3 days? Purulent nasal secretions: coryza (so no point for absence of cough/coryza)?',
    '131': 'CT angiography was done to exclude PE: was PE the #1 or an equally likely diagnosis?',
    '224': 'Leg pain and claudication with arterial thrombosis: localized tenderness along the deep veins?',
    '238': 'Cervical cancer treated 3 years earlier, metastasis found only after admission: active cancer at admission? Trochanteric tenderness: deep-vein tenderness?',
    '271': 'Father with cardiovascular disease, age at onset not stated: family history before age 65?',
    '355': 'Splenic rupture with perisplenic bleeding found at presentation: counts as major bleeding for HAS-BLED?',
    '405': 'Note edit removed "His vital signs were initially normal, however," so the pre-treatment 80/60 mm Hg and HR 100 are the admission vitals. Alternative: score the initially normal vitals (label 10).',
    '889': 'Note edit: "no urine output noted, raising concerns..." changed to "Urine output has not been recorded." Alternative: read it as anuria (renal 4 points).',
    '858': 'Note edit: "Pupillary evaluation was not possible" changed to "Eye opening could not be assessed" (the inputs score eye response as not testable).',
    '863': 'Note edit: "visual evaluation was not possible" changed to "eye opening could not be assessed" (the inputs score eye response as not testable).',
    '218': 'Note edit: ascites on examination had no stated severity; the note now says "slight ascites" (the severity v1.1\'s inputs use). Alternative: moderate ascites (label 8).',
    '269': 'Note replaced: troponin 0.44 ng/mL given without a reference limit. Alternative: keep the original note and score troponin >3x normal.',
}
KEPT = {
    'K1': "Wells' DVT, note with no suspected DVT: is \"alternative diagnosis as likely or more likely\" true (-2) or false (0)?",
    'K2': 'HEART: no ECG or troponin result documented, scored as normal ECG and troponin <= normal (0 points). Keep, or exclude the row?',
    'K3': 'GCS: a component documented as not testable is scored as full points (the code\'s rule). Keep this rule?',
    'K4': 'Caprini: previous DVT and previous PE are scored separately (3 points each). Keep, or count once?',
    'K5': 'APACHE II: "liver cirrhosis" (no biopsy or portal hypertension stated) counted as severe organ insufficiency. Keep?',
    'K6': 'Score applied outside its intended population (see the checker\'s reason). Keep the row, or exclude it from the benchmark?',
}
# rows whose change is not visible against the before snapshot, or whose alternative needs a different input
SPECIAL = {
    '218': (None, '8 (moderate ascites)'),
    '914': ('Surgery Type: laparoscopic -> none (applied before the snapshot; Verified label 8)', '8'),
    '405': (None, '10 (initial vitals scored as normal)'),
    '875': (None, '12 (potassium 139 as written)'),
    '889': (None, '14 (no urine output read as anuria)'),
}
K2_EXTRA = {'280'}  # original note restored 2026-10-17: "an ECG is performed" with no result, no troponin
OUT_OF_SCOPE = re.compile(r'intended for|validated (for|in)|outside (its|the) (intended|validated)|not validated|intended population|'
                          r'not clinically (valid|appropriate|applicable)|not applicable', re.I)


def compute(cid, entities, category):
    info = CALC[cid]
    params = entities if cid == '49' else {info[k]: v for k, v in entities.items()}
    spec = importlib.util.spec_from_file_location(Path(info['file path']).stem, IMPL / info['file path'])
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    ans = getattr(mod, info['explanation function'])(params)['Answer']
    return str(float(round_number(float(ans)))) if category in ('lab', 'lab test', 'physical', 'dosage') else str(ans)


def diff(before, after):
    out = []
    for k in sorted(set(before) | set(after)):
        if before.get(k, '<absent>') != after.get(k, '<absent>'):
            unit_only = isinstance(before.get(k), list) and before[k][1] in ('degrees celsisus', 'degrees fahreinheit')
            out.append(f"{k}: {before.get(k, '<absent>')} -> {after.get(k, '<absent>')}" + (' [global unit-spelling fix]' if unit_only else ''))
    return '; '.join(out)


def main():
    R = lambda p: pd.read_csv(p, dtype=str, keep_default_na=False)
    final, before = R(DATA / 'test_data_final.csv'), R(sys.argv[1]).set_index('Row Number')
    changes, overrides = R(DATA / 'test_data_corrected_changes.csv'), R(DATA / 'test_data_corrected_overrides.csv')
    ov_rows = final.merge(overrides, on=['Calculator ID', 'Note ID'])
    reasons = ov_rows.groupby('Row Number')['Reason'].agg(lambda s: ' | '.join(dict.fromkeys(s))).to_dict()
    replaced = {r['Row Number']: r['Reason'] for _, r in changes.iterrows() if 'checked' in r['Action']}
    f = final.set_index('Row Number')
    corrected = sorted({n for n in ov_rows['Row Number'] if f.loc[n, 'Category'] in RULE_BASED} |
                       {n for n in replaced if f.loc[n, 'Category'] in RULE_BASED}, key=int)

    rule_of = {n: k for k, (_, rows) in RULES.items() for n in rows}
    items = []
    for n in corrected:
        r, b = f.loc[n], before.loc[n]
        group, prompt = (('1 Rule', f'{rule_of[n]}: {RULES[rule_of[n]][0]}') if n in rule_of else
                         ('2 Case', CASES[n]) if n in CASES else ('3 Mechanical', 'Confirm the correction'))
        if n in replaced:
            kind, what, if_rejected = 'note replaced', f"Note {b['Note ID']} replaced by Verified note {r['Note ID']}", b['Ground Truth Answer']
        else:
            be, fe = ast.literal_eval(b['Relevant Entities']), ast.literal_eval(r['Relevant Entities'])
            what = diff(be, fe)
            kinds = (['inputs'] if what else []) + (['note edit'] if b['Patient Note'] != r['Patient Note'] else [])
            kind = ' + '.join(kinds)
            note_edits = ov_rows[(ov_rows['Row Number'] == n) & (ov_rows['Kind'] == 'note edit')]['Change']
            for c in note_edits:
                what = '; '.join(filter(None, [what] + [f'NOTE: "{o}" -> "{w}"' for o, w in json.loads(c)]))
            if 'celsisus' in b['Patient Note'] and 'note edit' not in kind:
                kind = (kind + ' + ' if kind else '') + 'note spelling (global)'
            reverted = {**fe, **{k: be[k] for k in be if be[k] != fe.get(k)}}
            for k in [k for k in fe if k not in be]:
                reverted.pop(k)
            for k, v in reverted.items():  # keep the global temperature-unit fix when reverting
                if isinstance(v, list) and len(v) == 2 and v[1] in ('degrees celsisus', 'degrees fahreinheit'):
                    reverted[k] = fe[k]
            if_rejected = compute(r['Calculator ID'], reverted, r['Category']) if what else r['Ground Truth Answer']
        if n in SPECIAL:
            what = SPECIAL[n][0] or what
            kind = kind or 'inputs'
            if_rejected = SPECIAL[n][1]
        items.append({'Group': group, 'Rule / Question for Reviewer': prompt, 'Row Number': n, 'Calculator Name': r['Calculator Name'],
                      'Note ID': r['Note ID'], 'Note Type': r['Note Type'], 'Change Type': kind, 'What Changed (before -> after)': what,
                      'Reason Given': replaced.get(n) or reasons.get(n, ''), 'Label Before Corrections': b['Ground Truth Answer'],
                      'Current Label': r['Ground Truth Answer'], 'Label If Rejected': if_rejected,
                      'Original Patient Note': b['Patient Note'] if b['Patient Note'] != r['Patient Note'] else ''})

    # judgment calls kept unchanged
    key = lambda c, note: (c, note.strip())
    by_note = {key(c, t): n for n, c, t in zip(final['Row Number'], final['Calculator ID'], final['Patient Note'])}
    ent_problems, oos = {}, {}
    for p in glob.glob(str(ROOT / 'entities_*.csv')):
        for _, x in R(p).iterrows():
            n = by_note.get(key(x['Calculator ID'], x['Patient Note']))
            if n and x['Relevant Entities'] == f.loc[n, 'Relevant Entities']:
                ent_problems.setdefault(n, []).extend(json.loads(x['Problems'] or '[]'))
    for p in glob.glob(str(ROOT / 'final_*_verify.csv')):
        if re.search(r'one_shot|oneshot|calib|candidates', p):
            continue
        for _, x in R(p).iterrows():
            n = by_note.get(key(x['Calculator ID'], x['Patient Note']))
            if n and x['Verdict'] == 'ambiguous' and OUT_OF_SCOPE.search(x['Explanation']):
                oos[n] = x['Explanation']
    kept = []
    for n, r in f.iterrows():
        e = ast.literal_eval(r['Relevant Entities']) if r['Note Type'] != 'Template' else {}
        probs = ent_problems.get(n, [])
        if r['Calculator ID'] == '16' and e.get('Alternative diagnosis to Deep Vein Thrombosis as likely or more likely') is True and \
                any(p.get('input', '').startswith('Alternative diagnosis') for p in probs):
            alt = compute('16', {**e, 'Alternative diagnosis to Deep Vein Thrombosis as likely or more likely': False}, r['Category'])
            kept.append(('K1', n, 'Alternative diagnosis: True', alt))
        if r['Calculator ID'] == '18' and (n in K2_EXTRA or any(p.get('input') in ('Electrocardiogram Test', 'Initial troponin') and p.get('issue') == 'not_in_note' for p in probs)):
            kept.append(('K2', n, f"ECG: {e.get('Electrocardiogram Test')}; troponin: {e.get('Initial troponin')}", ''))
        if r['Calculator ID'] == '21' and 'not testable' in e.values():
            kept.append(('K3', n, '; '.join(f'{k}: not testable' for k, v in e.items() if v == 'not testable'), ''))
        if r['Calculator ID'] == '36' and e.get('Previously documented Deep Vein Thrombosis') is True and e.get('Previously Documented Pulmonary Embolism') is True:
            kept.append(('K4', n, 'Previous DVT: True; previous PE: True', str(int(float(r['Ground Truth Answer'])) - 3)))
        if r['Calculator ID'] == '28' and n == '871':
            kept.append(('K5', n, 'History of severe organ failure or immunocompromise: True', compute('28', {**e, 'History of severe organ failure or immunocompromise': False}, r['Category'])))
        if n in oos:
            kept.append(('K6', n, oos[n][:400], ''))
    for k, n, what, alt in kept:
        r = f.loc[n]
        items.append({'Group': '4 Kept', 'Rule / Question for Reviewer': f'{k}: {KEPT[k]}', 'Row Number': n, 'Calculator Name': r['Calculator Name'],
                      'Note ID': r['Note ID'], 'Note Type': r['Note Type'], 'Change Type': 'none (kept)', 'What Changed (before -> after)': what,
                      'Reason Given': '', 'Label Before Corrections': before.loc[n, 'Ground Truth Answer'], 'Current Label': r['Ground Truth Answer'],
                      'Label If Rejected': alt, 'Original Patient Note': ''})

    out = pd.DataFrame(items)
    out['_g'] = out['Group'].str[0]; out['_r'] = out['Rule / Question for Reviewer'].str.extract(r'^(R\d|K\d)', expand=False).fillna('')
    out = out.sort_values(['_g', '_r', 'Row Number'], key=lambda s: s.astype(int) if s.name == 'Row Number' else s).drop(columns=['_g', '_r'])
    out.insert(0, 'Item', range(1, len(out) + 1))
    for c in ['Decision (Accept / Reject / Other)', 'Correct Value or Label If Other', 'Reviewer', 'Comment']:
        out[c] = ''
    out['Question'] = out['Row Number'].map(f['Question'])
    out['Patient Note'] = out['Row Number'].map(f['Patient Note'])
    out['Current Relevant Entities'] = out['Row Number'].map(f['Relevant Entities'])
    out.to_csv(DATA / 'physician_review.csv', index=False, encoding='utf-8-sig')
    print(len(out), 'items:', out['Group'].value_counts().sort_index().to_dict())
    print(out[out['Group'] == '4 Kept']['Rule / Question for Reviewer'].str[:2].value_counts().sort_index().to_dict())


if __name__ == '__main__':
    main()
