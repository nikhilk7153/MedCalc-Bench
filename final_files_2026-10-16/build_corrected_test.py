"""Build a corrected v1.0 test set by swapping flagged lab/physical rows and matched rule-based rows for MedCalc-Bench Verified rows.

Run: python3 equation_audit/build_corrected_test.py (after compare_verified.py).
dataset/test_data.csv is not modified. Rows are renumbered 1..N after top-ups, so v1.0 row numbers after an
inserted row shift; the changes log records both numbers for every changed or added row.
- Flagged rows whose note Verified kept and fixed: replaced by that Verified row.
- Flagged rows Verified removed, or whose Verified version is still wrong: replaced by an unused Verified row
  for the same calculator whose note is not in v1.0 (lowest Verified Row Number first).
- Rule-based rows (RULE_BASED categories) with the same Calculator ID + Note ID in Verified and a different
  Ground Truth Answer: replaced by that row. Clinician-handwritten notes (integer Note ID, not a template) are kept.
- REPLACE_CALCULATORS: every row is replaced by the Verified row with the same Calculator ID + Note ID.
- TOP_UP calculators get extra new Verified notes from the same pool, inserted after their last row.
- V11_CALCULATORS (handwritten notes): every row is replaced, in order, by the v1.1 test row for that calculator,
  or by Verified's row when Verified has the same Calculator ID + Note ID.
  The other handwritten calculators (SOFA, APACHE II, GCS, PSI, Caprini) stay as in v1.0.
- FROM_1_17 calculators (Charlson, Child-Pugh, GBS, HAS-BLED, RCRI): Patient Note and Relevant Entities from
  dataset/test_data_1_17_2025.csv, same Calculator ID + Note ID.
- Steroid Conversion rows that convert a steroid to itself (only the route differs): replaced by Verified's row
  with the same Note ID, which converts between different drugs.
- Rows kept from v1.0 whose Patient Note is identical to Verified's get Verified's Relevant Entities.
- Relevant Entities keys from v1.0 are renamed to the names Verified's name_to_python.json uses (ENTITY_RENAMES);
  keys with no current equivalent (ENTITY_DROPS) are removed.
- Clinician-written notes (integer Note ID, not template) get Note Type "Synthetic"; PMC/USMLE case reports get "Extracted"
  even where Verified labels them Synthetic.
- Every row's Calculator Name is set to Verified's name for its calculator.
- Every row's Question is Verified's question for its calculator without the time-point sentence (that sentence goes in
  the system prompt); Steroid Conversion keeps each row's own question (it names the note's drug pair).
- Test rows whose note is the one-shot example for the same calculator (dataset/one_shot_finalized.csv) are replaced:
  by Verified's row with the same Note ID when that note differs, otherwise by an unused Verified note.
- NOTE_TYPO_FIXES: spelling fixes applied to every Patient Note.
- Per-row corrections recorded in equation_audit/overrides/: replacements.json (replace a row by a specific
  Verified test row, or a row of extra_notes.csv, that passed verify_dataset.py and verify_entities.py), note_edits.json (exact text replacements in the
  Patient Note) and entity_overrides.json (set/remove Relevant Entities keys), keyed by Calculator ID + Note ID.
  Labels are then recomputed from the entities by run_generate_csv.py.
- All other rows are unchanged.
"""
from pathlib import Path

import ast
import json
import re

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / 'dataset'
KEY = ['Calculator ID', 'Note ID']
SWAP_SAME_NOTE = {'Resolved: label corrected to our value', 'Resolved: note edited',
                  'Resolved: missing inputs added and relabelled'}
SUBSTITUTE = {'Resolved: row removed or replaced', 'Still open', 'Changed, but new label questionable'}
# Whole calculators to take from Verified (none: the v1.0 QTc template rows are kept)
REPLACE_CALCULATORS = set()
RULE_BASED = {'diagnosis', 'risk', 'severity'}
V11_TEST = Path.home() / 'Downloads/MedCalc-Bench-version-1.1/dataset/test_data.csv'
STEROID_ID = '24'
# Rule-based rows found wrong or unusable on review: replaced by an unused, checked Verified note for the same calculator
RULE_SUBSTITUTE = {'595': 'SIRS: 14-year-old scored with adult thresholds', '600': 'SIRS: young child scored with adult thresholds',
                   '601': 'SIRS: young child scored with adult thresholds', '605': 'SIRS: young child scored with adult thresholds',
                   '602': 'SIRS: WBC is ascitic fluid, not blood', '607': 'SIRS: WBC and vital signs from encounters 27 days apart'}
# v1.0 rows kept even though Verified's same note has a different answer, because Verified's edit is wrong
KEEP_V10 = {'604': "SIRS: Verified changed WBC 910 to 9100 while the note still says leukopenia"}
# MME template notes that dose fentanyl in mg (impossible for buccal fentanyl or patches): use Verified's
# regenerated note with the same Note ID, which doses fentanyl in µg
MME_ID = '49'
FENTANYL_IN_MG = re.compile(r'\d+ mg of Fent', re.IGNORECASE)
# Flagged v1.0 rows in non-handwritten calculators whose Verified version of the same note passed the direct check
# (equation_audit/final_v10_flagged_candidates_verify.csv): use Verified's row
V10_FLAGGED_TO_VERIFIED = {
    ('20', 'usmle-12316'), ('20', 'usmle-6356'), ('26', 'usmle-5686'), ('33', 'pmc-7219994-1'),
    ('45', 'usmle-8555'), ('68', '1'), ('68', '10'), ('68', '11'),
    ('68', '12'), ('68', '14'), ('68', '15'), ('68', '16'),
    ('68', '17'), ('68', '18'), ('68', '19'), ('68', '2'),
    ('68', '20'), ('68', '3'), ('68', '4'), ('68', '5'),
    ('68', '6'), ('68', '7'), ('68', '8'), ('68', '9')}
# Verified rows not to use as replacements
VERIFIED_EXCLUDE = {'655': 'SIRS: WBC given in cells/mL',
                    '441': 'HOMA-IR: same note as the HOMA-IR one-shot example'}
ONE_SHOT = DATA / 'one_shot_finalized.csv'
OVERRIDES = ROOT / 'overrides'
# (calculator, entity, wrong unit) -> correct unit
ENTITY_UNIT_FIXES = {('51', 'White blood cell count', 'm^3'): 'mm^3',
                     # the code treats any temperature unit other than 'degrees celsius' as Fahrenheit
                     ('28', 'Temperature', 'degrees celsisus'): 'degrees celsius', ('29', 'Temperature', 'degrees celsisus'): 'degrees celsius',
                     ('20', 'Temperature', 'degrees fahreinheit'): 'degrees fahrenheit', ('29', 'Temperature', 'degrees fahreinheit'): 'degrees fahrenheit',
                     ('43', 'norEPINEPHrine', 'mc/kg/min'): 'mcg/kg/min'}
# Spelling fixes applied to every Patient Note
NOTE_TYPO_FIXES = {'degrees celsisus': 'degrees celsius'}
# v1.0 entity name -> Verified name, only for the calculators whose current code expects the new name
_QT = {'QT interval': 'QT Interval'}
_O2 = {'FiO₂': 'FiO2', 'Partial pressure of oxygen (PaO₂) for Apache II': 'PaO2'}
ENTITY_RENAMES = {
    '11': _QT, '56': _QT, '57': _QT, '58': _QT, '59': _QT,
    '28': {**_O2, 'Chronic Renal Failure': 'Chronic renal failure'},
    '43': {**_O2, 'Continous positive airway pressure': 'Continuous positive airway pressure'},
    '29': {'Partial pressure of oxygen (PaO₂) for Apache II': 'Partial pressure of oxygen',
           'Congestive Heart Faliure': 'Congestive Heart Failure', 'Renal disease': 'Renal disease history',
           'Liver disease severity': 'Liver disease history'},
    '18': {'Diabetes mellitus criteria for CCI rule': 'Diabetes mellitus'},
    '33': {'Symptom onset ≤3 days': 'Symptom onset <=3 days'},
    # v1.0 names in the handwritten Charlson, GBS, HAS-BLED and RCRI notes (back in use since 2026-10-20)
    '32': {'Congestive Heart Faliure': 'Congestive Heart Failure', 'Chronic Obstructive Pulmonary Disease': 'Chronic Pulmonary Disease',
           'Chronic Obstructive Disease': 'Chronic Pulmonary Disease', 'solid tumor': 'Solid tumor',
           'Diabetes mellitus criteria for CCI rule': 'Diabetes mellitus'},
    '27': {'Recent syncope': 'Recent Syncope', 'Cardiac failure present': 'Cardiac Failure Present', 'Melena present': 'Melena Present'},
    '25': {'Hypertension history': 'Hypertension'},
}
WELLS_DVT_COMBINED = 'Bedridden recently >3 days or major surgery within 12 weeks'  # split into the two Verified keys
# v1.0's RCRI rows carry an always-False duplicate 'Cerebrovascular disease history' next to the real 'History of
# cerebrovascular disease' read from the note; v1.0's code scored the duplicate, so prior stroke/TIA never counted
ENTITY_DROPS = {('17', 'Cerebrovascular disease history'), ('43', 'Continuous veno-venous hemodialysis for ≥24 hours in the past week'), ('68', 'cycle length'),
                ('33', 'Altered mental status for PECARN head injury criteria'),
                ('33', 'Occipital, parietal or temporal scalp hematoma; history of level of conciousness (LOC) ≥5 sec; '
                       'not acting normally per parent or severe mechanism of injury?')}
# cci.py matches these values exactly ('none', 'localized', 'metastatic', 'mild', 'moderate to severe')
LOWERCASE_VALUES = {('32', 'Solid tumor'), ('32', 'Liver disease severity')}
# Charlson, Child-Pugh, Glasgow-Blatchford, HAS-BLED and RCRI used v1.1's notes until 2026-10-20; reverted to v1.0's own
# (handwritten) notes to keep edits to v1.0 minimal (user decision)
V11_CALCULATORS = set()
# The same handwritten notes as corrected in test_data_1_17_2025.csv (RCRI creatinine in mg/dL, HAS-BLED bleeding history,
# Charlson inputs in the values cci.py expects): each row's Patient Note and Relevant Entities are taken from that file's row
# with the same Calculator ID + Note ID (user decision, 2026-10-20). Labels are recomputed by run_generate_csv.py.
TEST_1_17 = DATA / 'test_data_1_17_2025.csv'
FROM_1_17 = {'32', '15', '27', '25', '17'}
TIME_POINT = "You should use the patient's medical values and health status when they were first admitted to the hospital prior to any treatment."
TOP_UP = {'7': 1}  # Calcium Correction has 19 rows in v1.0; bring it to 20


def same_steroid(question):
    m = re.search(r"dose of (.+?) (?:PO|IV), what is the equivalent dosage in mg of (.+?) (?:PO|IV)\?", question)
    return bool(m) and m.group(1).lower() == m.group(2).lower()


def rename_entities(cid, text):
    entities = ast.literal_eval(text)
    out = {}
    for key, value in entities.items():
        if (cid, key) in ENTITY_DROPS:
            continue
        if cid == '16' and key == WELLS_DVT_COMBINED:
            out['Bedridden recently >3 days'], out['Major surgery within 12 weeks'] = value, False
            continue
        if isinstance(value, list) and len(value) == 2 and (cid, key, value[1]) in ENTITY_UNIT_FIXES:
            value = [value[0], ENTITY_UNIT_FIXES[cid, key, value[1]]]
        key = ENTITY_RENAMES.get(cid, {}).get(key, key)
        if (cid, key) in LOWERCASE_VALUES and isinstance(value, str):
            value = value.lower()
        out[key] = value
    return str(out) if out != entities else text


def main():
    v10 = pd.read_csv(DATA / 'test_data.csv', dtype=str)
    ver = pd.read_csv(DATA / 'test_verified.csv', dtype=str)
    audit = pd.read_csv(ROOT / 'verified_comparison.csv', encoding='utf-8-sig', dtype=str)
    audit = audit[audit['Flag'] != 'Clean'].set_index('v1.0 Row Number')
    ver_by_row = ver.set_index('Row Number')
    v11 = pd.read_csv(V11_TEST, dtype=str)
    v11_rows = {cid: [r for _, r in g.iterrows()] for cid, g in v11[v11['Calculator ID'].isin(V11_CALCULATORS)].groupby('Calculator ID')}
    assert all(len(v11_rows[c]) == (v10['Calculator ID'] == c).sum() for c in V11_CALCULATORS)
    t117 = pd.read_csv(TEST_1_17, dtype=str)
    t117 = t117[t117['Calculator ID'].isin(FROM_1_17)]
    assert set(zip(t117['Calculator ID'], t117['Note ID'])) == set(zip(*v10.loc[v10['Calculator ID'].isin(FROM_1_17), KEY].T.values))
    t117 = {(r['Calculator ID'], r['Note ID']): r for _, r in t117.iterrows()}
    ver_by_note = dict(zip(zip(ver['Calculator ID'], ver['Note ID']), ver['Row Number']))
    ver_answer = dict(zip(zip(ver['Calculator ID'], ver['Note ID']), ver['Ground Truth Answer']))

    new_notes = ver.merge(v10[KEY], on=KEY, how='left', indicator=True)
    pool = {cid: list(g.sort_values('Row Number', key=lambda s: s.astype(int))['Row Number'])
            for cid, g in new_notes[(new_notes['_merge'] == 'left_only') & ~new_notes['Row Number'].isin(VERIFIED_EXCLUDE)]
            .groupby('Calculator ID')}

    out, log = [], []
    for _, row in v10.iterrows():
        n = row['Row Number']
        outcome = audit['Outcome'].get(n)
        if row['Calculator ID'] in V11_CALCULATORS:
            new = v11_rows[row['Calculator ID']].pop(0).copy()
            src = ver_by_note.get((new['Calculator ID'], new['Note ID']), '')
            if src:
                new = ver_by_row.loc[src].copy()
            new['Row Number'], new['Output Type'], new['Category'] = n, row['Output Type'], row['Category']
            out.append(new[v10.columns])
            log.append({'v1.0 Row Number': n, 'Calculator ID': row['Calculator ID'], 'Calculator Name': row['Calculator Name'],
                        'Action': 'Replaced with v1.1 handwritten note' + (' (Verified version)' if src else ''),
                        'Reason': 'Handwritten calculator taken from v1.1', 'Verified Row Number': src, 'Old Note ID': row['Note ID'], 'New Note ID': new['Note ID'],
                        'Old Ground Truth Answer': row['Ground Truth Answer'], 'New Ground Truth Answer': new['Ground Truth Answer'],
                        'Audit Flag': audit['Flag'].get(n, 'Clean')})
            continue
        if row['Calculator ID'] in FROM_1_17:
            src = t117[row['Calculator ID'], row['Note ID']]
            new = row.copy()
            new['Patient Note'], new['Relevant Entities'] = src['Patient Note'], src['Relevant Entities']
            out.append(new)
            if (new['Patient Note'], new['Relevant Entities']) != (row['Patient Note'], row['Relevant Entities']):
                log.append({'v1.0 Row Number': n, 'Calculator ID': row['Calculator ID'], 'Calculator Name': row['Calculator Name'],
                            'Action': 'Note and inputs from test_data_1_17_2025 (same note)',
                            'Reason': 'Corrected version of the handwritten note', 'Verified Row Number': '',
                            'Old Note ID': row['Note ID'], 'New Note ID': row['Note ID'],
                            'Old Ground Truth Answer': row['Ground Truth Answer'], 'New Ground Truth Answer': src['Ground Truth Answer'],
                            'Audit Flag': audit['Flag'].get(n, 'Clean')})
            continue
        if row['Calculator ID'] == STEROID_ID and same_steroid(row['Question']):
            src = ver_by_note[STEROID_ID, row['Note ID']]
            assert not same_steroid(ver_by_row.loc[src, 'Question'])
            new = ver_by_row.loc[src].copy()
            new['Row Number'], new['Output Type'], new['Category'] = n, row['Output Type'], row['Category']
            out.append(new[v10.columns])
            log.append({'v1.0 Row Number': n, 'Calculator ID': row['Calculator ID'], 'Calculator Name': row['Calculator Name'],
                        'Action': 'Replaced with Verified version of same note', 'Reason': 'Steroid converted to itself',
                        'Verified Row Number': src, 'Old Note ID': row['Note ID'], 'New Note ID': new['Note ID'],
                        'Old Ground Truth Answer': row['Ground Truth Answer'], 'New Ground Truth Answer': new['Ground Truth Answer'],
                        'Audit Flag': audit['Flag'].get(n, 'Clean')})
            continue
        if row['Calculator ID'] == MME_ID and FENTANYL_IN_MG.search(row['Patient Note']):
            src = ver_by_note[MME_ID, row['Note ID']]
            assert not FENTANYL_IN_MG.search(ver_by_row.loc[src, 'Patient Note'])
            new = ver_by_row.loc[src].copy()
            new['Row Number'], new['Output Type'], new['Category'] = n, row['Output Type'], row['Category']
            out.append(new[v10.columns])
            log.append({'v1.0 Row Number': n, 'Calculator ID': row['Calculator ID'], 'Calculator Name': row['Calculator Name'],
                        'Action': 'Replaced with Verified version of same note', 'Reason': 'MME: fentanyl dosed in mg',
                        'Verified Row Number': src, 'Old Note ID': row['Note ID'], 'New Note ID': new['Note ID'],
                        'Old Ground Truth Answer': row['Ground Truth Answer'], 'New Ground Truth Answer': new['Ground Truth Answer'],
                        'Audit Flag': audit['Flag'].get(n, 'Clean')})
            continue
        if (row['Calculator ID'], row['Note ID']) in V10_FLAGGED_TO_VERIFIED:
            src = ver_by_note[row['Calculator ID'], row['Note ID']]
            new = ver_by_row.loc[src].copy()
            new['Row Number'], new['Output Type'], new['Category'] = n, row['Output Type'], row['Category']
            out.append(new[v10.columns])
            log.append({'v1.0 Row Number': n, 'Calculator ID': row['Calculator ID'], 'Calculator Name': row['Calculator Name'],
                        'Action': 'Replaced with Verified version of same note', 'Reason': 'Flagged in v1.0; Verified version checked correct',
                        'Verified Row Number': src, 'Old Note ID': row['Note ID'], 'New Note ID': new['Note ID'],
                        'Old Ground Truth Answer': row['Ground Truth Answer'], 'New Ground Truth Answer': new['Ground Truth Answer'],
                        'Audit Flag': audit['Flag'].get(n, 'Clean')})
            continue
        handwritten = row['Note ID'].isdigit() and row['Note Type'] != 'Template'
        note = (row['Calculator ID'], row['Note ID'])
        if n in RULE_SUBSTITUTE:
            src, action = pool[row['Calculator ID']].pop(0), 'Replaced with new Verified note (same calculator)'
            outcome = RULE_SUBSTITUTE[n]
        elif (n not in KEEP_V10 and row['Category'] in RULE_BASED and not handwritten and note in ver_by_note
                and float(ver_answer[note]) != float(row['Ground Truth Answer'])):
            src, action = ver_by_note[row['Calculator ID'], row['Note ID']], 'Replaced with Verified version of same note'
            outcome = 'Rule-based note with 1:1 Verified match and a different answer'
        elif row['Calculator ID'] in REPLACE_CALCULATORS:
            src, action = ver_by_note[row['Calculator ID'], row['Note ID']], 'Replaced with Verified version (whole calculator)'
            outcome = 'Calculator replaced with Verified version'
        elif outcome in SWAP_SAME_NOTE:
            src, action = audit.loc[n, 'Verified Row Number'].split('.')[0], 'Replaced with Verified version of same note'
        elif outcome in SUBSTITUTE:
            src, action = pool[row['Calculator ID']].pop(0), 'Replaced with new Verified note (same calculator)'
        else:
            out.append(row)
            continue
        new = ver_by_row.loc[src].copy()
        new['Row Number'] = n
        new['Output Type'], new['Category'] = row['Output Type'], row['Category']  # Verified renames 'lab' to 'lab test'
        out.append(new[v10.columns])
        log.append({'v1.0 Row Number': n, 'Calculator ID': row['Calculator ID'], 'Calculator Name': row['Calculator Name'],
                    'Action': action, 'Reason': outcome, 'Verified Row Number': src,
                    'Old Note ID': row['Note ID'], 'New Note ID': new['Note ID'],
                    'Old Ground Truth Answer': row['Ground Truth Answer'], 'New Ground Truth Answer': new['Ground Truth Answer'],
                    'Audit Flag': audit['Flag'].get(n, 'Clean')})

    out = pd.DataFrame(out)[v10.columns].reset_index(drop=True)
    assert out['Row Number'].tolist() == v10['Row Number'].tolist()
    for cid, extra in TOP_UP.items():
        block = out[out['Calculator ID'] == cid]
        add = []
        for src in (pool[cid].pop(0) for _ in range(extra)):
            new = ver_by_row.loc[src].copy()
            new['Row Number'] = f'new-{src}'  # placeholder until renumbering
            new['Output Type'], new['Category'] = block.iloc[0]['Output Type'], block.iloc[0]['Category']
            add.append(new[v10.columns])
            log.append({'v1.0 Row Number': '', 'Calculator ID': cid, 'Calculator Name': block.iloc[0]['Calculator Name'],
                        'Action': 'Added new Verified note (calculator top-up)', 'Reason': f'v1.0 has {len(block)} rows',
                        'Verified Row Number': src, 'Old Note ID': '', 'New Note ID': new['Note ID'],
                        'Old Ground Truth Answer': '', 'New Ground Truth Answer': new['Ground Truth Answer'], 'Audit Flag': ''})
        at = block.index.max() + 1
        out = pd.concat([out.iloc[:at], pd.DataFrame(add), out.iloc[at:]], ignore_index=True)
    one_shot = pd.read_csv(ONE_SHOT, dtype=str, keep_default_na=False)
    one_shot_note = dict(zip(one_shot['Calculator ID'], one_shot['Patient Note'].str.strip()))
    for i, row in out.iterrows():
        cid = row['Calculator ID']
        if one_shot_note.get(cid) != row['Patient Note'].strip():
            continue
        src = ver_by_note.get((cid, row['Note ID']), '')
        if not src or ver_by_row.loc[src, 'Patient Note'].strip() == one_shot_note[cid]:
            src = pool[cid].pop(0)
        new = ver_by_row.loc[src].copy()
        new['Row Number'], new['Output Type'], new['Category'] = row['Row Number'], row['Output Type'], row['Category']
        out.loc[i] = new[v10.columns]
        log = [e for e in log if e['v1.0 Row Number'] != row['Row Number']]
        log.append({'v1.0 Row Number': row['Row Number'], 'Calculator ID': cid, 'Calculator Name': row['Calculator Name'],
                    'Action': 'Replaced (duplicate of one-shot example)', 'Reason': 'Same note as the one-shot example for this calculator',
                    'Verified Row Number': src, 'Old Note ID': row['Note ID'], 'New Note ID': new['Note ID'],
                    'Old Ground Truth Answer': v10.set_index('Row Number').loc[row['Row Number'], 'Ground Truth Answer'] if not str(row['Row Number']).startswith('new') else '',
                    'New Ground Truth Answer': new['Ground Truth Answer'], 'Audit Flag': audit['Flag'].get(row['Row Number'], 'Clean')})
    # notes from outside Verified's test set (overrides/extra_notes.csv, with their source), used only via replacements.json
    extra = pd.read_csv(OVERRIDES / 'extra_notes.csv', dtype=str).set_index('Row Number')
    for rep in json.load(open(OVERRIDES / 'replacements.json')):
        m = (out['Calculator ID'] == rep['calculator_id']) & (out['Note ID'] == rep['note_id'])
        assert m.sum() == 1, rep
        i, src = out.index[m][0], rep['verified_row']
        row = out.loc[i]
        new = (extra.loc[src].copy() if src in extra.index else ver_by_row.loc[src].copy())
        assert new['Calculator ID'] == row['Calculator ID'] and src not in VERIFIED_EXCLUDE, rep
        new['Row Number'], new['Output Type'], new['Category'] = row['Row Number'], row['Output Type'], row['Category']
        out.loc[i] = new[v10.columns]
        prior = [e for e in log if e['v1.0 Row Number'] == row['Row Number']]
        log = [e for e in log if e['v1.0 Row Number'] != row['Row Number']]
        old_answer = v10.set_index('Row Number')['Ground Truth Answer'].get(row['Row Number'], '')
        log.append({'v1.0 Row Number': row['Row Number'], 'Calculator ID': row['Calculator ID'], 'Calculator Name': row['Calculator Name'],
                    'Action': 'Replaced with new Verified note (checked)', 'Reason': rep['reason'], 'Verified Row Number': src,
                    'Old Note ID': prior[0]['Old Note ID'] if prior else row['Note ID'], 'New Note ID': new['Note ID'],
                    'Old Ground Truth Answer': old_answer, 'New Ground Truth Answer': new['Ground Truth Answer'],
                    'Audit Flag': audit['Flag'].get(row['Row Number'], 'Clean')})
    ver_entities = out[KEY].merge(ver[KEY + ['Patient Note', 'Relevant Entities']], on=KEY, how='left')
    same_note = (out['Patient Note'].values == ver_entities['Patient Note'].values)
    entities_changed = same_note & (out['Relevant Entities'].values != ver_entities['Relevant Entities'].values)
    out.loc[entities_changed, 'Relevant Entities'] = ver_entities.loc[entities_changed, 'Relevant Entities'].values
    renamed = out.apply(lambda r: rename_entities(r['Calculator ID'], r['Relevant Entities']), axis=1)
    entities_renamed = renamed != out['Relevant Entities']
    out['Relevant Entities'] = renamed
    for old, new in NOTE_TYPO_FIXES.items():
        out['Patient Note'] = out['Patient Note'].str.replace(old, new, regex=False)
    applied = []
    for edit in json.load(open(OVERRIDES / 'note_edits.json')):
        m = (out['Calculator ID'] == edit['calculator_id']) & (out['Note ID'] == edit['note_id'])
        assert m.sum() == 1, edit
        i = out.index[m][0]; note = out.at[i, 'Patient Note']
        for old, new in edit['replace']:
            assert note.count(old) == 1, (edit['note_id'], old)
            note = note.replace(old, new)
        out.at[i, 'Patient Note'] = note
        applied.append({'Calculator ID': edit['calculator_id'], 'Note ID': edit['note_id'], 'Kind': 'note edit', 'Change': json.dumps(edit['replace'], ensure_ascii=False), 'Reason': edit['reason']})
    for fix in json.load(open(OVERRIDES / 'entity_overrides.json')):
        m = (out['Calculator ID'] == fix['calculator_id']) & (out['Note ID'] == fix['note_id'])
        assert m.sum() == 1, fix
        i = out.index[m][0]; ent = ast.literal_eval(out.at[i, 'Relevant Entities'])
        for k in fix.get('remove', []):
            assert k in ent, (fix['note_id'], k); ent.pop(k)
        ent.update(fix.get('set', {}))
        out.at[i, 'Relevant Entities'] = str(ent)
        applied.append({'Calculator ID': fix['calculator_id'], 'Note ID': fix['note_id'], 'Kind': 'entity override', 'Change': json.dumps({'set': fix.get('set', {}), 'remove': fix.get('remove', [])}, ensure_ascii=False), 'Reason': fix['reason']})
    # clinician-written notes (integer Note ID, not template) are labelled Synthetic, as Verified labels its written notes
    handwritten_rows = out['Note ID'].str.isdigit() & (out['Note Type'] != 'Template')
    out.loc[handwritten_rows, 'Note Type'] = 'Synthetic'
    # PMC/USMLE case reports that Verified labels Synthetic (e.g. HOMA-IR pmc-6128601-1) are extracted notes
    case_reports = out['Note ID'].str.match(r'(pmc|usmle)-') & (out['Note Type'] == 'Synthetic')
    out.loc[case_reports, 'Note Type'] = 'Extracted'
    names = ver.groupby('Calculator ID')['Calculator Name'].first()  # v1.0 misspells calculator 68 ("Estimated of Conception")
    out['Calculator Name'] = out['Calculator ID'].map(names).fillna(out['Calculator Name'])
    # Questions: Verified's wording without the time-point sentence, which goes in the system prompt instead (column
    # 'Time Point Instruction', added by run_generate_csv.py). Steroid Conversion keeps each row's own question.
    questions = ver.groupby('Calculator ID')['Question'].agg(lambda q: q.iloc[0] if q.nunique() == 1 else None).dropna()
    questions = questions.map(lambda q: q.replace(' ' + TIME_POINT, '').replace(TIME_POINT, '').rstrip())
    has_question = out['Calculator ID'].isin(questions.index)
    question_changed = has_question & (out['Question'] != out['Calculator ID'].map(questions))
    out.loc[has_question, 'Question'] = out.loc[has_question, 'Calculator ID'].map(questions)
    renumber = dict(zip(out['Row Number'], map(str, range(1, len(out) + 1))))
    out['Row Number'] = out['Row Number'].map(renumber)

    assert len(out) == len(v10) + sum(TOP_UP.values())
    assert not out.duplicated(KEY).any(), 'a Verified note was used twice'
    out.to_csv(DATA / 'test_data_corrected.csv', index=False)
    log = pd.DataFrame(log)
    log.insert(0, 'Row Number', [renumber[r if r else f'new-{v}'] for r, v in zip(log['v1.0 Row Number'], log['Verified Row Number'])])
    log.sort_values('Row Number', key=lambda s: s.astype(int)).to_csv(DATA / 'test_data_corrected_changes.csv', index=False)
    print(f'{len(out)} rows written; {len(log)} replaced or added; entities updated on {entities_changed.sum()} rows; '
          f'entity names updated on {entities_renamed.sum()} rows; '
          f'question updated on {question_changed.sum()} rows')
    print(log['Action'].value_counts().to_string())
    pd.DataFrame(applied, columns=['Calculator ID', 'Note ID', 'Kind', 'Change', 'Reason']).to_csv(DATA / 'test_data_corrected_overrides.csv', index=False)
    print(f'overrides applied: {len(applied)} (see dataset/test_data_corrected_overrides.csv)')


if __name__ == '__main__':
    main()
