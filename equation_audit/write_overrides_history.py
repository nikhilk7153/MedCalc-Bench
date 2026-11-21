"""Turn the reviewed decisions on the 135 flagged rows into equation_audit/overrides/*.json.
Row numbers refer to the build before these overrides (dataset/test_data_corrected.csv at that time, saved as
scratchpad/test_before_overrides.csv)."""
import ast, json, sys
import pandas as pd

S = sys.argv[1]
R = lambda p: pd.read_csv(p, dtype=str, keep_default_na=False)
base = R(f'{S}/test_before_overrides.csv').set_index('Row Number')
prop = R('equation_audit/proposals_fix1.csv').set_index('Row Number')
cand = R(f'{S}/replacement_candidates_checked.csv')
OUT = 'equation_audit/overrides'
key = lambda n: (base.loc[n, 'Calculator ID'], base.loc[n, 'Note ID'])
ent = lambda n: ast.literal_eval(base.loc[n, 'Relevant Entities'])

# ---- replacements: row -> reason; candidates taken in Verified row order per calculator
REPLACE = {
    '3': 'Creatinine Clearance: only a baseline creatinine, no admission value',
    '133': "Wells' PE: cancer treatment timing relative to admission not documented",
    '137': "Wells' PE: no pre-treatment heart rate; surgery happened after admission",
    '269': 'HEART: troponin 0.44 ng/mL given without a reference limit',
    '273': 'HEART: ECG only in an unavailable figure',
    '276': 'HEART: initial troponin inconclusive',
    '570': 'PERC: heart rate only after treatment',
    '573': 'PERC: oxygen saturation only on supplemental oxygen',
    '576': 'PERC: oxygen saturation only on supplemental oxygen',
    '586': 'PERC: oxygen saturation only on supplemental oxygen',
    '406': 'Serum Osmolality: only a glucose-corrected sodium is given',
    '483': 'Free Water Deficit: no pre-treatment sodium',
    '739': 'Delta Gap: electrolytes only after treatment',
    '793': 'Albumin Corrected Delta Gap: admission albumin not reported',
}
# only Extracted notes may come from Verified (user rule, 2026-10-17)
ver_type = R('dataset/test_verified.csv').set_index('Row Number')['Note Type']
clean = cand[(cand['clean'] == 'True') & (cand['Source'] == 'pool')].copy()
clean = clean[clean['Orig Row'].map(ver_type) == 'Extracted']
# no clean Extracted note available yet: keep the current choice until the user decides
PENDING = {}  # 280: original Extracted note kept with the HEART convention (user decision 2026-10-17)  # 555, 568: original Extracted notes kept, inputs match the notes (user decision 2026-10-17)
clean['vr'] = clean['Orig Row'].astype(int)
clean = clean.sort_values('vr')
pools = {c: list(g['Orig Row']) for c, g in clean.groupby('Calculator ID')}
replacements = []
for n, reason in REPLACE.items():
    cid, nid = key(n)
    src = pools[cid].pop(0) if pools.get(cid) else PENDING[n]
    replacements.append({'calculator_id': cid, 'note_id': nid, 'verified_row': src, 'reason': reason + (' [PENDING: Synthetic, no clean Extracted note]' if n in PENDING and src == PENDING[n] else '')})

replacements.append({'calculator_id': key('426')[0], 'note_id': key('426')[1], 'verified_row': 'extra-1',
                     'reason': 'HOMA-IR: Verified has no Extracted HOMA-IR note left; use the v1.0 train case pmc-3114100-1 '
                               '(insulin 21 pmol/L as written; v1.0 had 756) (user decision 2026-10-17)'})

# electrolytes taken from a blood gas instead of serum chemistry (equation_audit/blood_gas_final.csv; user rule 2026-10-17):
# replaced by unused Verified Extracted notes that passed the label, input and blood-gas checks (scratchpad bg_candidates_checked.csv)
final_now = R('dataset/test_data_final.csv').set_index('Row Number') if False else R('/private/tmp/claude-501/-Users-nikhilkhandekar-Downloads-MedCalc-Bench-version-1-0-2/341ceb4c-9399-4f09-944a-7a4f7963c93c/scratchpad/test_final_before_bloodgas.csv').set_index('Row Number')
BG_ROWS = ['375', '383', '472', '494', '729', '740', '750', '756', '761']  # 769, 798, 809, 811, 812: test_set_1_31 slot notes below
bgc = R('/private/tmp/claude-501/-Users-nikhilkhandekar-Downloads-MedCalc-Bench-version-1-0-2/341ceb4c-9399-4f09-944a-7a4f7963c93c/scratchpad/bg_candidates_checked.csv'); bgc = bgc[bgc['clean'] == 'True'].copy()
bgc['vr'] = bgc['Orig Row'].astype(int); bgc = bgc.sort_values('vr')
bg_pool = {c: list(g['Orig Row']) for c, g in bgc.groupby('Calculator ID')}
for n in BG_ROWS:
    cid, nid = final_now.loc[n, 'Calculator ID'], final_now.loc[n, 'Note ID']
    replacements.append({'calculator_id': cid, 'note_id': nid, 'verified_row': bg_pool[cid].pop(0),
                         'reason': f'{final_now.loc[n, "Calculator Name"]}: electrolyte input taken from a blood gas, no serum value in the note'})

# CKD-EPI row 30: patient on maintenance hemodialysis (creatinine-based GFR not meaningful; MD review 2026-10-17).
# Use the note v1.1 / test_set_1_31 / Verified have in this slot (Verified test row 30, pmc-2275218-1; checked)
replacements.append({'calculator_id': '3', 'note_id': 'pmc-7405101-3', 'verified_row': '30',
                     'reason': 'CKD-EPI: patient on maintenance hemodialysis; replaced with the v1.1/Verified note for this row (user decision 2026-10-17)'})

# Blood-gas rows: use test_set_1_31's note for the same slot (its row = our row - 1) where clean, else as noted
# (checked: label, inputs, blood gas; none reused within a calculator) (user decision 2026-10-18)
for cid, row_note, src, why in [
        ('65', 'pmc-3950226-1', 'extra-2', 'test_set_1_31 row 768 (its edit: bicarbonate listed with the serum panel)'),
        ('65', 'pmc-6363666-1', '833', 'Verified test row 833 (no clean test_set_1_31 note left for this calculator; Verified edit, negative anion gap)'),
        ('65', 'pmc-4411570-1', 'extra-3', 'test_set_1_31 row 779'),
        ('65', 'pmc-4047698-1', 'extra-4', 'test_set_1_31 row 785 (same note, edited: serum bicarbonate 18.4, albumin g/dL)'),
        ('66', 'pmc-3354932-1', 'extra-5', 'test_set_1_31 row 797'),
        ('67', 'pmc-3354932-1', 'extra-6', 'test_set_1_31 row 819 (its slot note, row 808, is already our row 821)'),
        ('67', 'pmc-4241748-1', 'extra-7', 'test_set_1_31 row 810'),
        ('67', 'usmle-2638', 'extra-8', 'test_set_1_31 row 811'),
        ('39', 'pmc-2657771-1', 'extra-9', 'test_set_1_31 row 499')]:
    replacements.append({'calculator_id': cid, 'note_id': row_note, 'verified_row': src,
                         'reason': f'Electrolyte input taken from a blood gas; replaced with {why}'})

# ---- note edits
edits = {}
for n in ['128', '224', '354', '360', '552', '871', '918']:  # model-proposed unit typos / contradictions, reviewed
    edits[n] = (json.loads(prop.loc[n, 'Note Edits']), {
        '128': 'Unit typos: electrolytes in mEq/dL, cortisol in mg/dL', '224': 'Unit typo: lipids in gr/dL',
        '354': 'Unit typo: homocysteine in mmol/L', '360': 'Unit typo: homocysteine in mmoL/L',
        '552': 'Unit typo: creatinine in gm/dl', '871': 'Contradiction: FiO2 63% and "room air"',
        '918': 'Contradiction: positive Factor V Leiden but "no congenital or acquired thrombophilia"'}[n])
edits['218'] = ([['splenomegaly, and ascites.', 'splenomegaly, and slight ascites.']],
                'Ascites severity was not stated; the note now states the slight ascites its inputs use (user decision 2026-10-17)')
edits['875'] = ([['potassium 139 mEq/L', 'potassium 5.0 mEq/L']], 'Typo: potassium 139 (same as sodium); v1.0 inputs use 5.0')
edits['858'] = ([['Pupillary evaluation was not possible', 'Eye opening could not be assessed']],
                'Note described pupils; the inputs (eye response not testable) refer to eye opening')
edits['863'] = ([['visual evaluation was not possible', 'eye opening could not be assessed']],
                'Note described vision; the inputs (eye response not testable) refer to eye opening')
edits['869'] = ([['ABG shows pH of 7.29 a 57 % FiO₂.', 'ABG shows pH of 7.29 and an A-a gradient of 375 mm Hg on 57 % FiO₂.']],
                'A-a gradient (needed at FiO2 >= 50%) missing from the note; v1.0 inputs use 375')
edits['870'] = ([['125/8mmHg', '125/83mmHg'],
                 ['ABG shows pH of 7.47 a 68 % FiO₂.', 'ABG shows pH of 7.47 and an A-a gradient of 240 mm Hg on 68 % FiO₂.']],
                'Diastolic BP typo (8; v1.0 inputs use 83) and A-a gradient missing (v1.0 inputs use 240)')
edits['889'] = ([['The patient has no urine output noted, raising concerns for renal or urinary complications.', 'Urine output has not been recorded.']],
                'Ambiguous between anuria and unrecorded output; v1.0 inputs do not use urine output')
edits['890'] = ([['1.57 mg/L', '1.57 mg/dL'],
                 ['41 % FiO₂ with continuous positive airway pressure.', '41 % FiO₂ with continuous positive airway pressure, with a PaO₂ of 90 mm Hg.']],
                'Bilirubin unit typo (v1.0 inputs use mg/dL) and PaO2 missing (v1.0 inputs use 90)')
edits['516'] = ([[', low density lipoprotein-cholesterol <10 mg/dL', '']],
                'The note reported a measured LDL (<10 mg/dL) that conflicts with the calculated LDL the question asks for')
# LDL Calculated notes that also state the measured LDL (the answer): remove that phrase only (user, 2026-10-18)
LDL_EDITS = {
    '513': ', low-density lipoproteins of 103.8 milligrams per deciliter (reference range: <100 milligrams per deciliter)',
    '517': 'the low-density lipoprotein cholesterol concentration was 214.4 mg/dl, ',
    '526': ('lipoprotein 49 mg/dL, low-density lipoprotein 63 mg/dL, and', 'lipoprotein 49 mg/dL, and'),
    '528': 'low-density lipoprotein 83 mg/dl, ',
    '529': 'low-density lipoprotein 89mg%, ',
    '530': ', low-density lipoprotein 90 mg/dL',
    '532': 'low density lipoprotein 83 mg/dL, '}
for n, e in LDL_EDITS.items():
    old, new = e if isinstance(e, tuple) else (e, '')
    edits[n] = ([[old, new]], 'The note stated the measured LDL, which is the answer the LDL Calculated question asks for')
edits['772'] = ([[' anion gap 25 mEq/L after correcting for albumin (normal: 8-12),', '']],
                'The note stated the albumin-corrected anion gap, which is the answer the question asks for')
edits['405'] = ([['His vital signs were initially normal, however, after an episode', 'After an episode']],
                'Admission time point ambiguous ("initially normal" vitals without values); keeps the documented pre-treatment vitals')

# ---- entity changes: accept the model's proposal (diff against current), with reviewed exceptions
ACCEPT = ('41 43 49 128 131 136 224 238 242 245 248 253 259 264 266 271 274 346 348 349 351 354 355 360 362 365 '
          '428 430 434 446 451 452 458 461 465 552 830 831 832 834 836 846 847 848 849 861 871 872 873 877 878 879 '
          '880 881 882 883 885 886 888 892 894 909 911 912 918 921 922 924 926 927').split()
SKIP_KEYS = {'Temperature'}  # handled by ENTITY_UNIT_FIXES
# same convention as Centor: cough/coryza not documented -> the 'absence' criterion is met
ADJUST = {'452': {'Absence of cough or coryza': True}, '465': {'Absence of cough or coryza': True}}
EXCLUDE = {'871': {'History of severe organ failure or immunocompromise'}}  # liver cirrhosis kept as qualifying
MANUAL = {
    '261': {'parent or sibling with Cardiovascular disease before age 65': False},
    '301': {'Tender/swollen anterior cervical lymph nodes': False, 'Cough Absent': True},
    '313': {'Tender/swollen anterior cervical lymph nodes': False, 'Cough Absent': True},
    '315': {'Tender/swollen anterior cervical lymph nodes': False, 'Cough Absent': True},
    '320': {'Tender/swollen anterior cervical lymph nodes': False, 'Cough Absent': True},
    '448': {'Severe tonsil inflammation': False, 'Absence of cough or coryza': True},
    '869': {'pH': 7.29, 'Diastolic Blood Pressure': [84.0, 'mm Hg']},
    '870': {'Sodium': [133.28, 'mmol/L']},
    '884': {'Surgery Type': 'Nonoperative'},
    '887': {'Surgery Type': 'Nonoperative'},
    '890': {'Continuous positive airway pressure': True},
    # from the recheck of changed rows (equation_audit/final_changed_fix1_verify.csv)
    '238': {'Active cancer': False},
    '870-arf': {'Acute renal failure': False},
    '909-surgery': {'Surgery Type': 'none'},
    # from the check of the 286 rows never checked in their final form (equation_audit/final_unchecked286_verify.csv)
    '910': {'Major Surgery in the last month': False},
    '913': {'Mobility': 'confined to bed >72 hours'},
    '221': {'Bedridden recently >3 days': False, 'Major surgery within 12 weeks': False},
    '227': {'Active cancer': False},
    '237': {'Active cancer': False},
    '222': {'Paralysis, paresis, or recent plaster immobilization of the lower extremity': False},
    '302': {'Temperature': [37.6, 'degrees celsius']},
    '748': {'Sodium': [141.0, 'mEq/L'], 'Chloride': [104.0, 'mEq/L'], 'Bicarbonate': [29.0, 'mEq/L']},
    '899': {'Diastolic Blood Pressure': [70.0, 'mm Hg']},
    '900': {'Systolic Blood Pressure': [93.0, 'mm Hg'], 'Diastolic Blood Pressure': [62.0, 'mm Hg']},
    '901': {'Diastolic Blood Pressure': [59.0, 'mm Hg']},
    '902': {},
    '906': {'Systolic Blood Pressure': [85.0, 'mm Hg'], 'Diastolic Blood Pressure': [45.0, 'mm Hg']},
}
MANUAL_REMOVE = {'902': ['Diastolic Blood Pressure']}
REASON = {'261': 'HEART: a nephew with PE is not a parent/sibling with CVD', '301': 'Centor: nodes not documented; no cough documented',
          '313': 'Centor: nodes not documented; no cough documented', '315': 'Centor: posterior (not anterior) cervical nodes; no cough documented',
          '320': 'Centor: nodes not documented; no cough documented', '448': 'FeverPAIN: no severe tonsil inflammation; no cough or coryza documented',
          '869': 'pH and diastolic BP did not match the note', '870': 'Sodium did not match the note',
          '884': 'APACHE II: nonoperative ED visit; surgery type was missing (0 instead of 5 chronic-health points)',
          '887': 'APACHE II: nonoperative clinic visit; surgery type was missing (0 instead of 5 chronic-health points)',
          '890': 'SOFA: note documents CPAP',
          '238': "Wells' DVT: cancer treated 3 years earlier; metastasis found only after admission",
          '870-arf': 'APACHE II: acute renal failure not documented (CKD with decreased urine output)',
          '910': 'Caprini: the knee replacement is the current operation; no other surgery in the last month',
          '913': 'Caprini: mostly confined to bed during a month-long stay',
          '221': "Wells' DVT: bed rest and recent surgery not documented (STEMI 10 days earlier)",
          '227': "Wells' DVT: a previous basal cell carcinoma is not active cancer",
          '237': "Wells' DVT: a WHO grade I meningioma is not active cancer",
          '222': "Wells' DVT: hemiparesis began six days after the first admission",
          '302': 'Centor: temperature at admission was 37.6 C (38.4 C was at home before admission)',
          '748': 'Delta Gap: use the initial (pre-treatment) electrolytes; the answer is unchanged',
          '899': 'SOFA: BP from the note (91/70)', '900': 'SOFA: BP from the note (93/62)', '901': 'SOFA: BP from the note (87/59)',
          '902': 'SOFA: the note gives no numeric diastolic pressure', '906': 'SOFA: BP from the note (85/45)',
          '909-surgery': 'Caprini: no current or planned surgery; the recent femoral operation is counted as major surgery in the last month'}
overrides = []
for n in ACCEPT:
    cur, new = ent(n), json.loads(prop.loc[n, 'Proposed Entities'])
    s = {k: v for k, v in new.items() if k not in SKIP_KEYS | EXCLUDE.get(n, set()) and cur.get(k, '<missing>') != v}
    s.update(ADJUST.get(n, {}))
    rm = [k for k in cur if k not in new and k not in SKIP_KEYS]
    if s or rm:
        overrides.append((n, s, rm, prop.loc[n, 'Calculator Name'].split(' ')[0] + ': ' + prop.loc[n, 'Explanation'][:300]))
for n, s in MANUAL.items():
    overrides.append((n.split('-')[0], s, MANUAL_REMOVE.get(n, []), REASON[n]))

cur_ent = json.load(open(f'{OUT}/entity_overrides.json'))  # keep the existing Caprini override
keep = [o for o in cur_ent if o.get('origin') != 'fix1']
out_ent = keep + [dict(calculator_id=key(n)[0], note_id=key(n)[1], set=s, **({'remove': rm} if rm else {}), reason=r, origin='fix1', row=n)
                  for n, s, rm, r in sorted(overrides, key=lambda o: int(o[0]))]
out_edit = [dict(calculator_id=key(n)[0], note_id=key(n)[1], replace=e, reason=r, row=n) for n, (e, r) in sorted(edits.items(), key=lambda o: int(o[0]))]
for name, obj in [('entity_overrides', out_ent), ('note_edits', out_edit), ('replacements', replacements)]:
    json.dump(obj, open(f'{OUT}/{name}.json', 'w'), indent=1, ensure_ascii=False)
print(len(out_ent), 'entity overrides;', len(out_edit), 'note edits;', len(replacements), 'replacements')
for o in out_ent:
    print(o.get('row'), o['calculator_id'], o['note_id'], json.dumps(o['set'], ensure_ascii=False)[:150], o.get('remove', ''))
