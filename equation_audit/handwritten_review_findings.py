"""Findings from the 2026-10-20 review of the 200 clinician-written rows (10 calculators) in dataset/test_data_final.csv.

Run: python3 equation_audit/handwritten_review_findings.py
Each finding was checked against the note text. 'Proposed Answer' is computed by the calculator code
(run_generate_csv.py) from the row's inputs with only that finding's change applied; for code bugs it is the
score the code's own stated rule gives. Nothing in the dataset is changed.
Writes dataset/handwritten_review_findings.csv.
"""
import ast
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / 'dataset'

# Kind: 'Input error' (input contradicts the note), 'Unrecognised value' (input string the code does not match, so it is
# silently mis-scored), 'Code bug', 'Judgement call' (defensible either way; for physician review), 'Note issue'
# (note wording or inconsistency; no input change proposed). set/remove: the input change that 'Proposed Answer' uses.
F = [
    # Child-Pugh (15)
    dict(row='204', kind='Unrecognised value', input='Encephalopathy', set={'Encephalopathy': 'No Encephalopathy'}, conf='high',
         quote='the patient is A&Ox3 all reflexes 2+ and absent asterixis', why="'Grade 0' is not a value the code matches, so encephalopathy scores 0 instead of 1"),
    dict(row='201', kind='Judgement call', input='Encephalopathy', set={'Encephalopathy': 'Grade 3-4'}, conf='medium',
         quote='lethargy and confusion ... The patient is oriented only to self. Asterixis is present.', why='Disoriented to place with confusion is West Haven grade 3; graded 1-2 while rows 216 and 220 with milder signs are graded 3-4'),
    dict(row='216', kind='Judgement call', input='Encephalopathy', set={'Encephalopathy': 'Grade 1-2'}, conf='medium',
         quote='the patient was disoriented and exhibited asterixis', why='Asterixis and disorientation at a routine visit read as grade 2; graded 3-4 (make consistent with 201 and 220)'),
    dict(row='220', kind='Judgement call', input='Encephalopathy', set={'Encephalopathy': 'Grade 1-2'}, conf='medium',
         quote='he exhibited confusion, asterixis, and slurred speech', why='Outpatient with asterixis and slurred speech reads as grade 2; graded 3-4'),
    # RCRI (17)
    dict(row='241', kind='Input error', input='Congestive Heart Failure criteria for the Cardiac Risk Index rule', set={'Congestive Heart Failure criteria for the Cardiac Risk Index rule': True}, conf='high',
         quote='well-controlled congestive heart failure', why='History of CHF is stated (row 243, the same note with other numbers, has True)'),
    dict(row='252', kind='Input error', input='Elevated-risk surgery', set={'Elevated-risk surgery': False}, conf='high',
         quote='high-risk orthopedic surgery for revision hip replacement', why='RCRI elevated-risk surgery is intraperitoneal, intrathoracic or suprainguinal vascular; hip replacement is none of these'),
    dict(row='248', kind='Input error', input='Pre-operative creatinine', set={'Pre-operative creatinine': [140.0, 'µmol/L']}, conf='high',
         quote='an elevated creatinine level of 140 µmol/L', why='Input holds the mg/dL conversion (1.6) while the note gives µmol/L; score unchanged'),
    dict(row='254', kind='Input error', input='Pre-operative creatinine', set={'Pre-operative creatinine': [150.0, 'µmol/L']}, conf='high',
         quote='an elevated creatinine of 150 µmol/L', why='Input holds the mg/dL conversion (1.7) while the note gives µmol/L; score unchanged'),
    dict(row='258', kind='Judgement call', input='Elevated-risk surgery', set={'Elevated-risk surgery': False}, conf='medium',
         quote='high-risk surgery for carotid artery stenosis', why='Carotid endarterectomy is often classed intermediate risk, not suprainguinal vascular'),
    dict(row='260', kind='Judgement call', input='Elevated-risk surgery', set={'Elevated-risk surgery': False}, conf='low',
         quote='major high-risk endovascular aneurysm repair', why='Aortic EVAR is usually counted as suprainguinal vascular (True defensible)'),
    # HAS-BLED (25)
    dict(row='349', kind='Input error', input='Medication usage predisposing to bleeding', set={'Medication usage predisposing to bleeding': False}, conf='high',
         quote='a specific anticoagulant recommended following a DVT', why='Only antiplatelets and NSAIDs count; an anticoagulant does not'),
    dict(row='350', kind='Input error', input='Medication usage predisposing to bleeding', set={'Medication usage predisposing to bleeding': False}, conf='high',
         quote='His medication regimen includes anticoagulants, initiated after his stroke.', why='Only antiplatelets and NSAIDs count; an anticoagulant does not'),
    dict(row='360', kind='Input error', input='Hypertension', set={'Hypertension': False}, conf='high',
         quote='has been doing well with controlled blood pressure readings on his antihypertensive medications', why='HAS-BLED counts uncontrolled hypertension (systolic >160) only'),
    dict(row='362', kind='Input error', input='Hypertension', set={'Hypertension': False}, conf='high',
         quote='currently his blood pressure profile shows well-controlled readings (118/61)', why='HAS-BLED counts uncontrolled hypertension (systolic >160) only'),
    dict(row='349', kind='Judgement call', input='Hypertension', set={'Hypertension': False}, conf='medium',
         quote='already diagnosed with hypertension which is being managed with antihypertensive medication', why='No BP value and nothing says uncontrolled'),
    dict(row='350', kind='Note issue', input='Hypertension', set={'Hypertension': False}, conf='medium',
         quote='uncontrolled hypertension - although his blood pressure has been consistently well-controlled in recent times', why='Note contradicts itself; edit the note or decide the point'),
    dict(row='348', kind='Judgement call', input='Liver disease criteria for the HAS-BLED rule', set={'Liver disease criteria for the HAS-BLED rule': False}, conf='medium',
         quote='AST and ALT were 4 times the normal value), suggesting liver disease', why='No cirrhosis or bilirubin given, so the bilirubin >2x with AST/ALT >3x rule is not shown'),
    dict(row='346', kind='Judgement call', input='Stroke; Hypertension; Liver disease', set=None, conf='medium',
         quote='transient ischemic attack of her brain last year; several recorded high blood pressure readings in the clinic; suggestive signs of hepatic cirrhosis', why='TIA counted as stroke; hypertension without a value >160; suspected (not confirmed) cirrhosis. Answer could be 2-4'),
    *[dict(row=r, kind='Judgement call', input='Renal disease criteria for the HAS-BLED rule', set={'Renal disease criteria for the HAS-BLED rule': False}, conf='low',
           quote=q, why='Renal disease needs dialysis, transplant or Cr >2.26 mg/dL; the note gives no value')
      for r, q in [('348', 'elevated creatinine'), ('351', 'laboratory evidence of chronic kidney disease'), ('354', 'high creatinine level'), ('357', 'elevated creatinine')]],
    dict(row='354', kind='Judgement call', input='Liver disease criteria for the HAS-BLED rule', set={'Liver disease criteria for the HAS-BLED rule': False}, conf='low',
         quote='repeated indications of hepatic impairment due to ALD', why='No cirrhosis, bilirubin or transaminase values given'),
    # Glasgow-Blatchford (27)
    dict(row='391', kind='Note issue', input='Blood Urea Nitrogen (BUN)', set={'Blood Urea Nitrogen (BUN)': [22.0, 'mg/dL']}, conf='low',
         quote='mildly elevated blood urea nitrogen levels at 22 mmol/L', why='22 mmol/L is 62 mg/dL, not mildly elevated; input follows the stated unit. Rows 394, 396, 398, 401, 403, 405 are similar but milder'),
    # Charlson (32)
    dict(row='434', kind='Input error', input='Liver disease severity', set={'Liver disease severity': 'mild'}, conf='high',
         quote='chronic liver disease and imaging has shown prominent changes consistent with long term liver damage', why='No cirrhosis with portal hypertension, varices or bleeding: mild'),
    dict(row='447', kind='Input error', input='Diabetes mellitus', set={'Diabetes mellitus': 'none or diet-controlled'}, conf='medium',
         quote='He has managed to keep his type II diabetes mellitus uncomplicated with diet management and regular exercise.', why='Diet-controlled scores 0 (row 428 codes the same wording that way)'),
    dict(row='440', kind='Input error', input='Cerebrovascular Accident', set={'Cerebrovascular Accident': False}, conf='high',
         quote='a transient ischemic attack that left him with no residual neurological deficits', why='TIA only, no stroke; score unchanged because TIA gives the same point'),
    dict(row='430', kind='Judgement call', input='Connective tissue disease', set={'Connective tissue disease': False}, conf='medium',
         quote='He was diagnosed with marfan syndrome when he was 19 years old.', why='CCI connective tissue disease means rheumatologic disease (SLE, RA, PMR, ...), not Marfan'),
    dict(row='428', kind='Judgement call', input='Moderate to severe Chronic Kidney Disease', set={'Moderate to severe Chronic Kidney Disease': False}, conf='low',
         quote='chronic kidney disease (CKD) ... worsening CKD', why='No stage, creatinine or dialysis to show moderate-severe'),
    dict(row='437', kind='Judgement call', input='Moderate to severe Chronic Kidney Disease', set={'Moderate to severe Chronic Kidney Disease': False}, conf='low',
         quote='nephropathy, indicated by increasing proteinuria and declining glomerular filtration rate', why='No stage, creatinine or dialysis to show moderate-severe'),
    dict(row='436', kind='Judgement call', input='Congestive Heart Failure', set={'Congestive Heart Failure': False}, conf='low',
         quote='Increasing dyspnoea on exertion and peripheral edema suggest a presenting sign of congestive heart failure, for which further investigation is planned', why='Suspected, not diagnosed'),
    dict(row='444', kind='Judgement call', input='Liver disease severity', set={'Liver disease severity': 'mild'}, conf='medium',
         quote='moderate liver disease without clinical manifestations', why='MDCalc moderate-severe means cirrhosis with portal hypertension'),
    dict(row='439', kind='Judgement call', input='Cerebrovascular Accident', set={'Cerebrovascular Accident': True}, conf='low',
         quote='left-sided weakness due to damage from a central nervous system pathology', why='Cause not named; counts only if it was a stroke'),
    # GCS (21)
    dict(row='861', kind='Input error', input='Best verbal response', set={'Best verbal response': 'not testable'}, conf='high',
         quote='does not attempt to verbalize due to the tracheostomy', why='Verbal response cannot be tested, not absent'),
    dict(row='849', kind='Judgement call', input='Best verbal response', set={'Best verbal response': 'confused'}, conf='medium',
         quote="When asked if she knows where she is, she responds, 'I am in the Lego aisle of Target.'", why='Coherent but disoriented answer is confused (4), not inappropriate words (3); note also says she is not able to be aroused verbally'),
    dict(row='866', kind='Judgement call', input='Best verbal response', set={'Best verbal response': 'incomprehensible sounds'}, conf='low',
         quote='kept mumbling nonsensical words', why='Could be inappropriate words (3, as coded) or incomprehensible sounds (2)'),
    *[dict(row=r, kind='Code bug', input='Best verbal response', set=None, manual=+1, conf='high', quote=q,
           why="glasgow_coma_score.py scores verbal 'not testable' as 4, but its explanation says the patient is assumed oriented (5)")
      for r, q in [('857', "Verbal responsiveness could not be evaluated due to the patient's intubation."),
                   ('864', 'Due to intubation verbal responsiveness could not be tested.')]],
    # APACHE II (28)
    *[dict(row=r, kind='Unrecognised value', input='Surgery Type', set={'Surgery Type': 'Nonoperative'}, conf='high', quote='',
           why="'Nonoperative ' has a trailing space, so the code does not add the 5 points for severe organ failure/immunocompromise")
      for r in ('871', '873')],
    dict(row='882', kind='Unrecognised value', input='Chronic renal failure', set={'Chronic renal failure': False}, conf='high', quote='',
         why="'Not mentinoed' is read as present by the code; score unchanged (creatinine 1.4)"),
    dict(row='869', kind='Input error', input='pH; A-a gradient; Diastolic Blood Pressure', set={'pH': 7.29, 'Diastolic Blood Pressure': [84.0, 'mm Hg'], 'A-a gradient': 100.0}, conf='high',
         quote='BP 122/84 mmHg ... ABG shows pH of 7.29 a 57 % FiO₂.', why='pH is 7.29 not 7.53; diastolic 84 not 81; the note gives no A-a gradient or PaO2 (375 is not in the note). The code needs an A-a gradient at FiO2 >=50%, so the proposed answer scores oxygenation as normal (0); the note needs a value added'),
    dict(row='870', kind='Input error', input='A-a gradient; Sodium', set={'Sodium': [133.28, 'mmol/L'], 'A-a gradient': 100.0}, conf='high',
         quote='sodium level of 133.28 mmol/L ... ABG shows pH of 7.47 a 68 % FiO₂.', why='The note gives no A-a gradient or PaO2 (240 is not in the note); sodium digits transposed (no score effect). The code needs an A-a gradient at FiO2 >=50%, so the proposed answer scores oxygenation as normal (0); the note needs a value added'),
    dict(row='872', kind='Input error', input='Systolic Blood Pressure', set={'Systolic Blood Pressure': [190.0, 'mm Hg']}, conf='high',
         quote='Blood pressure is elevated at 190/103 mm Hg.', why='Systolic is 190, not 132'),
    dict(row='872', kind='Judgement call', input='History of severe organ failure or immunocompromise; Surgery Type', set={'History of severe organ failure or immunocompromise': False}, conf='medium',
         quote='chronic renal failure, managed conservatively, and is scheduled for a kidney transplant ... presents to the emergency department',
         why="Severe organ insufficiency for renal means dialysis; if it is kept, the type is Nonoperative (+5), not 'Elective' (+2)"),
    *[dict(row=r, kind='Input error', input='Surgery Type', set={'Surgery Type': 'Nonoperative'}, conf='high', quote=q,
           why='Organ failure history is True but Surgery Type is missing, so the code adds 0 instead of 5 for a nonoperative patient')
      for r, q in [('884', 'history of significant chronic liver failure requiring transplant ... presented to the emergency unit'),
                   ('887', 'previously experienced severe liver failure managed with medications ... presented to the outpatient clinic')]],
    dict(row='870', kind='Judgement call', input='Acute renal failure', set={'Acute renal failure': False}, conf='medium',
         quote='chronic kidney disease not requiring dialysis ... presents with decreased urine output', why='Acute renal failure is never stated'),
    dict(row='871', kind='Judgement call', input='Acute renal failure', set={'Acute renal failure': False}, conf='medium',
         quote='a creatinine level of 4.58 mg/dL, indicating impaired renal function', why='Acute renal failure is never stated'),
    dict(row='877', kind='Judgement call', input='Acute renal failure', set={'Acute renal failure': False}, conf='medium',
         quote='the possible occurrence of acute renal failure', why='Only possible acute renal failure'),
    dict(row='869', kind='Judgement call', input='Acute renal failure', set={'Acute renal failure': False}, conf='medium',
         quote='Urine output is 104ml over last six hours.', why='Acute renal failure is never stated; no score effect (creatinine 1.258)'),
    dict(row='871', kind='Note issue', input='FiO2; A-a gradient', set=None, conf='medium',
         quote='His fraction of inspired air is 63 % ... an A-a gradient of 331 mm Hg while on room air', why='Note contradicts itself on FiO2 (63% vs room air)'),
    dict(row='870', kind='Note issue', input='Diastolic Blood Pressure', set=None, conf='low',
         quote='blood pressure is  125/8mmHg', why='Typo in the note; input uses 83'),
    dict(row='875', kind='Note issue', input='Potassium', set=None, conf='medium',
         quote='sodium 139 mEq/L, potassium 139 mEq/L', why='Impossible potassium in the note; input 5.0 is not in the note (scores the same as missing)'),
    # Caprini (36)
    dict(row='911', kind='Input error', input='Acute Myocardial infarction', set={'Acute Myocardial infarction': False}, conf='high',
         quote='a PMH of MI (2014)', why='A past MI, not an acute one'),
    dict(row='911', kind='Judgement call', input='Multiple trauma in the last month', set={'Multiple trauma in the last month': False}, conf='medium',
         quote='a recent motor vehicle accident (MVA) resulting in the immobilization of her left lower extremity with a plaster cast', why='One limb injury; no multiple injuries described'),
    dict(row='912', kind='Input error', input='Congestive Heart Failure in the last month', set={'Congestive Heart Failure in the last month': False}, conf='high',
         quote='', why='Heart failure is not mentioned in the note'),
    dict(row='912', kind='Input error', input='Acute spinal cord injury causing paralysis in the last month', set={'Acute spinal cord injury causing paralysis in the last month': False}, conf='high',
         quote='a recent fall resulting in a spinal injury ... she exhibits swollen legs but is able to ambulate normally', why='No cord injury or paralysis; she walks normally'),
    dict(row='924', kind='Input error', input='Major Surgery in the last month', set={'Major Surgery in the last month': False}, conf='high',
         quote='recently underwent minor surgery for bunion correction', why='Only minor surgery stated'),
    dict(row='927', kind='Input error', input='Major Surgery in the last month', set={'Major Surgery in the last month': False}, conf='medium',
         quote='She underwent a minor surgical procedure not long after admission', why='Only minor surgery stated (calling hip-fracture surgery minor is odd, but that is the note)'),
    dict(row='922', kind='Judgement call', input='Surgery Type', remove=['Surgery Type'], conf='medium',
         quote='He underwent minor surgery some months back for a trigger finger release.', why='Months ago and no current surgery'),
    dict(row='921', kind='Judgement call', input='Family history of thrombosis', set={'Family history of thrombosis': False}, conf='medium',
         quote='strong family history of thrombotic events with her mother and brother suffering MIs', why='Arterial events (MI), not venous thrombosis'),
    dict(row='921', kind='Judgement call', input='Major Surgery in the last month', set={'Major Surgery in the last month': False}, conf='low',
         quote='admitted following a stroke and thromboembolism removal', why='Probably endovascular thrombectomy, which may not be major surgery'),
    dict(row='926', kind='Judgement call', input='Surgery Type', remove=['Surgery Type'], conf='medium',
         quote='He was scheduled for elective major lower extremity arthroplasty prior to these episodes.', why='Medical admission; the surgery may have been postponed'),
    dict(row='913', kind='Judgement call', input='Mobility', set={'Mobility': 'confined to bed >72 hours'}, conf='low',
         quote='Over the past month ... he remained mostly confined to his bed', why="Could be 'confined to bed >72 hours' rather than 'on bed rest'"),
    dict(row='912', kind='Judgement call', input='Multiple trauma in the last month', set={'Multiple trauma in the last month': True}, conf='low',
         quote='a recent fall resulting in a spinal injury and nonoperative management of a lower leg fracture', why='Spinal injury plus leg fracture could count as multiple trauma'),
    dict(row='909', kind='Note issue', input='Surgery Type; Major Surgery in the last month', set=None, conf='low',
         quote='well-healed surgical incision over the left femoral shaft', why="Operation only implied, no timing; scored as Surgery Type 'major' and major surgery in the last month. Several rows score one operation in both items"),
    # PSI (29)
    dict(row='834', kind='Input error', input='Neoplastic disease; Heart Rate or Pulse', set={'Neoplastic disease': True, 'Heart Rate or Pulse': [95.0, 'beats per minute']}, conf='high',
         quote='he is currently under treatment for neoplasia ... heart rate of 95 beats per minute', why='Active cancer missing from inputs (+30); heart rate 85 should be 95 (no score effect)'),
    dict(row='831', kind='Note issue', input='Altered mental status', set={'Altered mental status': True}, conf='medium',
         quote='increased fatigue and confusion over the past week ... has no change in her mental status at present', why='Note contradicts itself'),
    dict(row='847', kind='Judgement call', input='Neoplastic disease', set={'Neoplastic disease': False}, conf='medium',
         quote='significant for prior breast cancer', why='PSI counts cancer active or diagnosed within a year; no dates given'),
    dict(row='829', kind='Judgement call', input='Altered mental status', set={'Altered mental status': False}, conf='low',
         quote='describes a recent episode of fever and altered mental state', why='Reads as history at a routine physical, not a finding at presentation'),
    dict(row='844', kind='Note issue', input='pH', set=None, conf='low',
         quote='a pH of 7.34 ... within normal limits', why='Wording contradicts the number; 7.34 correctly scores +30'),
    # SOFA (43)
    dict(row='892', kind='Input error', input='norEPINEPHrine', remove=['norEPINEPHrine'], conf='high',
         quote='She has been normotensive and has not required vasopressors.', why='No vasopressor in the note'),
    dict(row='894', kind='Input error', input='Continuous positive airway pressure', set={'Continuous positive airway pressure': True}, conf='high',
         quote='Our patient is currently on continuous positive airway pressure, with a fraction of inspired oxygen (FiO₂) of 50%', why='CPAP stated'),
    dict(row='890', kind='Input error', input='PaO2; Continuous positive airway pressure', set={'Continuous positive airway pressure': True, 'PaO2': [170.0, 'mm Hg']}, conf='medium',
         quote='The patient is currently receiving 41 % FiO₂ with continuous positive airway pressure.', why='The note gives no PaO2 (90 is not in the note); CPAP stated. The code needs a PaO2, so the proposed answer scores respiration as normal (0); the note needs a value added'),
    dict(row='890', kind='Note issue', input='Bilirubin', set={'Bilirubin': [1.57, 'mg/L']}, conf='low',
         quote='a total bilirubin level of 1.57 mg/L', why='Note says mg/L (0.157 mg/dL) next to jaundice; input uses mg/dL. Likely a note typo'),
    *[dict(row=r, kind='Input error', input=i, set=s, conf='high', quote=q, why='Blood pressure does not match the note; score unchanged (vasopressors set the cardiovascular points)')
      for r, i, s, q in [('899', 'Diastolic Blood Pressure', {'Diastolic Blood Pressure': [70.0, 'mm Hg']}, 'blood pressure of 91/70 mm Hg'),
                         ('900', 'Systolic/Diastolic Blood Pressure', {'Systolic Blood Pressure': [93.0, 'mm Hg'], 'Diastolic Blood Pressure': [62.0, 'mm Hg']}, 'His blood pressure was 93/62 mm Hg.'),
                         ('901', 'Diastolic Blood Pressure', {'Diastolic Blood Pressure': [59.0, 'mm Hg']}, 'systolic and diastolic measures at 87 and 59 mm Hg'),
                         ('906', 'Systolic/Diastolic Blood Pressure', {'Systolic Blood Pressure': [85.0, 'mm Hg'], 'Diastolic Blood Pressure': [45.0, 'mm Hg']}, 'Most recent BP was 85/45.')]],
    dict(row='902', kind='Input error', input='Diastolic Blood Pressure', remove=['Diastolic Blood Pressure'], conf='high',
         quote='In response to his low blood pressure readings', why='Note gives no BP numbers; score unchanged'),
    dict(row='889', kind='Judgement call', input='Urine Output', set={'Urine Output': [0, 'mL/day']}, conf='low',
         quote='The patient has no urine output noted', why='Could mean not recorded (current) or anuric (4 renal points)'),
    dict(row='889', kind='Judgement call', input='Hypotension', set={'Hypotension': False}, conf='low',
         quote='a significant medical history, including hypotension', why='History rather than current state; no BP given'),
]


def build(F, base, target):
    """Write the findings F, scored against the rows of base (keyed by Row Number), to dataset/<target>."""
    by_row = base.set_index('Row Number')
    scen = []
    for i, f in enumerate(F):
        r = by_row.loc[f['row']].copy()
        ent = ast.literal_eval(r['Relevant Entities'])
        for k in f.get('remove', []):
            assert k in ent, (f['row'], k)
            ent.pop(k)
        ent.update(f.get('set') or {})
        r['Relevant Entities'], r['Row Number'] = str(ent), str(i + 1)  # generate_csv.py numbers rows 1..N
        scen.append(r)
    with tempfile.TemporaryDirectory() as tmp:
        src, out = Path(tmp) / 'in.csv', Path(tmp) / 'out.csv'
        cols = [c for c in base.columns if c != 'Time Point Instruction']
        pd.DataFrame(scen)[cols].to_csv(src, index=False)
        run = subprocess.run([sys.executable, str(ROOT / 'run_generate_csv.py'), str(src), str(out)], capture_output=True, text=True)
        if run.returncode:
            sys.exit(run.stdout[-1500:] + run.stderr[-1500:])
        gen = pd.read_csv(out, dtype=str, keep_default_na=False)
    rows = []
    for f, g in zip(F, gen.itertuples()):
        r = by_row.loc[f['row']]
        ent = ast.literal_eval(r['Relevant Entities'])
        keys = list(f.get('set') or {}) + f.get('remove', [])
        cur = '; '.join(f'{k}={ent.get(k, "(missing)")}' for k in keys) if keys else ''
        new = '; '.join([f'{k}={v}' for k, v in (f.get('set') or {}).items()] + [f'{k}=(remove)' for k in f.get('remove', [])])
        if 'manual' in f:
            proposed = str(int(float(r['Ground Truth Answer'])) + f['manual'])
        elif f.get('set') is None and not f.get('remove'):
            proposed = ''
        else:
            proposed = g._asdict()['_11']  # Ground Truth Answer
            if float(proposed).is_integer() and '.' not in r['Ground Truth Answer']:
                proposed = str(int(float(proposed)))
        rows.append({'Row Number': f['row'], 'Calculator ID': r['Calculator ID'], 'Calculator Name': r['Calculator Name'], 'Note ID': r['Note ID'],
                     'Kind': f['kind'], 'Confidence': f['conf'], 'Input': f['input'], 'Current Value': cur, 'Proposed Value': new,
                     'Note Quote': f['quote'], 'Reason': f['why'], 'Current Answer': r['Ground Truth Answer'], 'Proposed Answer': proposed,
                     'Answer Changes': bool(proposed) and float(proposed) != float(r['Ground Truth Answer']),
                     **({'Found By': f.get('seen', 'Both passes')} if any('seen' in x for x in F) else {})})
    order = {'Input error': 0, 'Unrecognised value': 1, 'Code bug': 2, 'Judgement call': 3, 'Note issue': 4}
    out = pd.DataFrame(rows).sort_values(['Kind', 'Row Number'], key=lambda s: s.map(order) if s.name == 'Kind' else s.astype(int))
    out.to_csv(DATA / target, index=False, encoding='utf-8-sig')
    print(out.groupby('Kind', sort=False).agg(rows=('Row Number', 'nunique'), findings=('Row Number', 'size'), answer_changes=('Answer Changes', 'sum')).to_string())


def main():
    build(F, pd.read_csv(DATA / 'test_data_final.csv', dtype=str, keep_default_na=False), 'handwritten_review_findings.csv')


if __name__ == '__main__':
    main()
