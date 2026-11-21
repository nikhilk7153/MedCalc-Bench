"""Findings from the 2026-10-20 review of test_data_1_17_2025.csv's versions of the APACHE II, Caprini, GCS, PSI and SOFA
rows (same Calculator ID + Note ID as v1.0; Row Number is the row in dataset/test_data_final.csv).

Run: python3 equation_audit/handwritten_review_1_17_findings.py
The 1_17 rows get the same entity renames and note spelling fixes as the build, and 'Current Answer' is what the
calculator code computes from the 1_17 inputs (the file's own answers are not used). 'Proposed Answer' applies only that
finding's change. Each finding was checked against the 1_17 note text. Nothing in the dataset is changed.
Two independent passes were made (whole calculator, then row by row); 'Found By' says which found each finding.
Writes dataset/handwritten_review_1_17_findings.csv.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

import build_corrected_test as bct
from handwritten_review_findings import DATA, ROOT, build

CALCS = ['28', '36', '21', '29', '43']
DVT_PE = {'Previously documented Deep Vein Thrombosis': True, 'Previously Documented Pulmonary Embolism': True}

F = [
    # GCS (21)
    dict(row='861', kind='Input error', input='Best verbal response', set={'Best verbal response': 'not testable'}, conf='high',
         quote='does not attempt to verbalize due to the tracheostomy',
         why="Verbal cannot be tested, not absent. The code scores 'not testable' as 4 although its explanation assumes oriented (5), so this gives 12, or 13 if the code is fixed"),
    dict(row='857', kind='Judgement call', input='Best verbal response', set={'Best verbal response': 'not testable'}, conf='medium',
         quote='His intubation prevented further assessment of verbal responsiveness.',
         why="Dataset convention is 'not testable' = full score (gives 6, or 7 with the code fixed) for a cardiac-arrest patient with no eye or motor response; clinically this is GCS 3T (as now)"),
    dict(row='849', kind='Judgement call', input='Best verbal response', set={'Best verbal response': 'confused'}, conf='medium',
         quote="When asked if she knows where she is, she responds, 'I am in the Lego aisle of Target.'",
         why='A coherent but disoriented answer is confused (4), not inappropriate words (3). The note also says she is not able to be aroused verbally'),
    dict(row='866', kind='Judgement call', input='Best verbal response', set={'Best verbal response': 'incomprehensible sounds'}, conf='low',
         quote='kept mumbling nonsensical words', why='Inappropriate words (3, as coded) or incomprehensible sounds (2)'),
    dict(row='863', kind='Note issue', input='Best motor response', set=None, conf='low',
         quote='pulled his arm away when pressure was applied to his nail bed', why='Oriented, conversant patient whose only motor finding is withdrawal; inputs follow the note'),
    # PSI (29)
    dict(row='848', kind='Input error', input='Liver disease history', set={'Liver disease history': True}, conf='high',
         quote="The patient's medical history is indicative of both a prior stroke and MASH", why='MASH (steatohepatitis) is chronic liver disease'),
    dict(row='847', kind='Judgement call', input='Neoplastic disease', set={'Neoplastic disease': False}, conf='medium',
         quote='Her medical history is significant for prior breast cancer', why='PSI counts cancer active at presentation or diagnosed within a year; no dates given'),
    dict(row='829', kind='Judgement call', input='Altered mental status', set={'Altered mental status': False}, conf='low',
         quote='The patient describes a recent episode of fever and altered mental state.', why='A past episode reported at a routine physical, not a finding at presentation'),
    dict(row='838', kind='Judgement call', input='Altered mental status', set={'Altered mental status': False}, conf='low', seen='First pass only',
         quote='brought to the ER due to recent episodes of confusion', why='No mental-status finding at presentation recorded'),
    dict(row='844', kind='Note issue', input='pH', set=None, conf='low',
         quote='a pH of 7.34 and a PaO2 of 75 mm Hg, which are within normal limits', why='Wording contradicts the number; 7.34 correctly scores +30'),
    # SOFA (43)
    dict(row='897', kind='Input error', input='Continuous positive airway pressure', set={'Continuous positive airway pressure': True}, conf='high',
         quote='The patient is receiving a continuous positive airway pressure (CPAP) therapy, with a FiO₂ set at 50%, maintaining a PaO₂ of 80 mm Hg.',
         why='CPAP stated but missing from inputs (scored as no support)'),
    dict(row='890', kind='Note issue', input='Bilirubin', set={'Bilirubin': [1.57, 'mg/L']}, conf='medium',
         quote='a total bilirubin level of 1.57 mg/L', why='Note says mg/L (0.157 mg/dL) next to jaundice and scleral icterus; input uses mg/dL. Fix the note unit or the input'),
    *[dict(row=r, kind='Judgement call', input='Urine Output', set={'Urine Output': [0, 'mL/day']}, conf='low', quote=q,
           why='Could mean anuric (renal 4 points) or not recorded (current)')
      for r, q in [('889', 'The patient has no urine output noted'), ('890', 'there is no urine output recorded')]],
    *[dict(row=r, kind='Judgement call', input='Hypotension', set={'Hypotension': False}, conf='low', quote=q,
           why='Hypotension not stated; no score change (vasopressors set the cardiovascular points)')
      for r, q in [('899', 'Her blood pressure was 91/70 mm Hg.'), ('900', 'His blood pressure was 93/62 mm Hg.')]],
    dict(row='893', kind='Note issue', input='Platelet count', set=None, conf='low',
         quote='Thrombocytopenia is also present with a platelet count of 150000 µL.', why='150,000 is not thrombocytopenia; input follows the number (0 points)'),
    # Caprini (36)
    dict(row='911', kind='Input error', input='Acute Myocardial infarction', set={'Acute Myocardial infarction': False}, conf='high',
         quote='a PMH of MI (2014)', why='A remote MI, not acute'),
    dict(row='911', kind='Input error', input='Multiple trauma in the last month', set={'Multiple trauma in the last month': False}, conf='medium',
         quote='a recent motor vehicle accident (MVA) resulting in the immobilization of her left lower extremity with a plaster cast', why='One limb injured; no multiple trauma described'),
    dict(row='913', kind='Input error', input='Congestive Heart Failure in the last month', set={'Congestive Heart Failure in the last month': True}, conf='high',
         quote="Over the past month, heart failure exacerbation and pneumonia had prolonged the patient's hospital stay", why='CHF in the last month stated but missing from inputs'),
    dict(row='922', kind='Input error', input='Surgery Type', set={'Surgery Type': 'none'}, conf='high',
         quote='He underwent minor surgery some months back for a trigger finger release', why='Old surgery; no current surgery'),
    dict(row='913', kind='Judgement call', input='Mobility', set={'Mobility': 'confined to bed >72 hours'}, conf='medium',
         quote='during which he remained mostly confined to his bed', why="Could be 'confined to bed >72 hours' rather than 'on bed rest'"),
    dict(row='909', kind='Judgement call', input='Mobility', set={'Mobility': 'confined to bed >72 hours'}, conf='medium',
         quote='remains confined to bed due to her injuries', why="Bed-bound through the hospital stay and at rehab; could be 'confined to bed >72 hours'"),
    dict(row='926', kind='Input error', input='Congestive Heart Failure in the last month', set={'Congestive Heart Failure in the last month': True}, conf='medium',
         quote='recent heart failure exacerbation', why="'Recent' is not explicitly within a month"),
    dict(row='912', kind='Input error', input='Multiple trauma in the last month', set={'Multiple trauma in the last month': True}, conf='medium',
         quote='a recent fall resulting in a spinal injury and nonoperative management of a lower leg fracture', why='Spine and leg are two body regions'),
    dict(row='921', kind='Input error', input='Family history of thrombosis', set={'Family history of thrombosis': False}, conf='high',
         quote='strong family history of thrombotic events with her mother and brother suffering MIs', why='Arterial events (MI), not venous thrombosis'),
    dict(row='921', kind='Judgement call', input='Major Surgery in the last month', set={'Major Surgery in the last month': False}, conf='low',
         quote='admitted following a stroke and thromboembolism removal', why='Probably endovascular thrombectomy, which may not be major surgery'),
    dict(row='909', kind='Input error', input='Surgery Type; Major Surgery in the last month', set={'Surgery Type': 'none', 'Major Surgery in the last month': True}, conf='medium',
         quote='well-healed surgical incision over the left femoral shaft', why='Surgery was during the earlier admission, so a past major surgery rather than current'),
    dict(row='919', kind='Judgement call', input='Hip, pelvis, or leg fracture in the last month', set={'Hip, pelvis, or leg fracture in the last month': False}, conf='low', seen='First pass only',
         quote='follow-up visit after a leg fracture', why='No date, so within a month is not established'),
    dict(row='914', kind='Judgement call', input='Surgery Type', set={'Surgery Type': 'minor'}, conf='low', seen='First pass only',
         quote='a central venous catheter was surgically inserted', why='Could count as minor surgery; the line is already counted as central venous access'),
    dict(row='927', kind='Note issue', input='Surgery Type', set=None, conf='low',
         quote='a plaster cast for proper healing of her hip', why='Hip fracture repaired with minor surgery and a plaster cast is clinically implausible; inputs follow the note'),
    *[dict(row=r, kind='Code bug', input='Previous DVT; previous PE', set=None, manual=-3, conf='medium', quote='', seen='First pass only',
           why="caprini_score.py adds 3 for previous DVT and 3 for previous PE, but its criteria list (and MDCalc) has one item 'History of DVT or PE' = +3")
      for r in ('915', '920', '928')],
    # gpt-6-astra only (verify_entities.py on review_1_17_rows.csv)
    *[dict(row=r, kind='Judgement call', input='Major Surgery in the last month', set={'Major Surgery in the last month': True}, conf='low', seen='Astra only', quote=q,
           why="The operation is scored as the current 'Surgery Type' only; whether it also counts as major surgery in the last month is the same double-counting question as rows 909, 910, 913")
      for r, q in [('915', 'currently recovering in critical care following recent major laparoscopic surgery for ovarian cancer'),
                   ('920', 'on bed rest for 4 days after a bowel resection')]],
    # second pass only
    dict(row='855', kind='Judgement call', input='Best eye response', set={'Best eye response': 'eyes open spontaneously'}, conf='low', seen='Second pass only',
         quote='He was awake ... He opens his eyes upon verbal request ... His speech was disjointed',
         why='Awake (spontaneous, 4) vs opens to voice (3, as coded); verbal could also be inappropriate words (3). 12 is the most likely answer'),
    dict(row='871', kind='Judgement call', input='History of severe organ failure or immunocompromise', set={'History of severe organ failure or immunocompromise': False}, conf='low', seen='Second pass only',
         quote='liver cirrhosis', why='MDCalc requires biopsy-proven cirrhosis (or portal hypertension / prior hepatic failure); usually accepted'),
    # APACHE II (28)
    dict(row='884', kind='Input error', input='Surgery Type', set={'Surgery Type': 'Nonoperative'}, conf='high',
         quote='chronic liver failure requiring a scheduled transplant. The individual presented to the emergency unit', why='No surgery this admission; the transplant has not happened'),
    dict(row='872', kind='Input error', input='Surgery Type', set={'Surgery Type': 'Nonoperative'}, conf='high',
         quote='presents to the emergency department', why="ED admission without surgery: not 'Elective' (if the organ-failure history is kept)"),
    dict(row='876', kind='Input error', input='Surgery Type', set={'Surgery Type': 'Nonoperative'}, conf='high',
         quote='acute aortic disection which was treated with an emergency surgery in the past, presents to the emergency department', why='The emergency surgery was in the past; this admission is nonoperative (same points)'),
    dict(row='872', kind='Judgement call', input='History of severe organ failure or immunocompromise', set={'History of severe organ failure or immunocompromise': False}, conf='medium',
         quote='chronic renal failure, managed conservatively, and had a scheduled kidney transplant in the past', why='Severe renal insufficiency means chronic dialysis; no transplant immunosuppression stated'),
    *[dict(row=r, kind='Judgement call', input='Acute renal failure', set={'Acute renal failure': False}, conf=c, quote=q, why=w)
      for r, c, q, w in [('870', 'medium', 'chronic kidney disease not requiring dialysis ... decreased urine output', 'Acute renal failure never stated (chronic kidney disease is)'),
                         ('871', 'medium', 'a creatinine level of 4.58 mg/dL, indicating impaired renal function', 'Acute renal failure never stated'),
                         ('877', 'medium', 'the possible occurrence of acute renal failure', 'Only possible acute renal failure'),
                         ('869', 'low', 'Urine output over the past six hours was 104 mL, suggesting oliguria', 'Oliguria (about 17 mL/h) supports acute kidney injury, but it is not stated (second pass: OK as is)')]],
    dict(row='874', kind='Note issue', input='', set=None, conf='low', quote="A gentleman of 65 years ... Her hematocrit", why='Patient sex inconsistent in the note; no score effect'),
    dict(row='878', kind='Note issue', input='', set=None, conf='low', quote='presented to the clinic for a routine check-up',
         why='Routine check-up with no history, yet A-a gradient 430 on 52% FiO2 and acute renal failure: clinically implausible'),
]


def base_1_17():
    """The 1_17 rows for CALCS as the build would use them, with code-computed answers and test_data_final Row Numbers."""
    final = pd.read_csv(DATA / 'test_data_final.csv', dtype=str, keep_default_na=False)
    row = dict(zip(zip(final['Calculator ID'], final['Note ID']), final['Row Number']))
    t = pd.read_csv(DATA / 'test_data_1_17_2025.csv', dtype=str, keep_default_na=False)
    t = t[t['Calculator ID'].isin(CALCS)].reset_index(drop=True)
    t['Relevant Entities'] = [bct.rename_entities(c, e) for c, e in zip(t['Calculator ID'], t['Relevant Entities'])]
    for old, new in bct.NOTE_TYPO_FIXES.items():
        t['Patient Note'] = t['Patient Note'].str.replace(old, new, regex=False)
    final_rows = [row[k] for k in zip(t['Calculator ID'], t['Note ID'])]
    t['Row Number'] = [str(i + 1) for i in range(len(t))]  # generate_csv.py numbers rows 1..N
    with tempfile.TemporaryDirectory() as tmp:
        src, out = Path(tmp) / 'in.csv', Path(tmp) / 'out.csv'
        t.to_csv(src, index=False)
        subprocess.run([sys.executable, str(ROOT / 'run_generate_csv.py'), str(src), str(out)], check=True, capture_output=True)
        gen = pd.read_csv(out, dtype=str, keep_default_na=False)
    gen['Row Number'] = final_rows
    return gen


def add_astra(target):
    """Add 'Astra Flagged': whether verify_entities.py (gpt-6-astra) flagged the finding's input."""
    path = ROOT / 'entities_rows_1_17.csv'
    if not path.exists():
        return
    import json
    astra = pd.read_csv(path, dtype=str, keep_default_na=False)
    flagged = {(r['Row Number'], p['input']) for _, r in astra.iterrows() if r['Problems']
               for p in json.loads(r['Problems'])}
    out = pd.read_csv(DATA / target, dtype=str, keep_default_na=False)
    out['Astra Flagged'] = [any((r, k.strip()) in flagged for k in i.split(';')) for r, i in zip(out['Row Number'], out['Input'])]
    out.to_csv(DATA / target, index=False, encoding='utf-8-sig')


if __name__ == '__main__':
    build(F, base_1_17(), 'handwritten_review_1_17_findings.csv')
    add_astra('handwritten_review_1_17_findings.csv')
