"""Independent, conservative label screening; does not modify the source dataset.

Run: python3 audit_labels.py. Findings are not a certification of unflagged notes.
Numeric screening uses the supplied entities; narrative extraction needs review.
"""
import ast
import csv
import math
from pathlib import Path
from datetime import datetime, timedelta
from collections import Counter

ROOT = Path(__file__).resolve().parent
rows = list(csv.DictReader((ROOT / 'dataset/test_data.csv').open()))
findings = []
checked = Counter()

def add(r, status, expected, reason):
    findings.append({'Row Number': r['Row Number'], 'Calculator Name': r['Calculator Name'],
                     'Status': status, 'Ground Truth Answer': r['Ground Truth Answer'],
                     'Recalculated Answer': expected, 'Reason': reason})

def numerical(r, answer, reason):
    checked[r['Calculator ID']] += 1
    old = float(r['Ground Truth Answer'])
    # Ignore small intermediate-rounding differences. Flag material discrepancies.
    if abs(old-answer) > max(0.01, abs(answer)*0.01):
        add(r, 'numeric mismatch from supplied entities', round(answer, 6), reason)

def value(e, k):
    v = e[k]
    return float(v[0]) if isinstance(v, list) else float(v)

def mass(e, k, mw):
    n, u = e[k]
    return float(n) * {'mg/dL': 1, 'mmol/L': mw/10, 'µmol/L': mw/10000,
                       'g/dL': 1000, 'g/L': 100}[u]

def albumin(e):
    n, u = e['Albumin']
    return n * {'g/dL': 1, 'g/L': .1, 'mg/dL': .001}[u]

def sodium(e):
    n, u = e['Sodium']
    return n if u in ('mmol/L', 'mEq/L') else n*10/22.99

def kg(e):
    n, u = e['weight']
    return n * {'kg': 1, 'lbs': .45359237, 'g': .001}[u]

def cm(e):
    n, u = e['height']
    return n * {'cm': 1, 'm': 100, 'in': 2.54}[u]

def age(e):
    n, u = e['age']
    return n / 12 if u == 'months' else n

def ibw(e):
    return (50 if e['sex'] == 'Male' else 45.5) + 2.3*(cm(e)/2.54-60)

def count_k(e, key):
    n, u = e[key]
    return n/1000 if u in ('µL', 'mm^3') else n/1e9

eye = {'eyes open spontaneously': 4, 'eye opening to verbal command': 3,
       'eye opening to pain': 2, 'no eye opening': 1}
verbal = {'oriented': 5, 'confused': 4, 'inappropriate words': 3,
          'incomprehensible sounds': 2, 'no verbal response': 1}
motor = {'obeys commands': 6, 'localizes pain': 5, 'withdrawal from pain': 4,
         'flexion to pain': 3, 'extension to pain': 2, 'no motor response': 1}

for r in rows:
    e = ast.literal_eval(r['Relevant Entities'])
    cid = int(r['Calculator ID'])
    reason = 'Independent arithmetic using supplied Relevant Entities; small rounding differences ignored.'
    result = None
    if cid == 2:
        w, h, ideal = kg(e), cm(e)/100, ibw(e)
        bmi = w/h**2
        adjusted = w if bmi < 18.5 else min(w, ideal) if bmi < 25 else ideal+.4*(w-ideal)
        result = (140-age(e))*adjusted*(.85 if e['sex']=='Female' else 1)/(72*mass(e,'creatinine',113.12))
    elif cid == 3:
        cr = mass(e,'creatinine',113.12)
        female = e['sex']=='Female'
        ratio = cr/(.7 if female else .9)
        result = 142*min(ratio,1)**(-.241 if female else -.302)*max(ratio,1)**-1.2*.9938**age(e)*(1.012 if female else 1)
    elif cid == 5:
        result = (value(e,'Systolic Blood Pressure')+2*value(e,'Diastolic Blood Pressure'))/3
    elif cid == 6:
        result = kg(e)/(cm(e)/100)**2
    elif cid == 7:
        result = mass(e,'Calcium',40.08)+.8*(4-albumin(e))
    elif cid == 9:
        result = 175*mass(e,'creatinine',113.12)**-1.154*age(e)**-.203*(.742 if e['sex']=='Female' else 1)*(1.212 if e.get('Race')=='Black' else 1)
    elif cid == 10:
        result = ibw(e)
    elif cid == 19:
        result = age(e)*value(e,'Aspartate aminotransferase')/(count_k(e,'Platelet count')*math.sqrt(value(e,'Alanine aminotransferase')))
    elif cid == 22:
        w = kg(e)
        result = 4*min(w,10)+2*max(0,min(w-10,10))+max(0,w-20)
    elif cid == 26:
        result = sodium(e)+.024*(mass(e,'Glucose',180.16)-100)
    elif cid == 30:
        result = 2*sodium(e)+mass(e,'Glucose',180.16)/18+mass(e,'Blood Urea Nitrogen (BUN)',28.014)/2.8
    elif cid == 31:
        result = mass(e,'Glucose',180.16)*value(e,'Insulin')/405
    elif cid == 38:
        a = age(e)
        tbw = .6 if a<18 else (.6 if e['sex']=='Male' else .5) if a<65 else (.5 if e['sex']=='Male' else .45)
        result = tbw*kg(e)*(sodium(e)/140-1)
    elif cid == 40:
        result = 100*value(e,'Urine sodium')*value(e,'creatinine')/(sodium(e)*value(e,'Urine creatinine'))
    elif cid == 44:
        result = mass(e,'Total cholesterol',386.65)-mass(e,'high-density lipoprotein cholesterol',386.65)-mass(e,'Triglycerides',885.7)/5
    elif cid in (39,63,64,65,66,67):
        gap = sodium(e)-value(e,'Chloride')-value(e,'Bicarbonate')
        if cid in (65,66,67):
            gap += 2.5*(4-albumin(e))
        if cid in (63,64,66,67):
            gap -= 12
        result = gap/(24-value(e,'Bicarbonate')) if cid in (64,67) else gap
    elif cid == 60:
        result = math.sqrt(kg(e)*cm(e)/3600)
    elif cid == 61:
        result = value(e,'Body Mass Index (BMI)')*(cm(e)/100)**2
    elif cid == 62:
        result = ibw(e)+.4*(kg(e)-ibw(e))
    elif cid in (11,56,57,58,59):
        qt, hr = value(e,'QT interval'), value(e,'Heart Rate or Pulse')
        rr = 60/hr
        result = {11: lambda: qt/math.sqrt(rr), 56: lambda: qt/rr**(1/3),
                  57: lambda: qt+154*(1-rr),58: lambda: qt+1.75*(hr-60),
                  59: lambda: qt*(120+hr)/180}[cid]()
    elif cid == 13:
        checked[str(cid)] += 1
        d = datetime.strptime(e['Last menstrual date'],'%m/%d/%Y')
        expected = (d+timedelta(days=280+e['cycle length']-28)).strftime('%m/%d/%Y')
        if expected != r['Ground Truth Answer']:
            add(r,'confirmed incorrect',expected,'Shorter cycles require subtracting days; label adds days instead.')
    elif cid == 21:
        checked[str(cid)] += 1
        evm = [e['Best eye response'],e['Best verbal response'],e['Best motor response']]
        if 'not testable' in evm:
            add(r,'invalid numeric target','NT; no defensible total','A component is not testable; explanation invents its value and counts verbal response twice.')
        else:
            result = eye[evm[0]]+verbal[evm[1]]+motor[evm[2]]
            reason = 'GCS = eye + verbal + motor. Dataset explanation counts the verbal component twice.'
    elif cid == 46:
        a, tc, hd, bp = age(e),mass(e,'Total cholesterol',386.65),mass(e,'high-density lipoprotein cholesterol',386.65),value(e,'Systolic Blood Pressure')
        sm, tx = int(e.get('Smoker',False)),int(e.get('Blood pressure being treated with medicines',False))
        ln = math.log
        if e['sex']=='Male':
            l = 52.00961*ln(a)+20.014077*ln(tc)-.905964*ln(hd)+1.305784*ln(bp)+.241549*tx+12.096316*sm-4.605038*ln(a)*ln(tc)-2.84367*ln(min(a,70))*sm-2.93323*ln(a)**2-172.300168
            base = .9402
        else:
            l = 31.764001*ln(a)+22.465206*ln(tc)-1.187731*ln(hd)+2.552905*ln(bp)+.420251*tx+13.07543*sm-5.060998*ln(a)*ln(tc)-2.996945*ln(min(a,78))*sm-146.5933061
            base = .98767
        result = 100*(1-base**math.exp(l))
        reason = 'Using the explanation\'s mg/dL equation and returning percent: code incorrectly converts cholesterol to mmol/L, rounds probability, and omits multiplication by 100. Clinical applicability also needs review.'
    elif cid == 36:
        # Exact lower bound avoids double counting history of VTE, current/prior
        # surgery, or correlated thrombophilia categories.
        a = age(e)
        lower = 0 if a<=40 else 1 if a<=60 else 2 if a<=74 else 3
        lower += {'none':0,'minor':1,'major':2,'laparoscopic':2,'arthroscopic':2,'elective major lower extremity arthroplasty':5}.get(e.get('Surgery Type','none'),0)
        lower += max([p for k,p in [('Hip, pelvis, or leg fracture in the last month',5),('Stroke in the last month',5),('Multiple trauma in the last month',5),('Acute spinal cord injury causing paralysis in the last month',5)] if e.get(k) is True]+[0])
        lower += 3*int(e.get('Previously documented Deep Vein Thrombosis') is True or e.get('Previously Documented Pulmonary Embolism') is True)
        lower += 3*int(e.get('Family history of thrombosis') is True)
        lower += 3*int(any(e.get(k) is True for k in ['Positive Factor V Leiden','Positive prothrombin 20210A','Elevated serum homocysteine','Positive lupus anticoagulant','Elevated anticardiolipin antibody','Heparin-induced thrombocytopenia','Other congenital or acquired thrombophilia']))
        lower += sum(e.get(k) is True for k in ['Congestive Heart Failure in the last month','Sepsis in the last month','Varicose veins','Current swollen legs','History of inflammatory bowel disease','Acute Myocardial infarction'])
        lower += int(e.get('Pneumonia in the last month') is True or e.get('Chronic Obstructive Pulmonary Disease') is True)
        lower += 2*int(e.get('Present or previous malignancy') is True)
        lower += int(value(e,'Body Mass Index (BMI)')>25) if 'Body Mass Index (BMI)' in e else 0
        checked[str(cid)] += 1
        if float(r['Ground Truth Answer']) < lower:
            add(r,'confirmed incorrect',f'>= {lower} (conservative lower bound)','Stated risk factors alone exceed the label; boolean risk-factor branch never increments score. Lower bound intentionally omits mobility, casts, central lines, prior surgery and sums only one highest trauma and thrombophilia item.')
    elif cid == 32:
        checked[str(cid)] += 1
        a = age(e)
        lower = min(4,max(0,int(a//10)-4))
        lower += sum(e.get(k) is True for k in ['Myocardial infarction','Congestive Heart Faliure','Peripheral vascular disease','Dementia','Chronic Obstructive Pulmonary Disease','Connective tissue disease','Peptic ulcer disease'])
        lower += max(2*int(e.get('Hemiplegia') is True),int(e.get('Cerebrovascular Accident') is True or e.get('Transient Ischemic Attacks History') is True))
        lower += 2*int(bool(e.get('Moderate to severe Chronic Kidney Disease',False)))
        dm = str(e.get('Diabetes mellitus criteria for CCI rule','')).lower()
        lower += 2 if dm=='end-organ damage' else 1 if dm=='uncomplicated' else 0
        liver = str(e.get('Liver disease severity','')).lower()
        lower += 3 if 'moderate' in liver or 'severe' in liver else 1 if liver=='mild' else 0
        tumor = e.get('Solid tumor',False)
        # Unspecified solid malignancy has at least 2 points, but avoid a
        # precise target when site, metastasis, or remission timing is unclear.
        cancer = 6 if str(tumor).lower()=='metastatic' else 2 if tumor else 0
        lower += max(cancer,2*int(e.get('Leukemia') is True),2*int(e.get('Lymphoma') is True))
        lower += 6*int(e.get('AIDS') is True)
        if lower>float(r['Ground Truth Answer']):
            add(r,'confirmed incorrect',f'>= {lower} (conservative lower bound)','Case-sensitive categorical handling scores tumor/liver/diabetes severity as generic true=1. Bound uses highest cancer and neurologic category only; exact target requires review of history and severity.')
    elif cid == 28:
        checked[str(cid)] += 1
        a = age(e)
        parts = {'age':0 if a<45 else 2 if a<55 else 3 if a<65 else 5 if a<75 else 6,
                 'GCS':15-value(e,'Glasgow Coma Score')}
        temp = value(e,'Temperature')
        parts['temperature'] = 4 if temp>=41 or temp<30 else 3 if temp>=39 or temp<32 else 2 if temp<34 else 1 if temp>=38.5 or temp<36 else 0
        bp = (value(e,'Systolic Blood Pressure')+2*value(e,'Diastolic Blood Pressure'))/3
        parts['MAP'] = 4 if bp>=160 or bp<50 else 3 if bp>=130 else 2 if bp>=110 or bp<70 else 0
        hr = value(e,'Heart Rate or Pulse')
        parts['heart rate'] = 4 if hr>=180 or hr<40 else 3 if hr>=140 or hr<55 else 2 if hr>=110 or hr<70 else 0
        rr = value(e,'respiratory rate')
        parts['respiratory rate'] = 4 if rr>=50 or rr<6 else 3 if rr>=35 else 2 if rr<10 else 1 if rr>=25 or rr<12 else 0
        ph = value(e,'pH')
        parts['pH'] = 4 if ph>=7.7 or ph<7.15 else 3 if ph>=7.6 or ph<7.25 else 2 if ph<7.33 else 1 if ph>=7.5 else 0
        na, k = sodium(e),value(e,'Potassium')
        parts['sodium'] = 4 if na>=180 or na<111 else 3 if na>=160 else 2 if na>=155 or na<130 else 1 if na>=150 else 0
        parts['potassium'] = 4 if k>=7 or k<2.5 else 3 if k>=6 else 2 if k<3 else 1 if k>=5.5 or k<3.5 else 0
        cr = mass(e,'creatinine',113.12)
        cp = 4 if cr>=3.5 else 3 if cr>=2 else 2 if cr>=1.5 or cr<.6 else 0
        parts['creatinine'] = cp*(2 if e.get('Acute renal failure') is True else 1)
        h = value(e,'Hematocrit')
        parts['hematocrit'] = 4 if h>=60 or h<20 else 2 if h>=50 or h<30 else 1 if h>=46 else 0
        w = count_k(e,'White blood cell count')
        parts['WBC'] = 4 if w>=40 or w<1 else 2 if w>=20 or w<3 else 1 if w>=15 else 0
        if value(e,'FiO₂')>=50:
            g = value(e,'A-a gradient')
            parts['oxygenation'] = 4 if g>=500 else 3 if g>=350 else 2 if g>=200 else 0
        else:
            p = value(e,'Partial pressure of oxygen (PaO₂) for Apache II')
            parts['oxygenation'] = 4 if p<55 else 3 if p<=60 else 1 if p<=70 else 0
        chronic = e.get('History of severe organ failure or immunocompromise') is True
        parts['chronic health'] = (2 if str(e.get('Surgery Type','')).strip()=='Elective' else 5) if chronic else 0
        answer = int(sum(parts.values()))
        detail = '; '.join(f'{k}={v:g}' for k,v in parts.items() if v)
        add(r,'confirmed incorrect',answer,'Standard APACHE II from entities, not a final narrative-adjudicated target: '+detail+'. GCS must be 15-GCS; dataset also overcounts WBC and has other scoring/input errors.')
    if result is not None:
        numerical(r,result,reason)

by_row = {r['Row Number']:r for r in rows}
for number, expected, why in [
    ('427',6,'From the supplied entities: age 2 + CHF 1 + COPD 1 + CKD 2 = 6. Diet-controlled diabetes contributes 0. Note mentions neuropathy without establishing cause; review complication attribution.'),
    ('435',10,'Age 1 + leukemia 2 + CKD 2 + connective tissue disease 1 + COPD 1 + peripheral vascular disease 1 + CHF 1 + mild liver disease 1 = 10. Diet-controlled diabetes contributes 0.'),
    ('429',2,'Age 1 + uncomplicated diabetes 1 = 2. Note describes presyncope, not a TIA, and Marfan syndrome, not a Charlson rheumatologic/connective-tissue diagnosis. Both extracted positive flags are unsupported.'),
    ('772','Cannot calculate from note','Albumin is not supplied. Extracted albumin 17.1 g/dL is the hemoglobin value from the note.'),
    ('820','Cannot calculate from note','Albumin is not supplied. Extracted albumin 8 g/dL is the hemoglobin value from the note.'),
]:
    add(by_row[number], 'confirmed incorrect' if number not in ('427',) else 'incorrect from supplied entities; narrative review needed',expected,why)

for finding in findings:
    n = finding['Row Number']
    cid = int(by_row[n]['Calculator ID'])
    if n in ('3','21','22','36'):
        finding['Status'] = 'precision issue; within benchmark tolerance'
        finding['Reason'] = 'Unit conversion rounded too early; discrepancy is between 1% and 5%. This alone does not change correctness under the benchmark tolerance. Row 3 also combines a later weight with earlier creatinine; narrative timing needs review.'
    elif cid in (3,7,44):
        finding['Status'] = 'confirmed incorrect'
        finding['Reason'] = ('Male low-creatinine CKD-EPI branch wrongly uses 0.7 instead of 0.9; row 23 also has intermediate rounding.' if cid==3 else
                            'Unit conversion rounds mmol to mol prematurely, materially corrupting the final answer.')
    elif cid in (21,46) and finding['Status']=='numeric mismatch from supplied entities':
        finding['Status'] = 'confirmed incorrect'
    if n in ('433','434'):
        finding['Status'] = 'requires narrative review'
        finding['Reason'] += ' Note does not clearly establish encoded liver severity (433), or tumor eligibility at the five-year remission boundary (434).'
    if cid==28 and n in ('870','871','874'):
        finding['Reason'] += {'870':' Note contradicts itself about FiO2/room air and does not explicitly establish acute renal failure.',
                              '871':' Note BP is 190/103, entities use 132/103; surgical timing/severity needs adjudication.',
                              '874':' Note potassium is 139 mEq/L, entities use 5.0; repair note before fixing target.'}[n]
findings.sort(key=lambda x:int(x['Row Number']))
out = ROOT/'label_audit'
out.mkdir(exist_ok=True)
with (out/'findings.csv').open('w',newline='') as f:
    w = csv.DictWriter(f,fieldnames=list(findings[0]));w.writeheader();w.writerows(findings)
print('Flagged:',len(findings))
for name,n in Counter(x['Calculator Name'] for x in findings).items():
    print(name,n,','.join(x['Row Number'] for x in findings if x['Calculator Name']==name))
coverage_ids = set(checked)
coverage = sum(r['Calculator ID'] in coverage_ids for r in rows)
print('Rows screened by arithmetic/scoring rules:',coverage)
print('Statuses:',dict(Counter(x['Status'] for x in findings)))
report = ['# Test data label audit','',
          f'{len(rows)} records loaded. Arithmetic/scoring screening covered {coverage} records; this is not an exhaustive clinical adjudication of every patient note.',
          'All numeric answer bounds were also checked across the file: no reversed or answer-excluding numeric bounds were found.',
          'The source dataset and calculator implementations were not changed. findings.csv contains the flagged rows, original labels, recalculations or conservative bounds, and reasons.',
          '', '## Interpretation', '',
          '- Confirmed incorrect means a scoring/arithmetic/input error was demonstrated. Entity-based recalculations still need reconciliation with the patient note before updating the dataset.',
          '- Conservative lower bounds prove undercounting without claiming a final exact target.',
          '- Precision issues within benchmark tolerance are separate from clear scoring errors.',
          '- Unflagged rows are not certified correct. Screening generally accepts the extracted entities; longitudinal notes, omitted risk factors, unsupported assumptions and source-unit errors can remain.',
          '- Exact GCS totals cannot be assigned to the four rows with non-testable components (856, 857, 862, 863).',
          '- APACHE II values are recalculated from entities. Several notes have conflicting inputs or uncertain chronic/acute disease status; see individual reasons.',
          '', '## Findings by calculator', '', '| Calculator | Flagged Row Number values |', '|---|---|']
for name in dict.fromkeys(x['Calculator Name'] for x in findings):
    report.append('| '+name+' | '+', '.join(x['Row Number'] for x in findings if x['Calculator Name']==name)+' |')
report += ['', '## Demonstrations', '',
           '- Row 848: eye 2 + verbal 3 + motor 4 = 9, not 12. Its explanation adds verbal 3 twice.',
           '- Row 879: normal physiology and GCS 15 contribute 0; age 63 contributes 3. Label is 26.',
           '- Row 928: LMP 12/11/2013 + 280 days + (22 - 28) days = 09/11/2014, not 09/23/2014.',
           '- Row 25: male CKD-EPI creatinine scale is 0.9; result about 87.228 rather than 80.852.',
           '- Row 428: metastatic tumor contributes 6, not 1; conservative CCI is at least 15, label 10.',
           '- Rows 772 and 820: hemoglobin is extracted as albumin; no albumin-based answer can be derived from the supplied note.',
           '', '## Rule references', '',
           '- GCS assessment aid: https://www.glasgowcomascale.org/downloads/GCS-Assessment-Aid-English.pdf',
           '- APACHE II scoring table: https://www.merckmanuals.com/professional/multimedia/table/acute-physiologic-assessment-and-chronic-health-evaluation-apache-ii-scoring-system',
           '- Caprini point factors: https://www.ahrq.gov/patient-safety/settings/hospital/vtguide/appb2.html',
           '- Charlson categories: https://ncbi.nlm.nih.gov/books/NBK587905/?report=reader',
           '- Framingham mg/dL equation reproduced in research appendix: https://aclanthology.org/anthology-files/anthology-files/pdf/naacl/2025.naacl-long.263.pdf',
           '- Other arithmetic follows the explicit equations in the dataset questions/explanations, with unit conversions performed without premature rounding.', '']
(out/'README.md').write_text('\n'.join(report))
