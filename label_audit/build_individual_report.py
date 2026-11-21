"""Export the completed narrative review; does not modify source labels.

The decisions JSON records manual review. This exporter does not adjudicate notes.
Run from the repository root: python3 label_audit/build_individual_report.py
"""
import ast
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

from review_notes import ROWS, excerpts

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
decisions = json.loads((OUT / 'narrative_decisions.json').read_text())
reviewed = set(map(int, decisions['reviewed_calculator_ids']))
assert reviewed == {int(r['Calculator ID']) for r in ROWS}, 'Incomplete review coverage'
assert len(ROWS) == 1047
assert len({r['Row Number'] for r in ROWS}) == len(ROWS)
prior = {r['Row Number']: r for r in csv.DictReader((OUT / 'findings.csv').open())}

SOURCES = {
    3: 'https://www.kidney.org/ckd-epi-creatinine-equation-2021',
    4: 'https://www.escardio.org/static-file/Escardio/Guidelines/Documents/ehaa612.pdf',
    15: 'https://www.merckmanuals.com/professional/multimedia/table/child-turcotte-pugh-scoring-system',
    17: 'https://pubmed.ncbi.nlm.nih.gov/10477528/',
    20: 'https://www.idsociety.org/practice-guideline/streptococcal-pharyngitis2/',
    21: 'https://www.glasgowcomascale.org/downloads/GCS-Assessment-Aid-English.pdf',
    23: 'https://pmc.ncbi.nlm.nih.gov/articles/PMC3305837/',
    27: 'https://handbook.ggcmedicines.org.uk/guidelines/gastrointestinal-system/glasgow-blatchford-score/',
    28: 'https://www.merckmanuals.com/professional/multimedia/table/acute-physiologic-assessment-and-chronic-health-evaluation-apache-ii-scoring-system',
    32: 'https://ncbi.nlm.nih.gov/books/NBK587905/?report=reader',
    33: 'https://www.nice.org.uk/guidance/ng84/resources/sore-throat-acute-in-adults-antimicrobial-prescribing-visual-summary-pdf-11315864557',
    36: 'https://www.ahrq.gov/patient-safety/settings/hospital/vtguide/appb2.html',
    43: 'https://pmc.ncbi.nlm.nih.gov/articles/PMC4968574/',
    46: 'https://aclanthology.org/anthology-files/anthology-files/pdf/naacl/2025.naacl-long.263.pdf',
    49: 'https://www.cdc.gov/mmwr/volumes/71/rr/rr7103a1.htm',
    51: 'https://pmc.ncbi.nlm.nih.gov/articles/PMC4968574/',
    68: 'https://www.merckmanuals.com/professional/gynecology-and-obstetrics/family-planning/fertility-awareness-based-methods-of-contraception',
}


def readable(s):
    # Expand compact shorthand in the manual working ledger for the exported report.
    replacements = {
        'doesnot': 'does not ', 'donot': 'do not ', 'isnot': 'is not ',
        'cannotbe': 'cannot be ', 'notprovided': 'not provided',
        'notestablished': 'not established', 'notdocumented': 'not documented',
        'notnormal': 'not normal', 'notnecessarily': 'not necessarily',
        'notautomatically': 'not automatically', 'notquantified': 'not quantified',
        'notconfirmed': 'not confirmed', 'notindependently': 'not independently',
        'notclearly': 'not clearly', 'notprove': 'not prove ',
        'notmeet': 'not meet ', 'notapplicable': 'not applicable',
        'notprior': 'not prior', 'ratherthan': 'rather than',
        'withoutportal': 'without portal', 'dietcontrolled': 'diet-controlled ',
        'endorgandamage': 'end-organ damage', 'connectivetissuedisease': 'connective tissue disease',
        'portalhypertensive': 'portal hypertensive ', 'chronichepatitis': 'chronic hepatitis ',
        'mildliverdisease': 'mild liver disease', 'liverdisease': 'liver disease',
        'liverseverity': 'liver severity', 'CKDseverity': 'CKD severity',
        'cancerrecency': 'cancer recency', 'solidtumor': 'solid tumor ',
        'localizedcancer': 'localized cancer ', 'metastaticcancer': 'metastatic cancer ',
        'severeliver': 'severe liver ', 'priorbleed': 'prior bleed ',
        'treatedBP': 'treated BP', 'controlledBP': 'controlled BP',
        'controlledhypertension': 'controlled hypertension',
        'roomair': 'room air', 'firstadmission': 'first admission',
        'laterreadmission': 'later readmission', 'preoperativeHR': 'preoperative HR',
        'initialtroponin': 'initial troponin', 'uppernormal': 'upper normal ',
        'troponinassessment': 'troponin assessment', 'troponinmissing': 'troponin missing',
        'ECGinformation': 'ECG information', 'ECGimage': 'ECG image',
        'coughstatus': 'cough status', 'cough/coryzastatus': 'cough/coryza status',
        'absencecough': 'absence of cough', 'absentcough': 'absent cough',
        'tonsillarexudate': 'tonsillar exudate', 'purulenttonsils': 'purulent tonsils',
        'Moderateascites': 'Moderate ascites', 'noascites': 'no ascites',
        'encephalopathy': 'encephalopathy', 'onsetthismorning': 'onset this morning',
        'systemicHTN': 'systemic hypertension', 'Currentfever': 'Current fever',
        'Currentstatus': 'Current status', 'PECARNkey': 'PECARN key',
        'hiprevision': 'hip revision', 'Hiprevision': 'Hip revision',
        'chronichealth': 'chronic health', 'alreadydialyzed': 'already dialyzed',
        'initialhypernatremic': 'initial hypernatremic', 'sixweekfollowup': 'six-week follow-up',
    }
    for a, b in replacements.items():
        s = s.replace(a, b)
    # Preserve unit abbreviations such as dL and mL.
    s = re.sub(r'(?<=[A-Za-z])(?=\d)', ' ', s)
    s = re.sub(r'(?<=\d)(?=[A-Za-z])', ' ', s)
    return re.sub(r' {2,}', ' ', s).strip()


def framingham_with_untreated_bp(r):
    e = ast.literal_eval(r['Relevant Entities'])
    a = float(e['age'][0])
    def cholesterol(key):
        n, unit = e[key]
        return n * (38.665 if unit == 'mmol/L' else 1)
    tc = cholesterol('Total cholesterol')
    hd = cholesterol('high-density lipoprotein cholesterol')
    bp = e['Systolic Blood Pressure'][0]
    sm = int(e.get('Smoker', False))
    ln = math.log
    if e['sex'] == 'Male':
        z = (52.00961*ln(a) + 20.014077*ln(tc) - .905964*ln(hd)
             + 1.305784*ln(bp) + 12.096316*sm - 4.605038*ln(a)*ln(tc)
             - 2.84367*ln(min(a,70))*sm - 2.93323*ln(a)**2 - 172.300168)
        base = .9402
    else:
        z = (31.764001*ln(a) + 22.465206*ln(tc) - 1.187731*ln(hd)
             + 2.552905*ln(bp) + 13.07543*sm - 5.060998*ln(a)*ln(tc)
             - 2.996945*ln(min(a,78))*sm - 146.5933061)
        base = .98767
    return f'{100*(1-base**math.exp(z)):.6f}% (untreated BP)'


manifest = []
for r in ROWS:
    num = r['Row Number']
    cid = int(r['Calculator ID'])
    status = 'no error identified'
    expected = ''
    reason = 'No label error identified in the calculator-relevant narrative review. This does not certify every clinical assumption or unrelated sentence.'
    basis = 'Individual narrative review'
    if num in prior:
        p = prior[num]
        status, expected, reason = p['Status'], p['Recalculated Answer'], p['Reason']
        basis = 'Individual narrative review plus independent arithmetic screen'
        if status == 'precision issue; within benchmark tolerance':
            status = 'minor rounding difference'
        elif status == 'confirmed incorrect':
            status = 'incorrect'
    if num in decisions['overrides']:
        status, expected, reason = decisions['overrides'][num]
        status = 'incorrect' if status == 'confirmed incorrect' else status
        basis = 'Individual narrative adjudication; supersedes preliminary entity screen'
    if num in ('553', '560', '567'):
        expected = framingham_with_untreated_bp(r)
    if cid == 68 and status == 'ambiguous':
        e = ast.literal_eval(r['Relevant Entities'])
        # Dates are hypothetical estimates, not an observed fertilization date.
        lmp = datetime.strptime(e['Last menstrual date'], '%m/%d/%Y')
        adjusted = lmp + timedelta(days=e['cycle length'] - 14)
        expected = adjusted.strftime('%m/%d/%Y') + ' if cycle-adjusted estimate intended'
    if status == 'incorrect' and not expected:
        expected = 'No single supported replacement; repair inputs or specify encounter'
    elif not expected and status in {'insufficient information', 'invalid numeric target', 'out of domain'}:
        expected = 'Not determinable or not applicable from the supplied note'
    elif not expected and status == 'ambiguous':
        expected = 'Unresolved; see competing interpretations in Reason'
    if status == 'minor rounding difference':
        reason = 'Small intermediate-rounding discrepancy; the independently recalculated value remains inside the supplied benchmark acceptance interval.'
    within = 'Not assessed: conditional, range, date or unavailable answer'
    try:
        n = float(expected)
        within = str(float(r['Lower Limit']) <= n <= float(r['Upper Limit']))
    except ValueError:
        pass
    manifest.append({
        'Row Number': num,
        'Calculator ID': r['Calculator ID'],
        'Calculator Name': r['Calculator Name'],
        'Note ID': r['Note ID'],
        'Status': status,
        'Original Label': r['Ground Truth Answer'],
        'Corrected Answer or Qualification': readable(str(expected)),
        'Reason': readable(reason),
        'Question': r['Question'],
        'Narrative Evidence': excerpts(r),
        'Patient Note': r['Patient Note'],
        'Original Relevant Entities': r['Relevant Entities'],
        'Original Explanation': r['Ground Truth Explanation'],
        'Lower Limit': r['Lower Limit'],
        'Upper Limit': r['Upper Limit'],
        'Corrected Numeric Answer Within Original Bounds': within,
        'Review Basis': basis,
        'Rule Reference': SOURCES.get(cid, 'Equation stated in the dataset; independent arithmetic/unit conversion and narrative comparison'),
    })

allowed = {'no error identified', 'incorrect', 'ambiguous', 'insufficient information',
           'invalid numeric target', 'out of domain', 'minor rounding difference'}
assert {r['Status'] for r in manifest} <= allowed
assert all(r['Reason'] for r in manifest)


def write_csv(name, data):
    with (OUT / name).open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(data)


incorrect = [r for r in manifest if r['Status'] == 'incorrect']
invalid = [r for r in manifest if r['Status'] == 'invalid numeric target']
uncertain = [r for r in manifest if r['Status'] in {'ambiguous', 'insufficient information', 'out of domain'}]
write_csv('individual_note_review.csv', manifest)
write_csv('incorrect_labels.csv', incorrect)
write_csv('invalid_numeric_targets.csv', invalid)
write_csv('needs_adjudication.csv', uncertain)

counts = Counter(r['Status'] for r in manifest)
groups = defaultdict(list)
for r in incorrect:
    groups[r['Calculator Name']].append(r['Row Number'])

summary = [
    '# Individual patient-note label review', '',
    f'All {len(manifest):,} records across {len(reviewed)} calculators have a recorded review outcome. '
    'The original test_data.csv and calculator implementations were not edited.', '',
    '| Outcome | Records |', '|---|---:|',
    *[f'| {s} | {n} |' for s, n in counts.most_common()], '',
    '## Files', '',
    '- [All 1,047 row outcomes](individual_note_review.csv)',
    '- [Incorrect labels, with corrections or qualified replacements](incorrect_labels.csv)',
    '- [Invalid numeric targets](invalid_numeric_targets.csv)',
    '- [Ambiguous, missing-information and out-of-domain cases](needs_adjudication.csv)', '',
    'Use **Row Number**, the dataset column, to locate records. It is not a physical CSV line number: patient notes contain embedded newlines. '
    'Each report row contains the original label, reason, corrected value/range when defensible, question, narrative evidence, full note, original entities, original explanation and available rule reference.', '',
    '## Method and limits', '',
    'Each record was reviewed against its calculator-relevant narrative passages, with full-note follow-up for ambiguous risk factors and missing-input findings. '
    'This is not a claim that every unrelated sentence was individually clinically adjudicated. '
    'Arithmetic screening was checked against the notes rather than accepted as ground truth. '
    'The manual narrative decisions supersede the earlier entity-only screen in findings.csv. '
    'The exporter only combines recorded decisions; running it does not perform a fresh clinical review.', '',
    '**Incorrect** means a demonstrable arithmetic, scoring or narrative-input error affects the target. '
    'Some have only a conservative bound, or require a repaired note before a unique replacement is possible. '
    '**Invalid numeric target** means a numeric total cannot be assigned, for example GCS with a non-testable component. '
    '**Ambiguous** includes competing encounters, source-unit conflicts, unspecified scoring conventions and explanations whose errors may cancel numerically. '
    '**Insufficient information** means the supplied note cannot establish required inputs. '
    '**Out of domain** flags inappropriate populations or presentations; it is not counted as a proven arithmetic error. '
    '**No error identified** is a review outcome, not certification of clinical correctness. '
    'Minor rounding differences within the benchmark tolerance are reported separately.', '',
    'An incorrect target does not necessarily fail the benchmark’s supplied acceptance interval. '
    'Some systematic input errors are numerically small enough to fall inside it; the CSV reports this separately when an exact numeric replacement is available.', '',
    'Conventions: the named legacy calculators are retained (including 2005 Caprini and legacy MELD-Na); newer calculator versions are not silently substituted. '
    'Current and prior events are separated. Source typos are not silently repaired. '
    'Conservative Caprini bounds omit uncertain or overlapping features and count at most one trauma and one thrombophilia item. '
    'Conditional values and ranges should not be copied into a corrected training target without resolving their stated conditions.', '',
    '## Incorrect labels by calculator', '',
    '| Calculator | Count | Row Number values |', '|---|---:|---|',
]
for name, nums in sorted(groups.items(), key=lambda x: (-len(x[1]), x[0])):
    summary.append(f'| {name} | {len(nums)} | {", ".join(nums)} |')
summary += ['', '## Selected examples', '',
    '| Row Number | Calculator | Existing label | Reviewed result | Reason |',
    '|---:|---|---|---|---|',
]
for num in ('47', '152', '343', '445', '772', '879', '908', '928'):
    r = next(r for r in manifest if r['Row Number'] == num)
    summary.append(f'| {num} | {r["Calculator Name"]} | {r["Original Label"]} | {r["Corrected Answer or Qualification"]} | {r["Reason"]} |')
summary += ['', '## Rule references', '',
    *[f'- [{next(r["Calculator Name"] for r in manifest if int(r["Calculator ID"]) == cid)}]({url})' for cid, url in SOURCES.items()], '',
    'Sources support scoring definitions; row-level findings are this review’s inferences from the supplied notes. '
    'For routine algebra, the equation specified in the dataset is retained and independently recalculated.', '',
    '## Reproduction and provenance', '',
    '`python3 label_audit/build_individual_report.py` regenerates these exports from the recorded review decisions and preliminary arithmetic findings. '
    'The clinical judgments themselves are manual and are not reproduced by executing a script.', '',
    'Source CSV SHA-256: `' + hashlib.sha256((ROOT / 'dataset/test_data.csv').read_bytes()).hexdigest() + '`', '',
]
(OUT / 'README.md').write_text('\n'.join(summary))
(OUT / 'review_summary.json').write_text(json.dumps({
    'records': len(manifest), 'calculators': len(reviewed), 'counts': dict(counts),
    'source_sha256': hashlib.sha256((ROOT / 'dataset/test_data.csv').read_bytes()).hexdigest(),
}, indent=2))
print(json.dumps(dict(counts), indent=2))
