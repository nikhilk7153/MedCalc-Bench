"""Display individual records for narrative review, retaining all relevant matches.
This is a reading aid, not a label adjudicator. --full prints the entire note.
"""
import csv, re, sys, ast
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
ROWS = list(csv.DictReader((ROOT/'dataset/test_data.csv').open()))
TERMS = {
2:r'creatinin|weigh|\bkg\b|\bcm\b|\bbmi\b|\btall\b',
3:r'creatinin',4:r'hypertens|\bbp\b|blood pressure|diabet|stroke|ischemi|ischaemi|infarct|heart failure|embol|vascul|female|woman',
5:r'blood pressure|\bbp\b|\d\s*/\s*\d',6:r'weigh|\bkg\b|\bcm\b|\bbmi\b|\btall\b|height',
7:r'calcium|albumin|\bCa\b|\bAlb\b',8:r'.*',9:r'creatinin|black|african|race',10:r'height|\bcm\b|\btall\b',
19:r'platelet|aminotransferase|transaminase|\bALT\b|\bAST\b',22:r'weigh|\bkg\b|\blb|\bgram|\bbody mass',
26:r'sodium|glucose|\bNa\b',30:r'sodium|glucose|\bBUN\b|urea|\bNa\b',31:r'glucose|insulin|fasting',
38:r'weigh|\bkg\b|\blb|sodium|\bNa\b',39:r'sodium|chloride|bicarbonate|carbon dioxide|\bHCO3|\bNa\b|\bCl\b',
40:r'creatinin|sodium|\bNa\b',44:r'cholesterol|cholestrol|triglyceride|\bHDL\b|\bLDL\b|\bTC\b|\bTG\b',
46:r'cholesterol|cholestrol|\bHDL\b|\bLDL\b|smok|blood pressure|\bbp\b|hypertens|statin|infarct|coronary',
60:r'weigh|\bkg\b|\bcm\b|height|\btall\b',62:r'weigh|\bkg\b|\bcm\b|height|\btall\b',
63:r'sodium|chloride|bicarbonate|carbon dioxide|\bHCO3|\bNa\b|\bCl\b',
64:r'sodium|chloride|bicarbonate|carbon dioxide|\bHCO3|\bNa\b|\bCl\b',
65:r'sodium|chloride|bicarbonate|carbon dioxide|albumin|\bHCO3|\bNa\b|\bCl\b|\bAlb\b',
66:r'sodium|chloride|bicarbonate|carbon dioxide|albumin|\bHCO3|\bNa\b|\bCl\b|\bAlb\b',
67:r'sodium|chloride|bicarbonate|carbon dioxide|albumin|\bHCO3|\bNa\b|\bCl\b|\bAlb\b'}

def excerpts(r):
    note = r['Patient Note']
    term = TERMS.get(int(r['Calculator ID']),r'.*')
    if term == r'.*': return note
    spans = [(0,min(len(note),125))]
    term = term.replace('platelet','platelet|thrombocytopen').replace('glucose','glucose|blood sugar|glycemi|glycaemi').replace('creatinin','creatinin|creatine').replace('weigh','weigh|kilos')
    for m in re.finditer(term,note,re.I):
        spans.append((max(0,m.start()-65),min(len(note),m.end()+95)))
    spans.sort(); merged=[]
    for a,b in spans:
        if merged and a<=merged[-1][1]:merged[-1]=(merged[-1][0],max(b,merged[-1][1]))
        else:merged.append((a,b))
    return ' […] '.join(note[a:b] for a,b in merged)

if __name__=='__main__':
    ids = set(map(int,[a for a in sys.argv[1:] if a.isdigit()]))
    for r in ROWS:
        if int(r['Calculator ID']) not in ids:continue
        print('\nROW',r['Row Number'],'LABEL',r['Ground Truth Answer'],'ENTITIES',r['Relevant Entities'])
        print('NOTE:',r['Patient Note'] if '--full' in sys.argv else excerpts(r))
