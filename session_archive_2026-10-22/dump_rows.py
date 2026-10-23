import json,pandas as pd,sys
df=pd.read_csv('/Users/nikhilkhandekar/Downloads/MedCalc-Bench-version-1.0 2/dataset/test_data_10_22.csv')
d=json.load(open('audit_scores_1.json'))
rows=[int(a) for a in sys.argv[1:]]
for x in d:
    if x['Row Number'] not in rows: continue
    r=df[df['Row Number']==x['Row Number']].iloc[0]
    print('='*100);print('ROW',x['Row Number'],r['Calculator Name'],'GT',r['Ground Truth Answer'],'consensus',x['Consensus wrong answer from strong models'])
    print('TPI:',r['Time Point Instruction'])
    print('NOTE:',r['Patient Note'])
    print('ENT:',r['Relevant Entities'])
    for k,v in x['model reasoning samples'].items():
        print('--',k,v['answer'],':',v['reasoning'][:1200].replace('\n',' '))
