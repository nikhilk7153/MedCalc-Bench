import sys, json, copy, pandas as pd
sys.path.insert(0, "/workspace/medcalc/evaluation"); import run
model = "epfl-llm/meditron-70b"; tok = run.load_tokenizer(model)
df = pd.read_csv("/workspace/medcalc/extra_ctx/test_data_10_22_corrected.csv", dtype=str)
os_ = json.load(open("/workspace/medcalc/extra_ctx/one_shot_10_17_2026_corrected.json"))
out = {json.loads(l)["Row Number"]: json.loads(l) for l in open("/workspace/medcalc/evaluation/outputs/test_data_10_22/epfl-llm_meditron-70b_one_shot.jsonl")}
for _, row in df.iterrows():
    _, msgs = run.build_messages(row, "one_shot", model, tok, copy.deepcopy(os_))
    n = len(tok.encode(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True), add_special_tokens=False))
    if 4096 - n < 300:
        r = out[int(row["Row Number"])]
        print(f"row {row['Row Number']} {row['Calculator Name'][:28]:28s} | room for answer {4096-n} tokens | finish {r.get('Finish Reason')} | answer {str(r['LLM Answer'])[:20]!r} | {r['Result']} | output tail: {(r['LLM Raw Output'] or '')[-110:]!r}")
