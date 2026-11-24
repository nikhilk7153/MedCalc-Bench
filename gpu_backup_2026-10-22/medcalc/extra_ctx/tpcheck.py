import sys, json, copy, collections, pandas as pd
sys.path.insert(0, "/workspace/medcalc/evaluation"); import run
TP = "You should use the patient's medical values and health status when they were first admitted to the hospital prior to any treatment."
df = pd.read_csv("/workspace/medcalc/dataset/test_data_10_22.csv", dtype=str)
os_ = json.load(open("/workspace/medcalc/final_files_2026-10-16/one_shot_10_17_2026.json"))
models = ["OpenAI/gpt-4-0613", "meta-llama/Meta-Llama-3-70B-Instruct", "mistralai/Mixtral-8x7B-Instruct-v0.1", "epfl-llm/meditron-70b", "axiong/PMC_LLaMA_13B"]
for m in models:
    tok = None if m.startswith("OpenAI") else run.load_tokenizer(m)
    for s in ("direct_answer", "zero_shot", "one_shot"):
        c = collections.Counter()
        for _, r in df.iterrows():
            has = isinstance(r["Time Point Instruction"], str) and r["Time Point Instruction"].strip() != ""
            _, msgs = run.build_messages(r, s, m, tok, copy.deepcopy(os_))
            text = msgs[0]["content"] + "\n" + msgs[1]["content"]
            n = text.count(TP); q = r["Question"].rstrip()
            right_place = (q + " " + TP) in text
            want = (2 if s == "one_shot" and r["Calculator ID"] != "24" else 1) if has else 0
            if s == "one_shot" and r["Calculator ID"] == "24" and has: want = 1
            if has: c["rows with instruction"] += 1
            c["correct count" if n == want else f"WRONG count (has={has}, n={n}, want={want})"] += 1
            if has and not right_place: c["not directly after the question"] += 1
            if not has and n: c["instruction on a row without it"] += 1
            if TP in msgs[0]["content"] and s != "one_shot": c["instruction in system prompt"] += 1
        print(f"{m.split('/')[1]:28s} {s:13s} {dict(c)}")
