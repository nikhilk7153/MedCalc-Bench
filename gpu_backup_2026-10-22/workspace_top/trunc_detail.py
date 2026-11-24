import json, pandas as pd
from run_vllm import build_messages, load_tokenizer, max_length_for
df = pd.read_csv("../final_files_2026-10-16/test_data_final.csv").set_index("Row Number", drop=False)
osj = json.load(open("../final_files_2026-10-16/one_shot_finalized_explanation.json"))
for model, style in [("meta-llama/Meta-Llama-3-8B-Instruct","zero_shot"),("mistralai/Mistral-7B-Instruct-v0.2","direct_answer"),("mistralai/Mistral-7B-Instruct-v0.2","zero_shot"),("axiong/PMC_LLaMA_13B","zero_shot"),("axiong/PMC_LLaMA_13B","one_shot"),("epfl-llm/meditron-70b","direct_answer"),("epfl-llm/meditron-70b","one_shot")]:
    tok = load_tokenizer(model); L = max_length_for(model)
    budgets = []; loops = 0; ex = []
    for r in map(json.loads, open(f"outputs/test_data_final/{model.replace(chr(47), chr(95))}_{style}.jsonl")):
        _, msgs = build_messages(df.loc[r["Row Number"]], style, model, tok, osj)
        p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        budget = min(4096, L - len(tok.encode(p, add_special_tokens=True)))
        ids = tok.encode(r["LLM Raw Output"], add_special_tokens=False)
        if len(ids) < budget - 2: continue
        grams = [tuple(ids[i:i+20]) for i in range(0, len(ids) - 20)]
        uniq = len(set(grams)) / max(1, len(grams))
        budgets.append(budget); loops += uniq < 0.5; ex.append(r["LLM Raw Output"][-120:])
    print(f"{model} {style}: {len(budgets)} hits, budget min/median {min(budgets)}/{sorted(budgets)[len(budgets)//2]}, repetitive (<50% unique 20-grams): {loops}")
    print("   e.g.", ex[0])
