import sys, json, pandas as pd
sys.path.insert(0, "/workspace/medcalc/evaluation"); import run
df = pd.read_csv("/workspace/medcalc/dataset/test_data_10_22.csv", dtype=str)
os_ = json.load(open("/workspace/medcalc/final_files_2026-10-16/one_shot_10_17_2026.json"))
row = df[df["Row Number"] == "21"].iloc[0]
for m in ["meta-llama/Meta-Llama-3-8B-Instruct", "meta-llama/Meta-Llama-3-70B-Instruct", "mistralai/Mistral-7B-Instruct-v0.2", "mistralai/Mixtral-8x7B-Instruct-v0.1", "epfl-llm/meditron-70b", "axiong/PMC_LLaMA_13B"]:
    tok = run.load_tokenizer(m)
    _, msgs = run.build_messages(row, "zero_shot", m, tok, os_)
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok.encode(prompt, add_special_tokens=False)
    stops = [tok.eos_token_id] + ([tok.convert_tokens_to_ids("<|eot_id|>")] if "llama-3" in m.lower() else [])
    print(f"== {m.split('/')[1]}: BOS id {tok.bos_token_id} appears {ids.count(tok.bos_token_id)}x (first token is BOS: {ids[0] == tok.bos_token_id}) | prompt {len(ids)} tokens | stop ids {stops} = {[tok.convert_ids_to_tokens(i) for i in stops]}")
    print("   start:", repr(prompt[:150])); print("   end:  ", repr(prompt[-90:]))
