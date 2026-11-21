'''
Counts vLLM outputs that stopped because they used their whole token budget (max_tokens = min(4096, context - prompt))
instead of ending on their own, per model and prompt style.
'''

import json
import pandas as pd

from run import build_messages, load_tokenizer, max_length_for

MODELS = ["meta-llama/Meta-Llama-3-8B-Instruct", "mistralai/Mistral-7B-Instruct-v0.2", "axiong/PMC_LLaMA_13B",
          "mistralai/Mixtral-8x7B-Instruct-v0.1", "meta-llama/Meta-Llama-3-70B-Instruct", "epfl-llm/meditron-70b"]

df = pd.read_csv("../final_files_2026-10-16/test_data_final.csv").set_index("Row Number", drop=False)
with open("../final_files_2026-10-16/one_shot_finalized_explanation.json") as file:
    one_shot_json = json.load(file)

for model in MODELS:
    tokenizer = load_tokenizer(model)
    max_length = max_length_for(model)
    for prompt_style in ["direct_answer", "zero_shot", "one_shot"]:
        hit = correct_hit = 0
        rows = [json.loads(l) for l in open(f"outputs/test_data_final/{model.replace('/', '_')}_{prompt_style}.jsonl")]
        for r in rows:
            row = df.loc[r["Row Number"]]
            _, messages = build_messages(row, prompt_style, model, tokenizer, one_shot_json)
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            budget = min(4096, max_length - len(tokenizer.encode(prompt, add_special_tokens=True)))
            used = len(tokenizer.encode(r["LLM Raw Output"], add_special_tokens=False))
            if used >= budget - 2:
                hit += 1
                correct_hit += r["Result"] == "Correct"
        print(f"{model:40s} {prompt_style:14s} hit budget: {hit:4d} / {len(rows)} (of which Correct: {correct_hit})", flush=True)
