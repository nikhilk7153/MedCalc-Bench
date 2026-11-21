'''
Summarises compare_hf.py: per prompt style, how often the HF pipeline and vLLM generate the same text and
the same Correct/Incorrect result, and the accuracy of each on the sample.
'''

import re
import sys
import glob
import json
import pandas as pd

from run import score

tag = "test_data_final"
model = sys.argv[1] if len(sys.argv) > 1 else "meta-llama/Meta-Llama-3-8B-Instruct"
name = model.replace('/', '_')
df = pd.read_csv("../final_files_2026-10-16/test_data_final.csv")

hf = [json.loads(l) for f in glob.glob(f"outputs/{tag}/hf_compare/{name}_shard*.jsonl") for l in open(f)]
for prompt_style in ["direct_answer", "zero_shot", "one_shot"]:
    vllm = {r["Row Number"]: r for r in map(json.loads, open(f"outputs/{tag}/{name}_{prompt_style}.jsonl"))}
    rows = [r for r in hf if r["prompt_style"] == prompt_style]
    same_text = same_result = hf_correct = hf_completion_correct = vllm_correct = 0
    for r in rows:
        v = vllm[r["Row Number"]]
        prompt = re.sub(r"\s+", " ", r["Prompt"])
        full = r["LLM Raw Output"]
        completion = full[len(prompt):] if full.startswith(prompt) else full
        same_text += completion.strip() == v["LLM Raw Output"].strip()
        same_result += r["Result"] == v["Result"]
        hf_correct += r["Result"] == "Correct"
        hf_completion_correct += score(df.iloc[r["index"]], r["Patient Note"], completion, prompt_style)["Result"] == "Correct"
        vllm_correct += v["Result"] == "Correct"
    n = len(rows)
    if n:
        print(f"{prompt_style:14s} n={n:3d}  identical text {same_text/n:6.1%}  same result {same_result/n:6.1%}  "
              f"accuracy: HF {hf_correct/n:6.1%} (completion only {hf_completion_correct/n:6.1%})  vLLM {vllm_correct/n:6.1%}")
