import json, re
import pandas as pd
from run_vllm import score

df = pd.read_csv("../final_files_2026-10-16/test_data_final.csv")
m = "mistralai_Mixtral-8x7B-Instruct-v0.1"
v = {r["Row Number"]: r for r in map(json.loads, open(f"outputs/test_data_final/{m}_one_shot.jsonl"))}
hf = [json.loads(l) for l in open(f"outputs/test_data_final/hf_compare/{m}_shard0.jsonl")]
for r in hf:
    w = v[r["Row Number"]]
    if r["Result"] == w["Result"]:
        continue
    prompt = re.sub(r"\s+", " ", r["Prompt"])
    full = r["LLM Raw Output"]
    comp = full[len(prompt):] if full.startswith(prompt) else None
    comp_res = score(df.iloc[r["index"]], r["Patient Note"], comp, "one_shot")["Result"] if comp is not None else "prompt-not-prefix"
    print(f"row {r['Row Number']} calc {r['Calculator ID']} GT {r['Ground Truth Answer']} | HF {r['Result']} ans={str(r['LLM Answer'])[:30]} "
          f"(completion-only: {comp_res}) | vLLM {w['Result']} ans={str(w['LLM Answer'])[:30]}")
    print("   HF tail:  ", (comp or full)[-160:])
    print("   vLLM tail:", w["LLM Raw Output"][-160:])
