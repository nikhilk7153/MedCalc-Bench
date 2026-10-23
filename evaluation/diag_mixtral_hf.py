'''
Regenerates the Mixtral rows that stopped mid-sentence under vLLM with the original HF path (llm_inference.LLMInference),
set up as under transformers 4.41.2: the templated prompt is tokenized without an extra BOS and max_length is not
overridden by a default max_new_tokens. Writes outputs/<tag>/diag/mixtral_hf_<tag>_shard<k>.jsonl.
'''

import os
import re
import json
import argparse
import pandas as pd

from llm_inference import LLMInference
from run import build_messages, score

MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--num_shards', type=int, default=1)
    parser.add_argument('--tag', type=str, default="test_data_final", help='Run whose Mixtral rows without an answer are regenerated')
    args = parser.parse_args()

    df = pd.read_csv("../final_files_2026-10-16/test_data_final.csv").set_index("Row Number", drop=False)
    with open("../final_files_2026-10-16/one_shot_finalized_explanation.json") as file:
        one_shot_json = json.load(file)

    targets = []
    for prompt_style in ["zero_shot", "one_shot"]:
        for r in map(json.loads, open(f"outputs/{args.tag}/{MODEL.replace('/', '_')}_{prompt_style}.jsonl")):
            if not re.search(r'[Aa]nswer"', r["LLM Raw Output"]):
                targets.append((prompt_style, r["Row Number"]))
    targets = targets[args.shard::args.num_shards]

    llm = LLMInference(llm_name=MODEL, cache_dir=os.path.join(os.environ["HF_HOME"], "hub"))
    llm.model.generation_config.max_new_tokens = None
    tok = llm.tokenizer

    os.makedirs(os.path.join("outputs", args.tag, "diag"), exist_ok=True)
    out_path = os.path.join("outputs", args.tag, "diag", f"mixtral_hf_{args.tag}_shard{args.shard}.jsonl")
    with open(out_path, "w") as f:
        for prompt_style, row_number in targets:
            row = df.loc[row_number]
            patient_note, messages = build_messages(row, prompt_style, MODEL, tok, one_shot_json)
            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            # same call as LLMInference.generate (non-llama-3 branch), plus add_special_tokens=False as in 4.41.2
            response = llm.model(prompt, do_sample=False, eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id,
                                 max_length=min(llm.max_length, len(tok.encode(prompt, add_special_tokens=True)) + 4096),
                                 truncation=True, add_special_tokens=False, return_full_text=False)
            answer = re.sub(r"\s+", " ", response[0]["generated_text"])
            n_tokens = len(tok.encode(answer, add_special_tokens=False))
            result = score(row, patient_note, answer, prompt_style)["Result"]
            f.write(json.dumps({"Row Number": int(row_number), "prompt_style": prompt_style, "n_tokens": n_tokens,
                                "has_answer": bool(re.search(r'[Aa]nswer"', answer)), "Result": result, "text": answer}) + "\n")
            f.flush()
