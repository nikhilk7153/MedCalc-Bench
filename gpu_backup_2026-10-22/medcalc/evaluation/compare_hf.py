'''
Re-generates a random sample of rows with the original HF pipeline (llm_inference.LLMInference) and compares
them with the vLLM outputs of run.py. Writes outputs/<tag>/hf_compare/<model>_shard<k>.jsonl.
'''

import os
import time
import json
import random
import argparse
import pandas as pd

from llm_inference import LLMInference
from run import build_messages, score


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Compare the HF pipeline with vLLM on a sample')
    parser.add_argument('--model', type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument('--prompts', type=str, nargs='+', default=["direct_answer", "zero_shot", "one_shot"])
    parser.add_argument('--data', type=str, default="../final_files_2026-10-16/test_data_final.csv")
    parser.add_argument('--one_shot', type=str, default="../final_files_2026-10-16/one_shot_finalized_explanation.json")
    parser.add_argument('--tag', type=str, default="test_data_final")
    parser.add_argument('--n', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--num_shards', type=int, default=1)
    parser.add_argument('--cache_dir', type=str, default=os.path.join(os.getenv("HF_HOME", "~/.cache/huggingface"), "hub"))
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    with open(args.one_shot, "r") as file:
        one_shot_json = json.load(file)
    sample = sorted(random.Random(args.seed).sample(range(len(df)), args.n))
    jobs = [(p, i) for p in args.prompts for i in sample][args.shard::args.num_shards]

    llm = LLMInference(llm_name=args.model, cache_dir=args.cache_dir)
    # Newer transformers pipelines default to max_new_tokens=256 when the model's generation_config sets no limit,
    # which overrides llm_inference.py's max_length; transformers 4.41.2 (environment.yml) honoured max_length.
    llm.model.generation_config.max_new_tokens = None

    out_dir = os.path.join("outputs", args.tag, "hf_compare")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.model.replace('/', '_')}_shard{args.shard}.jsonl")
    done = set()
    if os.path.exists(out_path):
        done = {(r["prompt_style"], r["index"]) for r in map(json.loads, open(out_path))}

    for prompt_style, index in jobs:
        if (prompt_style, index) in done:
            continue
        row = df.iloc[index]
        patient_note, messages = build_messages(row, prompt_style, args.model, llm.tokenizer, one_shot_json)
        prompt = llm.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        start = time.time()
        full = llm.answer(messages)  # the HF pipeline returns prompt + completion, whitespace-collapsed
        seconds = time.time() - start
        outputs = score(row, patient_note, full, prompt_style)  # scored like run.py: on the full text
        outputs["prompt_style"] = prompt_style
        outputs["index"] = index
        outputs["Prompt"] = prompt
        outputs["Seconds"] = seconds
        with open(out_path, "a") as f:
            f.write(json.dumps(outputs) + "\n")
