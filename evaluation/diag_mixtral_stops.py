'''
Diagnoses the Mixtral rows whose vLLM output stopped mid-sentence: regenerates them with a given tensor-parallel size
and BOS handling and records vLLM's finish/stop reason. Writes outputs/<tag>/diag/mixtral_tp<tp>_<bos>.jsonl.
'''

import os
import re
import json
import argparse
import pandas as pd

from run import build_messages, load_tokenizer, max_length_for, score

MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"

if __name__ == "__main__":
    from vllm import LLM, SamplingParams

    parser = argparse.ArgumentParser()
    parser.add_argument('--tp', type=int, required=True)
    parser.add_argument('--single_bos', action='store_true', help='Tokenize the templated prompt without adding another BOS')
    parser.add_argument('--eager', action='store_true', help='No torch.compile/CUDA graphs and no custom all-reduce')
    parser.add_argument('--tag', type=str, default="test_data_final")
    args = parser.parse_args()

    df = pd.read_csv("../final_files_2026-10-16/test_data_final.csv").set_index("Row Number", drop=False)
    with open("../final_files_2026-10-16/one_shot_finalized_explanation.json") as file:
        one_shot_json = json.load(file)
    tokenizer = load_tokenizer(MODEL)
    max_length = max_length_for(MODEL)

    jobs = []
    for prompt_style in ["zero_shot", "one_shot"]:
        for r in map(json.loads, open(f"outputs/{args.tag}/{MODEL.replace('/', '_')}_{prompt_style}.jsonl")):
            if re.search(r'[Aa]nswer"', r["LLM Raw Output"]):
                continue
            row = df.loc[r["Row Number"]]
            patient_note, messages = build_messages(row, prompt_style, MODEL, tokenizer, one_shot_json)
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            ids = tokenizer.encode(prompt, add_special_tokens=not args.single_bos)
            jobs.append({"prompt_style": prompt_style, "row": row, "patient_note": patient_note, "ids": ids,
                         "old": r["LLM Raw Output"], "max_tokens": min(4096, max_length - len(ids))})
    print(f"{len(jobs)} rows without an answer key")

    llm = LLM(model=MODEL, tensor_parallel_size=args.tp, max_model_len=max_length, dtype="bfloat16",
              gpu_memory_utilization=0.9, generation_config="vllm",
              **({"enforce_eager": True, "disable_custom_all_reduce": True, "compilation_config": {"mode": 0}} if args.eager else {}))
    sampling = [SamplingParams(temperature=0.0, max_tokens=j["max_tokens"], stop_token_ids=[tokenizer.eos_token_id]) for j in jobs]
    generations = llm.generate([{"prompt_token_ids": j["ids"]} for j in jobs], sampling)

    os.makedirs(os.path.join("outputs", args.tag, "diag"), exist_ok=True)
    out_path = os.path.join("outputs", args.tag, "diag", f"mixtral_tp{args.tp}_{'single' if args.single_bos else 'double'}_bos{'_eager' if args.eager else ''}.jsonl")
    with open(out_path, "w") as f:
        for j, g in zip(jobs, generations):
            o = g.outputs[0]
            answer = re.sub(r"\s+", " ", o.text)
            result = score(j["row"], j["patient_note"], answer, j["prompt_style"])["Result"]
            f.write(json.dumps({"Row Number": int(j["row"]["Row Number"]), "prompt_style": j["prompt_style"],
                                "finish_reason": o.finish_reason, "stop_reason": o.stop_reason,
                                "n_tokens": len(o.token_ids), "last_token_ids": list(o.token_ids[-3:]),
                                "has_answer": bool(re.search(r'[Aa]nswer"', answer)), "Result": result,
                                "same_as_old": answer.strip() == j["old"].strip(), "text": answer}) + "\n")
    rows = [json.loads(l) for l in open(out_path)]
    print(f"tp={args.tp} bos={'single' if args.single_bos else 'double'} eager={args.eager}: has answer {sum(r['has_answer'] for r in rows)}/{len(rows)}, "
          f"correct {sum(r['Result'] == 'Correct' for r in rows)}, identical to old {sum(r['same_as_old'] for r in rows)}, "
          f"finish reasons {dict(pd.Series([r['finish_reason'] for r in rows]).value_counts())}")
