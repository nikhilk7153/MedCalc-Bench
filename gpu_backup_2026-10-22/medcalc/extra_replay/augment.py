import sys, json, copy, os, collections, pandas as pd
sys.path.insert(0, "/workspace/medcalc/evaluation")
import importlib.util
spec = importlib.util.spec_from_file_location("run_final", "/workspace/medcalc/evaluation/run_pre_fit_context.py"); run = importlib.util.module_from_spec(spec); spec.loader.exec_module(run)
df = pd.read_csv("/workspace/medcalc/dataset/test_data_10_22.csv", dtype={"Ground Truth Answer": str, "Lower Limit": str, "Upper Limit": str})
rows = {int(r["Row Number"]): r for _, r in df.iterrows()}
os_ = json.load(open("/workspace/medcalc/final_files_2026-10-16/one_shot_10_17_2026.json"))
SRC, OUT = "/workspace/medcalc/evaluation/outputs/test_data_10_22", "/workspace/medcalc/extra_replay/with_prompts"; os.makedirs(OUT, exist_ok=True)
MODELS = {"meta-llama/Meta-Llama-3-8B-Instruct": "meta-llama_Meta-Llama-3-8B-Instruct", "meta-llama/Meta-Llama-3-70B-Instruct": "meta-llama_Meta-Llama-3-70B-Instruct",
          "mistralai/Mistral-7B-Instruct-v0.2": "mistralai_Mistral-7B-Instruct-v0.2", "mistralai/Mixtral-8x7B-Instruct-v0.1": "mistralai_Mixtral-8x7B-Instruct-v0.1",
          "epfl-llm/meditron-70b": "epfl-llm_meditron-70b", "axiong/PMC_LLaMA_13B": "axiong_PMC_LLaMA_13B"}
for m, pre in MODELS.items():
    tok = run.load_tokenizer(m)
    for s in ("direct_answer", "zero_shot", "one_shot"):
        c = collections.Counter(); out = []
        for l in open(f"{SRC}/{pre}_{s}.jsonl"):
            r = json.loads(l); row = rows[int(r["Row Number"])]
            note, msgs = run.build_messages(row, s, m, tok, copy.deepcopy(os_))
            prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            c["note matches the run"] += note == r["Patient Note"]
            tp = row["Time Point Instruction"].strip() if isinstance(row["Time Point Instruction"], str) else ""
            new = {}
            for k, v in r.items():
                new[k] = v
                if k == "Question": new["Time Point Instruction"] = tp
            new["Prompt Messages (reconstructed)"] = msgs; new["Prompt (reconstructed)"] = prompt
            out.append(new)
        with open(f"{OUT}/{pre}_{s}.jsonl", "w") as f: f.writelines(json.dumps(x) + "\n" for x in out)
        print(f"{pre[:40]:40s} {s:13s} rows {len(out)} | rebuilt note == recorded note {c['note matches the run']}/{len(out)}")
