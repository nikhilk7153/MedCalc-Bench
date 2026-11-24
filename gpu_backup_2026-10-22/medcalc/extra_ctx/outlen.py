import sys, json, copy, numpy as np, pandas as pd
sys.path.insert(0, "/workspace/medcalc/evaluation"); import run
O = "/workspace/medcalc/evaluation/outputs/test_data_10_22/"
for model, pre, L in [("axiong/PMC_LLaMA_13B", "axiong_PMC_LLaMA_13B", 2048), ("epfl-llm/meditron-70b", "epfl-llm_meditron-70b", 4096)]:
    tok = run.load_tokenizer(model)
    for style in ["direct_answer", "zero_shot", "one_shot"]:
        rows = [json.loads(l) for l in open(f"{O}{pre}_{style}.jsonl")]
        n = np.array([len(tok.encode(r["LLM Raw Output"] or "", add_special_tokens=False)) for r in rows])
        fr = {}
        for r in rows: fr[r.get("Finish Reason")] = fr.get(r.get("Finish Reason"), 0) + 1
        print(f"{model.split('/')[1]:14s} {style:13s} answer tokens: median {int(np.median(n))}, 95th pct {int(np.percentile(n,95))}, max {n.max()} | finish reasons {fr}")
