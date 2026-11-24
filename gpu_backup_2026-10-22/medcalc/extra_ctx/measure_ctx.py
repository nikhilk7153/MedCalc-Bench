import sys, json, copy, numpy as np, pandas as pd
sys.path.insert(0, "/workspace/medcalc/evaluation"); import run
df = pd.read_csv("/workspace/medcalc/extra_ctx/test_data_10_22_corrected.csv", dtype={"Ground Truth Answer": str, "Lower Limit": str, "Upper Limit": str})
os_ = json.load(open("/workspace/medcalc/extra_ctx/one_shot_10_17_2026_corrected.json"))
orig_trunc = run.truncate
for model in ["axiong/PMC_LLaMA_13B", "epfl-llm/meditron-70b"]:
    tok = run.load_tokenizer(model); L = run.max_length_for(model)
    note_len = np.array([len(tok.encode(n, add_special_tokens=False)) for n in df["Patient Note"]])
    print(f"== {model} (context {L}) | test-note tokens: median {int(np.median(note_len))}, 90th pct {int(np.percentile(note_len,90))}, max {note_len.max()} | notes > 256 tokens: {(note_len>256).sum()}")
    for style in ["direct_answer", "zero_shot", "one_shot"]:
        res = {}
        for mode in ["current", "none"]:
            run.truncate = orig_trunc if mode == "current" else (lambda t, text, n: text)
            lens = []
            for _, row in df.iterrows():
                _, msgs = run.build_messages(row, style, model, tok, copy.deepcopy(os_))
                lens.append(len(tok.encode(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True), add_special_tokens=False)))
            lens = np.array(lens); res[mode] = lens
        run.truncate = orig_trunc
        c, n = res["current"], res["none"]
        print(f"   {style:13s} current: median {int(np.median(c))}, max {c.max()}, over context {(c>=L).sum()}, <300 tokens left {(L-c<300).sum()} | "
              f"no truncation: median {int(np.median(n))}, max {n.max()}, over context {(n>=L).sum()}, <300 tokens left {(L-n<300).sum()}")
