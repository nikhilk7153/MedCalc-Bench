import sys, json, copy, importlib.util, numpy as np, pandas as pd
sys.path.insert(0, "/workspace/medcalc/evaluation")
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
old = load("run_old", "/workspace/medcalc/evaluation/run.py"); new = load("run_new", "/workspace/medcalc/extra_ctx/run_new.py")
df = pd.read_csv("/workspace/medcalc/extra_ctx/test_data_10_22_corrected.csv", dtype=str)
os_ = json.load(open("/workspace/medcalc/extra_ctx/one_shot_10_17_2026_corrected.json"))
styles = ["direct_answer", "zero_shot", "one_shot"]
print("== default path, old vs new")
for model in ["axiong/PMC_LLaMA_13B", "epfl-llm/meditron-70b", "OpenAI/gpt-4-0613"]:
    tok = None if model.startswith("OpenAI") else old.load_tokenizer(model)
    diff = sum(old.build_messages(r, s, model, tok, copy.deepcopy(os_)) != new.build_messages(r, s, model, tok, copy.deepcopy(os_)) for s in styles for _, r in df.iterrows())
    print(f"  {model:24s} prompts that differ: {diff}/{3*len(df)}")
print("== fit_context 512")
for model in ["axiong/PMC_LLaMA_13B", "epfl-llm/meditron-70b"]:
    tok = new.load_tokenizer(model); L = new.max_length_for(model)
    for s in styles:
        full = cut = changed = 0; lens = []; kept = []
        for _, r in df.iterrows():
            note, msgs = new.build_messages(r, s, model, tok, copy.deepcopy(os_), 512)
            _, dm = new.build_messages(r, s, model, tok, copy.deepcopy(os_))
            n = new.prompt_length(tok, msgs); lens.append(n); changed += msgs != dm
            nt = len(tok.encode(r["Patient Note"], add_special_tokens=False)); kt = len(tok.encode(note, add_special_tokens=False))
            full += note == r["Patient Note"]; kept.append(min(kt / nt, 1.0) if nt else 1.0)
        lens = np.array(lens)
        print(f"  {model.split('/')[1]:14s} {s:13s} prompts changed vs default {changed:4d} | notes kept whole {full:4d}/1048 | mean share of note kept {np.mean(kept):.2f} | "
              f"prompt max {lens.max()} | room for answer min {L - lens.max()}")
