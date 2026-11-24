import json, re, glob, collections
import transformers
from transformers import AutoTokenizer

print("transformers", transformers.__version__)
for model in ["meta-llama/Meta-Llama-3-8B-Instruct", "mistralai/Mixtral-8x7B-Instruct-v0.1", "meta-llama/Meta-Llama-3-70B-Instruct"]:
    tok = AutoTokenizer.from_pretrained(model)
    rows = [json.loads(l) for f in glob.glob(f"outputs/test_data_final/hf_compare/{model.replace('/', '_')}_shard*.jsonl") for l in open(f)]
    lens = collections.Counter()
    by_style = collections.defaultdict(list)
    for r in rows:
        prompt = re.sub(r"\s+", " ", r["Prompt"])
        full = r["LLM Raw Output"]
        comp = full[len(prompt):] if full.startswith(prompt) else full
        n = len(tok.encode(comp, add_special_tokens=False))
        by_style[r["prompt_style"]].append(n)
    for style, ns in by_style.items():
        print(f"{model} {style}: n={len(ns)} max={max(ns)} at >=250 tokens: {sum(n >= 250 for n in ns)}  top lengths: {collections.Counter(ns).most_common(3)}")
