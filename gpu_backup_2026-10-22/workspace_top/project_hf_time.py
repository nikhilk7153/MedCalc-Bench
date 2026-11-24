import json, re, glob
import numpy as np
from transformers import AutoTokenizer

OUT = "outputs/test_data_final"
STYLES = ["direct_answer", "zero_shot", "one_shot"]


def name(m):
    return m.replace("/", "_")


def fit(model):
    """Fit seconds = a + b * completion_tokens on the HF rows measured so far."""
    tok = AutoTokenizer.from_pretrained(model)
    rows = [json.loads(l) for f in glob.glob(f"{OUT}/hf_compare/{name(model)}_shard*.jsonl") for l in open(f)]
    rows = [r for r in rows if "Seconds" in r]
    x, y = [], []
    for r in rows:
        prompt = re.sub(r"\s+", " ", r["Prompt"])
        full = r["LLM Raw Output"]
        comp = full[len(prompt):] if full.startswith(prompt) else full
        x.append(len(tok.encode(comp, add_special_tokens=False)))
        y.append(r["Seconds"])
    b, a = np.polyfit(x, y, 1)
    return tok, a, b, len(rows)


def output_tokens(model):
    tok = AutoTokenizer.from_pretrained(model)
    lens = [len(tok.encode(json.loads(l)["LLM Raw Output"], add_special_tokens=False))
            for p in STYLES for l in open(f"{OUT}/{name(model)}_{p}.jsonl")]
    return len(lens), sum(lens)


for model in ["mistralai/Mixtral-8x7B-Instruct-v0.1", "meta-llama/Meta-Llama-3-70B-Instruct"]:
    tok, a, b, n = fit(model)
    n_rows, toks = output_tokens(model)
    hours = (n_rows * a + toks * b) / 3600
    print(f"{model}: fitted on {n} rows: {a:.1f} s overhead + {1 / b:.1f} tok/s; full run {n_rows} answers, {toks} output tokens -> {hours:.1f} h")

for model in ["meta-llama/Meta-Llama-3-8B-Instruct", "mistralai/Mistral-7B-Instruct-v0.2", "axiong/PMC_LLaMA_13B", "epfl-llm/meditron-70b"]:
    n_rows, toks = output_tokens(model)
    print(f"{model}: {n_rows} answers, {toks} output tokens")

for m in ["mistralai_Mixtral-8x7B-Instruct-v0.1", "meta-llama_Meta-Llama-3-70B-Instruct"]:
    v = {r["Row Number"]: r for r in map(json.loads, open(f"{OUT}/{m}_one_shot.jsonl"))}
    hf = [json.loads(l) for l in open(f"{OUT}/hf_compare/{m}_shard0.jsonl")]
    n = len(hf)
    same = sum(r["Result"] == v[r["Row Number"]]["Result"] for r in hf)
    hf_acc = sum(r["Result"] == "Correct" for r in hf)
    v_acc = sum(v[r["Row Number"]]["Result"] == "Correct" for r in hf)
    print(f"{m} one-shot n={n}: same result {same}/{n}, correct HF {hf_acc} vs vLLM {v_acc}")
