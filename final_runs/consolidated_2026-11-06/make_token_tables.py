"""Token-usage tables (paper Appendix B.3): input and output tokens per model for direct answer, zero-shot CoT and
one-shot CoT, over the consolidated outputs in outputs/ (run consolidate.py first).

Input tokens: each prompt is rebuilt with the as-run run.py (run_py_as_run_5f7e8a9d.py, = git HEAD evaluation/run.py) from
the test row and one-shot file of the run the output came from, then counted with the model's own tokenizer:
- open models: the templated prompt the model received, tokenized as run.py did (chat template, no added special tokens);
- GPT-3.5 / GPT-4: tiktoken cl100k_base over the chat messages, with OpenAI's per-message overhead (3 per message + 3).
Output tokens: open models, the `Output Tokens` vLLM recorded; GPT, tiktoken cl100k_base over the raw output.
Check: for main-run rows the rebuilt prompts must equal the ones in final_runs/test_data_10_22_with_prompts.

Needs transformers, tiktoken, sentencepiece, protobuf and jinja2 (no torch), and Hugging Face access to the gated
Llama 3 and Mistral tokenizers. Writes token_counts.csv and tables/token_tables.tex."""
import importlib.util, json, os, re, sys
import pandas as pd
import tiktoken

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))  # repository root
EVAL = f"{ROOT}/evaluation"
sys.path.insert(0, EVAL)  # run.py imports evaluate and table_stats
spec = importlib.util.spec_from_file_location("run_as_run", f"{HERE}/run_py_as_run_5f7e8a9d.py")
run = importlib.util.module_from_spec(spec); spec.loader.exec_module(run)

MODELS = [  # table name, model name for run.py, output file prefix
    ("PMC-LLaMA", "axiong/PMC_LLaMA_13B", "axiong_PMC_LLaMA_13B"),
    (r"\textsc{MediTron}", "epfl-llm/meditron-70b", "epfl-llm_meditron-70b"),
    ("Mistral", "mistralai/Mistral-7B-Instruct-v0.2", "mistralai_Mistral-7B-Instruct-v0.2"),
    ("Mixtral", "mistralai/Mixtral-8x7B-Instruct-v0.1", "mistralai_Mixtral-8x7B-Instruct-v0.1"),
    ("Llama 3-8B", "meta-llama/Meta-Llama-3-8B-Instruct", "meta-llama_Meta-Llama-3-8B-Instruct"),
    ("Llama 3-70B", "meta-llama/Meta-Llama-3-70B-Instruct", "meta-llama_Meta-Llama-3-70B-Instruct"),
    ("GPT-3.5", "OpenAI/gpt-3.5-turbo-0125", "OpenAI_gpt-3.5-turbo-0125"),
    ("GPT-4", "OpenAI/gpt-4-0613", "OpenAI_gpt-4"),
]
STYLES = [("direct_answer", "Number of tokens used for zero-shot direct prompting", "tab:tokens_direct"),
          ("zero_shot", "Number of tokens used for zero-shot CoT prompting", "tab:tokens_zero"),
          ("one_shot", "Number of tokens used for one-shot CoT prompting", "tab:tokens_one")]
# test set and one-shot file each source run used (md5s in the runs' manifests)
AS_RUN_TEST = f"{ROOT}/extra_runs/gpt-4o-2024-08-06_benchmark_2026-10-24/input/test_data_10_22_as_run.csv"
SOURCES = {
    "final_runs/test_data_10_22": (AS_RUN_TEST, f"{ROOT}/gpu_backup_2026-10-22/medcalc/final_files_2026-10-16/one_shot_10_17_2026.json"),
    "corrected_rows_ckdepi_rerun_2026-10-22": (AS_RUN_TEST, f"{ROOT}/extra_runs/corrected_rows_ckdepi_rerun_2026-10-22/input/one_shot_10_17_2026_corrected.json"),
    "mdrd_sodium_wording_rerun_2026-10-22": (f"{ROOT}/dataset/test_data_10_22.csv", f"{ROOT}/extra_runs/mdrd_sodium_wording_rerun_2026-10-22/input/one_shot_10_17_2026_corrected.json"),
}
tests = {p: pd.read_csv(p, dtype=str, keep_default_na=False).set_index("Row Number") for p in {t for t, _ in SOURCES.values()}}
shots = {p: json.load(open(p)) for p in {s for _, s in SOURCES.values()}}
enc = tiktoken.get_encoding("cl100k_base")


def openai_prompt_tokens(messages):
    return sum(3 + len(enc.encode(m["role"])) + len(enc.encode(m["content"])) for m in messages) + 3


def main_run_prompts(prefix, style, key):
    path = f"{ROOT}/final_runs/test_data_10_22_with_prompts/outputs/{prefix}_{style}.jsonl"
    return {str(x["Row Number"]): x[key] for x in map(json.loads, open(path))}


os.chdir(EVAL)  # run.load_tokenizer reads the chat templates from ../templates
counts, mismatches = [], []
for label, model, prefix in MODELS:
    openai = run.is_openai(model)
    tokenizer = None if openai else run.load_tokenizer(model)
    for style, _, _ in STYLES:
        check = main_run_prompts(prefix, style, "Prompt Messages (reconstructed)" if openai else "Prompt (reconstructed)")
        n_in = n_out = 0
        for x in map(json.loads, open(f"{HERE}/outputs/{prefix}_{style}.jsonl")):
            r = str(x["Row Number"])
            test, shot = SOURCES[x["Source Run"]]
            row = tests[test].loc[r].to_dict() | {"Row Number": r}
            _, messages = run.build_messages(row, style, model, tokenizer, shots[shot])
            if openai:
                n_in += openai_prompt_tokens(messages)
                n_out += len(enc.encode(x["LLM Raw Output"]))
                rebuilt = messages
            else:
                rebuilt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                n_in += len(tokenizer.encode(rebuilt, add_special_tokens=False))
                n_out += int(x["Output Tokens"])
            if x["Source Run"] == "final_runs/test_data_10_22":
                expected = check[r]
                if isinstance(expected, str) and openai:
                    expected = json.loads(expected)
                if rebuilt != expected:
                    mismatches.append((label, style, r))
        counts.append({"Model": label, "Style": style, "Input": n_in, "Output": n_out})
        print(label, style, n_in, n_out, flush=True)

print("prompt mismatches against the main-run reconstructions:", len(mismatches), mismatches[:10])
d = pd.DataFrame(counts); d.to_csv(f"{HERE}/token_counts.csv", index=False)


def fmt(n):
    if n >= 1_000_000: return f"{n / 1e6:.1f}M"
    if n >= 1_000: return f"{n / 1e3:.1f}k"
    return str(n)


tex = []
for style, caption, tab in STYLES:
    s = d[d.Style == style].set_index("Model")
    tex += [r"\begin{table}[h!]", r"    \centering", rf"    \caption{{{caption}}}",
            r"    \resizebox{\textwidth}{!}{ % Adjust table to fit within the text width", r"    \begin{tabular}{ccccccccc}", r"    \toprule",
            "         & " + " & ".join(m for m, _, _ in MODELS) + r" \\", r"    \midrule"]
    for kind in ("Input", "Output"):
        tex.append(f"        {kind} & " + " & ".join(fmt(int(s.at[m, kind])) for m, _, _ in MODELS) + r" \\")
    tex += [r"    \bottomrule", r"    \end{tabular}", r"    }", rf"    \label{{{tab}}}", r"\end{table}", ""]
open(f"{HERE}/tables/token_tables.tex", "w").write("\n".join(tex))
print("\n".join(tex))
