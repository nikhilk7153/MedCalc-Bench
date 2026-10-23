import json, os, random, html, sys
import pandas as pd

EVAL = "/Users/nikhilkhandekar/Downloads/MedCalc-Bench-version-1.0 2/evaluation"
os.chdir(EVAL)
sys.path.insert(0, EVAL)
import run

FINAL16 = ("../final_files_2026-10-16/test_data_final.csv", "../final_files_2026-10-16/one_shot_finalized_explanation.json")
NOCRIT = ("../final_files_2026-10-20_no_criteria_lists/test_data_final.csv", "../final_files_2026-10-20_no_criteria_lists/one_shot_finalized_explanation.json")
D1022 = ("../dataset/test_data_10_22.csv", "../final_files_2026-10-16/one_shot_10_17_2026.json")


class NoTruncate:
    """Stands in for a tokenizer: keeps text whole (PMC notes are taken already truncated from the output file)."""
    def encode(self, t, add_special_tokens=False): return [t]
    def decode(self, ids): return "".join(ids)


# (label, output file, inputs, time-point placement used by that run)
RUNS = []
for m, label in [("OpenAI_gpt-3.5-turbo-0125", "GPT-3.5 (0125)"), ("OpenAI_gpt-4-0613", "GPT-4 (0613)")]:
    for s in ["direct_answer", "zero_shot", "one_shot"]:
        RUNS.append((label, s, f"outputs/test_data_10_22/{m}_{s}.jsonl", D1022, "after opening instruction", "10/22 run"))
for m, label in [("meta-llama_Meta-Llama-3-70B-Instruct", "Llama3-70B"), ("meta-llama_Meta-Llama-3-8B-Instruct", "Llama3-8B"),
                 ("mistralai_Mixtral-8x7B-Instruct-v0.1", "Mixtral-8x7B"), ("mistralai_Mistral-7B-Instruct-v0.2", "Mistral-7B"),
                 ("epfl-llm_meditron-70b", "Meditron-70B"), ("axiong_PMC_LLaMA_13B", "PMC-LLaMA-13B")]:
    for s in ["direct_answer", "zero_shot", "one_shot"]:
        if s == "one_shot" and label in ("Meditron-70B", "PMC-LLaMA-13B"):
            continue
        if s == "one_shot":
            RUNS.append((label, s, f"outputs/test_data_final_r2_nocrit/{m}_{s}.jsonl", NOCRIT, "end of system prompt", "fixed re-run, no criteria"))
        else:
            RUNS.append((label, s, f"outputs/test_data_final_r2/{m}_{s}.jsonl", FINAL16, "end of system prompt", "fixed re-run"))

cache = {}
examples = []
for label, style, path, (data, one_shot), placement, run_name in RUNS:
    if not os.path.exists(path):
        continue
    if data not in cache:
        cache[data] = pd.read_csv(data).set_index("Row Number", drop=False)
    df = cache[data]
    one_shot_json = json.load(open(one_shot))
    rows = [json.loads(l) for l in open(path)]
    rng = random.Random(f"{label}-{style}")
    correct = [r for r in rows if r["Result"] == "Correct"]
    wrong = [r for r in rows if r["Result"] != "Correct"]
    picks = rng.sample(correct, min(2, len(correct))) + rng.sample(wrong, min(4 - min(2, len(correct)), len(wrong)))
    for r in sorted(picks, key=lambda r: r["Row Number"]):
        row = df.loc[r["Row Number"]].copy()
        row["Patient Note"] = r["Patient Note"]  # as the run used it (PMC: already truncated to 256 tokens)
        time_point = row.get("Time Point Instruction")
        model_id = "OpenAI/x" if label.startswith("GPT") else {"Meditron-70B": "epfl-llm/meditron-70b", "PMC-LLaMA-13B": "x/pmc_llama"}.get(label, "x/other")
        if placement == "end of system prompt":
            row["Time Point Instruction"] = None
            _, messages = run.build_messages(row, style, model_id, NoTruncate(), one_shot_json)
            if isinstance(time_point, str) and time_point.strip():
                messages[0]["content"] += " " + time_point.strip()
        else:
            _, messages = run.build_messages(row, style, model_id, NoTruncate(), one_shot_json)
        examples.append({
            "model": label, "style": style, "run": run_name, "placement": placement,
            "row": int(r["Row Number"]), "calc": r["Calculator Name"], "category": r["Category"],
            "system": messages[0]["content"], "user": messages[1]["content"],
            "time_point": time_point.strip() if isinstance(time_point, str) else "",
            "output": r["LLM Raw Output"], "answer": str(r["LLM Answer"]), "truth": str(r["Ground Truth Answer"]),
            "lower": str(row["Lower Limit"]), "upper": str(row["Upper Limit"]), "result": r["Result"],
            "version": r.get("LLM Model Version", ""), "finish": r.get("Finish Reason", ""),
        })

print(len(examples), "examples from", len({(e['model'], e['style']) for e in examples}), "runs")
json.dump(examples, open(os.path.join(os.path.dirname(__file__), "samples.json"), "w"))
