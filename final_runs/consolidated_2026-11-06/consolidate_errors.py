"""Error classifications for the consolidated outputs: for every wrong zero/one-shot answer, take the classification
made on that same output (from the run it came from), and record whether it used the current reference explanation. MEDCALC_VARIANT=astra: see variant.py."""
import json, os, re
import pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root
HERE = os.path.dirname(os.path.abspath(__file__))
import variant
CONS = f"{variant.OUT}/outputs"
EA = {"final_runs/test_data_10_22": f"{ROOT}/evaluation/error_analysis_10_22",
      "corrected_rows_ckdepi_rerun_2026-10-22": f"{ROOT}/extra_runs/corrected_rows_ckdepi_rerun_2026-10-22/error_analysis",
      "mdrd_sodium_wording_rerun_2026-10-22": f"{ROOT}/extra_runs/mdrd_sodium_wording_rerun_2026-10-22/error_analysis",
      "astra_corrections_rerun_2026-10-22": f"{variant.ASTRA_RUN}/error_analysis"}
MODELS = [("PMC-LLaMA", "axiong_PMC_LLaMA_13B", "pmc-llama-13b"), ("MediTron", "epfl-llm_meditron-70b", "meditron-70b"),
          ("Mistral", "mistralai_Mistral-7B-Instruct-v0.2", "mistral-7b"), ("Mixtral", "mistralai_Mixtral-8x7B-Instruct-v0.1", "mixtral-8x7b"),
          ("Llama 3 8B", "meta-llama_Meta-Llama-3-8B-Instruct", "llama-3-8b"), ("Llama 3 70B", "meta-llama_Meta-Llama-3-70B-Instruct", "llama-3-70b"),
          ("GPT-3.5", "OpenAI_gpt-3.5-turbo-0125", "gpt-3.5-turbo"), ("GPT-4", "OpenAI_gpt-4", "gpt-4")]
norm = lambda s: re.sub(r"\s+", " ", str(s)).strip()
cache = {}
def ea(source, name):
    if (source, name) not in cache:
        d = pd.read_csv(f"{EA[source]}/{name}", dtype=str, keep_default_na=False)
        d.index = d["Calculator ID"] + "_" + d["Note ID"]
        cache[source, name] = (d, json.load(open(f"{EA[source]}/{name}.error_type.json")))
    return cache[source, name]

def types(out):
    if not isinstance(out, dict) or not isinstance(out.get("error_type"), list):
        return None
    t = sorted({m.group(1) for e in out["error_type"] for m in [re.match(r"\s*([A-D])\b", str(e))] if m})
    return t or None

lines = []
for model, prefix, ename in MODELS:
    for style in ["zero_shot", "one_shot"]:
        name = f"{style}_{ename}.csv"
        for l in open(f"{CONS}/{prefix}_{style}.jsonl"):
            x = json.loads(l)
            if x["Result"] == "Correct": continue
            uid = f'{x["Calculator ID"]}_{x["Note ID"]}'
            line = {"Model": model, "Style": style, "Row Number": int(x["Row Number"]), "Calculator ID": x["Calculator ID"],
                    "Note ID": x["Note ID"], "Source Run": x["Source Run"], "Error Types": "", "Status": ""}
            d, j = ea(x["Source Run"], name)
            if uid not in d.index:
                line["Status"] = "not classified (was correct under the old label)"
            elif norm(d.at[uid, "raw_answer"]) != norm(x["LLM Raw Output"]):
                line["Status"] = "classification is of a different output"
            else:
                t = types(j.get(uid))
                if t is None:
                    line["Status"] = "unclassified (empty or blocked)"
                else:
                    line["Error Types"] = "".join(t)
                    line["Status"] = ("classified" if norm(d.at[uid, "explanation"]) == norm(x["Ground Truth Explanation"])
                                      else "classified against the old explanation")
            lines.append(line)
d = pd.DataFrame(lines)
d.to_csv(f"{variant.OUT}/error_classifications.csv", index=False)
print(d.groupby("Status").size().to_string(), "\n")
print(d[d.Status != "classified"].groupby(["Status", "Source Run"]).size().to_string(), "\n")
ok = d[d.Status == "classified"]
summ = []
for (m, s), g in ok.groupby(["Model", "Style"], sort=False):
    row = {"Model": m, "Style": s, "Wrong": int((d.Model.eq(m) & d.Style.eq(s)).sum()), "Classified": len(g)}
    for t in "ABCD":
        row[t] = round(100 * g["Error Types"].str.contains(t).mean(), 1)
    summ.append(row)
summ = pd.DataFrame(summ); summ.to_csv(f"{variant.OUT}/error_type_summary.csv", index=False)
print(summ.to_string(index=False))
