"""
Accuracy vs. number of attributes (paper Appendix E.1, Figures 2 and 3).

The number of attributes of a calculator is the number of its attributes in calculator_implementations_verified/
name_to_python.json (its entries other than file path, explanation function, calculator name, type and question), and
each row gets the count of its calculator. For each model, the accuracy at each attribute count is the share of rows with that
count answered correctly; a second-degree polynomial is fitted to these points (np.polyfit) and drawn over the observed
range, one figure for equation-based (lab, physical, date, dosage) and one for rule-based (risk, severity, diagnosis)
calculators.

  python attribute_analysis.py                 (zero-shot, as in the paper's figures)
  python attribute_analysis.py --setting one_shot
Writes <out_dir>/accuracy_vs_attributes_<setting>_{equation,rule}.png, the points (attribute count, rows, correct,
accuracy) to <out_dir>/accuracy_vs_attributes_<setting>.csv and the fitted coefficients to
<out_dir>/fit_coefficients_<setting>.csv.
"""

import argparse
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# legend label -> output file stem (same order, and so the same default colours, as the paper's figures)
MODELS = {
    "PMC Llama 13b": "axiong_PMC_LLaMA_13B",
    "Meditron 70b": "epfl-llm_meditron-70b",
    "Llama3 8b": "meta-llama_Meta-Llama-3-8B-Instruct",
    "Llama3 70b": "meta-llama_Meta-Llama-3-70B-Instruct",
    "Mistral 7b": "mistralai_Mistral-7B-Instruct-v0.2",
    "Mixtral 8x7b": "mistralai_Mixtral-8x7B-Instruct-v0.1",
    "GPT 4": "OpenAI_gpt-4",
    "GPT-3.5-turbo": "OpenAI_gpt-3.5-turbo-0125",
}
TYPES = {
    "equation": ("Equation-Based", ["lab", "physical", "date", "dosage"]),
    "rule": ("Rule-Based", ["risk", "severity", "diagnosis"]),
}
SETTINGS = ["direct_answer", "zero_shot", "one_shot"]
NOT_ATTRIBUTES = {"file path", "explanation function", "calculator name", "type", "question"}


def load_results(path):
    return {int(r["Row Number"]): r["Result"] == "Correct" for r in map(json.loads, open(path))}


def run(setting, args, n_attributes, category):
    files = {label: os.path.join(args.outputs_dir, f"{stem}_{setting}.jsonl") for label, stem in MODELS.items()}

    points, coefficients = [], []
    for kind, (title, categories) in TYPES.items():
        rows = category[category.isin(categories)].index
        plt.figure(figsize=(12, 8))
        for label, path in files.items():
            correct = load_results(path)
            missing = set(rows) - set(correct)
            if missing:
                raise SystemExit(f"{path}: no result for {len(missing)} rows, e.g. {sorted(missing)[:5]}")
            df = pd.DataFrame({"n": n_attributes[rows], "correct": [correct[r] for r in rows]})
            acc = df.groupby("n")["correct"].agg(["size", "sum", "mean"])
            fit = np.polyfit(acc.index, acc["mean"], 2)
            x = np.linspace(acc.index.min(), acc.index.max(), 200)
            plt.plot(x, np.polyval(fit, x), label=label)
            for n, r in acc.iterrows():
                points.append({"Setting": setting, "Calculator Type": kind, "Model": label, "Attributes": n,
                               "Rows": int(r["size"]), "Correct": int(r["sum"]), "Accuracy": round(r["mean"], 4)})
            coefficients.append({"Setting": setting, "Calculator Type": kind, "Model": label,
                                 "a (x^2)": fit[0], "b (x)": fit[1], "c": fit[2]})
        plt.title(f"Accuracy vs. Number of Parameters for {title} Calculators")
        plt.xlabel("Number of Parameters")
        plt.ylabel("Accuracy")
        plt.legend(loc="upper right")
        out = os.path.join(args.out_dir, f"accuracy_vs_attributes_{setting}_{kind}.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print("wrote", out)

    pd.DataFrame(points).to_csv(os.path.join(args.out_dir, f"accuracy_vs_attributes_{setting}.csv"), index=False)
    pd.DataFrame(coefficients).to_csv(os.path.join(args.out_dir, f"fit_coefficients_{setting}.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Accuracy vs. number of attributes (Appendix E.1)")
    parser.add_argument("--setting", choices=SETTINGS, default="zero_shot")
    parser.add_argument("--data", default="../dataset/test_data_10_22.csv")
    parser.add_argument("--outputs_dir", default="../final_runs/test_data_10_22/outputs")
    parser.add_argument("--calculators", default="../calculator_implementations_verified/name_to_python.json")
    parser.add_argument("--out_dir", default="attribute_analysis_10_22")
    args = parser.parse_args()

    data = pd.read_csv(args.data).set_index("Row Number")
    calculators = json.load(open(args.calculators))
    attributes = {int(calc_id): len(set(calc) - NOT_ATTRIBUTES) for calc_id, calc in calculators.items()}
    n_attributes = data["Calculator ID"].map(attributes)
    if n_attributes.isna().any():
        raise SystemExit(f"calculators missing from {args.calculators}: {sorted(data['Calculator ID'][n_attributes.isna()].unique())}")
    os.makedirs(args.out_dir, exist_ok=True)
    run(args.setting, args, n_attributes, data["Category"])
