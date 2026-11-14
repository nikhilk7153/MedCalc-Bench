"""
Accuracy vs. number of parameters (paper Appendix E.1, Figures 2 and 3).

A row's number of parameters is the number of its Relevant Entities (the calculator inputs extracted from its note), as
for the attribute columns of Table 1. Zero-shot chain-of-thought outputs, as in the paper. For each model, the accuracy
at each parameter count is the share of rows with that count answered correctly; a second-degree polynomial is fitted
to these points (np.polyfit) and drawn over the observed range, one figure for equation-based (lab, physical, date,
dosage) and one for rule-based (risk, severity, diagnosis) calculators.

  python attribute_analysis.py
Writes <out_dir>/accuracy_vs_attributes_zero_shot_{equation,rule}.png, the points (parameter count, rows, correct,
accuracy) to <out_dir>/accuracy_vs_attributes_zero_shot.csv and the fitted coefficients to
<out_dir>/fit_coefficients_zero_shot.csv.
"""

import argparse
import ast
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


def count_parameters(entities):
    try:
        return len(json.loads(entities))
    except ValueError:
        return len(ast.literal_eval(entities))


def load_results(path):
    """Read a model's output file and return, for each row number, whether the model answered correctly."""
    is_correct = {}
    with open(path) as f:
        for line in f:
            output = json.loads(line)
            row_number = int(output["Row Number"])
            is_correct[row_number] = output["Result"] == "Correct"
    return is_correct


def accuracy_by_parameter_count(is_correct, rows, n_parameters):
    """For each number of parameters: how many rows have it, how many the model got right, and the accuracy."""
    table = pd.DataFrame({"parameters": n_parameters[rows], "correct": [is_correct[row] for row in rows]})
    return table.groupby("parameters")["correct"].agg(rows="size", correct="sum", accuracy="mean")


def make_figure(calculator_type, rows, n_parameters, outputs_dir, out_dir):
    """Draw one figure with a fitted curve per model. Returns the plotted points and the curves' coefficients."""
    points, curves = [], []
    plt.figure(figsize=(12, 8))

    for model, file_stem in MODELS.items():
        is_correct = load_results(os.path.join(outputs_dir, f"{file_stem}_zero_shot.jsonl"))
        missing = [row for row in rows if row not in is_correct]
        if missing:
            raise SystemExit(f"{model}: no result for {len(missing)} rows, e.g. {missing[:5]}")

        accuracy = accuracy_by_parameter_count(is_correct, rows, n_parameters)
        a, b, c = np.polyfit(accuracy.index, accuracy["accuracy"], 2)  # accuracy ≈ a·x² + b·x + c
        x = np.linspace(accuracy.index.min(), accuracy.index.max(), 200)
        plt.plot(x, np.polyval([a, b, c], x), label=model)

        for count, point in accuracy.iterrows():
            points.append({"Calculator Type": calculator_type, "Model": model, "Attributes": count,
                           "Rows": int(point["rows"]), "Correct": int(point["correct"]),
                           "Accuracy": round(point["accuracy"], 4)})
        curves.append({"Calculator Type": calculator_type, "Model": model, "a (x^2)": a, "b (x)": b, "c": c})

    title = TYPES[calculator_type][0]
    plt.title(f"Accuracy vs. Number of Parameters for {title} Calculators")
    plt.xlabel("Number of Parameters")
    plt.ylabel("Accuracy")
    plt.legend(loc="upper right")
    path = os.path.join(out_dir, f"accuracy_vs_attributes_zero_shot_{calculator_type}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print("wrote", path)
    return points, curves


def run(args, n_parameters, category):
    """Make the equation-based and rule-based figures and save their points and curves as CSV."""
    all_points, all_curves = [], []
    for calculator_type, (_, categories) in TYPES.items():
        rows = category[category.isin(categories)].index
        points, curves = make_figure(calculator_type, rows, n_parameters, args.outputs_dir, args.out_dir)
        all_points += points
        all_curves += curves

    pd.DataFrame(all_points).to_csv(os.path.join(args.out_dir, "accuracy_vs_attributes_zero_shot.csv"), index=False)
    pd.DataFrame(all_curves).to_csv(os.path.join(args.out_dir, "fit_coefficients_zero_shot.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Accuracy vs. number of parameters (Appendix E.1)")
    parser.add_argument("--data", default="../dataset/test_data_10_22_corrected.csv",
                        help="test set whose Relevant Entities the outputs were scored against")
    parser.add_argument("--outputs_dir", default="../final_runs/consolidated_2026-11-06/outputs")
    parser.add_argument("--out_dir", default="../final_runs/consolidated_2026-11-06/accuracy_vs_param_count")
    args = parser.parse_args()

    data = pd.read_csv(args.data, keep_default_na=False).set_index("Row Number")
    n_parameters = data["Relevant Entities"].map(count_parameters)
    os.makedirs(args.out_dir, exist_ok=True)
    run(args, n_parameters, data["Category"])
