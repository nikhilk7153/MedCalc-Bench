"""
Scores a fine-tuned model's outputs on the corrected test set. The paper's fine-tuned models end their response with
"Answer: <value>" (Appendix C.2) instead of the JSON the benchmark prompts ask for, so the text after the last "Answer:"
is passed to evaluation/run.py's score() as {"step_by_step_thinking": <text before it>, "answer": <it>}: the same answer
extraction, 5% limits and hedge rule as every other run. A response with no "Answer:" is scored Incorrect.

  python score_sft.py --outputs outputs/<model>.jsonl     (lines: {"Row Number": n, "Output": "..."})
Writes results/<model>.jsonl (one scored line per row, as in final_runs) and prints the accuracy.
"""
import argparse
import json
import os
import re
import sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, f"{ROOT}/evaluation")
from run import score  # noqa: E402

ANSWER = re.compile(r"Answer:\s*(.*)\s*$", re.S)


def as_json(output):
    cut = output.rfind("Answer:")
    if cut < 0:
        return output
    answer = output[cut + len("Answer:"):].strip().splitlines()[0].strip() if output[cut + len("Answer:"):].strip() else ""
    return json.dumps({"step_by_step_thinking": output[:cut].strip(), "answer": answer})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", required=True)
    parser.add_argument("--data", default=f"{ROOT}/final_release_2026-11-08/dataset/test_data.csv")
    parser.add_argument("--out_dir", default=os.path.join(HERE, "results"))
    args = parser.parse_args()

    test = pd.read_csv(args.data).set_index("Row Number", drop=False)
    outputs = {int(r["Row Number"]): r["Output"] for r in map(json.loads, open(args.outputs))}
    missing = set(test.index) - set(outputs)
    if missing:
        sys.exit(f"{args.outputs}: no output for {len(missing)} rows, e.g. {sorted(missing)[:5]}")

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, os.path.basename(args.outputs))
    correct = 0
    with open(out_path, "w") as f:
        for n, row in test.iterrows():
            scored = score(row, row["Patient Note"], as_json(outputs[n]), "zero_shot")
            scored["LLM Raw Output"] = outputs[n]
            correct += scored["Result"] == "Correct"
            f.write(json.dumps(scored) + "\n")
    print(f"{out_path}: {correct}/{len(test)} correct ({100 * correct / len(test):.2f}%)")
