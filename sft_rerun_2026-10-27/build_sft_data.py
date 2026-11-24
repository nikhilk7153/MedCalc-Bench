"""
Builds the fine-tuning data in the format of the paper's Appendix C.2 (chatML, as read by CodeAct's
Megatron-LLM/tools/preprocess_instruct_data.py: {"id", "conversations": [{"role", "content"}, ...]}):
  system:    the paper's system message
  user:      "Patient Note:\n<note>\nQuestion: <question>" (the time-point instruction, when the row has one, follows the
             question, as in evaluation/run.py)
  assistant: "<Ground Truth Explanation>\nAnswer: <Ground Truth Answer>" (training only; the loss is on this turn)

Writes data/train_chatml.jsonl from dataset/train_data_10_22.csv and data/test_chatml.jsonl (system and user turns,
plus the Row Number) from the release test set, final_release_2026-11-08/dataset/test_data.csv (the prompts every
Table 2 baseline saw; --test picks another, e.g. the Astra release). No training row has the calculator and the note ID
or note text of a test row or one-shot example (dataset/build_train_data_10_22.py removed them), and the script stops
if one does. By default every training row is kept, as in the paper; --dedupe leaves out later rows with the same
calculator and note as an earlier one.
"""
import argparse
import json
import os
import re
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SYSTEM = ("Below is a patient note as well as a medical question about the patient. Provide an accurate answer to the "
          "question based on the note. Explain your reasoning before stating your final answer and put your final answer "
          "at the end of your response in the format Answer: INSERT_ANSWER")

norm = lambda s: re.sub(r"\s+", " ", str(s)).strip()


def user_message(row):
    question = row["Question"]
    time_point = row.get("Time Point Instruction")
    if isinstance(time_point, str) and time_point.strip():
        question = question.rstrip() + " " + time_point.strip()
    return f"Patient Note:\n{row['Patient Note']}\nQuestion: {question}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dedupe", action="store_true")
    parser.add_argument("--out_dir", default=os.path.join(HERE, "data"))
    parser.add_argument("--test", default=f"{ROOT}/final_release_2026-11-08/dataset/test_data.csv")
    args = parser.parse_args()

    train = pd.read_csv(f"{ROOT}/dataset/train_data_10_22.csv")
    test = pd.read_csv(args.test, keep_default_na=False)
    one_shot = json.load(open(f"{os.path.dirname(args.test)}/one_shot_examples.json"))

    held_ids = set(zip(test["Calculator ID"], test["Note ID"].astype(str)))
    held_notes = set(zip(test["Calculator ID"], test["Patient Note"].map(norm))) | {(int(c), norm(e["Patient Note"])) for c, e in one_shot.items()}
    overlap = [rn for rn, c, i, n in zip(train["Row Number"], train["Calculator ID"], train["Note ID"].astype(str), train["Patient Note"].map(norm))
               if (c, i) in held_ids or (c, n) in held_notes]
    if overlap:
        raise SystemExit(f"{len(overlap)} training rows share a calculator and note ID or note with a held-out row, e.g. {overlap[:5]}")

    keep = pd.Series(True, index=train.index)
    if args.dedupe:
        keep &= ~train.assign(n=train["Patient Note"].map(norm)).duplicated(["Calculator ID", "n"])
    train = train[keep]

    os.makedirs(args.out_dir, exist_ok=True)
    with open(f"{args.out_dir}/train_chatml.jsonl", "w") as f:
        for _, r in train.iterrows():
            f.write(json.dumps({"id": f"medcalc-train-{r['Row Number']}", "conversations": [  # Megatron reads "id"
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user_message(r)},
                {"role": "assistant", "content": f"{str(r['Ground Truth Explanation']).strip()}\nAnswer: {r['Ground Truth Answer']}"},
            ]}) + "\n")
    with open(f"{args.out_dir}/test_chatml.jsonl", "w") as f:
        for _, r in test.iterrows():
            f.write(json.dumps({"Row Number": int(r["Row Number"]), "conversations": [
                {"role": "system", "content": SYSTEM}, {"role": "user", "content": user_message(r)}]}) + "\n")
    print(f"train: {len(train)} conversations ({(~keep).sum()} left out); test: {len(test)} prompts -> {args.out_dir}")
