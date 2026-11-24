'''
Checks a test set and one-shot file before a paid run: required columns, unique rows, a one-shot example for every
calculator, no time-point sentence left in the questions, every prompt builds, and the ground truth itself scores Correct.
Exits non-zero with the reasons if anything is wrong.
'''

import sys
import json
import argparse
import pandas as pd

from run import build_messages, score

REQUIRED = ["Row Number", "Calculator ID", "Calculator Name", "Category", "Note ID", "Patient Note", "Question",
            "Ground Truth Answer", "Ground Truth Explanation", "Lower Limit", "Upper Limit"]
TIME_POINT = "first admitted to the hospital"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--one_shot', type=str, required=True)
    args = parser.parse_args()

    problems = []
    df = pd.read_csv(args.data)
    with open(args.one_shot) as file:
        one_shot_json = json.load(file)

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        sys.exit(f"FAILED: {args.data} lacks columns {missing}")
    if df["Row Number"].duplicated().any():
        problems.append(f"{df['Row Number'].duplicated().sum()} duplicate Row Numbers")
    no_example = sorted(set(df["Calculator ID"].astype(str)) - set(one_shot_json), key=int)
    if no_example:
        problems.append(f"no one-shot example for calculators {no_example}")
    in_question = df["Question"].str.contains(TIME_POINT, case=False).sum()
    if in_question:
        problems.append(f"{in_question} questions still contain the time-point sentence")
    if "Time Point Instruction" not in df.columns or df["Time Point Instruction"].isna().all():
        problems.append("no Time Point Instruction column/values, so no prompt would get the time-point sentence")

    if not problems:
        not_correct = []
        for i in range(len(df)):
            row = df.iloc[i]
            for prompt_style in ["direct_answer", "zero_shot", "one_shot"]:
                build_messages(row, prompt_style, "OpenAI/check", None, one_shot_json)
            answer = json.dumps({"step_by_step_thinking": "check", "answer": str(row["Ground Truth Answer"])})
            if score(row, row["Patient Note"], answer, "zero_shot")["Result"] != "Correct":
                not_correct.append(int(row["Row Number"]))
        if not_correct:
            problems.append(f"{len(not_correct)} rows whose own ground truth does not score Correct, e.g. {not_correct[:10]}")

    if problems:
        sys.exit("FAILED: " + "; ".join(problems))
    time_point_rows = df["Time Point Instruction"].notna().sum() if "Time Point Instruction" in df.columns else 0
    print(f"OK: {len(df)} rows, {df['Calculator ID'].nunique()} calculators, {time_point_rows} rows with a Time Point Instruction")
