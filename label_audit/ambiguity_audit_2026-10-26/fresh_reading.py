"""Ambiguity audit, part B: a fresh reading of every non-template row by gpt-4o-2024-08-06 (temperature 0).

The model sees the note, the question, the time-point instruction and our extracted inputs, and lists every input whose
value the note does not clearly determine, with the alternative value(s) in the same format. Each alternative is then
scored with our calculator code; a row is flagged when an alternative label falls outside our label's accepted range.

  python3 fresh_reading.py --build        # write batch_input.jsonl and print a cost estimate (no API call)
  python3 fresh_reading.py --score FILE   # score a batch output file -> fresh_reading_flags.csv
"""
import argparse, ast, json, os, sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "session_archive_2026-10-22"))
MODEL = "gpt-4o-2024-08-06"

SYSTEM = """You audit a medical-calculation benchmark for ambiguous patient notes. You are given a patient note, the calculation question, an instruction on which time point to use, and the input values that were extracted from the note to compute the answer.

Conventions of the benchmark (these are rules, not sources of ambiguity):
- Use values at the time point the instruction names; later events and later measurements do not count.
- A criterion the note does not mention is absent; for "absence" criteria (e.g. "Cough Absent") an unmentioned cough means the criterion is met.
- HEART: no documented ECG/troponin result counts as normal ECG / troponin <= normal limit.
- Electrolytes come from the serum chemistry panel, not a blood gas.

Task: identify every input whose value is NOT clearly determined by the note under these conventions, i.e. where a careful clinician could reasonably choose a different value. Typical causes: several measurements of the same quantity with no clear one for the time point, vague or hedged wording, a clinical judgement call (e.g. a suspicion level), unclear timing, or a unit that is implausible or unclear. Do not list inputs that are merely extracted wrongly in an obvious way unless the note is genuinely unclear; flag those separately as errors.

Answer with JSON only:
{"ambiguous_inputs": [{"input": "<input name exactly as given>", "alternatives": [<alternative value(s) in exactly the same format as the given value>], "quote": "<verbatim text from the note>", "reason": "<one sentence>"}],
 "clear_errors": [{"input": "<input name>", "correct_value": <value in the same format>, "quote": "<verbatim text>", "reason": "<one sentence>"}]}
Use empty lists when there is nothing to report. For an input that was not extracted but should be, use its name from the calculator and give the value(s)."""


def user_prompt(r):
    tp = r["Time Point Instruction"] if isinstance(r["Time Point Instruction"], str) else "(none)"
    return (f"Patient note:\n{r['Patient Note']}\n\nQuestion: {r['Question']}\nTime-point instruction: {tp}\n\n"
            f"Extracted inputs: {r['Relevant Entities']}")


def build(data):
    df = pd.read_csv(data, dtype=str)
    df = df[df["Note Type"] != "Template"]
    with open(os.path.join(HERE, "batch_input.jsonl"), "w") as f:
        for _, r in df.iterrows():
            body = {"model": MODEL, "temperature": 0, "response_format": {"type": "json_object"},
                    "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_prompt(r)}]}
            f.write(json.dumps({"custom_id": f"row-{r['Row Number']}", "method": "POST", "url": "/v1/chat/completions", "body": body}) + "\n")
    chars = sum(len(SYSTEM) + len(user_prompt(r)) for _, r in df.iterrows())
    tin, tout = chars / 4, len(df) * 300
    print(f"{len(df)} requests; about {tin/1e6:.2f}M input and {tout/1e6:.2f}M output tokens")
    print(f"estimated cost: ${tin/1e6*2.5 + tout/1e6*10:.2f} regular API, ${(tin/1e6*2.5 + tout/1e6*10)/2:.2f} Batch API")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--build", action="store_true")
    p.add_argument("--score")
    p.add_argument("--data", default=os.path.join(ROOT, "dataset/test_data_10_22_corrected.csv"))
    a = p.parse_args()
    if a.build:
        build(a.data)
