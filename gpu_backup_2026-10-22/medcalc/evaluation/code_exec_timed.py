"""
Code-augmented MedCalc-Bench run (copy of generate_code_prompt.py with timing), for a timing trial.

Run: python3 code_exec_timed.py --model gpt-4 --limit 50 [--workers 8] [--data ../dataset/test_data_10_20_2026.csv]
Same system prompt, user prompt, feedback messages, 20-round cap and scoring as generate_code_prompt.py. Differences:
- the OpenAI >= 1.0 client (generate_code_prompt.py uses the removed openai.ChatCompletion); key from OPENAI_API_KEY
  or ~/.openai_key;
- rows run in parallel (--workers); --limit N takes the first row of each of the first N calculators;
- the model's code runs in a separate isolated Python process with a 60 s timeout instead of exec() in this process;
- a row whose API call fails (e.g. the conversation outgrows the context) is scored Incorrect with the error recorded;
- answer parsing: </answer> is accepted as well as <\answer>, and an answer tag in a message that also has code is only
  taken once that code has been run (generate_code_prompt.py took e.g. "<answer> YOUR_ANSWER_HERE <\answer>" written
  next to not-yet-run code as the final answer);
- prints wall-clock time, rounds and accuracy; writes outputs/<tag>/code_exec_<model>.jsonl.
"""

import json
import os
import re
import subprocess
import sys
import tempfile
import time
import argparse
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from evaluate import check_correctness
from table_stats import compute_overall_accuracy
from generate_code_prompt import extract_python_code


def api_key():
    key = os.getenv("OPENAI_API_KEY")
    if not key and os.path.exists(os.path.expanduser("~/.openai_key")):
        key = open(os.path.expanduser("~/.openai_key")).read().strip()
    if not key:
        raise SystemExit("Set OPENAI_API_KEY or put the key in ~/.openai_key")
    return key


client = OpenAI(api_key=api_key(), max_retries=8)


def capture_exec_output_and_errors(code):
    """Run the code in a separate isolated Python process and return its printed output and any error messages."""
    with tempfile.TemporaryDirectory() as tmp:
        try:
            run = subprocess.run([sys.executable, "-I", "-c", code], cwd=tmp, capture_output=True, text=True, timeout=60)
            return run.stdout + run.stderr
        except subprocess.TimeoutExpired:
            return "TimeoutError: the code did not finish within 60 seconds"


def apply_calc(question, patient_note, model_name):
    system = f"You are a helpful assistant. Your task is to read a patient note and compute a medical value based on the following the question: {question}.\n"
    system += "If there are multiple values for a given measurement or attribute, then please use the value recorded based on when the patient note was written. You should not be using values that the patient had post-treatment or values from a patient's history in the past. "
    system += "Additionally, if the problem doesn't directly imply or provide information regarding a particular patient attribute, assume the patient does not have it."
    system += "Do not perform any computations yourself. Do not state a numerical answer. First write code for any equations you are using and then plug in values based on the patient note. Make sure the code prints all of its outputs."
    system += "If there are any errors with compiling your script, you may need to re-write your code to obtain the output. Note that all of the necessary information is provided in the patient note and you should not need to prompt the user for any information."
    system += "When you are finished with all the computations, please output your final answer value in the following format: <answer> YOUR_ANSWER_HERE <\\answer>, where YOUR_ANSWER_HERE is your final answer value without any units (examples: <answer> 17.29 <\\answer> (an example answer where the output can be a decimal), <answer> 5 <\\answer> (an example answer for score-based problems), <answer> 5/4/2021 <\\answer> (an example answer for estimated date problems), <answer> (4 weeks, 3 days) <\\answer> (an example answer for estimated gestational age))."
    system += "Asides for the step where you give your final answer in the <answer> <\\answer> tags, all your other responses must ALWAYS have code with the ```python tag as part of your response. This code should all be written in a single block used for computing the final answer value. The last statement in your code should be a print() statement so that the user can execute your code and provide you with the final answer. "

    prompt = "Patient Note:\n\n"
    prompt += patient_note + "\n\n"
    prompt += "Question:\n\n"
    prompt += question + "\n\n"

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]

    n = 0
    executed = set()

    while True:
        response = client.chat.completions.create(model=model_name, messages=messages)
        output = response.choices[0].message.content
        n += 1
        messages.append({"role": "assistant", "content": output})

        match_ = re.search(r'<answer>(.*?)<[\\/]answer>', output, re.DOTALL)
        message_code = extract_python_code(output)
        # an answer is accepted only once the code in the same message has been run; a message that writes new code and
        # also shows the answer format (e.g. "<answer> YOUR_ANSWER_HERE <\answer>") gets its code executed first.
        # A code block holding nothing but the answer tag is not code.
        only_tag = not re.sub(r'<answer>.*?<[\\/]answer>', '', message_code, flags=re.DOTALL).strip()
        if match_ and ("```python" not in output or only_tag or (re.search(r"\d", match_.group(1)) and message_code in executed)):
            answer = match_.group(1).strip()
            messages.append({"role": "user", "content": str(answer)})
            return str(answer), messages, n
        else:

            if "```python" not in output:
                new_output = f"It seems that you have not written any code as part of your response. This was your last thought:\n\n\n{output}\n\n\n. Based on this, please write a single block of code which the user will execute for you so that you can obtain the final answer. To get the final answer value from the console, please add a print() statement at the end, i.e. print(creatinine_clearance), print(bmi), print(curb-65 score)"
                messages.append({"role": "user", "content": new_output})

            elif "print" not in message_code:
                new_output = f"This was your previous response:\n\n\n{output}\n\n\n. There is no print() statement in your code. Please add a print statement to your code so that the user can execute your code for you to print out the value of the final answer value, i.e. print(creatinine_clearance), print(bmi), print(curb-65 score) "
                messages.append({"role": "user", "content": new_output})

            elif "input(" in message_code:
                return "N/A", messages, n
            else:

                console_output = capture_exec_output_and_errors(message_code)
                executed.add(message_code)

                new_output = f"""I have executed your code, and the output is:

                {console_output}

                If there was an error, or the computed answer is obviously incorrect, please revise your code. Otherwise please output your final answer in the following format:

                <answer> YOUR_ANSWER_HERE <\\answer> where YOUR_ANSWER_HERE is your final answer.

                Decimal Example:
                <answer> 17.29 <\\answer>

                Score-Based Example:
                <answer> 5 <\\answer>

                Estimated Date Example:
                <answer> 5/21/2021 <\\answer>

                Estimated Age Example:
                <answer> (4 weeks, 3 days) <\\answer>

                All of the information needed is in the patient note and you should not need to prompt the user for any more information.
                """

                messages.append({"role": "user", "content": new_output})

        if n >= 20:
            return None, messages, n


def process_row(row, model_name):
    start = time.time()
    calc_id = str(row["Calculator ID"])
    try:
        answer, messages, rounds = apply_calc(row["Question"], row["Patient Note"], model_name)
        error = None
    except Exception as e:
        answer, messages, rounds, error = None, [], 0, str(e)

    if not answer:
        extracted_answer = "None"
        result = "Incorrect"
    else:
        # as generate_code_prompt.py: its extract_answer call raises, so the tagged answer is scored as given
        extracted_answer = answer
        try:
            status = check_correctness(extracted_answer, row["Ground Truth Answer"], calc_id, row["Upper Limit"], row["Lower Limit"])
            result = "Correct" if status else "Incorrect"
        except:
            result = "Incorrect"

    return {
        "Row Number": int(row["Row Number"]),
        "Calculator Name": row["Calculator Name"],
        "Calculator ID": calc_id,
        "Category": row["Category"],
        "Note ID": str(row["Note ID"]),
        "Patient Note": row["Patient Note"],
        "Question": row["Question"],
        "LLM Answer": extracted_answer,
        "LLM Chat History": messages,
        "Ground Truth Answer": row["Ground Truth Answer"],
        "Ground Truth Explanation": row["Ground Truth Explanation"],
        "Result": result,
        "Rounds": rounds,
        "Seconds": round(time.time() - start, 1),
        "API Error": error,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Code-augmented run with timing')
    parser.add_argument('--model', type=str, required=True, help='OpenAI model, e.g. gpt-4 or gpt-3.5-turbo')
    parser.add_argument('--data', type=str, default="../dataset/test_data_10_20_2026.csv")
    parser.add_argument('--limit', type=int, default=None, help='First row of each of the first N calculators')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--tag', type=str, default="code_exec_trial")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    if args.limit:
        df = df.groupby("Calculator ID", sort=False).head(1).head(args.limit)

    os.makedirs(os.path.join("outputs", args.tag), exist_ok=True)
    output_path = os.path.join(args.tag, f"code_exec_{args.model}.jsonl")
    if os.path.exists(os.path.join("outputs", output_path)):
        os.remove(os.path.join("outputs", output_path))

    start = time.time()
    rows = []
    with ThreadPoolExecutor(args.workers) as ex:
        futures = [ex.submit(process_row, row, args.model) for _, row in df.iterrows()]
        for future in as_completed(futures):
            rows.append(future.result())
    elapsed = time.time() - start

    rows.sort(key=lambda r: r["Row Number"])
    with open(os.path.join("outputs", output_path), "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)

    secs = sorted(r["Seconds"] for r in rows)
    errors = [r for r in rows if r["API Error"]]
    print(f"{args.model}: {len(rows)} rows in {elapsed:.0f}s wall clock ({elapsed / 60:.1f} min, {args.workers} workers)")
    print(f"per row: median {secs[len(secs) // 2]:.0f}s, max {secs[-1]:.0f}s; rounds median {sorted(r['Rounds'] for r in rows)[len(rows) // 2]}")
    print(f"correct {sum(r['Result'] == 'Correct' for r in rows)}/{len(rows)}; API errors {len(errors)}")
    for r in errors[:3]:
        print("  error:", r["Row Number"], r["API Error"][:200])
    print(json.dumps(compute_overall_accuracy(output_path, args.model, "code_exec_trial")))
