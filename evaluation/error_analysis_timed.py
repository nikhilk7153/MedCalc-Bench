"""
classify the error types with GPT-4 (copy of ~/Documents/MedCalc-Bench/error_analysis.py with timing)

Run: python3 error_analysis_timed.py <folder> <setting> [workers]
Same prompt, model (gpt-4, temperature 0), inputs and outputs as error_analysis.py. Differences: uses Azure OpenAI when
OPENAI_ENDPOINT is set, else the OpenAI API (key from OPENAI_API_KEY or ~/.openai_key); the number of parallel requests
is an argument (default 3, as in error_analysis.py); failed requests are counted; the elapsed time is printed per file.
"""

import glob
import json
import os
import sys
import time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import AzureOpenAI, OpenAI


def api_key():
	key = os.getenv("OPENAI_API_KEY")
	if not key and os.path.exists(os.path.expanduser("~/.openai_key")):
		key = open(os.path.expanduser("~/.openai_key")).read().strip()
	if not key:
		raise SystemExit("Set OPENAI_API_KEY or put the key in ~/.openai_key")
	return key


if os.getenv("OPENAI_ENDPOINT"):
	client = AzureOpenAI(api_version="2023-09-01-preview", azure_endpoint=os.getenv("OPENAI_ENDPOINT"), api_key=api_key())
else:
	client = OpenAI(api_key=api_key(), max_retries=8)

system = 'You are a helpful assistant for error analysis in medical calculation, and your task is to classify the type of errors in the prediction, using the ground-truth explanation as the reference. There are four potential error types: A. wrong calculator / rules used, or wrong knowledge about the calculation; B. extracted different parameters than the ground-truth; C. performed the arithmetic incorrectly; D. no clear error in the prediction. Please output a JSON dict formatted as {"analysis": Str, "error_type": List[types]}.'

user_msg_template = lambda truth, pred: f"Here is the ground-truth:\n{truth}\n\nHere is the prediction:\n{pred}\n\nJSON output:"

def process_row(row, outputs, system, client):
	uid = str(row["Calculator ID"]) + "_" + str(row["Note ID"])

	if uid in outputs:
		return None

	truth = row["explanation"]
	pred = row["raw_answer"]
	prompt = user_msg_template(truth, pred)

	messages = [
		{"role": "system", "content": system},
		{"role": "user", "content": prompt}
	]

	try:
		response = client.chat.completions.create(
			model="gpt-4",
			messages=messages,
			temperature=0,
		)
	except Exception as e:
		return "failed", str(e)

	output = response.choices[0].message.content

	try:
		output = json.loads(output)
	except:
		output = output

	return uid, output

if __name__ == "__main__":
	folder = sys.argv[1]
	setting = sys.argv[2]
	max_workers = int(sys.argv[3]) if len(sys.argv) > 3 else 3

	total_start, total_rows = time.time(), 0
	for errors_path in sorted(glob.glob(os.path.join(folder, setting + "*"))):

		if errors_path[-4:] != ".csv":
			continue

		errors = pd.read_csv(errors_path)
		output_path = errors_path + ".error_type.json"

		if os.path.exists(output_path):
			outputs = json.load(open(output_path))
		else:
			outputs = {}

		start, failed, errors_seen = time.time(), 0, []
		with ThreadPoolExecutor(max_workers=max_workers) as executor:
			future_to_row = {
				executor.submit(process_row, row, outputs, system, client): row
				for _, row in errors.iterrows()
			}

			for future in as_completed(future_to_row):
				result = future.result()
				if result is None:
					continue
				if result[0] == "failed":
					failed += 1
					errors_seen.append(result[1])
					continue
				uid, output = result
				outputs[uid] = output
				with open(output_path, "w") as f:
					json.dump(outputs, f, indent=4)

		elapsed = time.time() - start
		total_rows += len(errors)
		print(f"{os.path.basename(errors_path)}: {len(errors)} rows, {failed} failed, {elapsed:.0f}s ({elapsed / max(len(errors), 1):.2f}s per row, {max_workers} workers)")
		for e in errors_seen[:3]:
			print("  error:", e[:300])
	elapsed = time.time() - total_start
	print(f"total: {total_rows} rows in {elapsed:.0f}s ({elapsed / 60:.1f} min)")
