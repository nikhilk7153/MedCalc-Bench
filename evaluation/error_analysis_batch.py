"""
classify the error types with GPT-4 through the OpenAI Batch API (same prompt and settings as error_analysis.py)

Run:
  python3 error_analysis_batch.py submit <folder> <setting>   # writes <folder>/batch_input.jsonl and <folder>/batch_job.json
  python3 error_analysis_batch.py fetch <folder>              # prints status; when done writes <csv>.error_type.json files
Key from OPENAI_API_KEY or ~/.openai_key. Outputs use the same format as error_analysis.py (uid -> parsed JSON or raw
text, uid = "<Calculator ID>_<Note ID>"). fetch prints the batch's own turnaround (created -> completed).
"""

import glob
import json
import os
import sys
import time
import pandas as pd

from openai import OpenAI

system = 'You are a helpful assistant for error analysis in medical calculation, and your task is to classify the type of errors in the prediction, using the ground-truth explanation as the reference. There are four potential error types: A. wrong calculator / rules used, or wrong knowledge about the calculation; B. extracted different parameters than the ground-truth; C. performed the arithmetic incorrectly; D. no clear error in the prediction. Please output a JSON dict formatted as {"analysis": Str, "error_type": List[types]}.'

user_msg_template = lambda truth, pred: f"Here is the ground-truth:\n{truth}\n\nHere is the prediction:\n{pred}\n\nJSON output:"


def client():
	key = os.getenv("OPENAI_API_KEY")
	if not key and os.path.exists(os.path.expanduser("~/.openai_key")):
		key = open(os.path.expanduser("~/.openai_key")).read().strip()
	if not key:
		raise SystemExit("Set OPENAI_API_KEY or put the key in ~/.openai_key")
	return OpenAI(api_key=key)


def submit(folder, setting):
	requests = []
	for errors_path in sorted(glob.glob(os.path.join(folder, setting + "*"))):
		if errors_path[-4:] != ".csv":
			continue
		for _, row in pd.read_csv(errors_path).iterrows():
			uid = str(row["Calculator ID"]) + "_" + str(row["Note ID"])
			requests.append({
				"custom_id": os.path.basename(errors_path) + "|" + uid,
				"method": "POST",
				"url": "/v1/chat/completions",
				"body": {"model": "gpt-4", "temperature": 0, "messages": [
					{"role": "system", "content": system},
					{"role": "user", "content": user_msg_template(row["explanation"], row["raw_answer"])}]},
			})
	input_path = os.path.join(folder, "batch_input.jsonl")
	with open(input_path, "w") as f:
		f.writelines(json.dumps(r) + "\n" for r in requests)
	c = client()
	file = c.files.create(file=open(input_path, "rb"), purpose="batch")
	batch = c.batches.create(input_file_id=file.id, endpoint="/v1/chat/completions", completion_window="24h")
	json.dump({"batch_id": batch.id, "input_file_id": file.id, "requests": len(requests), "submitted_at": time.time()},
	          open(os.path.join(folder, "batch_job.json"), "w"), indent=1)
	print(f"submitted {len(requests)} requests: batch {batch.id} ({batch.status})")


def fetch(folder):
	job = json.load(open(os.path.join(folder, "batch_job.json")))
	c = client()
	batch = c.batches.retrieve(job["batch_id"])
	rc = batch.request_counts
	print(f"batch {batch.id}: {batch.status}; {rc.completed} completed, {rc.failed} failed of {rc.total}")
	end = batch.completed_at or batch.failed_at or batch.expired_at
	if end:
		print(f"turnaround: {(end - batch.created_at) / 60:.1f} min (created -> {batch.status})")
	else:
		print(f"running for {(time.time() - batch.created_at) / 60:.1f} min")
	if batch.errors and batch.errors.data:
		for e in batch.errors.data[:5]:
			print("  error:", e.code, e.message)
	if batch.status != "completed":
		return
	outputs = {}
	for line in c.files.content(batch.output_file_id).text.splitlines():
		r = json.loads(line)
		csv_name, uid = r["custom_id"].split("|", 1)
		if r.get("error") or r["response"]["status_code"] != 200:
			continue
		output = r["response"]["body"]["choices"][0]["message"]["content"]
		try:
			output = json.loads(output)
		except:
			output = output
		outputs.setdefault(csv_name, {})[uid] = output
	for csv_name, out in outputs.items():
		with open(os.path.join(folder, csv_name + ".error_type.json"), "w") as f:
			json.dump(out, f, indent=4)
		print(f"{csv_name}.error_type.json: {len(out)} rows")
	if batch.error_file_id:
		print("some requests failed; details:", c.files.content(batch.error_file_id).text[:1000])


if __name__ == "__main__":
	if sys.argv[1] == "submit":
		submit(sys.argv[2], sys.argv[3])
	elif sys.argv[1] == "fetch":
		fetch(sys.argv[2])
	else:
		raise SystemExit(__doc__)
