import json, glob, collections

T = "test_data_final_r2"
rows = [json.loads(l) for f in glob.glob(f"outputs/{T}/diag/mixtral_hf_{T}_shard*.jsonl") for l in open(f)]
by = collections.defaultdict(lambda: [0, 0, 0])
for r in rows:
    b = by[r["prompt_style"]]
    b[0] += 1
    b[1] += r["has_answer"]
    b[2] += r["Result"] == "Correct"
for s, (n, a, c) in by.items():
    print(f"{s}: {n} rows vLLM (single BOS) left without an answer -> original code: has answer {a}, correct {c}")
