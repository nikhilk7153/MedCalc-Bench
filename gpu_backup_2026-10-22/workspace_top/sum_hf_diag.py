import json, glob, collections
rows = [json.loads(l) for f in glob.glob("outputs/test_data_final/diag/mixtral_hf_shard*.jsonl") for l in open(f)]
bad = [r for r in rows if not r["has_answer"]]
print(f"HF original path (single BOS): {len(rows)} rows, has answer {len(rows) - len(bad)}, correct {sum(r['Result'] == 'Correct' for r in rows)}")
print("no-answer by style:", collections.Counter(r["prompt_style"] for r in bad), "median tokens:", sorted(r["n_tokens"] for r in bad)[len(bad) // 2] if bad else None)
for r in bad[:4]:
    print("   ", r["prompt_style"], r["Row Number"], r["n_tokens"], r["text"][-100:])
single = {(r["prompt_style"], r["Row Number"]): r for r in map(json.loads, open("outputs/test_data_final/diag/mixtral_tp2_single_bos.jsonl"))}
both_bad = sum(1 for r in bad if not single[(r["prompt_style"], r["Row Number"])]["has_answer"])
print(f"rows with no answer in BOTH HF and vLLM single-BOS: {both_bad}; vLLM-single no-answer total: {sum(not v['has_answer'] for v in single.values())}")
