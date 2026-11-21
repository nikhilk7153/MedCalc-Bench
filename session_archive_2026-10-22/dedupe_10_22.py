import os, json, glob, collections
from table_stats import compute_overall_accuracy

TAG = "test_data_10_22"
for path in sorted(glob.glob(f"outputs/{TAG}/OpenAI_*.jsonl")):
    rows = [json.loads(l) for l in open(path)]
    first = collections.OrderedDict()
    for r in rows:
        first.setdefault(r["Row Number"], r)
    kept = sorted(first.values(), key=lambda r: r["Row Number"])
    with open(path, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in kept)
    name = os.path.basename(path)[:-len(".jsonl")]
    style =next(s for s in ["direct_answer", "zero_shot", "one_shot"] if name.endswith(s))
    model = "OpenAI/" + name[len("OpenAI_"):-len("_" + style)]
    stats = compute_overall_accuracy(os.path.join(TAG, os.path.basename(path)), model, f"{style}_{TAG}")
    print(f"{name}: {len(rows)} lines -> {len(kept)} unique rows; overall {stats['overall']['average']}")
