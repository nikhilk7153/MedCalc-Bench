import json, glob, collections, sys

folder = sys.argv[1] if len(sys.argv) > 1 else "outputs/test_data_10_22"
for p in sorted(glob.glob(f"{folder}/*.jsonl")):
    rows = [json.loads(l) for l in open(p)]
    finish = collections.Counter(r.get("Finish Reason") for r in rows)
    unparsed = sum(str(r["LLM Answer"]) in ("N/A", "Not Found") or "is not defined" in str(r["LLM Answer"]) for r in rows)
    empty = sum(not (r["LLM Raw Output"] or "").strip() for r in rows)
    accuracy = 100 * sum(r["Result"] == "Correct" for r in rows) / len(rows)
    unique = len({r["Row Number"] for r in rows})
    name = p.split("/")[-1][:-6]
    print(f"{name:52s} rows {unique:4d} | acc {accuracy:6.2f} | no answer {unparsed:4d} | empty {empty:4d} | hit length {finish.get('length', 0):3d}")
