"""Recomputes the Open-Patients table (paper Table 1: # Patients and Average Token Length per source) from the
Hugging Face dataset ncbi/Open-Patients. Sources come from the `_id` prefix (trec-cds-, trec-ct-, usmle-, pmc-);
lengths are NLTK word_tokenize tokens of `description`, which reproduces the published averages.

Needs nltk with the punkt_tab data (nltk.download('punkt_tab')) and huggingface_hub (or the dataset in the HF cache)."""
import collections
import json

from huggingface_hub import hf_hub_download
from nltk.tokenize import word_tokenize

PUBLISHED = {"TREC Clinical Decision Support": (90, 105.1), "TREC Clinical Trials": (125, 137.7),
             "MedQA-USMLE": (12893, 135.8), "PMC-Patients": (167034, 484)}


def source(_id):
    if _id.startswith("trec-cds"): return "TREC Clinical Decision Support"
    if _id.startswith("trec-ct"): return "TREC Clinical Trials"
    if _id.startswith("usmle"): return "MedQA-USMLE"
    if _id.startswith("pmc"): return "PMC-Patients"
    raise ValueError(_id)


path = hf_hub_download("ncbi/Open-Patients", "Open-Patients.jsonl", repo_type="dataset")
n, tokens = collections.Counter(), collections.Counter()
for line in open(path):
    x = json.loads(line)
    s = source(x["_id"])
    n[s] += 1
    tokens[s] += len(word_tokenize(x["description"]))

print(f"{'Source':32} {'# Patients':>10} {'published':>10} {'Avg tokens':>11} {'published':>10}")
for s, (pub_n, pub_len) in PUBLISHED.items():
    print(f"{s:32} {n[s]:>10,} {pub_n:>10,} {tokens[s] / n[s]:>11.2f} {pub_len:>10}")
print(f"{'Total':32} {sum(n.values()):>10,}")
