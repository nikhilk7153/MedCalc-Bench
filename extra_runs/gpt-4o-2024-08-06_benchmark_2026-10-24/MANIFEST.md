# GPT-4o (gpt-4o-2024-08-06) on MedCalc-Bench, 2026-10-24

An extra model, run on the exact files the paper's 8 models used. It is not a replacement for GPT-4 (0613).
- **Test set:** `input/test_data_10_22_as_run.csv`, md5 e53d936a15725196eae1272d132bd302 (1,048 rows).
- **One-shot examples:** `input/one_shot_10_17_2026_as_run.json`, md5 b3ddc1c7f7fc62d21f2fbff0b8c80480.
- **Direct / zero-shot / one-shot:** `input/run.py` (md5 8192b62b…), OpenAI SDK 2.8.1, temperature 0. Outputs also record the
  time-point instruction and the full prompt.
- **Code-augmented:** `input/code_exec_timed.py` (md5 e7da8218…), openai==0.28.0 (the paper's SDK), temperature 0.

Folders: `outputs/`, `results/` and `logs/` (direct / zero-shot / one-shot), plus `code_exec/{outputs,results,logs}`.

**Checks:** all four runs have 1,048 unique rows. Every response came from `gpt-4o-2024-08-06`, there were no API errors, and
re-scoring reproduces every verdict. No error analysis was run (the paper's classifier, GPT-4 0613, is retired).
