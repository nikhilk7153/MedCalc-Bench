# Replay check: was the time-point instruction in the prompts of the final runs? (2026-10-24)

The main-run output files store the bare question, not the prompt. OpenAI kept no copies (0 stored completions;
`store=true` was never set), and vLLM does not log prompts. So the prompts were checked by replaying a run.

## Method
Meta-Llama-3-8B-Instruct was re-run on all 1,048 rows in all 3 prompt styles.
- **Replay:** the exact final command and files, run on GPU 0:
  - `run.py` as used for the final runs (md5 5f7e8a9d…, kept on the instance as `run_pre_fit_context.py`);
  - `dataset/test_data_10_22.csv` as run (md5 e53d936a…);
  - `final_files_2026-10-16/one_shot_10_17_2026.json` as run (md5 b3ddc1c7…).
- **Control:** the same, but with the Time Point Instruction column emptied, run on GPU 1.

Both outputs were compared with `final_runs/test_data_10_22/outputs/` (exact string equality of the raw output).

## Result: share of rows whose output is identical to the original run
| Style | Rows | Replay (with instruction) | Control (instruction removed) |
|---|---|---|---|
| Direct | 768 with instruction | 768 (100%) | 442 (58%) |
| Direct | 280 without | 280 (100%) | 280 (100%) |
| Zero-shot | 768 with instruction | 768 (100%) | 4 (0.5%) |
| Zero-shot | 280 without | 280 (100%) | 279 (99.6%) |
| One-shot | 768 with instruction | 753 (98%) | 278 (36%) |
| One-shot | 280 without | 255 (91%) | 251 (90%) |

## Reading
- **The replay reproduces the original byte for byte** in direct and zero-shot, and in 98% of one-shot rows. The remaining one-shot
  differences also occur on rows without the instruction (91%), so they come from vLLM batching variation in long generations,
  not from a prompt difference.
- **Removing the instruction breaks the match** only on the 768 rows that have it (as low as 0.5%). On the other 280 rows the
  control still matches. So the original prompts contained the instruction on exactly those rows.
- **For the GPT models,** which cannot be replayed (retired), the saved code-augmented conversations show the instruction in
  every applicable prompt. The main GPT runs used the same prompt code.

Folders: `replay_with_instruction/`, `control_without_instruction/`, `logs/`.
