# Main-run outputs with the time-point instruction and the prompts added (2026-10-24)

Copies of the 24 output files in `../test_data_10_22/outputs/` (8 models x direct / zero-shot / one-shot) with fields added.
**Every answer, verdict and other field is unchanged.** The original files are untouched, and their checksums still hold.

## Added fields
- `Time Point Instruction`: placed right after `Question`, taken from `dataset/test_data_10_22.csv` by row number
  (768 rows have it; the other 280 are empty).
- `Prompt Messages (reconstructed)`: the system and user messages, rebuilt with the exact final `run.py` (md5 5f7e8a9d…) from the
  as-run test set (md5 e53d936a…) and one-shot file (md5 b3ddc1c7…).
- `Prompt (reconstructed)`: open models only; the templated text the model received, using its tokenizer and chat template.

## Why "reconstructed", and why it can be trusted
The runs did not save their prompts (the output format of the original repository); `run.py` now saves them for future runs.
- **Consistency check:** for every row, the rebuilt patient note equals the note the run recorded (25,152 of 25,152, including
  PMC-LLaMA's 256-token truncated notes).
- **Replay check:** an exact replay of Llama3-8B reproduces the original outputs (100% direct and zero-shot, 98% one-shot);
  removing the instruction does not. See `../../extra_runs/time_point_replay_check_2026-10-24/MANIFEST.md`.
