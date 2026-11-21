# Fine-tuning re-run (paper Appendix C) on the corrected data, 2026-10-27

Recipe as in Appendix C.2: Mistral-7B and Llama-2-7B, CodeAct's Megatron-LLM training scripts on one 4xA100 40GB node,
chatML, loss on the assistant turn only, packing, flash attention, 2 epochs, global batch 32, lr 1e-5, 50 warmup steps,
cosine decay (CodeAct's scripts also set min lr 1e-6).

- Training code: `~/Downloads/sft_2026-10-27/code-act` (github.com/xingyaoww/code-act, commit d607f56; submodule
  Megatron-LLM at 1b07577). Its conversion scripts take `Llama-2-7b-hf` and the base `Mistral-7B-v0.1`.
- `build_sft_data.py` writes `data/train_chatml.jsonl` (9,871 rows of `dataset/train_data_10_22.csv`, as the paper used
  its whole training set) and `data/test_chatml.jsonl` (the 1,048 corrected test rows). No training row has the
  calculator and the note ID or note text of a test row or one-shot example; the script checks this and stops if one
  does. `data/clean/` is the same with `--dedupe` (repeated calculator + note removed). Rows with implausible v1.0
  inputs stay, as the training inputs are not verified.
- `score_sft.py` scores "...Answer: <value>" outputs with `evaluation/run.py`'s `score()`. Check: the ground truth
  written in that format scores 1,048/1,048.

About 12M tokens per epoch, so roughly 45 minutes of training per model at CodeAct's reported 9k tokens/s.

Not run yet: needs a 4xA100 40GB machine.
