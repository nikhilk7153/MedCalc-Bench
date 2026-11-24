# Fine-tuning re-run (paper Appendix C) on the corrected data, 2026-10-27

Recipe as in Appendix C.2: Mistral-7B and Llama-2-7B, CodeAct's Megatron-LLM training scripts on one 4xA100 40GB node,
chatML, loss on the assistant turn only, packing, flash attention, 2 epochs, global batch 32, lr 1e-5, 50 warmup steps,
cosine decay (CodeAct's scripts also set min lr 1e-6).

- Training code: `~/Downloads/sft_2026-10-27/code-act` (github.com/xingyaoww/code-act, commit d607f56; submodule
  Megatron-LLM at 1b07577). Its conversion scripts take `Llama-2-7b-hf` and the base `Mistral-7B-v0.1`.
- `build_sft_data.py` writes `data/train_chatml.jsonl` (9,871 rows of `dataset/train_data_10_22.csv`, as the paper used
  its whole training set) and `data/test_chatml.jsonl` (the 1,048 rows of the release test set, `final_release_2026-11-08/dataset/test_data.csv`:
the same prompts every Table 2 baseline saw). No training row has the
  calculator and the note ID or note text of a test row or one-shot example; the script checks this and stops if one
  does. `data/clean/` is the same with `--dedupe` (repeated calculator + note removed). Rows with implausible v1.0
  inputs stay, as the training inputs are not verified.
- `score_sft.py` scores "...Answer: <value>" outputs with `evaluation/run.py`'s `score()`. Check: the ground truth
  written in that format scores 1,048/1,048.

- `megatron/run_sft.sh <Mistral-7b|Llama-2-7b>` (with `CODEACT=<code-act checkout>`, inside CodeAct's Megatron docker
  image) runs the whole training: Hugging Face -> Megatron conversion and sharding, chatML -> packed Megatron data,
  training with the paper's settings, and conversion of the last checkpoint back to Hugging Face. It replaces CodeAct's
  `finetune_4xA100_4tp_mixture_*.sh`: MedCalc data, 2 epochs instead of 5, the packed example count read from
  preprocessing, and two broken paths in CodeAct fixed (the Llama-2 script's `source.sh`, and `add_chat_format.py`
  in the conversion back). See the script's header for every setting.
- Packing length: CodeAct uses 16,384 (Mistral) and 4,096 (Llama-2); the paper gives none. Megatron-LLM's own
  preprocessing on `data/train_chatml.jsonl` (run on CPU 2026-11-24, GPU-only imports stubbed) packs it into:
  Mistral 815 examples at 16,384 (52 steps over 2 epochs, so the 50 warmup steps take almost the whole run) or 3,967 at
  4,096 (248 steps); Llama-2 4,096 examples at 4,096 (258 steps). Recommended: `SEQ_LENGTH=4096` for both. No training
  conversation is longer than about 4,000 tokens, so nothing is truncated.

About 12.7M tokens per epoch (13.0M with the Llama-2 tokenizer), so roughly 45 minutes of training per model at
CodeAct's reported 9k tokens/s.

- `generate_sft.py --model <model> --out outputs/<name>.jsonl` writes the test responses `score_sft.py` scores: the
  chatML test prompts in exactly the training format, greedy decoding, up to 4,096 new tokens within the model's context.
  The same script gives the baselines when pointed at the base models. Every test prompt fits (longest about 2,950
  tokens with the Llama-2 tokenizer, leaving at least 1,000 tokens to answer).

Order on the GPU machine: `megatron/run_sft.sh` (per model) -> `generate_sft.py` (fine-tuned and base models) ->
`score_sft.py`. Not run yet: needs a 4xA100 40GB machine and vllm.
