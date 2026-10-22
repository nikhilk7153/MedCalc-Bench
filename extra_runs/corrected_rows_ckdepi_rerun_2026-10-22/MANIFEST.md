# Re-run of the corrected rows and the CKD-EPI rows (2026-10-22)

Separate from the main results. Nothing here is merged into `final_runs/`.

## Why these rows
- **13 label-corrected rows:** 240, 241, 304, 318, 350, 360, 362, 447, 452, 461, 465, 500, 849.
  These are the inputs and labels fixed in `dataset/test_data_10_22_corrected.csv`; see
  `dataset/label_corrections_review_10_22.csv` for the evidence. Their prompts are unchanged; only the labels changed.
- **20 CKD-EPI rows (21-40):** their one-shot example's explanation was regenerated after the wording fix in
  `ckd-epi_2021_creatinine.py` ("greater than or equal to" -> "greater than"; "coefficient" spelling). Only these 20
  one-shot prompts differ from the final runs; their labels are unchanged.

## Inputs (`input/`)
| File | md5 | What |
|---|---|---|
| `test_data_10_22_corrected.csv` | 18dffe88f06dfb51a780337490b28fab | Full corrected test set (13 label fixes + CKD-EPI explanation wording), 1,048 rows |
| `test_rows.csv` | b3919362ebec07bc592f00526aa634ce | The 33 rows above, copied unchanged from the corrected test set |
| `test_rows_corrected_only.csv` | 07b7e870e74018f4235362a779dc4ffd | The 13 label-corrected rows (code-augmented runs) |
| `one_shot_10_17_2026_corrected.json` | ededd6264b1d449805069e8a0af7e9f5 | `one_shot_10_17_2026.json` with only the CKD-EPI (calculator 3) explanation regenerated; answer unchanged (39.99991) |
| `ckd-epi_2021_creatinine.py` | f3afbbed9dfa6187e8631a24c1412143 | Calculator code after the wording fix (logic unchanged) |
| `as_text.py` | 19cdf79f90b69bb88110b497afcee5c7 | Wrapper that reads the label columns as text, as when the full test set is read (row subsets contain only numeric labels) |
| `queue_h100.sh`, `queue_h100.log` | | Open-model queue and its log |

## Code (same files as the final runs)
| File | md5 |
|---|---|
| `evaluation/run.py` | 5f7e8a9d5b4f3b95ebeae90fe27fdac4 (identical on the GPU instance) |
| `evaluation/evaluate.py` | b2f3e1ad90b2bffc5808f72ba247783a (identical on the GPU instance) |
| `evaluation/code_exec_timed.py` | e7da8218d1d4961ee95aea0f88721f75 |
| `evaluation/error_analysis_batch.py` | 73dce14b27b91b8d59412e40ad186255 |
| `calculator_implementations_verified/*.py` (concatenated) | 3f5d8155d67f39ec426ccc4e2c309113 |

## Runs
Settings as in `final_runs/test_data_10_22/`:
| Models | Prompt styles | Engine |
|---|---|---|
| `gpt-3.5-turbo-0125`, `gpt-4-0613` | direct_answer, zero_shot, one_shot (33 rows) | `run.py`, OpenAI SDK 2.8.1, temperature 0 |
| `gpt-3.5-turbo-0125`, `gpt-4-0613` | code-augmented (13 rows) | `code_exec_timed.py`, OpenAI SDK 0.28.0 (paper's environment) |
| Meta-Llama-3-70B-Instruct, meditron-70b, Mixtral-8x7B-Instruct-v0.1 | 3 styles (33 rows) | `run.py`, vLLM 0.32.0, greedy; tensor parallel 2 on an NVLink pair of H100 NVL |
| Meta-Llama-3-8B-Instruct, Mistral-7B-Instruct-v0.2, PMC_LLaMA_13B | 3 styles (33 rows) | `run.py`, vLLM 0.32.0, greedy; 1 H100 NVL |

transformers 5.19.0, torch 2.13.0+cu130. PMC-LLaMA notes are truncated to 256 tokens, as in the original `run.py`.

Each model has its own folder: `outputs/` (row-level answers), `results/` (accuracy by category) and `logs/`.

## Error analysis (`error_analysis/`)
- **Inputs:** every wrong zero-shot and one-shot answer from all 8 models on the 33 rows, 457 requests in 16 CSVs.
- **Reference text:** the Ground Truth Explanation from `input/test_data_10_22_corrected.csv`.
- **Classifier:** the paper's `error_analysis.py` prompt word for word, model `gpt-4` (resolves to gpt-4-0613),
  temperature 0, through the OpenAI Batch API (`batch_input.jsonl`, `batch_job.json`).
- **Results:** one `<setting>_<model>.csv.error_type.json` per CSV, keyed `<Calculator ID>_<Note ID>`.

## Checks done
- Every output has the expected rows (33, or 13 for code-augmented) and labels equal to the corrected test set.
- Re-scoring every answer reproduces the stored verdicts (0 mismatches). There were no API errors.
- The 42 open-model files are byte-identical to the GPU instance copies.
- Prompts: the direct and zero-shot prompts are identical to the final runs; the one-shot prompts differ only for the 20 CKD-EPI rows.
- Answers can differ from the final runs even where the prompt is unchanged. This is run-to-run variation (API
  nondeterminism; vLLM batch composition), largest for Mixtral.
