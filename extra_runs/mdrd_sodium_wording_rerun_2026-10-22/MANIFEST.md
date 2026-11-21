# Re-run of the MDRD and Sodium Correction rows after the question rewording (2026-10-22)

Separate from the main results. Nothing here is merged into `final_runs/`.

## Why these rows
- **20 MDRD rows (calculator 9, rows 141-160):** the question now states the race coefficient.
  - Old: "... If the patient is black, please use the MDRD GFR Equation for blacks."
  - New: "... If the patient is Black, apply the race coefficient of 1.212; otherwise, use a race coefficient of 1."
- **20 Sodium Correction rows (calculator 26):** the question now states Hillier's correction factor.
  - Old: "... Use the sodium correction equation based on the one derived in Hillier, 1999."
  - New: "... Use the sodium correction equation based on the one derived in Hillier, 1999, with a correction factor of 0.024."
- Also fixed: "Hiller, 1999" -> "Hillier, 1999" in the 20 Sodium Correction explanations and in the calculator 26
  one-shot example.

Labels are unchanged. Two MDRD notes describe Black patients (rows 143 and 152); both already used 1.212.

The one-shot JSON stores no question. `run.py` builds the one-shot "example task" from the test row's question
(except calculator 24), so the example task carries the new wording too.

## Inputs (`input/`)
| File | md5 | What |
|---|---|---|
| `test_data_10_22.csv` | d880d87f79156aafc902ba657d9dae27 | Full test set after the rewording and the Hillier fix, 1,048 rows |
| `test_rows.csv` | 1f5a0bd7e9332e2997ced8dcc4048934 | The 40 rows above, copied unchanged (identical in `test_data_10_22_corrected.csv`) |
| `one_shot_10_17_2026_corrected.json` | 3855d2fe15099775655fff89a92200e7 | Corrected one-shot file with the calculator 26 "Hillier" fix; the calculator 9 and 26 examples are identical in `one_shot_10_17_2026.json` |
| `as_text.py` | 19cdf79f90b69bb88110b497afcee5c7 | Reads the label columns as text, as when the full test set is read |
| `queue_h100.sh`, `queue_h100.log` | | Open-model queue and its log |

## Code (same files as the final runs)
| File | md5 |
|---|---|
| `evaluation/run.py` | 5f7e8a9d5b4f3b95ebeae90fe27fdac4 (identical on the GPU instance) |
| `evaluation/evaluate.py` | b2f3e1ad90b2bffc5808f72ba247783a (identical on the GPU instance) |
| `evaluation/code_exec_timed.py` | e7da8218d1d4961ee95aea0f88721f75 |
| `evaluation/error_analysis_batch.py` | 73dce14b27b91b8d59412e40ad186255 |

## Runs
Settings as in `final_runs/test_data_10_22/`, zero-shot and one-shot only (no direct answer):
| Models | Prompt styles | Engine |
|---|---|---|
| `gpt-3.5-turbo-0125`, `gpt-4-0613` | zero_shot, one_shot (40 rows) | `run.py`, OpenAI SDK 2.8.1, temperature 0 |
| `gpt-3.5-turbo-0125`, `gpt-4-0613` | code-augmented (40 rows) | `code_exec_timed.py`, OpenAI SDK 0.28.0 (paper's environment) |
| Meta-Llama-3-70B-Instruct, meditron-70b, Mixtral-8x7B-Instruct-v0.1 | zero_shot, one_shot (40 rows) | `run.py`, vLLM 0.32.0, greedy; tensor parallel 2 on an NVLink pair of H100 NVL |
| Meta-Llama-3-8B-Instruct, Mistral-7B-Instruct-v0.2, PMC_LLaMA_13B | zero_shot, one_shot (40 rows) | `run.py`, vLLM 0.32.0, greedy; 1 H100 NVL |

transformers 5.19.0, torch 2.13.0+cu130, vast.ai container 56977929. PMC-LLaMA notes are truncated to 256 tokens, as
in the original `run.py`.

Each model has its own folder: `outputs/` (row-level answers), `results/` (accuracy) and `logs/`.

## Accuracy, old prompt (main runs) vs new prompt (this re-run), % of 20 rows
| Model | Setting | MDRD old | MDRD new | Sodium old | Sodium new |
|---|---|---|---|---|---|
| gpt-4-0613 | zero-shot | 5 | 0 | 100 | 90 |
| gpt-4-0613 | one-shot | 20 | 15 | 90 | 90 |
| gpt-4-0613 | code-augmented | 95 | 100 | 100 | 95 |
| gpt-3.5-turbo-0125 | zero-shot | 15 | 15 | 90 | 75 |
| gpt-3.5-turbo-0125 | one-shot | 10 | 0 | 100 | 100 |
| gpt-3.5-turbo-0125 | code-augmented | 20 | 65 | 85 | 95 |
| Llama-3-70B | zero-shot | 5 | 0 | 70 | 100 |
| Llama-3-70B | one-shot | 5 | 0 | 100 | 100 |
| Llama-3-8B | zero-shot | 0 | 0 | 50 | 70 |
| Llama-3-8B | one-shot | 10 | 5 | 80 | 75 |
| Mixtral-8x7B | zero-shot | 10 | 5 | 50 | 55 |
| Mixtral-8x7B | one-shot | 0 | 0 | 85 | 85 |
| Mistral-7B | zero-shot | 0 | 0 | 35 | 40 |
| Mistral-7B | one-shot | 5 | 5 | 75 | 70 |
| meditron-70b | zero-shot | 0 | 0 | 0 | 0 |
| meditron-70b | one-shot | 5 | 5 | 80 | 80 |
| PMC-LLaMA-13B | zero-shot | 0 | 0 | 0 | 0 |
| PMC-LLaMA-13B | one-shot | 0 | 0 | 75 | 75 |

"Old" is the main run (`final_runs/test_data_10_22/`; code-augmented: `evaluation/outputs/test_data_10_22/`) on the same
40 rows. With 20 rows per cell, one row is 5 points.

## Error analysis (`error_analysis/`)
- **Inputs:** every wrong zero-shot and one-shot answer from all 8 models on the 40 rows, 409 requests in 16 CSVs.
- **Reference text:** the Ground Truth Explanation from `input/test_rows.csv`.
- **Classifier:** the paper's `error_analysis.py` prompt word for word, model `gpt-4` (resolves to gpt-4-0613),
  temperature 0, through the OpenAI Batch API (`batch_input.jsonl`, `batch_job.json`).
- **Results:** one `<setting>_<model>.csv.error_type.json` per CSV, keyed `<Calculator ID>_<Note ID>`.
  Batch `batch_6adaa1e125208190a49943872160dcb4`: 409 of 409 completed, 0 failed, all outputs parsed as JSON.

Error types summed over the 8 models (an answer can have more than one type; A = wrong calculator/knowledge,
B = wrong parameters, C = arithmetic, D = no clear error). "Old" is `evaluation/error_analysis_10_22/` on the same rows.
| Calculator | Setting | Wrong answers old / new | A old / new | B old / new | C old / new | D old / new |
|---|---|---|---|---|---|---|
| MDRD | zero-shot | 153 / 156 | 83 / 64 | 38 / 32 | 47 / 70 | 22 / 22 |
| MDRD | one-shot | 149 / 154 | 35 / 29 | 47 / 48 | 93 / 101 | 1 / 0 |
| Sodium | zero-shot | 81 / 74 | 43 / 33 | 26 / 26 | 5 / 10 | 35 / 33 |
| Sodium | one-shot | 23 / 25 | 2 / 4 | 9 / 10 | 18 / 21 | 0 / 0 |

## Checks done
- Every output has the expected 40 rows, the new question wording for every row, and labels equal to the test set.
- Re-scoring every answer reproduces the stored verdicts (0 mismatches). There were no API errors.
- The 12 open-model files are byte-identical to the GPU instance copies.
- Answers can differ from the final runs for reasons other than the wording: run-to-run variation (API
  nondeterminism; vLLM batch composition).
