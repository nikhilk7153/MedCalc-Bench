# Re-run of the LDL rows with the Friedewald question (2026-10-24)

Separate from the main results. Nothing here is merged into `final_runs/`.

## Why these rows
The LDL calculator (ID 44, 20 rows) was renamed and its question now names the method:
- Name: "LDL Calculated" -> "Friedewald Equation for Low-Density Lipoprotein Cholesterol (LDL-C)".
- Question: "What is the patient's LDL cholestrol concentration? Please outuput your answer in terms of mg/dL." ->
  "Using the Friedewald equation, what is the patient's LDL cholesterol concentration? Please output your answer in terms of mg/dL."

The prompts contain the question but not the calculator name, so all three prompt styles change for these rows. Labels are unchanged.
The change is in `dataset/test_data_10_22_corrected.csv`, `dataset/test_data_10_22_astra.csv`, the corrected one-shot CSV and
`calculator_implementations_verified/name_to_python.json`. It is **not** in `dataset/test_data_10_22.csv`.

## Inputs (`input/`)
| File | md5 | What |
|---|---|---|
| `test_data_10_22_corrected.csv` | 394de441595e8561667cf12ad8532189 | Full corrected test set at the time of the run |
| `test_rows.csv` | de1603fdb8dd83d7d353a833687f07c3 | The 20 LDL rows, copied unchanged from it |
| `one_shot_10_17_2026_corrected.json` | 63c12391ea4f399ac1a75ff6e9fe596d | Corrected one-shot file (also contains the corrected SOFA example; the LDL example is unchanged) |
| `name_to_python.json` | 78e6363defc163df15156d639cf3b958 | Calculator names and questions after the rename |
| `as_text.py` | 19cdf79f90b69bb88110b497afcee5c7 | Reads the label columns as text, as when the full test set is read |
| `queue_h100.sh`, `queue_h100.log` | | Open-model queue (also ran the SOFA one-shot rows) and its log |

## Runs
- **Models:** the six open models, direct / zero-shot / one-shot on the 20 rows.
- **Code and environment:** the same `run.py`/`evaluate.py` as the final runs (identical md5 on the GPU instance); vLLM 0.32.0, greedy,
  on H100 NVL; 70B models use an NVLink pair (tp 2); PMC-LLaMA notes are truncated to 256 tokens, as in the original `run.py`.
- **Not run:** GPT-4 (0613) and GPT-3.5 (0125). OpenAI retired them and every GPT-3.5 alias by 2026-10-24, so `code_exec/` is empty.

## Layout
| Folder | What |
|---|---|
| `outputs/<model>/` | Row-level answers (direct, zero-shot, one-shot) |
| `results/<model>/` | Accuracy by category |
| `logs/<model>/` | Run log |
| `code_exec/` | Empty: code-augmented runs need GPT-3.5/GPT-4 |
| `error_analysis/` | Empty: the paper's classifier (`gpt-4`) is retired; to be decided |

All 42 files are byte-identical to the instance copies. Every output has the 20 rows, with labels and questions equal to
`input/test_rows.csv`; re-scoring reproduces every verdict.

## Correct answers on the 20 LDL rows, old question (main run) -> Friedewald question (this re-run)
| Model | Direct | Zero-shot | One-shot |
|---|---|---|---|
| Meta-Llama-3-70B-Instruct | 3 -> 5 | 20 -> 20 | 12 -> 9 |
| Meta-Llama-3-8B-Instruct | 0 -> 3 | 8 -> 5 | 3 -> 4 |
| Mistral-7B-Instruct-v0.2 | 0 -> 2 | 3 -> 2 | 0 -> 3 |
| Mixtral-8x7B-Instruct-v0.1 | 3 -> 2 | 11 -> 14 | 6 -> 8 |
| PMC_LLaMA_13B | 0 -> 0 | 0 -> 0 | 1 -> 1 |
| meditron-70b | 0 -> 0 | 0 -> 0 | 4 -> 4 |
