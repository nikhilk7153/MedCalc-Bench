# Open-model re-runs, 2026-10-24: SOFA one-shot example and LDL Friedewald question

The six open models re-run on two sets of updated rows. Nothing here is merged into `final_runs/`.
GPT-4 (0613) and GPT-3.5 (0125) were not run: OpenAI retired both, and every GPT-3.5 alias, by 2026-10-24.
GPT-3.5's SOFA one-shot re-run (made through an alias the day before) is in `../sofa_oneshot_rerun_2026-10-23/`.

## Layout
| Folder | What |
|---|---|
| `sofa_oneshot/outputs/<model>/`, `results/<model>/`, `logs/<model>/` | 20 SOFA rows, one-shot, corrected SOFA example |
| `ldl_friedewald/outputs/<model>/`, `results/<model>/`, `logs/<model>/` | 20 LDL rows, direct / zero-shot / one-shot, Friedewald question |
| `ldl_friedewald/code_exec/` | Empty: code-augmented runs need GPT-3.5/GPT-4 (retired) |
| `ldl_friedewald/error_analysis/` | Empty: the paper's classifier (`gpt-4`) is retired; to be decided |
| `input/` | Inputs of both runs, plus the shared queue script and its log |

## 1. SOFA one-shot example (calculator 43, rows 889-908)
- **What changed:** the one-shot example for SOFA scored an unreported urine output as 0 mL/day (answer 9). It was corrected on
  2026-10-23: urine output is not reported, and the kidneys score on creatinine 3.12 alone (answer 7).
- **Prompts affected:** only the one-shot prompt of the 20 SOFA rows. The test labels are unchanged.
- **Inputs:** the same files as the GPT-3.5 SOFA re-run, copied from `../sofa_oneshot_rerun_2026-10-23/input/`:
  - `input/sofa_oneshot/test_rows.csv`, md5 45be60ae5c824e79e9b75e9b32651c89;
  - `input/sofa_oneshot/one_shot_10_17_2026.json`, md5 769a266c921fd4da4e4e171a3945f811.
  The full test set is `../sofa_oneshot_rerun_2026-10-23/input/test_data_10_22.csv`.

| Model | Old example (main run) | Corrected example | Answers that changed |
|---|---|---|---|
| Meta-Llama-3-70B-Instruct | 2/20 | 2/20 | 17/20 |
| Meta-Llama-3-8B-Instruct | 3/20 | 4/20 | 17/20 |
| Mistral-7B-Instruct-v0.2 | 1/20 | 0/20 | 14/20 |
| Mixtral-8x7B-Instruct-v0.1 | 2/20 | 1/20 | 15/20 |
| PMC_LLaMA_13B | 0/20 | 0/20 | 7/20 |
| meditron-70b | 4/20 | 2/20 | 16/20 |

## 2. LDL Friedewald question (calculator 44, 20 rows)
- **Name:** "LDL Calculated" -> "Friedewald Equation for Low-Density Lipoprotein Cholesterol (LDL-C)".
- **Question:** "What is the patient's LDL cholestrol concentration? Please outuput your answer in terms of mg/dL." ->
  "Using the Friedewald equation, what is the patient's LDL cholesterol concentration? Please output your answer in terms of mg/dL."
- **Where the change is:** `dataset/test_data_10_22_corrected.csv`, `dataset/test_data_10_22_astra.csv`, the corrected one-shot CSV
  and `calculator_implementations_verified/name_to_python.json`. It is **not** in `dataset/test_data_10_22.csv`.
- **Prompts affected:** all three prompt styles (prompts carry the question, not the name). Labels are unchanged.

**Inputs (`input/ldl_friedewald/`):**
| File | md5 |
|---|---|
| `test_rows.csv` | de1603fdb8dd83d7d353a833687f07c3 |
| `test_data_10_22_corrected.csv` | 394de441595e8561667cf12ad8532189 |
| `one_shot_10_17_2026_corrected.json` | 63c12391ea4f399ac1a75ff6e9fe596d |
| `name_to_python.json` | 78e6363defc163df15156d639cf3b958 |
| `as_text.py` | 19cdf79f90b69bb88110b497afcee5c7 |

**Correct answers on the 20 LDL rows, old question (main run) -> Friedewald question:**
| Model | Direct | Zero-shot | One-shot |
|---|---|---|---|
| Meta-Llama-3-70B-Instruct | 3 -> 5 | 20 -> 20 | 12 -> 9 |
| Meta-Llama-3-8B-Instruct | 0 -> 3 | 8 -> 5 | 3 -> 4 |
| Mistral-7B-Instruct-v0.2 | 0 -> 2 | 3 -> 2 | 0 -> 3 |
| Mixtral-8x7B-Instruct-v0.1 | 3 -> 2 | 11 -> 14 | 6 -> 8 |
| PMC_LLaMA_13B | 0 -> 0 | 0 -> 0 | 1 -> 1 |
| meditron-70b | 0 -> 0 | 0 -> 0 | 4 -> 4 |

## Code, environment and checks
- **Code:** `evaluation/run.py` (md5 5f7e8a9d…) and `evaluate.py` (md5 b2f3e1ad…), identical to the final runs and on the GPU instance.
- **Environment:** vLLM 0.32.0, greedy, on H100 NVL. 70B models use an NVLink pair (tp 2). PMC-LLaMA notes are truncated to 256 tokens.
- **Queue:** `input/queue_h100.sh` ran both re-runs (md5 258560c5232bf8993634c3f48fc4ae6f); `input/queue_h100.log` covers 14:11-14:26 UTC.
- **Checks:**
  - All 60 result files are byte-identical to the GPU instance copies (also in `gpu_backup_2026-10-22/`).
  - Every output has its 20 rows, with labels and questions equal to its input.
  - Re-scoring reproduces every verdict.
