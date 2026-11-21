# Main results with the reworded MDRD and Sodium Correction rows merged in (2026-10-22)

The main runs (`final_runs/test_data_10_22/`, code-augmented: `evaluation/outputs/test_data_10_22/`) with the 40
MDRD (calculator 9) and Sodium Correction (calculator 26) rows replaced by the re-run on the reworded questions
(`extra_runs/mdrd_sodium_wording_rerun_2026-10-22/`). The source folders are unchanged.

## What changed in the questions
- MDRD: "If the patient is black, please use the MDRD GFR Equation for blacks." ->
  "If the patient is Black, apply the race coefficient of 1.212; otherwise, use a race coefficient of 1."
- Sodium Correction: "... derived in Hillier, 1999." -> "... derived in Hillier, 1999, with a correction factor of 0.024."
- "Hiller, 1999" -> "Hillier, 1999" in the Sodium Correction explanations and its one-shot example.

Labels are unchanged. The test set these outputs correspond to is `dataset/test_data_10_22.csv`
(md5 d880d87f79156aafc902ba657d9dae27).

## Contents
| Path | What |
|---|---|
| `outputs/` | 16 zero/one-shot runs (8 models) + 2 code-augmented runs (GPT-3.5, GPT-4), 1,048 rows each |
| `results/` | Accuracy by category for each run (`table_stats.compute_overall_accuracy`), tag `test_data_10_22_reworded` |
| `accuracy_summary.csv` | Overall, MDRD and Sodium accuracy per run, main vs merged |
| `error_analysis/` | 16 zero/one-shot CSVs and `.error_type.json` files: the main error analysis with the calculator 9 and 26 entries replaced by the re-run's (12,079 - 406 + 409 = 12,082 entries) |
| `merge_reworded.py` | The script that built this folder (md5 d86dcda18d8c1355626065c46b1d5f3a) |

Direct-answer runs are not included: they were not re-run, so their 40 rows still have the old wording.

## Overall accuracy, main -> merged (%)
| Model | Zero-shot | One-shot | Code-augmented |
|---|---|---|---|
| gpt-4-0613 | 45.04 -> 44.75 | 62.02 -> 61.93 | 62.02 -> 62.02 |
| gpt-3.5-turbo-0125 | 30.44 -> 30.15 | 41.89 -> 41.70 | 42.27 -> 43.32 |
| Meta-Llama-3-70B-Instruct | 40.84 -> 41.32 | 45.52 -> 45.42 | |
| Meta-Llama-3-8B-Instruct | 16.60 -> 16.98 | 32.63 -> 32.44 | |
| Mixtral-8x7B-Instruct-v0.1 | 23.57 -> 23.57 | 36.55 -> 36.55 | |
| Mistral-7B-Instruct-v0.2 | 12.31 -> 12.40 | 22.14 -> 22.04 | |
| meditron-70b | 0.19 -> 0.19 | 26.62 -> 26.62 | |
| PMC_LLaMA_13B | 0.10 -> 0.10 | 10.97 -> 10.97 | |

Per-calculator numbers for the 40 rows are in `accuracy_summary.csv` and in the re-run's `MANIFEST.md`.

## Checks done
- Every merged output has 1,048 rows in the main run's order; every row outside calculators 9 and 26 is
  byte-identical to the main run; all 40 replaced rows have the new question wording.
- The "Main accuracy" column reproduces the stored main results files.
- Every error-analysis JSON has exactly one entry per row of its CSV.
