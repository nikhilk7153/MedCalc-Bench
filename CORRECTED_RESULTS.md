# Corrected MedCalc-Bench runs (branch `corrected-results-2026-10-22`)

Re-run of the MedCalc-Bench (NeurIPS 2024) experiments on a corrected test set. `PROJECT_INDEX.csv` lists every file
with its status (CURRENT / EXPERIMENTAL / SUPERSEDED / REFERENCE). Items marked BACKUP, the old superseded runs and
`.env` are not in this branch.

## Main results
- **Test set:** `dataset/test_data_10_22.csv` (1,048 rows, 55 calculators).
- **One-shot examples:** `final_files_2026-10-16/one_shot_10_17_2026.json`.
- **Runs:** 8 models x direct / zero-shot / one-shot, in `final_runs/test_data_10_22/`.
- **Code-augmented** (GPT-3.5 / GPT-4, openai==0.28.0 as in the paper): `evaluation/outputs/test_data_10_22/`.
- **Error analysis** (paper's prompt, gpt-4, temperature 0): `evaluation/error_analysis_10_22/`.

## Corrections (not yet physician-reviewed)
- `dataset/test_data_10_22_corrected.csv` has 13 label fixes and the CKD-EPI explanation wording fix.
  - `final_runs/test_data_10_22_corrected/` holds the original answers re-scored against it.
  - `extra_runs/corrected_rows_ckdepi_rerun_2026-10-22/` holds all models re-run on the 33 affected rows.
- `dataset/test_data_10_22_astra.csv` additionally applies the pending gpt-6-astra corrections (experimental).
  - `extra_runs/astra_corrections_rerun_2026-10-22/` holds the re-run on those 45 rows.
- Review list: `dataset/label_corrections_review_10_22.csv`.

Each `extra_runs/` folder has a `MANIFEST.md` with exact inputs, code checksums, model versions and settings.
