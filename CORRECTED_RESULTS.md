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

## Later wording changes and re-runs (each in its own folder, not merged into the main results)
- **Question and explanation wording:** MDRD (race coefficient), Sodium Correction (0.024 factor) and LDL (Friedewald)
  questions, and the "Hillier" spelling, are updated in the datasets.
  - `extra_runs/mdrd_sodium_wording_rerun_2026-10-22/` holds the MDRD/Sodium re-run.
  - `final_runs/test_data_10_22_reworded/` holds the main runs with those 40 rows merged in.
- **SOFA one-shot example:** corrected in the one-shot files (answer 9 -> 7).
  - `extra_runs/sofa_oneshot_rerun_2026-10-23/` has GPT-3.5 only.
  - `extra_runs/oss_rerun_2026-10-24/` has the six open models for SOFA and for the Friedewald LDL question.
  - The GPT-4 / GPT-3.5 0125 snapshots are retired, so they could not be re-run.
- **PMC-LLaMA longer context:** `extra_runs/pmc_llama_fit_context_2026-10-24/` (`run.py --fit_context`).
- **Audit copies:** `final_runs/test_data_10_22_with_prompts/` (main outputs plus the exact prompts) and
  `extra_runs/time_point_replay_check_2026-10-24/`.
- **Extra model:** GPT-4o (2024-08-06), all settings plus code-augmented, in
  `extra_runs/gpt-4o-2024-08-06_benchmark_2026-10-24/`.

## Appendix E.1 (accuracy vs. number of attributes)
- **Script:** `evaluation/attribute_analysis.py`.
  - Each calculator's attribute count comes from `calculator_implementations_verified/name_to_python.json`.
  - Accuracy at each count is fitted with a second-degree polynomial per model.
- **Figures:** the zero-shot equation- and rule-based figures are in `evaluation/attribute_analysis_10_22/`, with the
  plotted points and fit coefficients. They replace the paper's Figures 2-3, whose values did not agree with Table 2.
- **Model files:** `session_archive_2026-10-22/model_verification_2026-10-24.jsonl` records SHA-256 checks of all
  open-model weights against Hugging Face.

Each `extra_runs/` folder has a `MANIFEST.md` with exact inputs, code checksums, model versions and settings.
