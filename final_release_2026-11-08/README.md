# MedCalc-Bench correction: finalized dataset, outputs and error analysis (2026-11-08)

Everything the corrected paper's numbers come from, in one place. Built by
`final_runs/consolidated_2026-11-06/build_final_release.py` (copies only; rerun it to rebuild this folder). The scripts
that produce each file are in `final_runs/consolidated_2026-11-06/` (see its README). `MANIFEST.csv` lists every file
with its source, row count and SHA-256.

## dataset/
| File | Contents |
|---|---|
| `test_data.csv` | Revised test set: 1,048 rows, 55 calculators. Notes, questions and Time Point Instruction from `dataset/test_data_10_22.csv`; Relevant Entities, labels, limits and explanations from `dataset/test_data_10_22_corrected.csv`. Every table and every output is scored against this file |
| `one_shot_examples.json` | One-shot examples (55). Same as the main run's file except the CKD-EPI example wording (used by the CKD-EPI one-shot re-run) |
| `train_data.csv` | Training set, 9,871 rows, 40 calculators (open items below) |
| `CHANGELOG.md` | Dataset change log |

## outputs/
One output per model, prompt style and row (1,048 each), each with a `Source Run` field.
- 24 files: 8 models × `direct_answer` / `zero_shot` / `one_shot`.
- `code_exec_gpt-3.5-turbo-0125.jsonl`, `code_exec_gpt-4-0613.jsonl`: code-interpreter prompt.
- `accuracy_by_category.csv`: the accuracy behind the main table.

Which run each row comes from:

| Rows | Output |
|---|---|
| Default | Main run, `final_runs/test_data_10_22/` |
| CKD-EPI rows 21–40, one-shot | `extra_runs/corrected_rows_ckdepi_rerun_2026-10-22/` |
| MDRD and Sodium rows 141–160, 366–385, zero-shot, one-shot and code | `extra_runs/mdrd_sodium_wording_rerun_2026-10-22/` |
| 13 label-fix rows | Main-run answers, re-scored |

## error_analysis/
- `error_classifications.csv`: every wrong zero- and one-shot answer (12,059), with its `gpt-4` error types and status.
  - Used: 11,793. That is 11,647 `classified` plus the 146 CKD-EPI zero-shot answers classified against the old explanation wording (logic unchanged).
  - Excluded: 266. That is 124 label-fix answers classified against the old explanation, 30 newly wrong ones, and 112 empty or blocked classifications.
- `error_type_summary.csv`, `error_type_table_values.csv`: per model and style, each error counted once as its earliest type (A > B > C > D).

## figures/
Appendix E.1 Figures 2 and 3 (accuracy vs. number of parameters, zero-shot), their points and fit coefficients.

## tables/
LaTeX for the main results, error types, dataset statistics, code interpreter, token counts (B.3) and training set (C).

## Open items, not yet reflected here
- **Astra corrections** (89 input corrections, 15 note edits): set aside pending physician review. Applying them would change the test set and the scores.
- **Blood-gas rows:** 1,048 rows kept. The alternative is dropping 9.
- **Training set:**
  - de-duplication against the test set and one-shot examples is undecided (173 / 144 / 72 rows);
  - the LDL questions use the Friedewald wording while the test set keeps v1.0's;
  - about 50 Serum Osmolality rows take urea in mg/dL as BUN;
  - 2 steroid rows convert a drug to itself;
  - see `final_runs/consolidated_2026-11-06/verify_v10_fixes.csv`.
