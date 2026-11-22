# MedCalc-Bench correction: dataset, outputs and error analysis (2026-11-08)

> **Work in progress.** This release is not final. The open items at the end of this file (including the Astra
> corrections awaiting physician review) are not yet reflected here, and the dataset, outputs and numbers may still change.

Everything the corrected paper's numbers come from, in one place. Built by
`final_runs/consolidated_2026-11-06/build_final_release.py` (copies only; rerun it to rebuild this folder). The scripts
that produce each file are in `final_runs/consolidated_2026-11-06/` (see its README). `MANIFEST.csv` lists every file
with its source, row count and SHA-256.

## dataset/
| File | Contents |
|---|---|
| `test_data.csv` | Revised test set: 1,048 rows, 55 calculators. Notes, questions and Time Point Instruction from `dataset/test_data_10_22.csv`; Relevant Entities, labels, limits and explanations from `dataset/test_data_10_22_corrected.csv`. Every table and every output is scored against this file |
| `one_shot_examples.json` | One-shot examples (55), as the one-shot outputs saw them: the main run's file with the CKD-EPI example's wording and the Sodium Correction example's "Hillier" spelling fixed (both used by the re-runs spliced in). The corrected SOFA example is not included: GPT-4 never ran it |
| `train_data.csv` | Training set, 9,871 rows, 40 calculators (open items below) |
| `CHANGELOG.md` | Dataset change log |

## outputs/
One output per model, prompt style and row (1,048 each), each with a `Source Run` field.
- 24 files: 8 models × `direct_answer` / `zero_shot` / `one_shot`.
- `code_exec_gpt-3.5-turbo-0125.jsonl`, `code_exec_gpt-4-0613.jsonl`: code-interpreter prompt.
- `accuracy_by_category.csv`: the accuracy behind the main table.

Each row's output is its latest run (by timestamp) among the runs all 8 models took part in:

| Rows | Output |
|---|---|
| Default | Main run, `final_runs/test_data_10_22/` |
| 13 label-fix rows and CKD-EPI rows 21–40, all styles | `extra_runs/corrected_rows_ckdepi_rerun_2026-10-22/` |
| MDRD and Sodium rows 141–160, 366–385, zero-shot, one-shot and code | `extra_runs/mdrd_sodium_wording_rerun_2026-10-22/` |

Kept on the main run: the 45 Astra rows (pending physician review), and the LDL Friedewald and SOFA one-shot re-runs
(GPT-4 never ran them).

`calculator_sources.csv` (top level) gives the same per calculator ID: the changes it uses, how many of its outputs
come from each run, and any later change left out with the reason. 42 of the 55 calculators use their latest change;
the other 13 have a later change left out: Astra corrections in 12 (SOFA also has its corrected one-shot example) and
the LDL Friedewald question.

## error_analysis/
- `error_classifications.csv`: every wrong zero- and one-shot answer (12,068), with its `gpt-4` error types and status.
  - Used: 11,953, each classified against its current reference explanation.
  - Excluded: 115 whose classification is empty or blocked.
- `error_type_summary.csv`, `error_type_table_values.csv`: per model and style, each error counted once as its earliest type (A > B > C > D).

## figures/
Appendix E.1 Figures 2 and 3 (accuracy vs. number of parameters, zero-shot), their points and fit coefficients.

## tables/
LaTeX for the main results, error types, dataset statistics, code interpreter, token counts (B.3) and training set (C).

## Open items, not yet reflected here
- **Astra corrections** (89 input corrections, 15 note edits): set aside pending physician review. Applying them would change the test set and the scores;
  the same release with the 45 Astra rows applied (from the Astra re-run) is `../final_release_2026-11-22_astra/`.
- **Blood-gas rows:** 1,048 rows kept. The alternative is dropping 9.
- **Training set:**
  - de-duplication against the test set and one-shot examples is undecided (173 / 144 / 72 rows);
  - the LDL questions use the Friedewald wording while the test set keeps v1.0's;
  - about 50 Serum Osmolality rows take urea in mg/dL as BUN;
  - 2 steroid rows convert a drug to itself;
  - see `final_runs/consolidated_2026-11-06/verify_v10_fixes.csv`.
