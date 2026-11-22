# MedCalc-Bench correction, Astra variant: dataset, outputs and error analysis (2026-11-22)

> **Work in progress.** This release is not final. The Astra corrections in it have **not been reviewed by a physician**,
> and the dataset, outputs and numbers may still change.

The same as `../final_release_2026-11-08/` (the current set), except for the 45 rows changed by the gpt-6-astra
corrections. Those rows take the Astra re-run (`extra_runs/astra_corrections_rerun_2026-10-22/`, 10-22 14:24): its
notes, Relevant Entities, labels and explanations, its outputs, and its error classifications. With this, every row uses
its latest run among the runs all 8 models took part in.

Built by `final_runs/consolidated_2026-11-06/` with `MEDCALC_VARIANT=astra` (see `variant.py` and that folder's README),
from `final_runs/consolidated_2026-11-06/astra/`. `MANIFEST.csv` lists every file with its source, row count and SHA-256.

## dataset/
| File | Contents |
|---|---|
| `test_data.csv` | Astra test set: 1,048 rows, 55 calculators. The current set's `test_data.csv` with the 45 Astra rows taken from the Astra re-run's test set: 6 notes edited, 42 rows' Relevant Entities, 26 answers (with limits) and 39 explanations changed. Nothing outside the 45 rows differs. Every table and every output is scored against this file |
| `astra_changes_applied.csv` | One line per Astra row: answer before and after, inputs changed, note edits, and Astra's reasoning |
| `one_shot_examples.json` | One-shot examples (55), identical to the current set's. The Astra re-run's file differs only in the Sodium Correction example, which no Astra row uses |
| `train_data.csv`, `CHANGELOG.md` | As in the current set |

## outputs/
One output per model, prompt style and row (1,048 each), each with a `Source Run` field.
- 24 files: 8 models × `direct_answer` / `zero_shot` / `one_shot`.
- `code_exec_gpt-3.5-turbo-0125.jsonl`, `code_exec_gpt-4-0613.jsonl`: code-interpreter prompt.
- `accuracy_by_category.csv`: the accuracy behind the main table.

| Rows | Output |
|---|---|
| Default | Main run, `final_runs/test_data_10_22/` (code prompt: `evaluation/outputs/test_data_10_22/`) |
| 13 label-fix rows and CKD-EPI rows 21–40, all styles | `extra_runs/corrected_rows_ckdepi_rerun_2026-10-22/` (13:53) |
| 45 Astra rows, all styles and code prompt | `extra_runs/astra_corrections_rerun_2026-10-22/` (14:24) |
| MDRD and Sodium rows 141–160, 366–385, zero-shot, one-shot and code | `extra_runs/mdrd_sodium_wording_rerun_2026-10-22/` (16:51) |

Kept on the main run: the LDL Friedewald and SOFA one-shot re-runs (GPT-4 never ran them). `calculator_sources.csv`
gives the same per calculator ID: 53 of the 55 calculators use their latest change; SOFA (corrected one-shot example) and
LDL (Friedewald question) do not.

Every output's question, note, label and explanation equal this `test_data.csv`, except the direct-answer outputs of the
40 MDRD/Sodium rows (320 outputs), which saw the old question wording (direct answer was not re-run).

## error_analysis/
- `error_classifications.csv`: every wrong zero- and one-shot answer (12,088), with its `gpt-4` error types and status.
  - Used: 11,969, each classified against its current reference explanation (Astra rows: the Astra explanations).
  - Excluded: 119 whose classification is empty or blocked (10 of them from the Astra re-run).
- `error_type_summary.csv`, `error_type_table_values.csv`: per model and style, each error counted once as its earliest type (A > B > C > D).

## figures/, tables/
As in the current set, computed on this test set and these outputs.

## Difference from the current set
Overall accuracy moves by at most 0.47 points (Llama 3 8B zero-shot). GPT-4: zero-shot 44.66% → 44.47%, one-shot
62.31% → 61.93%; code prompt 62.21% → 61.55%. Dataset statistics: average attributes 10.0 → 10.1 (risk) and
7.4 → 7.6 (severity).
