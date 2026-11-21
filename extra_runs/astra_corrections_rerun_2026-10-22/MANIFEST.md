# Re-run with the gpt-6-astra corrections applied (2026-10-22)

Separate from the main results; nothing here is merged into `final_runs/`. The astra corrections are **unreviewed**.
They are applied here so their effect can be measured. Keep or drop each one later using `astra_changes_applied.csv`.

## What was applied
- **Source:** all pending gpt-6-astra corrections in `equation_audit/overrides/pending_physician_review/` (89 input corrections,
  15 note edits; copied to `input/astra_entity_overrides.json` and `input/astra_note_edits.json`).
- **Matching:** each correction was matched to the current row by Calculator ID + Note ID and applied on top of
  `dataset/test_data_10_22_corrected.csv` (13 label fixes + CKD-EPI wording).
- **Skipped:** corrections whose note is no longer in the set (19 input, 3 note edits), corrections already present
  (25 input, 2 note edits), and corrections whose text was not found (4 note edits). Also skipped are 3 corrections on rows that
  already carry our own fix (452, 465, 849; for 461 we keep our 2, where astra had 1).
- **Result:** 45 rows changed; 26 answers change and 6 notes are edited, so only those 6 prompts differ.
  Labels were regenerated with the verified calculator code (0 mismatches); integer answers keep integer format.
  Per-row details: `astra_changes_applied.csv`. Matching plan: `input/astra_apply_plan.json`.
- **Caveat:** 12 of these rows were flagged by astra on an earlier version of the row (review CSV group C). The same input
  change is applied to the current row, so the resulting value can differ from the one astra proposed.

## Inputs (`input/`)
| File | md5 |
|---|---|
| `test_data_10_22_astra.csv` (full set, 1,048 rows; same as `dataset/test_data_10_22_astra.csv`) | 129cbc704efa7850951042b065d6aaef |
| `test_rows.csv` (the 45 rows) | 1db6b734b0d0493c2afb1e662705cc0f |
| `one_shot_10_17_2026_corrected.json` | ededd6264b1d449805069e8a0af7e9f5 |
| `as_text.py` (reads label columns as text) | 19cdf79f90b69bb88110b497afcee5c7 |
| `astra_entity_overrides.json` / `astra_note_edits.json` / `astra_apply_plan.json` | 527cfb46… / 754e3949… / b5499e07… |
| `queue_h100.sh`, `queue_h100.log` | open-model queue and log |

## Code and settings
Identical to `../corrected_rows_ckdepi_rerun_2026-10-22/MANIFEST.md`:
- **Code:** `run.py` 5f7e8a9d…, `evaluate.py` b2f3e1ad…, `code_exec_timed.py` e7da8218…, `error_analysis_batch.py` 73dce14b….
- **GPT:** `gpt-3.5-turbo-0125` and `gpt-4-0613`, temperature 0, OpenAI SDK 2.8.1. Code-augmented uses SDK 0.28.0, and here runs on all 45 rows.
- **Open models:** vLLM 0.32.0, greedy, on H100 NVL. 70B models use an NVLink pair (tp 2). PMC-LLaMA notes are truncated to 256 tokens.
- **Rows:** all 8 models x direct / zero-shot / one-shot on the 45 rows.
- **Layout:** one folder per model with `outputs/`, `results/` and `logs/`.

## Error analysis (`error_analysis/`)
- **Inputs:** every wrong zero-shot and one-shot answer on the 45 rows, 630 requests.
- **Reference text:** explanations from `input/test_data_10_22_astra.csv`.
- **Classifier:** the paper's prompt, `gpt-4`, temperature 0, through the Batch API.

## Checks done
- Every output has the 45 rows. Labels, questions and notes equal the astra dataset (PMC-LLaMA notes are 256-token prefixes, as designed).
- Re-scoring every answer reproduces the stored verdicts (0 mismatches). There were no API errors.
- The 42 open-model files are byte-identical to the GPU instance copies.
