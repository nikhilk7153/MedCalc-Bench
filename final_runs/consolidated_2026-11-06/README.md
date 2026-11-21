# Consolidated results and paper tables (2026-11-06)

One output per model, prompt style and row, spliced from runs that already exist (no new model calls), re-scored
against the corrected labels, and the three paper tables built from them. Source folders are not modified.

## Run (from this folder, in this order)
| Script | Writes |
|---|---|
| `run_coverage.py` | `../../equation_audit/rerun_coverage_2026-11-01.csv`: per re-run row, which models ran which styles |
| `consolidate.py` | `outputs/` (24 files: 8 models x direct / zero-shot / one-shot, 1,048 rows each, with a `Source Run` field) and `accuracy_by_category.csv` |
| `make_calculator_sources.py` | `calculator_sources.csv`: per calculator ID, the changes made after the main run that it uses, which run each of its outputs comes from, and later changes not used (with the reason) |
| `consolidate_errors.py` | `error_classifications.csv` (every wrong zero/one-shot answer, its classification and status) and `error_type_summary.csv` |
| `make_main_results_table.py` | `tables/main_results_table.tex` (main accuracy table) |
| `make_error_table.py` | `tables/error_analysis_table.tex` (error type table) and `error_type_table_values.csv` |
| `make_stats_table.py` | `tables/dataset_statistics_table.tex` (dataset statistics table) |
| `make_code_table.py` | `outputs/code_exec_*.jsonl` (GPT-3.5 / GPT-4 code prompt, spliced the same way) and `tables/code_interpreter_table.tex` |
| `make_train_table.py` | `tables/train_table.tex` (Appendix C training-set table, from `dataset/train_data_10_22.csv`) and the numbers for the C.1 text |
| `accuracy_vs_param_count/run.sh` | Appendix E.1 figures (accuracy vs. number of parameters, zero-shot; equation and rule) from `outputs/`, with `evaluation/attribute_analysis.py`; points and fit coefficients as CSV in the same folder |
| `count_admission_extraction_errors.py` | How many equation-based calculators had v1.0 input extractions wrong for the admission (pre-treatment) time point, at three levels of confirmation, and how each row was fixed; writes `admission_input_extraction_rows.csv` |
| `count_v10_equation_errors.py` | Every v1.0 equation-based row followed into the revised set: kept, inputs wrong, formula wrong or replaced, with the reason; writes `v10_equation_rows.csv` |
| `verify_v10_fixes.py` | Checks that v1.0's equation-based errors are fixed in the test set, training set and one-shot examples (independent formula recomputation for Due Date, Framingham, LDL, Calcium, MELD Na, CKD-EPI; fentanyl-in-mg and steroid-to-itself templates; urea taken as BUN); writes `verify_v10_fixes.csv` |
| `build_final_release.py` | `../../final_release_2026-11-08/`: the revised test set, one-shot examples, training set, all outputs, error analysis and tables in one folder, with `MANIFEST.csv` (run last) |
| `check_open_patients_table.py` | Prints the Open-Patients table (Table 1) recomputed from `ncbi/Open-Patients` next to the published values. Needs `nltk` with `punkt_tab` |
| `make_token_tables.py` | `token_counts.csv` and `tables/token_tables.tex` (Appendix B.3 token tables). Needs `transformers`, `tiktoken`, `sentencepiece`, `protobuf`, `jinja2` (no torch) and Hugging Face access to the gated Llama 3 / Mistral tokenizers |

`run_py_as_run_5f7e8a9d.py` is the `evaluation/run.py` the final runs used (= `git show 80dc902:evaluation/run.py`);
`make_token_tables.py` rebuilds prompts with it, since `evaluation/run.py` has changed since.

## Which output is used for each row
Rule (2026-11-14): the latest run of the row, by timestamp, among the runs all 8 models took part in.
| Rows | Output | Run time |
|---|---|---|
| All rows by default | Main run, `final_runs/test_data_10_22/` | 10-22 04:51-06:06 (code prompt 09:10-09:31) |
| 13 label-fix rows and CKD-EPI rows 21-40, all styles (code prompt: the 13 label-fix rows) | `extra_runs/corrected_rows_ckdepi_rerun_2026-10-22/` | 10-22 13:53 |
| MDRD and Sodium rows 141-160, 366-385, zero- and one-shot and code prompt | `extra_runs/mdrd_sodium_wording_rerun_2026-10-22/` (reworded questions) | 10-22 16:51 |

Kept on the main run although a later run exists:
- **Astra rows (45):** re-run 10-22 14:24 on unreviewed corrections; physician review pending.
- **LDL rows 513-532 (Friedewald question) and the SOFA one-shot rows 889-908 (corrected example):** re-run 10-23/10-24
  without GPT-4 (and, for LDL, without GPT-3.5), so the models would have answered different prompts.
- Side experiments (`gpt-4o`, PMC-LLaMA fit-context, time-point replay) and the superseded GPT-only
  `corrected_rows_rerun_2026-10-22`.

MDRD/Sodium direct-answer cells keep the old question wording (not re-run).
The re-runs of unchanged prompts (13 label-fix rows; CKD-EPI direct and zero-shot) do not reproduce the main run's
answers exactly (temperature 0, but neither the OpenAI API nor vLLM is deterministic): overall accuracy moves by at most
0.39 points (Mixtral zero-shot); GPT-4 one-shot is 62.31% (62.60% with main-run answers on those rows).

Labels: `dataset/test_data_10_22_corrected.csv`. Questions and notes: `dataset/test_data_10_22.csv`.
`consolidate.py` checks every output's question and note against the row it is scored on (0 mismatches) and that
re-scoring reproduces the stored verdict on unchanged labels (0 differences).

## Methods, as in the published paper
- **Accuracy table:** accuracy by category; std = sqrt(acc(1-acc)/n); rounded half up.
- **Error table:** `gpt-4` classifications of the same outputs (no new classification). Each wrong answer counts once, as its
  earliest type (A > B > C > D). Type share of the errors in parentheses; Error Rate = 1 - accuracy.
  Excluded and counted: 115 of 12,068 wrong answers whose classification is empty or blocked. Every other wrong answer
  was classified against its current reference explanation (the CKD-EPI re-run was classified with the corrected labels
  and explanation wording).
- **Code-interpreter table:** main code-prompt run (`evaluation/outputs/test_data_10_22/`) with the 13 label-fix rows from the
  CKD-EPI re-run and the reworded MDRD/Sodium rows from their re-run; zero-shot CoT columns are the consolidated zero-shot outputs. GPT-4 row 564 (Framingham)
  exceeded GPT-4's 8,192-token context and is scored Incorrect, as in every version of the run.
- **Token tables:** every prompt is rebuilt from the test row and one-shot file of the run it came from and counted
  with the model's own tokenizer (open models: the templated prompt, as run.py tokenized it; GPT: tiktoken
  `cl100k_base` with OpenAI's per-message overhead). Output: the `Output Tokens` vLLM recorded for open models;
  tiktoken over the raw output for GPT. Rebuilt main-run prompts equal those in `../test_data_10_22_with_prompts/`.
- **Dataset statistics:** lengths in GPT-4 tokens (tiktoken `cl100k_base`); attributes = number of Relevant Entities.
  On v1.0 `test_data.csv` this reproduces the published attribute columns exactly and nearly all lengths.

## Paper map (arXiv 2406.12036v4 = `../../2406.12036v4.pdf`)
| Paper item | Script | Status |
|---|---|---|
| Figure 1: example instances | none (drawing: `../../instance_illustration/`) | CHA2DS2-VASc example shows answer 7; its note (row 44) is labelled 4 in v1.0 and now. Albumin-corrected AG example (row 769, 19.25) still correct |
| Table 1: dataset statistics | `make_stats_table.py` | done |
| Table 2: main accuracy | `consolidate.py` -> `make_main_results_table.py` | done |
| Table 3: error-type examples | none (quotes from the 2024 GPT-4 run) | quotes are from the 2024 run, but all three are still errors of their type under the corrected labels (checked 2026-11-13). Wells' DVT (row 233): label -1 unchanged; the table's question needs the current wording ("score of Wells' criteria"). PERC (row 569): label 3 -> 4 because the note now says the hernia repair was "3 weeks ago" (was "in the past", inside the quote's [...]); the prior-PE miss still holds. LDL (row 524): unchanged, 128 vs the model's 142 |
| Table 4: error types | `consolidate_errors.py` -> `make_error_table.py` | done |
| Table 5: dataset comparison | none (literature) | unaffected |
| Table 6: Open-Patients | `check_open_patients_table.py` | verified, unchanged |
| Tables 7-8 | do not appear in the PDF (numbering skips) | - |
| Table 9: calculator list | none saved (checked ad hoc, 2026-11-07) | needs a script; MAP 3->2, BMI 3->2, Due Date 1->2 |
| Tables 10-12: tokens (B.3) | `make_token_tables.py` | done |
| Table 13: training set (C.1) | `make_train_table.py` | done; training-set open items pending |
| Table 14: fine-tuning (C.3) | `../../sft_rerun_2026-10-27/` (`build_sft_data.py`, `score_sft.py`) | not run (needs 4xA100) |
| Table 15: code interpreter (D.1) | `make_code_table.py` | done |
| Figures 2-3 (E.1) | `accuracy_vs_param_count/run.sh` -> `evaluation/attribute_analysis.py` | zero-shot; 2026-11-13: a row's parameter count is its number of Relevant Entities, as in Table 1 (was the number of `name_to_python.json` entries, which counted inputs the code never reads and gave MME 1) |
| Sec. 4.2 text: 89% classifier accuracy on 200 errors | `evaluation/sample_judge_agreement.py` | sample drawn (from `evaluation/error_analysis_10_22`, not the consolidated errors); no human labels yet |
