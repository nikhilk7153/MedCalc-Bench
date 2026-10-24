# Re-run of the SOFA rows in one-shot with the corrected SOFA example (2026-10-23)

Separate from the main results. Nothing here is merged into `final_runs/`.

## Why these rows
The one-shot example for SOFA (calculator 43) used in all reported runs scored an unreported urine output as
0 mL/day, which adds renal points (example answer 9). It was corrected on 2026-10-23: urine output is not reported,
hypotension is true, and the kidneys score on creatinine 3.12 alone (example answer 7). The example only appears in
the one-shot prompt of the 20 SOFA rows (889-908), so only those rows are re-run, and only in one-shot.
The 20 SOFA test labels are unchanged.

## Inputs (`input/`)
| File | md5 | What |
|---|---|---|
| `test_data_10_22.csv` | d880d87f79156aafc902ba657d9dae27 | Full test set the main results correspond to, 1,048 rows |
| `test_rows.csv` | 45be60ae5c824e79e9b75e9b32651c89 | The 20 SOFA rows, copied unchanged (identical in `test_data_10_22_corrected.csv`) |
| `one_shot_10_17_2026.json` | 769a266c921fd4da4e4e171a3945f811 | One-shot file with the corrected SOFA example (copy of `final_files_2026-10-16/one_shot_10_17_2026.json`) |
| `as_text.py` | 19cdf79f90b69bb88110b497afcee5c7 | Reads the label columns as text, as when the full test set is read |

The example used before the correction is in `gpu_backup_2026-10-22/medcalc/final_files_2026-10-16/one_shot_10_17_2026.json`.
For every row the prompt is identical to the main run's except inside the SOFA example; the user message is identical.

## Code (same files as the final runs)
| File | md5 |
|---|---|
| `evaluation/run.py` | 5f7e8a9d5b4f3b95ebeae90fe27fdac4 |
| `evaluation/evaluate.py` | b2f3e1ad90b2bffc5808f72ba247783a |

## Runs
| Model | Prompt style | Engine |
|---|---|---|
| gpt-3.5-turbo-0125 | one_shot (20 rows) | `run.py`, OpenAI SDK 2.8.1, temperature 0 |

OpenAI retired `gpt-3.5-turbo` and `gpt-3.5-turbo-0125` on 2026-10-23. The run was made through the alias
`gpt-3.5-turbo-16k`, which OpenAI still served with `gpt-3.5-turbo-0125`: every row's `LLM Model Version` is
`gpt-3.5-turbo-0125`. That is why the output and results file names say `gpt-3.5-turbo-16k`.

Not re-run: `gpt-4-0613` (retired by OpenAI on 2026-10-23; no alias still serves it) and the six open models
(the GPU instance was not available).

Command (from `evaluation/`):
`python3 ../extra_runs/sofa_oneshot_rerun_2026-10-23/input/as_text.py run.py --model OpenAI/gpt-3.5-turbo-16k
--prompt one_shot --data <input>/test_rows.csv --one_shot <input>/one_shot_10_17_2026.json --tag sofa_oneshot_rerun`

`check_inputs.py` reports "no Time Point Instruction values" for these rows. That is expected: SOFA questions carry no
time-point sentence in the full test set either, and the check passes on the full test set.

## Accuracy, old example (main run) vs corrected example (this re-run), 20 SOFA rows
| Model | Old example | Corrected example |
|---|---|---|
| gpt-3.5-turbo-0125 | 2/20 (10%) | 2/20 (10%) |

"Old" is `final_runs/test_data_10_22_reworded/outputs/OpenAI_gpt-3.5-turbo-0125_one_shot.jsonl` on the same rows.
12 of the 20 answers changed. Row 897 is correct in both runs; row 890 was correct before and is wrong now; row 907
was wrong before and is correct now. On row 889 (no urine output in the note) the answer is 15 in both runs (ground truth 11).

No error analysis was run on these rows.
