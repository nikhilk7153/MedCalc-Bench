# PMC-LLaMA re-run with fit-to-context truncation (2026-10-24)

Separate from the main results. Nothing here is merged into `final_runs/`.

## Why
The original `run.py` (and ours, by default) cuts every PMC-LLaMA test note to 256 tokens. That affects 665 of the 1,048 notes
(median note 473 tokens), even though the cut prompts are only 380-1,120 tokens long in a 2,048-token context.

`run.py --fit_context 512` instead keeps as much of each note as fits while leaving at least 512 tokens for the answer.
- The one-shot example keeps the original 256-token trim; only the test note changes.
- Without the flag, `run.py` builds byte-identical prompts to the version used for every earlier run. This was checked on all
  3,144 PMC-LLaMA, Meditron and GPT prompts.

## Inputs (`input/`)
| File | md5 |
|---|---|
| `test_data_10_22_corrected.csv` (full corrected set, 1,048 rows) | 394de441595e8561667cf12ad8532189 |
| `one_shot_10_17_2026_corrected.json` | 63c12391ea4f399ac1a75ff6e9fe596d |
| `run.py` (with the `--fit_context` option) | 5cb3f889c37e0002cd7685106313bd06 |

The earlier runs used `run.py` md5 5f7e8a9d5b4f3b95ebeae90fe27fdac4. Its default code path is identical to this file's.

## Run
- **Command:** `run.py --model axiong/PMC_LLaMA_13B --fit_context 512`, all three prompt styles, 1,048 rows.
- **Environment:** vLLM 0.32.0, greedy, 1 H100 NVL.
- **Folders:** `outputs/`, `results/`, `logs/`. All 7 files are byte-identical to the GPU instance copies.

## Result
Every prompt fit (none scored Incorrect for length). Re-scoring reproduces every verdict.

| Setting | 256-token cut (`final_runs/test_data_10_22_corrected/`) | Fit to context | Notes kept whole | Empty answers |
|---|---|---|---|---|
| Direct | 0.00% | 0.10% | 921 | 957 |
| Zero-shot | 0.10% | 0.19% | 887 | 678 |
| One-shot | 11.16% | 11.07% | 583 | 8 |

Longer notes do not help PMC-LLaMA: it still returns mostly empty answers without an example, and one-shot is unchanged.

## Meditron
Meditron never truncates the test note, and all its prompts fit in 4,096 tokens. With `--fit_context 512` only 11 one-shot
prompts change: the example is trimmed further so the answer has at least 512 tokens. That covers rows 809, 823 and 828,
which were cut off mid-answer in the main run. Not run.
