# Check of the frozen inputs (2026-10-26)

Checks the parts of the corrected test set that can no longer change: patient notes, questions, time-point instructions,
prompt templates and one-shot examples. Base set: `dataset/test_data_10_22_corrected.csv`. No dataset was changed.

Scripts: `check_frozen.py` (note hygiene, questions, inputs written in the note, one-shot examples) and
`note_provenance.py` (each note against every earlier copy of it in v1.0, 10/15/2024, 1/17/2025, 1_31, Verified and the
training sets).

## Clean
- **Prompt templates:** the six prompt functions in `evaluation/run.py` are word for word the paper's (main branch).
- **Notes and time-point instructions:** identical between the as-run set (md5 e53d936a…) and the corrected set.
- **Changes after the main runs:** the corrected set has 60 reworded questions (MDRD, sodium correction, LDL) and 3 changed one-shot examples
  (CKD-EPI, sodium correction, SOFA). Each has its own re-run in `extra_runs/`, except GPT-4 one-shot on the 20 SOFA rows, because
  gpt-4-0613 was retired.
- **One-shot examples:** all 55 answers reproduce from their inputs with the calculator code (Wells DVT and MME were checked by hand).
- **Notes:** none is empty or has encoding residue. 807 of the 828 non-template notes are word for word an earlier published copy.
- **Inputs:** every numeric input is written in its note, apart from 26 that differ only in format (e.g. "12,35", µmol/L
  creatinine, "fifty-year-old", platelets × 10⁹/L).

## To disclose or decide
| Rows | Finding |
|---|---|
| 468, 470–475 (Free Water Deficit) | Sodium in the note raised by 20 (row 475: by 30). The values are in no source or published version (user decision 10-22). Rows 414 and 750 contain the same notes with the original sodium. |
| 598 (SIRS) | The sentence "white blood cell count of 8,500/mm³" was added to the note to match v1.0's input. It is in no published version. |
| 749 (Delta Ratio) | "HCO3 4.88 mmol/L (4-6)" was added to the serum panel and the blood-gas HCO3 of 5 removed. The input stays 5.0; both values fall within the 5% tolerance. |
| 500, 779 | Bicarbonate relabelled or moved within the note ("Carbon dioxide 26 mm Hg" → "bicarbonate 26 mEq/L"; blood-gas HCO3 → "serum HCO3"). |
| 595, 599, 427, 518, 528, 532 | Minor edits: a self-contradicting °F value removed; "× 104/L" → "× 10⁹/L"; phrases that stated the answer removed; units added. |
| 191, 193, 197, 620, 626, 647, 680, 683 (QTc templates) | New heart rates on 10/22 (`session_archive_2026-10-22/qtc_dedupe.py`) to remove duplicate template notes. `dataset/CHANGELOG.md` still says the QTc templates are kept as in v1.0. |
| 791 | The note says albumin "4.2 g/L"; the same note in row 811 says g/dL. The label reads it literally (28.95; 19.5 if g/dL). |
| 166 rows | The time-point instruction ("first admitted to the hospital") is attached to notes that never mention an admission (clinic or office vignettes). |
| 19 of 55 calculators | Questions use Verified's wording, not v1.0's. Ideal Body Weight no longer states the unit, and Free Water Deficit asks for kg. |
| One-shot 23, 56, 57 | The example note is also a test note for a different calculator (rows 802, 652, 674, 618). v1.0 had 9 such overlaps. |

**Effect** (`accuracy_without_edited_values.csv`, all 26 runs): excluding the 9 rows with values in no source (468, 470–475,
598, 749) changes accuracy by at most 0.4 points.
