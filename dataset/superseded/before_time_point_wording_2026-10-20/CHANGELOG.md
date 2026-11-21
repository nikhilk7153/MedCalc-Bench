# Corrected MedCalc-Bench test set and one-shot set — frozen 2026-10-14

## Status 2026-10-20: unreviewed per-row corrections removed
The per-row corrections from the gpt-6-astra review have not been reviewed by a physician, so they are NOT applied:
89 input corrections and 15 note edits are set aside in equation_audit/overrides/pending_physician_review/
(see physician_review.csv and rule_based_extraction_changes.* for the list). Labels for those rows are back to the
values computed from their original inputs. Still applied: the replacement notes (blood gas, dialysis, Extracted-only,
unusable notes), the note edits the user asked for (row 218; answer-leak removals in rows 513, 516, 517, 526, 528, 529,
530, 532, 772), the calculator code fixes and the global unit/spelling fixes. Statements below about the 89 input
corrections and the 15 other note edits describe the pending set.

## Status 2026-10-20 (2): Charlson, Child-Pugh, GBS, HAS-BLED and RCRI back on v1.0's own notes
These 100 rows use v1.0's handwritten notes and inputs again (labelled Synthetic), not v1.1's notes. v1.0's input names
are mapped to the calculator code's names; RCRI's always-False duplicate 'Cerebrovascular disease history' is dropped so
the real 'History of cerebrovascular disease' value is scored (v1.0's code scored the duplicate, so prior stroke/TIA never
counted; 13 RCRI labels change). Child-Pugh (14 rows) and GBS (3 rows) labels also differ from v1.0 because v1.0's code
mis-scored them (hand-checked). The row-218 note edit applied to v1.1's note and is set aside.

## Status 2026-10-20 (3): equation-based rows from the user's review sheet
19 equation-based rows use the user's versions from the review Google Sheet (tab gid 548716547; copy in
equation_audit/overrides/user_sheet_2026-10-20.csv): 18, 29, 375, 383, 427, 472, 500, 518, 528, 529, 532, 729, 740, 749,
750, 756, 761, 779, 809. Rows 375, 383, 472, 500, 729, 750, 756 and 761 again take electrolytes from a blood gas (user
decision); 500 and 779 relabel the bicarbonate in the note; 749 lists HCO3 4.88 in the note while its input is 5.0.

## Status 2026-10-20 (4): v1.0 questions; time-point instruction moved to the system prompt
Questions use v1.0's exact wording (973 rows; steroid rows keep their own), except Framingham (46: labels are 10-year
risk %), Estimated Date of Conception (68: notes give no cycle length) and Sodium Correction (26: labels use Hillier 1999),
which keep Verified's wording without the time-point sentence. The sentence "You should use the patient's medical values
and health status when they were first admitted to the hospital prior to any treatment." is in the new column
'Time Point Instruction' for the 768 non-template rows where Verified's question uses it (all but APACHE II, SOFA,
Caprini); the run script appends it to the system prompt.

## Status 2026-10-20 (5): Charlson, Child-Pugh, GBS, HAS-BLED and RCRI from test_data_1_17_2025.csv
These 100 rows take Patient Note and Relevant Entities from dataset/test_data_1_17_2025.csv (same Calculator ID + Note ID
as v1.0, user decision); labels are recomputed by the calculator code. Notes change in 11 rows: RCRI creatinine in mg/dL
instead of µmol/L (8), HAS-BLED "major bleeding"/"significant bleeding" → "DVT" (5, 7), Charlson 5 (one sentence removed).
Inputs change in Charlson (values in the form cci.py matches: 'localized', 'metastatic', 'mild', 'moderate to severe' —
the v1.0 forms 'Metastatic', True, 'Moderate' scored 0), HAS-BLED 5 and 7, RCRI 7 and 21. Charlson 10 keeps its v1.0
inputs (entity override): the 1_17 file repeats note 2's inputs there. The 1_17 file's own answers/explanations were not
used: 29 of them do not follow from its inputs (e.g. HAS-BLED labile INR and bleeding medications scored as missing;
Charlson metastatic tumor not scored). 30 rows change from status (4); 21 labels change: Charlson 18, HAS-BLED 2 (349, 351), RCRI 1 (260).
Previous files: dataset/superseded/before_1_17_swap_2026-10-20/.

## Status 2026-10-20 (6): Verified's questions, time-point sentence in the system prompt
Supersedes the question part of status (4). Every row uses Verified's question for its calculator with the time-point
sentence removed (Steroid Conversion rows keep their own question). The sentence stays in 'Time Point Instruction'
(768 rows) and the run script appends it to the system prompt. Only the Question column changes from status (5):
265 rows in 14 calculators (8, 9, 10, 11, 13, 16, 17, 18, 19, 23, 30, 38, 49, 51); notes, inputs and labels are identical.
Previous files: dataset/superseded/before_verified_questions_2026-10-20/.

## Files
- `test_data_final.csv`: 1,048 rows (v1.0 has 1,047; Calcium Correction topped up to 20).
- `one_shot_finalized_explanation.json` (what `evaluation/run.py` reads) and `one_shot_finalized.csv`: 55 one-shot examples.
- `CHECKSUMS.sha256`: checksums of the three files above.
- Per-row records: `test_data_corrected_changes.csv` (every replaced or added row: old/new note ID, source, reason),
  `test_data_corrected_overrides.csv` (every note edit and input correction, with its reason).

## How it is built (reproducible; two rebuilds give byte-identical output)
1. `python3 equation_audit/build_corrected_test.py`: starts from v1.0 `test_data.csv`, applies the rules in the script
   and the reviewed per-row overrides in `equation_audit/overrides/` (`replacements.json`, `note_edits.json`,
   `entity_overrides.json`).
2. `python3 equation_audit/run_generate_csv.py`: recomputes every Ground Truth Answer, limit and explanation from the
   Relevant Entities with the calculator code at `medcalc_verified_code` (branch `fix/anion-gap-and-sofa`, commit a58c754).

## Changes from v1.0
Rows: 872 keep a v1.0 note ID (760 with identical note text); 176 are notes new to this set.

Replaced or added rows (300):
- 121: replaced by MedCalc-Bench Verified's version of the same note (Verified fixed the note or label; non-handwritten
  rule-based rows with a different answer; steroid conversions of a drug to itself; MME notes dosing fentanyl in mg;
  flagged v1.0 rows whose Verified version passed the check).
- 99: handwritten Charlson, Child-Pugh, Glasgow-Blatchford, HAS-BLED and RCRI notes taken from v1.1 (Verified's version
  where the same note ID exists). SOFA, APACHE II, GCS, PSI and Caprini keep the v1.0 handwritten notes.
- 76: replaced by an unused Verified note for the same calculator (rows wrong or unusable in both v1.0 and Verified,
  including 18 chosen after passing both checks below).
- 3: test rows identical to the one-shot example for their calculator.
- 1: Calcium Correction top-up.

Edits to kept rows:
- All rows use Verified's question for their calculator (except Steroid Conversion, whose question names the drug pair).
- Input names follow Verified's `name_to_python.json`; inputs copied from Verified where the note text is identical.
- Temperature unit "degrees celsisus" corrected in 22 APACHE II/PSI rows. The code converts any unit other than
  "degrees celsius" as Fahrenheit, so these labels were 4–15 points too high. The same misspelling is fixed in 26 notes.
- 16 note edits: unit typos (e.g. mEq/dL, gr/dL, gm/dl), contradictions (FiO2 63% "on room air"; positive Factor V
  Leiden alongside "no thrombophilia"), and handwritten notes missing a value their v1.0 inputs use (A-a gradient,
  diastolic BP, PaO2), restored from those inputs; and a measured LDL removed from a calculated-LDL note.
- 89 input corrections: 76 from the review of 135 rows flagged by the gpt-6-astra label and input checks, and 13 from
  checking the 286 non-template rows that had not been checked in their final form (see overrides CSV).

Calculator code fixes on `fix/anion-gap-and-sofa` (they affect labels and explanations):
- Anion Gap one-shot chloride; SOFA with urine output but no creatinine; MME "FentaNYL patch" key; PSI PaO2 wording.
- SOFA: hypotension without BP now scores 1; the MAP < 70 branch no longer erases the explanation; boundary fixes for
  PaO2/FiO2 199–200 and creatinine = 5.0; renal wording.

## Conventions used for labels
- Values are taken at the time point the calculator's (Verified) question names, e.g. admission before treatment.
- A criterion the note does not mention is absent; for "absence" criteria (Centor "Cough Absent", FeverPAIN
  "Absence of cough or coryza") this means the criterion is met.
- HEART: no documented ECG/troponin result is scored as normal ECG / troponin ≤ normal limit.
- GCS: a component documented as not testable gets full points (the code's documented rule).

## Known and accepted
- About 50 rows apply a score outside its intended population (e.g. HEART without chest pain, PSI in a clinic visit).
  They are kept; their labels follow the conventions above.
- Note Type: clinician-written notes (integer Note ID; APACHE II, SOFA, GCS, PSI, Caprini) are labelled "Synthetic" (100 rows); PMC/USMLE notes are "Extracted" (728); templates "Template" (220).
- Only Extracted (PMC/USMLE) notes are taken from Verified; Synthetic notes come only from v1.0/v1.1's clinician-written
  calculators. HOMA-IR row 426 uses v1.0's train-set case pmc-3114100-1 (equation_audit/overrides/extra_notes.csv; insulin
  21 pmol/L as written, v1.0 had 756), because Verified has no unused Extracted HOMA-IR note. It was also v1.0's original
  HOMA-IR one-shot example, which the final one-shot set no longer uses.
- Electrolytes (sodium, chloride, bicarbonate, glucose) must come from the serum chemistry panel, not a blood gas.
  Rows that used blood-gas values were replaced. Where possible the note test_set_1_31 has in the same slot is used
  (Anion Gap 494, 500; Albumin Corrected Anion Gap 769, 780, 786; Albumin Corrected Delta Gap 798; Albumin Corrected
  Delta Ratio 811, 812), else another clean test_set_1_31 note for that calculator (809: row 819), else a Verified
  Extracted note (Sodium Correction 375, 383; Free Water Deficit 472; Delta Gap 729, 740; Delta Ratio 750, 756, 761).
  769 and 786 use test_set_1_31's edited notes (bicarbonate listed with the serum panel). Albumin Corrected Anion Gap 779
  has no clean test_set_1_31 option and uses Verified row 833 (pmc-5885426-1; Verified's edit, negative anion gap) pending
  a decision. All replacement notes are in equation_audit/overrides/ (extra_notes.csv records the source rows).
  CKD-EPI row 30 (patient on maintenance hemodialysis) uses the note v1.1, test_set_1_31 and Verified have in that row.
  One-shot examples 65 (Albumin Corrected Anion Gap) and 66 (Albumin Corrected Delta Gap) took electrolytes from a blood
  gas; both were replaced by Verified's one-shot examples (Extracted, serum values; explanations from the calculator code).
- Notes that stated the answer itself were edited to remove only that phrase: the measured LDL in 8 LDL Calculated notes
  (rows 513, 516, 517, 526, 528, 529, 530, 532) and the albumin-corrected anion gap in row 772.
- QTc template rows are kept as in v1.0 (15 template notes repeat within their calculator, as in v1.0).
- The one-shot example for Calculator ID 43 (SOFA): explanation regenerated with the fixed code (answer unchanged, 9).

## Checks
- Every non-template row (828) has been checked in its final form by at least one of the two checks; all rows changed
  in the last two review rounds (106 + 14) were rechecked with
  `equation_audit/verify_dataset.py` and `equation_audit/verify_entities.py` (gpt-6-astra). Remaining flags are
  out-of-population cases, the conventions above, or reviewed judgment calls. The checker also flags Delta Gap rows
  because it uses the delta-delta formula; the benchmark follows MDCalc's Delta Gap (anion gap - 12).
- 0 test rows duplicate their calculator's one-shot example; 0 duplicate Calculator ID + Note ID pairs.
