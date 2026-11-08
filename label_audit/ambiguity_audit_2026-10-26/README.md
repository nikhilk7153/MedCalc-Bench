# Ambiguity audit (2026-10-26)

Looks for test rows whose note supports more than one answer. Base set: `dataset/test_data_10_22_corrected.csv`.
No dataset was changed.

## Part A: earlier annotation rounds (done)
- **Method:** `cross_version.py` compares our inputs with the earlier readings of the identical note in v1.0, 1/17/2025,
  1_31 and Verified, then recomputes each alternative with our calculator code. Recomputing removes differences caused
  by the earlier versions' code bugs, and spelling or unit-encoding differences are ignored.
- **Coverage:** 997 rows have an earlier reading of the identical note.
- **Flags:** 78 rows have an alternative label outside our accepted range (`cross_version_flags.csv`).
- **Adjudication:** four independent reviewers read every flagged note (`adjudicate_batch_*.jsonl` ->
  `adjudication_*.csv`). Corrected labels were recomputed with the calculator code.

**Results** (`ambiguity_audit_results.csv`):

| Verdict | Rows |
|---|---|
| Our label holds | 58 of the 78 flagged rows. Rows 268 and 350 hold on the disputed input but are label errors on another input. |
| Label error in our set | 6: 230, 261, 266, 268, 278, 350 |
| Ambiguous (more than one defensible answer) | 15: 131, 222, 236, 265, 269, 275, 276, 306, 310, 317, 349, 454, 562, 909, 915 |

- Rows 268, 349 and 350 are side findings: a risk factor neither reading scored, and HAS-BLED counting anticoagulants,
  which the score excludes.

**Effect on accuracy** (`accuracy_effect.csv`, all 26 runs): applying the 6 fixes, and then either excluding the
ambiguous rows or accepting any defensible answer, changes accuracy by at most 0.52 points.

## Part B: fresh reading of all 828 non-template rows (not run)
`fresh_reading.py --build` wrote `batch_input.jsonl` (gpt-4o-2024-08-06, temperature 0; about $2.55 via the Batch API).
It finds ambiguity that all earlier rounds missed in the same way, such as rows nobody disputed. It needs the user's
go-ahead before submission, and the scoring step is still to be written.

## Caveats
- The adjudicators are models, not physicians. The 6 errors and 15 ambiguous rows still need physician sign-off.
- Row 266: the adjudicator found 9; the earlier astra review proposed 7.
