# Snapshot of the MedCalc-Bench-Verified calculator code used by the audit scripts

`calculator_implementations/` is a copy of `medcalc_verified_code/calculator_implementations/` as it was on disk on
2026-11-24: a clone of https://github.com/nikhilk7153/MedCalc-Bench-Verified, branch `fix/anion-gap-and-sofa`,
commit `810287c` (2026-10-14, "SOFA: let a measured MAP take precedence over the hypotension flag"), plus two
uncommitted edits, saved in `local_edits.diff`:
- `sOsm.py`: the explanation text showed BUN where glucose belongs in "(glucose / 18)" (the computed value was right);
- `name_to_python.json`: three question typos ("body mass mass index", "the the" twice).

That branch is not in the public MedCalc-Bench-Verified repository; it and `fix/one-shot-anion-gap-chloride` are kept in
the authors' private repository `nikhilk7153/medcalc-bench-private` (2026-11-24). It is read by
`equation_audit/verify_entities.py`, `propose_fixes.py` and `build_physician_review.py` (`$MEDCALC_IMPL` overrides it).
The paper's labels do not depend on it: they come from `calculator_implementations_verified/` (the same commit with
the criteria lists removed from the explanations, as in v1.0), via `equation_audit/run_generate_csv.py`.
