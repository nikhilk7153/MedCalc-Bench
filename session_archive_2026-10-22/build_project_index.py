"""Write PROJECT_INDEX.csv: every important file/folder of the project with its status, size and checksum."""
import csv, glob, hashlib, os
import pandas as pd

R = [  # (group, path, status, description)
 ("1 Data", "dataset/test_data_10_22.csv", "CURRENT (used by all final runs)", "Final test set, 1,048 rows / 55 calculators / 768 with a Time Point Instruction. Only change after the runs: PERC wording 'do increment' -> 'do not increment' in 19 explanations (no answers changed)."),
 ("1 Data", "dataset/test_data_10_22_corrected.csv", "CURRENT (corrected copy)", "test_data_10_22.csv + 13 label fixes (rows 240,241,304,318,350,360,362,447,452,461,465,500,849) + CKD-EPI explanation wording. Not used for the main runs."),
 ("1 Data", "final_files_2026-10-16/one_shot_10_17_2026.json", "CURRENT (used by all final runs)", "One-shot examples, 55 calculators (CSV twin: one_shot_10_17_2026.csv)."),
 ("1 Data", "final_files_2026-10-16/one_shot_10_17_2026_corrected.json", "CURRENT (corrected copy)", "Same, with only the CKD-EPI (calculator 3) explanation regenerated after the wording fix (CSV twin: one_shot_10_17_2026_corrected.csv)."),
 ("1 Data", "dataset/test_data_10_22_astra.csv", "EXPERIMENTAL (unreviewed astra corrections)", "test_data_10_22_corrected.csv + all pending gpt-6-astra corrections that match current rows (45 rows changed: 26 answers, 6 notes). For measuring their effect only."),
 ("1 Data", "dataset/test_data.csv", "REFERENCE", "MedCalc-Bench v1.0 test set (1,047 rows), the paper's original."),
 ("1 Data", "dataset/test_data_1_17_2025.csv", "REFERENCE", "January 2025 version (source of the 1/17 handwritten notes)."),
 ("1 Data", "dataset/test_data_10_22_2026.csv", "OTHER VERSION (not used for runs)", "A second '10/22' build from the 10/20 pipeline; not the file the runs used."),
 ("1 Data", "dataset/test_data_10_20_2026.csv", "SUPERSEDED", "Earlier build."),
 ("2 Review", "dataset/label_corrections_review_10_22.csv", "CURRENT (decisions pending)", "54 review items: 13 wrong labels (11 audit + 2 sweep), 5 ambiguous, 17 astra corrections not applied, 12 astra flags on older row versions, 7 blood-gas rows. Blank Decision columns."),
 ("2 Review", "dataset/physician_review.csv", "PENDING (no decisions yet)", "198-item physician review sheet from the 10/20 pipeline."),
 ("2 Review", "equation_audit/astra_rule_based_corrections.csv", "REFERENCE", "gpt-6-astra review: 104 rule-based corrections (88 input, 16 note edits) with status."),
 ("2 Review", "equation_audit/overrides/pending_physician_review/", "PENDING", "Machine-readable astra corrections set aside for physician review (89 entity overrides, 15 note edits)."),
 ("2 Review", "dataset/CHANGELOG.md", "REFERENCE", "Changelog of the 10/14-10/20 corrected-dataset pipeline (conventions, blood-gas rule, astra status)."),
 ("3 Code", "evaluation/run.py", "CURRENT", "Main runner (vLLM for open models, OpenAI SDK 2.x for GPT), direct/zero-shot/one-shot. Optional --fit_context (default behaviour unchanged). Since 2026-10-24 also saves Time Point Instruction and the prompt per row."),
 ("3 Code", "evaluation/evaluate.py", "CURRENT", "Scoring (check_correctness)."),
 ("3 Code", "evaluation/code_exec_timed.py", "CURRENT", "Code-augmented runner (paper's openai==0.28.0; run with .venv_openai028)."),
 ("3 Code", "evaluation/error_analysis_batch.py", "CURRENT", "Error analysis via the OpenAI Batch API (paper's prompt, gpt-4, temperature 0)."),
 ("3 Code", "evaluation/check_inputs.py", "CURRENT", "Pre-flight input check."),
 ("3 Code", "evaluation/table_stats.py", "CURRENT", "Accuracy by category."),
 ("3 Code", "evaluation/attribute_analysis.py", "CURRENT", "Accuracy vs. number of attributes (paper Appendix E.1, Figures 2-3): attributes per calculator from name_to_python.json, second-degree fit per model; zero-shot by default."),
 ("3 Code", "calculator_implementations_verified/", "CURRENT", "Calculator code that generates labels (Serum Osm, PERC wording and CKD-EPI wording fixed)."),
 ("3 Code", ".venv_openai028/", "CURRENT", "Python env with openai==0.28.0 for code-augmented runs."),
 ("3 Code", "session_archive_2026-10-22/regen_labels.py", "CURRENT", "Regenerates labels/explanations from Relevant Entities with the verified code (dry run unless --write)."),
 ("4 Results", "final_runs/test_data_10_22/", "CURRENT (main results)", "8 models x 3 prompt styles on test_data_10_22.csv: outputs/ (24), results/ (24), logs/ (7)."),
 ("4 Results", "evaluation/outputs/test_data_10_22/", "CURRENT (code-augmented)", "code_exec_gpt-3.5-turbo-0125.jsonl, code_exec_gpt-4-0613.jsonl (results: evaluation/results/results_*_code_augmented_test_data_10_22.json)."),
 ("4 Results", "final_runs/test_data_10_22_corrected/", "CURRENT (re-scored)", "All 26 runs (24 main + 2 code-augmented), original model answers re-scored against test_data_10_22_corrected.csv (13 fixes)."),
 ("4 Results", "extra_runs/corrected_rows_ckdepi_rerun_2026-10-22/", "CURRENT (separate re-run)", "All 8 models re-run on the 13 corrected + 20 CKD-EPI rows with the corrected one-shot; per-model folders, error_analysis/, MANIFEST.md."),
 ("4 Results", "extra_runs/astra_corrections_rerun_2026-10-22/", "EXPERIMENTAL (separate re-run)", "All 8 models (+ code-augmented) on the 45 astra-corrected rows; per-model folders, error_analysis/, astra_changes_applied.csv, MANIFEST.md."),
 ("4 Results", "extra_runs/mdrd_sodium_wording_rerun_2026-10-22/", "CURRENT (separate re-run, other session)", "MDRD + Sodium Correction rows (40) re-run with the reworded questions: 8 models zero/one-shot, GPT code-augmented, error analysis, MANIFEST.md."),
 ("4 Results", "final_runs/test_data_10_22_reworded/", "CURRENT (merged, other session)", "Main runs with the 40 reworded MDRD/Sodium rows merged in; zero/one-shot + code-augmented; merged error analysis; MANIFEST.md."),
 ("4 Results", "extra_runs/sofa_oneshot_rerun_2026-10-23/", "CURRENT (separate re-run, other session)", "20 SOFA rows, one-shot, corrected SOFA example (answer 7): GPT-3.5 only (via gpt-3.5-turbo-16k alias). GPT-4 retired, not run."),
 ("4 Results", "extra_runs/oss_rerun_2026-10-24/", "CURRENT (separate re-run)", "Six open models: sofa_oneshot/ (20 SOFA rows, one-shot, corrected example) and ldl_friedewald/ (20 LDL rows x 3 styles, Friedewald question), each with outputs/ results/ logs/; MANIFEST.md. GPT models retired, not run; no error analysis yet."),
 ("4 Results", "extra_runs/pmc_llama_fit_context_2026-10-24/", "CURRENT (separate re-run)", "PMC-LLaMA, all 1,048 rows x 3 styles, with run.py --fit_context 512 (note kept whole when it fits) instead of the fixed 256-token cut. Accuracy essentially unchanged; MANIFEST.md."),
 ("4 Results", "final_runs/test_data_10_22_with_prompts/", "CURRENT (audit copy)", "The 24 main-run outputs with Time Point Instruction and the reconstructed prompts added; all original fields unchanged; MANIFEST.md."),
 ("4 Results", "extra_runs/time_point_replay_check_2026-10-24/", "AUDIT", "Llama3-8B replay of the final run with and without the time-point instruction: replay reproduces the original outputs, the control does not; MANIFEST.md."),
 ("4 Results", "extra_runs/gpt-4o-2024-08-06_benchmark_2026-10-24/", "CURRENT (extra model)", "GPT-4o (2024-08-06) on the as-run test set: direct/zero-shot/one-shot + code-augmented; MANIFEST.md."),
 ("4 Results", "extra_runs/question_clarifications_rerun_2026-10-22/", "UNUSED SKELETON", "Inputs only, from an interrupted step; nothing was run. Safe to delete."),
 ("4 Results", "extra_runs/corrected_rows_rerun_2026-10-22/", "SUPERSEDED", "Earlier GPT-only re-run of 11 rows (see its README)."),
 ("4 Results", "evaluation/attribute_analysis_10_22/", "CURRENT (Appendix E.1 figures)", "Zero-shot accuracy vs. number of attributes for the 24 main runs: equation- and rule-based figures, points CSV, fit coefficients. Replaces the paper's Figures 2-3, which were inconsistent with Table 2."),
 ("5 Error analysis", "evaluation/error_analysis_10_22/", "CURRENT (main)", "12,079 wrong zero/one-shot answers from the 16 main runs, classified (16 *.error_type.json); references: test_data_10_22.csv explanations."),
 ("5 Error analysis", "extra_runs/corrected_rows_ckdepi_rerun_2026-10-22/error_analysis/", "CURRENT (re-run)", "457 wrong answers from the re-run; references: corrected explanations."),
 ("5 Error analysis", "extra_runs/astra_corrections_rerun_2026-10-22/error_analysis/", "EXPERIMENTAL (astra re-run)", "630 wrong answers from the astra re-run; references: astra-dataset explanations."),
 ("5 Error analysis", "evaluation/error_analysis_1k/", "SUPERSEDED", "1,000-request trial."),
 ("6 Backups", "gpu_backup_2026-10-22/", "BACKUP", "Full copy of the GPU instance's /workspace/medcalc and helper files (checksum-verified; no model weights)."),
 ("6 Backups", "session_archive_2026-10-22/", "BACKUP", "Working files: dataset backups before every edit (incl. test_data_10_22.before_perc_wording.csv), audit inputs, scripts."),
 ("7 Old runs", "evaluation/outputs/test_data_final/", "SUPERSEDED", "First open-model run (double BOS, PMC stop token)."),
 ("7 Old runs", "evaluation/outputs/test_data_final_r2/", "SUPERSEDED", "Second run, one-shot with criteria lists."),
 ("7 Old runs", "evaluation/outputs/test_data_final_r2_nocrit/", "SUPERSEDED", "Second run, one-shot without criteria lists."),
 ("7 Old runs", "evaluation/outputs/test_data_10_22_tp_in_system/", "SUPERSEDED", "Time-point sentence in the system prompt (old placement)."),
 ("7 Old runs", "evaluation/outputs/test_data_10_22_code_exec_openai_v2/", "SUPERSEDED", "Partial code-augmented runs with the newer SDK (not the paper's)."),
 ("7 Old runs", "evaluation/outputs/test_data_10_20_2026/", "SUPERSEDED", "Earlier dataset build."),
 ("7 Old runs", "evaluation/outputs/pmc_first_run_eos1/", "SUPERSEDED", "PMC-LLaMA before the stop-token fix."),
 ("7 Old runs", "evaluation/outputs/code_exec_trial*", "SUPERSEDED", "Code-augmented trials."),
 ("8 Secrets", ".env", "KEEP PRIVATE", "OpenAI API key used by the scripts. Do not share."),
]

def info(p):
    paths = glob.glob(p.rstrip("/")) if "*" in p else [p.rstrip("/")]
    files = size = 0; md5 = ""
    for q in paths:
        if os.path.isdir(q):
            for root, _, fs in os.walk(q):
                for f in fs:
                    fp = os.path.join(root, f)
                    if os.path.isfile(fp): files += 1; size += os.path.getsize(fp)  # skips broken symlinks
        elif os.path.isfile(q):
            files += 1; size += os.path.getsize(q)
            if q != ".env": md5 = hashlib.md5(open(q, "rb").read()).hexdigest()
    return files, size, md5, bool(paths) and all(os.path.exists(x) for x in paths)

if __name__ == "__main__":
    with open("PROJECT_INDEX.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["Group", "Path", "Status", "Description", "Files", "Size (MB)", "md5 (files)", "Exists"])
        for g, p, s, desc in R:
            n, size, md5, ok = info(p); w.writerow([g, p, s, desc, n, round(size / 1e6, 2), md5, ok])
    x = pd.read_csv("PROJECT_INDEX.csv")
    print("PROJECT_INDEX.csv:", len(x), "entries | missing:", x.loc[~x["Exists"], "Path"].tolist())
