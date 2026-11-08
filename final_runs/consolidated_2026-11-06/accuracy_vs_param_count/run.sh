#!/bin/sh
# Accuracy vs. number of parameters (paper Appendix E.1, Figures 2 and 3) on the consolidated outputs.
# Uses evaluation/attribute_analysis.py unchanged; zero-shot as in the paper (pass --setting one_shot etc. for others).
# Run from anywhere: sh final_runs/consolidated_2026-11-06/accuracy_vs_param_count/run.sh [--setting zero_shot]
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
cd "$ROOT/evaluation" && python3 attribute_analysis.py \
  --data "$ROOT/dataset/test_data_10_22.csv" \
  --outputs_dir "$HERE/../outputs" \
  --calculators "$ROOT/calculator_implementations_verified/name_to_python.json" \
  --out_dir "$HERE" "$@"
