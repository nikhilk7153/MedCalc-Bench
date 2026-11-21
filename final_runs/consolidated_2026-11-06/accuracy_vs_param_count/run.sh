#!/bin/sh
# Accuracy vs. number of parameters (paper Appendix E.1, Figures 2 and 3) on the consolidated outputs.
# Uses evaluation/attribute_analysis.py (parameters = Relevant Entities per row, as in Table 1); zero-shot, as in the paper.
# Run from anywhere: sh final_runs/consolidated_2026-11-06/accuracy_vs_param_count/run.sh
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
cd "$ROOT/evaluation" && python3 attribute_analysis.py \
  --data "$ROOT/dataset/test_data_10_22_corrected.csv" \
  --outputs_dir "$HERE/../outputs" \
  --out_dir "$HERE"
