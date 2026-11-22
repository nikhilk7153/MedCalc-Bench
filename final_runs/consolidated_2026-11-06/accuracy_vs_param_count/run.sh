#!/bin/sh
# Accuracy vs. number of parameters (paper Appendix E.1, Figures 2 and 3) on the consolidated outputs.
# Uses evaluation/attribute_analysis.py (parameters = Relevant Entities per row, as in Table 1); zero-shot, as in the paper.
# Run from anywhere: sh final_runs/consolidated_2026-11-06/accuracy_vs_param_count/run.sh
# MEDCALC_VARIANT=astra (see ../variant.py): the Astra set's Relevant Entities and outputs, written to ../astra/accuracy_vs_param_count.
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
DATA="$ROOT/dataset/test_data_10_22_corrected.csv"
OUT="$HERE/.."
if [ "$MEDCALC_VARIANT" = "astra" ]; then
  DATA="$ROOT/extra_runs/astra_corrections_rerun_2026-10-22/input/test_data_10_22_astra.csv"
  OUT="$HERE/../astra"
fi
mkdir -p "$OUT/accuracy_vs_param_count"
cd "$ROOT/evaluation" && python3 attribute_analysis.py \
  --data "$DATA" \
  --outputs_dir "$OUT/outputs" \
  --out_dir "$OUT/accuracy_vs_param_count"
