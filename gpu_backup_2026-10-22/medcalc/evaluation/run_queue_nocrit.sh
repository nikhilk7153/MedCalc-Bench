#!/bin/bash
# Runs on the vast.ai instance: one-shot only, with the one-shot examples whose criteria lists were removed
# (final_files_2026-10-20_no_criteria_lists), into outputs/test_data_final_r2_nocrit/ for comparison with test_data_final_r2.
cd /workspace/medcalc/evaluation
export HF_HOME=/workspace/.hf_home HF_HUB_OFFLINE=1
TAG=test_data_final_r2_nocrit
DIR=../final_files_2026-10-20_no_criteria_lists
ARGS="--prompt one_shot --data $DIR/test_data_final.csv --one_shot $DIR/one_shot_finalized_explanation.json --tag $TAG"
PY=/workspace/venv/bin/python
while pgrep -f "[d]iag_mixtral_hf.py" > /dev/null; do sleep 15; done
echo "$(date) start nocrit"
(
    CUDA_VISIBLE_DEVICES=0,1 $PY run.py --model meta-llama/Meta-Llama-3-70B-Instruct --tp 2 $ARGS > logs/nocrit_Meta-Llama-3-70B-Instruct.log 2>&1
    CUDA_VISIBLE_DEVICES=0,1 $PY run.py --model mistralai/Mixtral-8x7B-Instruct-v0.1 --tp 2 $ARGS > logs/nocrit_Mixtral-8x7B-Instruct-v0.1.log 2>&1
) &
(
    sleep 60
    CUDA_VISIBLE_DEVICES=2,3 $PY run.py --model epfl-llm/meditron-70b --tp 2 $ARGS > logs/nocrit_meditron-70b.log 2>&1
    CUDA_VISIBLE_DEVICES=2 $PY run.py --model meta-llama/Meta-Llama-3-8B-Instruct $ARGS > logs/nocrit_Meta-Llama-3-8B-Instruct.log 2>&1 &
    sleep 60
    CUDA_VISIBLE_DEVICES=3 $PY run.py --model mistralai/Mistral-7B-Instruct-v0.2 $ARGS > logs/nocrit_Mistral-7B-Instruct-v0.2.log 2>&1
    wait
    CUDA_VISIBLE_DEVICES=2 $PY run.py --model axiong/PMC_LLaMA_13B $ARGS > logs/nocrit_PMC_LLaMA_13B.log 2>&1
) &
wait
echo "$(date) nocrit done"
