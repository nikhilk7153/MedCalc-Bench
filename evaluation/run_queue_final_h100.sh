#!/bin/bash
# Runs on the 4x H100 NVL instance: all six open models, all three prompt styles, on the final test set and one-shot
# examples, into outputs/test_data_10_22/. GPUs 0-1 and 2-3 are NVLink pairs, so each 70B model gets a pair (tp 2).
cd /workspace/medcalc/evaluation
export HF_HOME=/workspace/.hf_home HF_HUB_OFFLINE=1
PY=/workspace/venv/bin/python
ARGS="--data ../dataset/test_data_10_22.csv --one_shot ../final_files_2026-10-16/one_shot_10_17_2026.json --tag test_data_10_22"
mkdir -p logs
$PY check_inputs.py --data ../dataset/test_data_10_22.csv --one_shot ../final_files_2026-10-16/one_shot_10_17_2026.json || { echo "$(date) input check failed"; exit 1; }
echo "$(date) start"
(
    CUDA_VISIBLE_DEVICES=0,1 $PY run.py --model meta-llama/Meta-Llama-3-70B-Instruct --tp 2 $ARGS > logs/final_Meta-Llama-3-70B-Instruct.log 2>&1
    CUDA_VISIBLE_DEVICES=0,1 $PY run.py --model mistralai/Mixtral-8x7B-Instruct-v0.1 --tp 2 $ARGS > logs/final_Mixtral-8x7B-Instruct-v0.1.log 2>&1
) &
(
    sleep 60
    CUDA_VISIBLE_DEVICES=2,3 $PY run.py --model epfl-llm/meditron-70b --tp 2 $ARGS > logs/final_meditron-70b.log 2>&1
    CUDA_VISIBLE_DEVICES=2 $PY run.py --model meta-llama/Meta-Llama-3-8B-Instruct $ARGS > logs/final_Meta-Llama-3-8B-Instruct.log 2>&1 &
    sleep 60
    CUDA_VISIBLE_DEVICES=3 $PY run.py --model mistralai/Mistral-7B-Instruct-v0.2 $ARGS > logs/final_Mistral-7B-Instruct-v0.2.log 2>&1
    wait
    CUDA_VISIBLE_DEVICES=2 $PY run.py --model axiong/PMC_LLaMA_13B $ARGS > logs/final_PMC_LLaMA_13B.log 2>&1
) &
wait
echo "$(date) all done"
