#!/bin/bash
# Runs on the vast.ai instance: re-runs all open models with the fixed run.py (single BOS, explicit eos ids,
# Steroid Conversion one-shot question) into outputs/test_data_final_r2/. GPUs 0-1 and 2-3 are NVLink pairs.
cd /workspace/medcalc/evaluation
export HF_HOME=/workspace/.hf_home HF_HUB_OFFLINE=1
TAG=test_data_final_r2
PY=/workspace/venv/bin/python
echo "$(date) start r2"
(
    CUDA_VISIBLE_DEVICES=0,1 $PY run.py --model meta-llama/Meta-Llama-3-70B-Instruct --tp 2 --tag $TAG > logs/r2_Meta-Llama-3-70B-Instruct.log 2>&1
    CUDA_VISIBLE_DEVICES=0,1 $PY run.py --model mistralai/Mixtral-8x7B-Instruct-v0.1 --tp 2 --tag $TAG > logs/r2_Mixtral-8x7B-Instruct-v0.1.log 2>&1
) &
(
    sleep 60
    CUDA_VISIBLE_DEVICES=2,3 $PY run.py --model epfl-llm/meditron-70b --tp 2 --tag $TAG > logs/r2_meditron-70b.log 2>&1
    CUDA_VISIBLE_DEVICES=2 $PY run.py --model meta-llama/Meta-Llama-3-8B-Instruct --tag $TAG > logs/r2_Meta-Llama-3-8B-Instruct.log 2>&1 &
    sleep 60
    CUDA_VISIBLE_DEVICES=3 $PY run.py --model mistralai/Mistral-7B-Instruct-v0.2 --tag $TAG > logs/r2_Mistral-7B-Instruct-v0.2.log 2>&1
    wait
    CUDA_VISIBLE_DEVICES=2 $PY run.py --model axiong/PMC_LLaMA_13B --tag $TAG > logs/r2_PMC_LLaMA_13B.log 2>&1
) &
wait
echo "$(date) r2 done"
