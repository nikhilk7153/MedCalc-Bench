#!/bin/bash
# Runs on the vast.ai instance. GPUs 0-1 and 2-3 are NVLink pairs, so each 70B model gets one pair (tp 2).
cd /workspace/medcalc/evaluation
export HF_HOME=/workspace/.hf_home HF_HUB_OFFLINE=1
while pgrep -f "[r]un.py --model mistralai/Mixtral" > /dev/null; do sleep 15; done
echo "$(date) Mixtral finished; start Llama3-70B (GPUs 0,1) and Meditron-70B (GPUs 2,3)"
CUDA_VISIBLE_DEVICES=0,1 /workspace/venv/bin/python run.py --model meta-llama/Meta-Llama-3-70B-Instruct --tp 2 > logs/Meta-Llama-3-70B-Instruct.log 2>&1 &
CUDA_VISIBLE_DEVICES=2,3 /workspace/venv/bin/python run.py --model epfl-llm/meditron-70b --tp 2 > logs/meditron-70b.log 2>&1 &
wait
echo "$(date) 70B runs finished; start PMC rerun"
CUDA_VISIBLE_DEVICES=0 /workspace/venv/bin/python run.py --model axiong/PMC_LLaMA_13B --tp 1 > logs/PMC_LLaMA_13B.log 2>&1
echo "$(date) all done"
