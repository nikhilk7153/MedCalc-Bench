#!/bin/bash
# Runs on the vast.ai instance: 300 one-shot rows through the original HF pipeline for Llama3-70B (GPUs 0,1)
# and Mixtral (GPUs 2,3), after the Llama3-8B comparison has finished.
cd /workspace/medcalc/evaluation
export HF_HOME=/workspace/.hf_home HUGGINGFACE_TOKEN=$(cat /workspace/.hf_home/token)
while pgrep -f "[c]ompare_hf.py --shard" > /dev/null; do sleep 15; done
echo "$(date) start HF one-shot runs"
CUDA_VISIBLE_DEVICES=0,1 /workspace/venv/bin/python compare_hf.py --model meta-llama/Meta-Llama-3-70B-Instruct --prompts one_shot --n 300 > logs/compare_hf_llama70b.log 2>&1 &
CUDA_VISIBLE_DEVICES=2,3 /workspace/venv/bin/python compare_hf.py --model mistralai/Mixtral-8x7B-Instruct-v0.1 --prompts one_shot --n 300 > logs/compare_hf_mixtral.log 2>&1 &
wait
echo "$(date) HF one-shot runs finished"
