#!/bin/bash
# Runs on the vast.ai instance: restarts the Mixtral HF one-shot run without the 256-token cap and marks
# logs/queue_hf.log when every compare_hf.py run has finished.
cd /workspace/medcalc/evaluation
export HF_HOME=/workspace/.hf_home HUGGINGFACE_TOKEN=$(cat /workspace/.hf_home/token)
mkdir -p outputs/test_data_final/hf_compare_capped256
mv outputs/test_data_final/hf_compare/mistralai_Mixtral-8x7B-Instruct-v0.1_shard0.jsonl outputs/test_data_final/hf_compare_capped256/
CUDA_VISIBLE_DEVICES=2,3 /workspace/venv/bin/python compare_hf.py --model mistralai/Mixtral-8x7B-Instruct-v0.1 --prompts one_shot --n 300 > logs/compare_hf_mixtral.log 2>&1 &
sleep 60
while pgrep -f "[c]ompare_hf.py --model" > /dev/null; do sleep 30; done
echo "$(date) HF one-shot runs finished" >> logs/queue_hf.log
