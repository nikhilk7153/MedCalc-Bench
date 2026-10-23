#!/bin/bash
# Runs on the vast.ai instance: pauses the Llama3-70B HF run, runs the Mixtral HF one-shot sample as two copies
# (GPUs 0,1 and 2,3), then resumes Llama3-70B (compare_hf.py skips rows already written).
cd /workspace/medcalc/evaluation
export HF_HOME=/workspace/.hf_home HUGGINGFACE_TOKEN=$(cat /workspace/.hf_home/token)
pkill -f "[r]estart_mixtral_hf.sh"
pkill -f "[c]ompare_hf.py --model"
sleep 10
OUT=outputs/test_data_final/hf_compare
mkdir -p outputs/test_data_final/hf_compare_partial
mv $OUT/mistralai_Mixtral-8x7B-Instruct-v0.1_shard0.jsonl outputs/test_data_final/hf_compare_partial/
echo "$(date) start Mixtral HF one-shot, 2 copies" >> logs/queue_hf.log
for k in 0 1; do
    CUDA_VISIBLE_DEVICES=$((2 * k)),$((2 * k + 1)) /workspace/venv/bin/python compare_hf.py --model mistralai/Mixtral-8x7B-Instruct-v0.1 --prompts one_shot --n 300 --shard $k --num_shards 2 > logs/compare_hf_mixtral_$k.log 2>&1 &
done
wait
echo "$(date) Mixtral HF one-shot finished; resume Llama3-70B" >> logs/queue_hf.log
CUDA_VISIBLE_DEVICES=0,1 /workspace/venv/bin/python compare_hf.py --model meta-llama/Meta-Llama-3-70B-Instruct --prompts one_shot --n 300 > logs/compare_hf_llama70b.log 2>&1
echo "$(date) HF one-shot runs finished" >> logs/queue_hf.log
