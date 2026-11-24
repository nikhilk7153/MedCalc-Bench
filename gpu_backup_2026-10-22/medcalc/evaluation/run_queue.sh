#!/bin/bash
cd /workspace/medcalc/evaluation
while pgrep -f "run_vllm.py --model axiong" > /dev/null; do sleep 15; done
for m in mistralai/Mixtral-8x7B-Instruct-v0.1 meta-llama/Meta-Llama-3-70B-Instruct epfl-llm/meditron-70b; do
    n=$(basename $m)
    until grep -q "DOWNLOADED $m" /workspace/dl_$n.log; do sleep 15; done
    echo "$(date) start $m"
    HF_HOME=/workspace/.hf_home HF_HUB_OFFLINE=1 /workspace/venv/bin/python run_vllm.py --model $m --tp 4 > logs/$n.log 2>&1
    echo "$(date) done $m exit=$?"
done
