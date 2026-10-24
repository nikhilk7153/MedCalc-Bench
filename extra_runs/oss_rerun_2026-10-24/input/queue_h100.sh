#!/bin/bash
# Six open models: (1) SOFA rows, one-shot, corrected SOFA example -> tag sofa_oneshot_rerun
#                  (2) LDL rows, 3 prompt styles, Friedewald question -> tag ldl_friedewald
# GPUs 0-1 and 2-3 are NVLink pairs; 70B models use a pair (tp 2).
cd /workspace/medcalc/evaluation
export HF_HOME=/workspace/.hf_home HF_HUB_OFFLINE=1
PY=/workspace/venv/bin/python
W=/workspace/medcalc/extra_ldl/as_text.py
SOFA="--prompt one_shot --data ../extra_sofa/test_rows.csv --one_shot ../extra_sofa/one_shot_10_17_2026.json --tag sofa_oneshot_rerun"
LDL="--data ../extra_ldl/test_rows.csv --one_shot ../extra_ldl/one_shot_10_17_2026_corrected.json --tag ldl_friedewald"
mkdir -p logs
$PY $W check_inputs.py --data ../extra_ldl/test_rows.csv --one_shot ../extra_ldl/one_shot_10_17_2026_corrected.json || { echo "$(date) LDL input check failed"; exit 1; }
both() {  # $1 = CUDA devices, $2 = model, $3 = extra args, $4 = log name
    CUDA_VISIBLE_DEVICES=$1 $PY $W run.py --model $2 $3 $SOFA > logs/sofa_$4.log 2>&1
    CUDA_VISIBLE_DEVICES=$1 $PY $W run.py --model $2 $3 $LDL > logs/ldl_$4.log 2>&1
}
echo "$(date) start"
(
    both 0,1 meta-llama/Meta-Llama-3-70B-Instruct "--tp 2" Meta-Llama-3-70B-Instruct
    both 0,1 mistralai/Mixtral-8x7B-Instruct-v0.1 "--tp 2" Mixtral-8x7B-Instruct-v0.1
) &
(
    sleep 60
    both 2,3 epfl-llm/meditron-70b "--tp 2" meditron-70b
    both 2 meta-llama/Meta-Llama-3-8B-Instruct "" Meta-Llama-3-8B-Instruct &
    sleep 60
    both 3 mistralai/Mistral-7B-Instruct-v0.2 "" Mistral-7B-Instruct-v0.2
    wait
    both 2 axiong/PMC_LLaMA_13B "" PMC_LLaMA_13B
) &
wait
echo "$(date) all done"
