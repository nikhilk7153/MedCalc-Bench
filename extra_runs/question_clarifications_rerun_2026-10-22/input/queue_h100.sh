#!/bin/bash
# 60 rows (LDL, MDRD, Sodium Correction question clarifications) x 3 prompt styles, six open models, corrected one-shot file.
# GPUs 0-1 and 2-3 are NVLink pairs; 70B models use a pair (tp 2).
cd /workspace/medcalc/evaluation
export HF_HOME=/workspace/.hf_home HF_HUB_OFFLINE=1
PY=/workspace/venv/bin/python
W=/workspace/medcalc/extra_qclar/as_text.py
ARGS="--data ../extra_qclar/test_rows.csv --one_shot ../extra_qclar/one_shot_10_17_2026_corrected.json --tag qclar_rows"
mkdir -p logs
$PY $W check_inputs.py --data ../extra_qclar/test_rows.csv --one_shot ../extra_qclar/one_shot_10_17_2026_corrected.json || { echo "$(date) input check failed"; exit 1; }
echo "$(date) start"
(
    CUDA_VISIBLE_DEVICES=0,1 $PY $W run.py --model meta-llama/Meta-Llama-3-70B-Instruct --tp 2 $ARGS > logs/qclar_Meta-Llama-3-70B-Instruct.log 2>&1
    CUDA_VISIBLE_DEVICES=0,1 $PY $W run.py --model mistralai/Mixtral-8x7B-Instruct-v0.1 --tp 2 $ARGS > logs/qclar_Mixtral-8x7B-Instruct-v0.1.log 2>&1
) &
(
    sleep 60
    CUDA_VISIBLE_DEVICES=2,3 $PY $W run.py --model epfl-llm/meditron-70b --tp 2 $ARGS > logs/qclar_meditron-70b.log 2>&1
    CUDA_VISIBLE_DEVICES=2 $PY $W run.py --model meta-llama/Meta-Llama-3-8B-Instruct $ARGS > logs/qclar_Meta-Llama-3-8B-Instruct.log 2>&1 &
    sleep 60
    CUDA_VISIBLE_DEVICES=3 $PY $W run.py --model mistralai/Mistral-7B-Instruct-v0.2 $ARGS > logs/qclar_Mistral-7B-Instruct-v0.2.log 2>&1
    wait
    CUDA_VISIBLE_DEVICES=2 $PY $W run.py --model axiong/PMC_LLaMA_13B $ARGS > logs/qclar_PMC_LLaMA_13B.log 2>&1
) &
wait
echo "$(date) all done"
