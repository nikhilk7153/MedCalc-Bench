#!/bin/bash
# Runs on the vast.ai instance: stops the HF comparison runs (their outputs stay on disk), then regenerates the
# Mixtral rows that stopped mid-sentence under three settings.
cd /workspace/medcalc/evaluation
export HF_HOME=/workspace/.hf_home HF_HUB_OFFLINE=1
pkill -f "[m]ixtral_first_hf.sh"
pkill -f "[c]ompare_hf.py --model"
sleep 15
echo "$(date) HF comparison stopped; start Mixtral stop diagnosis" >> logs/queue_hf.log
/workspace/venv/bin/python diag_mixtral_stops.py --tp 4 > logs/diag_tp4_double.log 2>&1
CUDA_VISIBLE_DEVICES=0,1 /workspace/venv/bin/python diag_mixtral_stops.py --tp 2 > logs/diag_tp2_double.log 2>&1 &
CUDA_VISIBLE_DEVICES=2,3 /workspace/venv/bin/python diag_mixtral_stops.py --tp 2 --single_bos > logs/diag_tp2_single.log 2>&1 &
wait
echo "$(date) Mixtral stop diagnosis finished" >> logs/queue_hf.log
