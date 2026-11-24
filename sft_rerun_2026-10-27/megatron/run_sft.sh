#!/bin/bash
# Fine-tune one model on the MedCalc-Bench training set as in the paper's Appendix C.2, with CodeAct's Megatron-LLM
# scripts (github.com/xingyaoww/code-act, commit d607f56; Megatron-LLM submodule at 1b07577).
#
#   CODEACT=/path/to/code-act bash run_sft.sh Mistral-7b     (or Llama-2-7b)
#
# Run inside CodeAct's Megatron environment (docker image xingyaoww/pt-megatron-llm:v1.1.1, see code-act's
# docs/MODEL_TRAINING.md) on one 4xA100 node. Before running, download the base models into code-act:
#   data/models/raw_hf/Mistral-7B-v0.1   (huggingface.co/mistralai/Mistral-7B-v0.1)
#   data/models/raw_hf/Llama-2-7b-hf     (huggingface.co/meta-llama/Llama-2-7b-hf)
# Optional: TRAIN_DATA=<chatML jsonl> (default ../data/train_chatml.jsonl; ../data/clean/train_chatml.jsonl is the
# de-duplicated set); SEQ_LENGTH=<packing length> (default: CodeAct's, below).
#
# Differences from CodeAct's own finetune_4xA100_4tp_mixture_*.sh:
#   - data: the MedCalc chatML training set instead of CodeAct's mixture; the number of packed examples is read from
#     the preprocessing output instead of being hard-coded;
#   - 2 epochs (paper) instead of 5;
#   - sources scripts/models/megatron/source.sh (CodeAct's Llama-2 script points at scripts/train/..., which does not exist);
#   - the conversion back to Hugging Face calls scripts/models/add_chat_format.py (CodeAct's convert_sharded_to_hf.sh
#     points at scripts/chat_interface/..., which does not exist).
# Unchanged (paper and CodeAct agree): lr 1e-5, 50 warmup steps, cosine decay (min lr 1e-6), global batch 32, 4 GPUs with
# tensor parallel 4, loss on the assistant turn only, packing, flash attention, bf16, full fine-tuning.
# Sequence length: CodeAct's 16,384 for Mistral and 4,096 for Llama-2 (the paper gives none). Packing truncates any
# single conversation longer than that; no MedCalc training conversation is (longest: about 4,000 tokens).
# On the 9,871-row set (Megatron-LLM's preprocessing, run 2026-11-24): Mistral at 16,384 packs into 815 examples, i.e.
# 52 steps over 2 epochs, so the 50 warmup steps take almost the whole run; at 4,096, Mistral 3,967 examples (248 steps)
# and Llama-2 4,096 examples (258 steps). SEQ_LENGTH=4096 is recommended for both.
set -euo pipefail

MODEL=${1:?usage: run_sft.sh Mistral-7b|Llama-2-7b}
CODEACT=${CODEACT:?set CODEACT to the code-act checkout}
HERE=$(cd "$(dirname "$0")" && pwd)
TRAIN_DATA=$(realpath "${TRAIN_DATA:-$HERE/../data/train_chatml.jsonl}")

case $MODEL in
    Mistral-7b) MODEL_TYPE=mistral; HF_DIR=Mistral-7B-v0.1; SEQ_LENGTH=${SEQ_LENGTH:-16384} ;;
    Llama-2-7b) MODEL_TYPE=llama2;  HF_DIR=Llama-2-7b-hf;   SEQ_LENGTH=${SEQ_LENGTH:-4096} ;;
    *) echo "unknown model $MODEL (Mistral-7b or Llama-2-7b)"; exit 1 ;;
esac
DATA_NAME=medcalc_$(basename "$(dirname "$TRAIN_DATA")")_${MODEL}_pack${SEQ_LENGTH}
EXP_ID=${MODEL}-medcalc-${DATA_NAME}-ep2
cd "$CODEACT"
[ -d "data/models/raw_hf/$HF_DIR" ] || { echo "download the base model to $CODEACT/data/models/raw_hf/$HF_DIR first"; exit 1; }

# 1. Hugging Face -> Megatron, then shard over 4 GPUs (CodeAct's scripts)
[ -d "data/models/raw/${MODEL}-megatron" ] || bash "scripts/models/megatron/convert_${MODEL_TYPE/llama2/llama}.sh"
[ -d "data/models/sharded/${MODEL}-megatron-tp4-pp1" ] || bash scripts/models/megatron/shard_model_4tp.sh "$MODEL"

# 2. chatML -> Megatron's packed format (as in CodeAct's process_mixture_*.sh)
OUT_PREFIX=data/megatron_format/$DATA_NAME/data
mkdir -p "$(dirname "$OUT_PREFIX")"
python Megatron-LLM/tools/preprocess_instruct_data.py \
    --input "$TRAIN_DATA" \
    --output_prefix "$OUT_PREFIX" \
    --tokenizer_type SentencePieceTokenizer \
    --vocab_file "data/models/raw/${MODEL}-megatron/tokenizer.model" \
    --vocab_extra_ids_list "<|im_start|>,<|im_end|>" \
    --chunk_size 32 \
    --workers 16 \
    --no_new_tokens \
    --do_packing \
    --max_seq_length "$SEQ_LENGTH" | tee "$(dirname "$OUT_PREFIX")/preprocess.log"
# the script ends with "Packed <n> documents into <N> documents."; N is the number of training examples
N_EXAMPLES=$(sed -n 's/^Packed [0-9]* documents into \([0-9]*\) documents\.$/\1/p' "$(dirname "$OUT_PREFIX")/preprocess.log")
[ -n "$N_EXAMPLES" ] || { echo "could not read the packed example count"; exit 1; }
echo "packed training examples: $N_EXAMPLES"

# 3. Train (CodeAct's train function with the paper's settings)
export WANDB_MODE=${WANDB_MODE:-offline}  # CodeAct always logs to wandb; offline needs no account
set +u  # source.sh reads optional variables that may be unset
source scripts/models/megatron/source.sh
train \
    --tp 4 --pp 1 \
    --lr 1e-5 \
    --n_warmup_steps 50 \
    --seq_length "$SEQ_LENGTH" \
    --model_name "$MODEL" \
    --data_name "$DATA_NAME" \
    --n_examples "$N_EXAMPLES" \
    --global_batch_size 32 \
    --n_epochs 2 \
    --n_gpu_per_node 4 \
    --recompute_num_layers 2 \
    --exp_id "$EXP_ID" \
    --use_flash_attn \
    --packed_input
set -u

# 4. Last checkpoint -> Hugging Face with the chatML template (CodeAct's convert_sharded_to_hf.sh, path fixed)
CKPT=data/ckpts/$EXP_ID
ITER=$(cat "$CKPT/latest_checkpointed_iteration.txt")
HF_OUT=$CKPT/hf/model_iter_$ITER
TMP=tmp/megatron-conversion/$CKPT
mkdir -p "$TMP"
python Megatron-LLM/tools/checkpoint_util.py \
    --target_tensor_parallel_size 1 --target_pipeline_parallel_size 1 \
    --load_dir "$CKPT" --load_iters "$ITER" --save_dir "$TMP" \
    --model_type "$MODEL_TYPE" --true_vocab_size 32002 --use_distributed_optimizer --bf16
python Megatron-LLM/weights_conversion/megatron_to_hf.py \
    --model="$MODEL_TYPE" --input_dir="$TMP" --output_dir="$HF_OUT" \
    --vocab_file="data/models/raw/${MODEL}-megatron/tokenizer.model" \
    --vocab_extra_ids_list "<|im_start|>,<|im_end|>" --no_new_tokens
python3 scripts/models/add_chat_format.py "$HF_OUT"
rm -rf tmp/megatron-conversion
echo "fine-tuned model: $CODEACT/$HF_OUT"
