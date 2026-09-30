#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TP="${TP:-1}"
MODEL_PATH="${MODEL_PATH:-$SCRIPT_DIR/models/zstanjj/MemSifter-4B-Thinking}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/data/results}"
METHOD="${METHOD:-DAPO-GenRank-MemSifter-4B-Thinking}"

mkdir -p "$SCRIPT_DIR/logs" || exit 1

# 在启动 Ray 和模型之前检查所有输入文件。
for DATASET in locomo split_longmemeval_s split_longmemeval_m; do
  INPUT_FILE="$OUTPUT_DIR/bge-m3/${DATASET}_test_embed.parquet"
  if [ ! -f "$INPUT_FILE" ]; then
    echo "输入数据不存在：$INPUT_FILE" >&2
    exit 1
  fi
done

if [ ! -f "$MODEL_PATH/config.json" ]; then
  echo "模型配置不存在：$MODEL_PATH/config.json" >&2
  exit 1
fi

for DATASET in \
  locomo \
  split_longmemeval_s \
  split_longmemeval_m
do
  LOG_FILE="$PWD/logs/${DATASET}_full_$(date +%Y%m%d_%H%M%S).log"

  python -u -B memsifter/ray_vllm_infer.py \
    --local \
    --prompt_name gen_retrieve_instruct_prompt.yaml \
    --dataset_name "$DATASET" \
    --dataset_split test \
    --embed_model bge-m3 \
    --output_dir "$OUTPUT_DIR" \
    --model_path "$MODEL_PATH" \
    --model_name "$METHOD" \
    --override_num 1 \
    --concurrency 1 \
    --tp "$TP" \
    --pp 1 \
    --batch_size 1 \
    --context_limit 128k \
    --filter_col similarities \
    --max_num_batched_tokens 133120 \
    --max_model_len 149504 \
    --max_output_token 16384 \
    --temperature 0.6 \
    2>&1 | tee "$LOG_FILE"

  # 当前数据集运行失败时停止循环
  if [ $? -ne 0 ]; then
    echo "运行失败，日志：$LOG_FILE"
    exit 1
  fi
done
