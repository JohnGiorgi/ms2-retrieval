#!/bin/bash

INPUT_FP="$1"
OUTPUT_DIR="$2"
DEVICE="$3"

# See: https://github.com/castorini/pyserini#dense-indexes
python -m pyserini.encode \
  input   --corpus "$INPUT_FP" \
          --fields text \
          --delimiter "\n" \
          --shard-id 0 \
          --shard-num 1 \
  output  --embeddings "$OUTPUT_DIR" \
          --to-faiss \
  encoder --encoder castorini/tct_colbert-v2-hnp-msmarco \
          --fields text \
          --batch 32 \
          --device "$DEVICE"