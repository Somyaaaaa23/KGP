#!/bin/bash
# Fast training with performance tweaks and reduced data

echo "🚀 Fast Training (optimized for speed)"

python3 train.py \
    --train_csv train.csv \
    --novels_dir novels \
    --batch_size 4 \
    --epochs 5 \
    --max_chunks 50 \
    --chunk_size 800 \
    --overlap 100 \
    --hidden_dim 256 \
    --learning_rate 2e-4 \
    --threads 4 \
    --interop_threads 1 \
    --encoder_batch_size 128 \
    --cache_dir .cache/embeddings \
    --output_dir checkpoints

echo "✓ Training done! Model in checkpoints/"
