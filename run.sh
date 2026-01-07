#!/bin/bash

# Quick Start Script for Narrative Consistency Model
# Track B - Kharagpur Data Science Hackathon 2026

echo "================================="
echo "Narrative Consistency Model"
echo "Track B - BDH-Inspired Reasoning"
echo "================================="
echo ""

# Check if novels directory exists
if [ ! -d "novels" ]; then
    echo "⚠️  Warning: 'novels' directory not found"
    echo "Please create a 'novels' directory and add your novel .txt files"
    echo ""
fi

# Install dependencies (use system python3)
echo "📦 Checking dependencies..."
if ! python3 -c "import torch, transformers, sentence_transformers, pathway" 2>/dev/null; then
    echo "Installing missing packages (this may take 10-20 minutes on first run)..."
    pip3 install -q -r requirements.txt
else
    echo "✓ Core dependencies already installed"
fi
echo ""

# Train the model
echo "🚀 Starting training..."
python3 train.py \
    --train_csv train.csv \
    --novels_dir novels \
    --batch_size 2 \
    --epochs 10 \
    --max_chunks 80 \
    --chunk_size 1000 \
    --overlap 200 \
    --hidden_dim 512 \
    --learning_rate 1e-4 \
    --use_focal_loss \
    --output_dir checkpoints

echo ""
echo "✓ Training completed"
echo ""

# Generate predictions
echo "🔮 Generating predictions..."
python3 predict.py \
    --test_csv test.csv \
    --novels_dir novels \
    --model_path checkpoints/best_model.pt \
    --batch_size 2 \
    --max_chunks 80 \
    --output_csv results.csv

echo ""
echo "================================="
echo "✓ All done!"
echo "Results saved to: results.csv"
echo "================================="
