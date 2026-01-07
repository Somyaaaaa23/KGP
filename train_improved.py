"""
Improved training with better handling of class imbalance
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
from tqdm import tqdm
import json
import pandas as pd
import numpy as np

from text_processing import NarrativeChunker, SemanticEncoder, ChunkProcessor
from performance import apply_optimizations
from narrative_memory import NarrativeMemory, MemoryUpdater
from model import ConsistencyClassifier, HybridReasoningModel, FocalLoss
from utils import (
    NarrativeDataset, 
    create_dataloaders, 
    MetricsTracker,
    save_checkpoint,
    load_checkpoint
)


def find_best_threshold(model, val_loader, device):
    """Find optimal classification threshold"""
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            narrative_chunks = batch['narrative_chunks'].to(device)
            backstory = batch['backstory'].to(device)
            chunk_mask = batch['chunk_mask'].to(device)
            labels = batch['label'].to(device).float()
            
            logits, _ = model(narrative_chunks, backstory, chunk_mask)
            logits = logits.squeeze(-1)
            probs = torch.sigmoid(logits)
            
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # Try different thresholds
    best_f1 = 0
    best_threshold = 0.5
    
    for threshold in np.arange(0.3, 0.7, 0.05):
        preds = (all_probs > threshold).astype(int)
        
        tp = ((preds == 1) & (all_labels == 1)).sum()
        fp = ((preds == 1) & (all_labels == 0)).sum()
        fn = ((preds == 0) & (all_labels == 1)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    return best_threshold, best_f1


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    metrics = MetricsTracker()
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    
    for batch in pbar:
        narrative_chunks = batch['narrative_chunks'].to(device)
        backstory = batch['backstory'].to(device)
        chunk_mask = batch['chunk_mask'].to(device)
        labels = batch['label'].to(device).float()
        
        optimizer.zero_grad()
        logits, _ = model(narrative_chunks, backstory, chunk_mask)
        logits = logits.squeeze(-1)
        
        loss = criterion(logits, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        probs = torch.sigmoid(logits)
        predictions = (probs > 0.5).long()
        
        metrics.update(predictions, labels.long(), loss.item())
        
        current_metrics = metrics.get_metrics()
        pbar.set_postfix({
            'loss': f"{current_metrics['loss']:.4f}",
            'acc': f"{current_metrics['accuracy']:.4f}"
        })
    
    return metrics.get_metrics()


def validate(model, val_loader, criterion, device, epoch, threshold=0.5):
    """Validate the model"""
    model.eval()
    metrics = MetricsTracker()
    
    pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]")
    
    with torch.no_grad():
        for batch in pbar:
            narrative_chunks = batch['narrative_chunks'].to(device)
            backstory = batch['backstory'].to(device)
            chunk_mask = batch['chunk_mask'].to(device)
            labels = batch['label'].to(device).float()
            
            logits, _ = model(narrative_chunks, backstory, chunk_mask)
            logits = logits.squeeze(-1)
            
            loss = criterion(logits, labels)
            
            probs = torch.sigmoid(logits)
            predictions = (probs > threshold).long()
            
            metrics.update(predictions, labels.long(), loss.item())
            
            current_metrics = metrics.get_metrics()
            pbar.set_postfix({
                'loss': f"{current_metrics['loss']:.4f}",
                'acc': f"{current_metrics['accuracy']:.4f}"
            })
    
    return metrics.get_metrics()


def main(args):
    """Main training function"""
    try:
        apply_optimizations(threads=args.threads, interop_threads=args.interop_threads)
    except Exception:
        pass

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Initializing components...")
    
    chunker = NarrativeChunker(
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        strategy=args.chunk_strategy
    )
    
    encoder = SemanticEncoder(
        model_name=args.encoder_model,
        device=device
    )
    
    processor = ChunkProcessor(
        chunker,
        encoder,
        cache_dir=args.cache_dir,
        encoder_batch_size=args.encoder_batch_size,
    )
    
    print(f"Encoder embedding dimension: {encoder.embedding_dim}")
    
    print("Loading data...")
    train_loader, val_loader = create_dataloaders(
        train_csv=args.train_csv,
        processor=processor,
        batch_size=args.batch_size,
        max_chunks=args.max_chunks,
        novels_dir=args.novels_dir,
        val_split=args.val_split
    )
    
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    print("Building model...")
    model = ConsistencyClassifier(
        embedding_dim=encoder.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Class weights
    train_df = pd.read_csv(args.train_csv)
    n_consistent = (train_df['label'] == 'consistent').sum()
    n_contradict = (train_df['label'] == 'contradict').sum()
    pos_weight = torch.tensor([n_contradict / n_consistent]).to(device)
    
    print(f"Class distribution - Consistent: {n_consistent}, Contradict: {n_contradict}")
    print(f"Using pos_weight: {pos_weight.item():.3f}")
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=5,
        T_mult=1,
        eta_min=1e-6
    )
    
    print(f"\nStarting training for {args.epochs} epochs...")
    best_val_f1 = 0.0
    best_threshold = 0.5
    history = {'train': [], 'val': [], 'thresholds': []}
    
    for epoch in range(args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*60}")
        
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch+1
        )
        history['train'].append(train_metrics)
        
        print(f"\nTrain Metrics:")
        for key, value in train_metrics.items():
            print(f"  {key}: {value:.4f}")
        
        # Find best threshold every 3 epochs
        if epoch % 3 == 0:
            threshold, _ = find_best_threshold(model, val_loader, device)
            print(f"\nOptimal threshold: {threshold:.3f}")
            best_threshold = threshold
        
        val_metrics = validate(
            model, val_loader, criterion, device, epoch+1, threshold=best_threshold
        )
        history['val'].append(val_metrics)
        history['thresholds'].append(best_threshold)
        
        print(f"\nValidation Metrics (threshold={best_threshold:.3f}):")
        for key, value in val_metrics.items():
            print(f"  {key}: {value:.4f}")
        
        scheduler.step()
        
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            save_checkpoint(
                model, optimizer, epoch,
                {'train': train_metrics, 'val': val_metrics, 'threshold': best_threshold},
                os.path.join(args.output_dir, 'best_model.pt')
            )
            print(f"  ✓ New best model saved (F1: {best_val_f1:.4f}, threshold: {best_threshold:.3f})")
        
        save_checkpoint(
            model, optimizer, epoch,
            {'train': train_metrics, 'val': val_metrics, 'threshold': best_threshold},
            os.path.join(args.output_dir, 'latest_model.pt')
        )
        
        with open(os.path.join(args.output_dir, 'history.json'), 'w') as f:
            json.dump(history, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Training completed!")
    print(f"Best validation F1: {best_val_f1:.4f}")
    print(f"Best threshold: {best_threshold:.3f}")
    print(f"Models saved in: {args.output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--train_csv', type=str, default='train.csv')
    parser.add_argument('--novels_dir', type=str, default='novels')
    parser.add_argument('--val_split', type=float, default=0.2)
    
    parser.add_argument('--chunk_size', type=int, default=800)
    parser.add_argument('--overlap', type=int, default=100)
    parser.add_argument('--chunk_strategy', type=str, default='sliding')
    parser.add_argument('--max_chunks', type=int, default=50)
    parser.add_argument('--encoder_model', type=str, default='all-MiniLM-L6-v2')
    parser.add_argument('--encoder_batch_size', type=int, default=128)
    parser.add_argument('--cache_dir', type=str, default='.cache/embeddings')
    
    parser.add_argument('--hidden_dim', type=int, default=384)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.4)
    
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--interop_threads', type=int, default=1)
    
    parser.add_argument('--output_dir', type=str, default='checkpoints')
    
    args = parser.parse_args()
    
    main(args)
