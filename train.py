"""
Training pipeline for narrative consistency model
Track B - BDH-inspired continuous reasoning
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


def train_epoch(model: ConsistencyClassifier,
                train_loader: DataLoader,
                criterion: nn.Module,
                optimizer: optim.Optimizer,
                device: str,
                epoch: int) -> dict:
    """
    Train for one epoch
    """
    model.train()
    metrics = MetricsTracker()
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    
    for batch in pbar:
        # Move to device
        narrative_chunks = batch['narrative_chunks'].to(device)
        backstory = batch['backstory'].to(device)
        chunk_mask = batch['chunk_mask'].to(device)
        labels = batch['label'].to(device).float()
        
        # Forward pass
        optimizer.zero_grad()
        logits, _ = model(narrative_chunks, backstory, chunk_mask)
        logits = logits.squeeze(-1)
        
        # Compute loss
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Get predictions
        probs = torch.sigmoid(logits)
        predictions = (probs > 0.5).long()
        
        # Update metrics
        metrics.update(predictions, labels.long(), loss.item())
        
        # Update progress bar
        current_metrics = metrics.get_metrics()
        pbar.set_postfix({
            'loss': f"{current_metrics['loss']:.4f}",
            'acc': f"{current_metrics['accuracy']:.4f}"
        })
    
    return metrics.get_metrics()


def validate(model: ConsistencyClassifier,
            val_loader: DataLoader,
            criterion: nn.Module,
            device: str,
            epoch: int) -> dict:
    """
    Validate the model
    """
    model.eval()
    metrics = MetricsTracker()
    
    pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]")
    
    with torch.no_grad():
        for batch in pbar:
            # Move to device
            narrative_chunks = batch['narrative_chunks'].to(device)
            backstory = batch['backstory'].to(device)
            chunk_mask = batch['chunk_mask'].to(device)
            labels = batch['label'].to(device).float()
            
            # Forward pass
            logits, _ = model(narrative_chunks, backstory, chunk_mask)
            logits = logits.squeeze(-1)
            
            # Compute loss
            loss = criterion(logits, labels)
            
            # Get predictions
            probs = torch.sigmoid(logits)
            predictions = (probs > 0.5).long()
            
            # Update metrics
            metrics.update(predictions, labels.long(), loss.item())
            
            # Update progress bar
            current_metrics = metrics.get_metrics()
            pbar.set_postfix({
                'loss': f"{current_metrics['loss']:.4f}",
                'acc': f"{current_metrics['accuracy']:.4f}"
            })
    
    return metrics.get_metrics()


def main(args):
    """Main training function"""
    # Apply performance optimizations early
    try:
        apply_optimizations(threads=args.threads, interop_threads=args.interop_threads)
    except Exception:
        pass

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Initializing components...")
    
    # Initialize text processing components
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
    
    # Create data loaders
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
    
    # Initialize model
    print("Building model...")
    model = ConsistencyClassifier(
        embedding_dim=encoder.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)
    
    # Print model size
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer with class weights for imbalance
    # Calculate class weights from training data
    train_df = pd.read_csv(args.train_csv)
    n_consistent = (train_df['label'] == 'consistent').sum()
    n_contradict = (train_df['label'] == 'contradict').sum()
    pos_weight = torch.tensor([n_contradict / n_consistent]).to(device)
    
    print(f"Class distribution - Consistent: {n_consistent}, Contradict: {n_contradict}")
    print(f"Using pos_weight: {pos_weight.item():.3f} to balance classes")
    
    if args.use_focal_loss:
        criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=3
    )
    
    # Load checkpoint if resuming
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        start_epoch = load_checkpoint(model, optimizer, args.resume, device)
        start_epoch += 1
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    best_val_f1 = 0.0
    history = {'train': [], 'val': []}
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*60}")
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch+1
        )
        history['train'].append(train_metrics)
        
        print(f"\nTrain Metrics:")
        for key, value in train_metrics.items():
            print(f"  {key}: {value:.4f}")
        
        # Validate
        val_metrics = validate(
            model, val_loader, criterion, device, epoch+1
        )
        history['val'].append(val_metrics)
        
        print(f"\nValidation Metrics:")
        for key, value in val_metrics.items():
            print(f"  {key}: {value:.4f}")
        
        # Learning rate scheduling
        scheduler.step(val_metrics['f1'])
        
        # Save best model
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            save_checkpoint(
                model, optimizer, epoch,
                {'train': train_metrics, 'val': val_metrics},
                os.path.join(args.output_dir, 'best_model.pt')
            )
            print(f"  ✓ New best model saved (F1: {best_val_f1:.4f})")
        
        # Save latest checkpoint
        save_checkpoint(
            model, optimizer, epoch,
            {'train': train_metrics, 'val': val_metrics},
            os.path.join(args.output_dir, 'latest_model.pt')
        )
        
        # Save history
        with open(os.path.join(args.output_dir, 'history.json'), 'w') as f:
            json.dump(history, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Training completed!")
    print(f"Best validation F1: {best_val_f1:.4f}")
    print(f"Models saved in: {args.output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train narrative consistency model (Track B)'
    )
    
    # Data arguments
    parser.add_argument('--train_csv', type=str, default='train.csv',
                       help='Path to training CSV')
    parser.add_argument('--novels_dir', type=str, default='novels',
                       help='Directory containing novel files')
    parser.add_argument('--val_split', type=float, default=0.2,
                       help='Validation split ratio')
    
    # Processing arguments
    parser.add_argument('--chunk_size', type=int, default=1000,
                       help='Chunk size in words')
    parser.add_argument('--overlap', type=int, default=200,
                       help='Overlap between chunks')
    parser.add_argument('--chunk_strategy', type=str, default='sliding',
                       choices=['sliding', 'chapter'],
                       help='Chunking strategy')
    parser.add_argument('--max_chunks', type=int, default=100,
                       help='Maximum chunks per narrative')
    parser.add_argument('--encoder_model', type=str, 
                       default='all-MiniLM-L6-v2',
                       help='Sentence transformer model')
    parser.add_argument('--encoder_batch_size', type=int, default=64,
                       help='Batch size for sentence encoder')
    parser.add_argument('--cache_dir', type=str, default='.cache/embeddings',
                       help='Directory to cache computed embeddings')
    
    # Model arguments
    parser.add_argument('--hidden_dim', type=int, default=512,
                       help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=3,
                       help='Number of classifier layers')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay')
    parser.add_argument('--use_focal_loss', action='store_true',
                       help='Use focal loss instead of BCE')
    parser.add_argument('--focal_alpha', type=float, default=0.25,
                       help='Focal loss alpha')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                       help='Focal loss gamma')

    # Performance arguments
    parser.add_argument('--threads', type=int, default=4,
                       help='Max compute threads for BLAS/torch')
    parser.add_argument('--interop_threads', type=int, default=1,
                       help='PyTorch inter-op threads')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='checkpoints',
                       help='Output directory for checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    main(args)
