"""
Improved prediction with optimal threshold
"""

import torch
import pandas as pd
import argparse
import json
import os

from text_processing import NarrativeChunker, SemanticEncoder, ChunkProcessor
from model import ConsistencyClassifier
from utils import NarrativeDataset
from performance import apply_optimizations
from tqdm import tqdm


def main(args):
    try:
        apply_optimizations(threads=args.threads, interop_threads=args.interop_threads)
    except Exception:
        pass

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
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
    
    print("Loading test data...")
    test_dataset = NarrativeDataset(
        csv_path=args.test_csv,
        novels_dir=args.novels_dir,
        processor=processor,
        max_chunks=args.max_chunks,
        is_test=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    model = ConsistencyClassifier(
        embedding_dim=encoder.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)
    
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Get optimal threshold from checkpoint
    threshold = checkpoint.get('metrics', {}).get('threshold', 0.5)
    if args.threshold is not None:
        threshold = args.threshold
    
    print(f"Model loaded successfully")
    print(f"Using threshold: {threshold:.3f}")
    
    # Generate predictions
    print("\nGenerating predictions...")
    predictions_dict = {}
    
    pbar = tqdm(test_loader, desc="Predicting")
    
    with torch.no_grad():
        for batch in pbar:
            story_ids = batch['story_id']
            narrative_chunks = batch['narrative_chunks'].to(device)
            backstory = batch['backstory'].to(device)
            chunk_mask = batch['chunk_mask'].to(device)
            
            logits, _ = model(narrative_chunks, backstory, chunk_mask)
            logits = logits.squeeze(-1)
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).long()
            
            for story_id, pred in zip(story_ids, preds):
                predictions_dict[story_id] = pred.item()
    
    # Create results
    results_df = pd.DataFrame([
        {'story_id': story_id, 'prediction': pred}
        for story_id, pred in predictions_dict.items()
    ])
    
    # Sort by test.csv order
    test_df = pd.read_csv(args.test_csv)
    results_df = results_df.set_index('story_id').reindex(test_df['id']).reset_index()
    results_df.columns = ['story_id', 'prediction']
    
    # Save results
    results_df.to_csv(args.output_csv, index=False)
    print(f"\n✓ Results saved to {args.output_csv}")
    print(f"  Total predictions: {len(results_df)}")
    print(f"  Consistent: {(results_df['prediction'] == 1).sum()}")
    print(f"  Inconsistent: {(results_df['prediction'] == 0).sum()}")
    
    print("\nSample predictions:")
    print(results_df.head(10).to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--test_csv', type=str, default='test.csv')
    parser.add_argument('--novels_dir', type=str, default='novels')
    parser.add_argument('--model_path', type=str, default='checkpoints/best_model.pt')
    parser.add_argument('--hidden_dim', type=int, default=384)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--chunk_size', type=int, default=800)
    parser.add_argument('--overlap', type=int, default=100)
    parser.add_argument('--chunk_strategy', type=str, default='sliding')
    parser.add_argument('--max_chunks', type=int, default=50)
    parser.add_argument('--encoder_model', type=str, default='all-MiniLM-L6-v2')
    parser.add_argument('--encoder_batch_size', type=int, default=128)
    parser.add_argument('--cache_dir', type=str, default='.cache/embeddings')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--interop_threads', type=int, default=1)
    parser.add_argument('--threshold', type=float, default=None,
                       help='Classification threshold (default: from checkpoint)')
    parser.add_argument('--output_csv', type=str, default='results.csv')
    
    args = parser.parse_args()
    
    main(args)
