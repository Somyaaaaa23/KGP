"""
Prediction pipeline for test data
Generates results.csv with predictions
"""

import torch
import pandas as pd
import argparse
import os
from tqdm import tqdm

from text_processing import NarrativeChunker, SemanticEncoder, ChunkProcessor
from performance import apply_optimizations
from narrative_memory import NarrativeMemory, MemoryUpdater
from model import ConsistencyClassifier, HybridReasoningModel
from utils import NarrativeDataset, create_test_loader


def predict(model: ConsistencyClassifier,
           test_loader,
           device: str,
           use_hybrid: bool = False,
           processor = None) -> dict:
    """
    Generate predictions for test data
    
    Returns:
        Dictionary mapping story_id -> prediction
    """
    model.eval()
    predictions_dict = {}
    
    pbar = tqdm(test_loader, desc="Predicting")
    
    with torch.no_grad():
        for batch in pbar:
            story_ids = batch['story_id']
            narrative_chunks = batch['narrative_chunks'].to(device)
            backstory = batch['backstory'].to(device)
            chunk_mask = batch['chunk_mask'].to(device)
            
            batch_size = narrative_chunks.shape[0]
            
            if use_hybrid:
                # Use hybrid reasoning with narrative memory
                hybrid_model = HybridReasoningModel(
                    model,
                    embedding_dim=model.embedding_dim,
                    use_rules=True,
                    rule_weight=0.3
                )
                
                for i in range(batch_size):
                    # Build narrative memory for this example
                    memory = NarrativeMemory(embedding_dim=model.embedding_dim)
                    
                    # Simple memory building (could be enhanced)
                    chunks_np = narrative_chunks[i].cpu().numpy()
                    for chunk_emb in chunks_np:
                        if chunk_mask[i][0]:  # if valid chunk
                            memory.update({
                                'events': [chunk_emb],
                                'temporal_marker': chunk_emb
                            })
                    
                    # Get prediction
                    pred, conf, evidence = hybrid_model.predict(
                        narrative_chunks[i:i+1],
                        backstory[i:i+1],
                        narrative_memory=memory,
                        character_name=batch['character'][i],
                        chunk_mask=chunk_mask[i:i+1]
                    )
                    
                    predictions_dict[story_ids[i]] = pred
            else:
                # Pure neural prediction
                logits, _ = model(narrative_chunks, backstory, chunk_mask)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).long().squeeze()
                
                for story_id, pred in zip(story_ids, preds):
                    predictions_dict[story_id] = pred.item()
    
    return predictions_dict


def main(args):
    """Main prediction function"""
    # Apply performance optimizations early
    try:
        apply_optimizations(threads=args.threads, interop_threads=args.interop_threads)
    except Exception:
        pass

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
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
    
    # Load test data
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
    
    print("Model loaded successfully")
    
    # Generate predictions
    print("\nGenerating predictions...")
    predictions_dict = predict(
        model,
        test_loader,
        device,
        use_hybrid=args.use_hybrid,
        processor=processor
    )
    
    # Create results dataframe
    results_df = pd.DataFrame([
        {'story_id': story_id, 'prediction': pred}
        for story_id, pred in predictions_dict.items()
    ])
    
    # Sort by story_id to match test.csv order
    test_df = pd.read_csv(args.test_csv)
    results_df = results_df.set_index('story_id').reindex(test_df['id']).reset_index()
    results_df.columns = ['story_id', 'prediction']
    
    # Save results
    results_df.to_csv(args.output_csv, index=False)
    print(f"\n✓ Results saved to {args.output_csv}")
    print(f"  Total predictions: {len(results_df)}")
    print(f"  Consistent: {(results_df['prediction'] == 1).sum()}")
    print(f"  Inconsistent: {(results_df['prediction'] == 0).sum()}")
    
    # Display sample
    print("\nSample predictions:")
    print(results_df.head(10).to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate predictions for test data'
    )
    
    # Data arguments
    parser.add_argument('--test_csv', type=str, default='test.csv',
                       help='Path to test CSV')
    parser.add_argument('--novels_dir', type=str, default='novels',
                       help='Directory containing novel files')
    
    # Model arguments
    parser.add_argument('--model_path', type=str, 
                       default='checkpoints/best_model.pt',
                       help='Path to trained model checkpoint')
    parser.add_argument('--hidden_dim', type=int, default=512,
                       help='Hidden dimension (must match training)')
    parser.add_argument('--num_layers', type=int, default=3,
                       help='Number of layers (must match training)')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate (must match training)')
    
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
    
    # Prediction arguments
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for prediction')
    parser.add_argument('--use_hybrid', action='store_true',
                       help='Use hybrid reasoning (neural + rules)')
    parser.add_argument('--threads', type=int, default=4,
                       help='Max compute threads for BLAS/torch')
    parser.add_argument('--interop_threads', type=int, default=1,
                       help='PyTorch inter-op threads')
    
    # Output arguments
    parser.add_argument('--output_csv', type=str, default='results.csv',
                       help='Output CSV file for predictions')
    
    args = parser.parse_args()
    
    main(args)
