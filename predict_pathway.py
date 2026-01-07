"""
Pathway-enhanced prediction pipeline
Demonstrates incremental streaming prediction with Pathway
"""

import torch
import pandas as pd
import argparse
import os
from tqdm import tqdm
import pathway as pw

from text_processing import NarrativeChunker, SemanticEncoder, ChunkProcessor
from narrative_memory import NarrativeMemory
from model import ConsistencyClassifier, HybridReasoningModel
from pathway_processor import PathwayNarrativeProcessor, create_pathway_pipeline
from performance import apply_optimizations


def predict_with_pathway(model: ConsistencyClassifier,
                         encoder: SemanticEncoder,
                         chunker: NarrativeChunker,
                         test_csv: str,
                         output_csv: str,
                         device: str):
    """
    Use Pathway for incremental streaming prediction
    This demonstrates Track B's continuous reasoning approach
    """
    print("\n" + "=" * 70)
    print("PATHWAY-BASED PREDICTION (Incremental Processing)")
    print("=" * 70)
    
    # Create Pathway pipeline
    processor = PathwayNarrativeProcessor(
        model=model.to(device),
        encoder=encoder,
        chunker=chunker,
        use_streaming=False  # Static mode for batch prediction
    )
    
    # Process using Pathway
    print("\nProcessing test data with Pathway...")
    print("(Pathway handles incremental updates automatically)")
    
    try:
        processor.process_stream_from_csv(
            csv_path=test_csv,
            output_path=output_csv,
            mode="static"
        )
    except Exception as e:
        print(f"\n⚠️  Pathway processing encountered an issue: {e}")
        print("Falling back to traditional batch processing...")
        return False
    
    return True


def predict_traditional(model: ConsistencyClassifier,
                       test_loader,
                       device: str) -> dict:
    """
    Traditional batch prediction (fallback)
    """
    model.eval()
    predictions_dict = {}
    
    pbar = tqdm(test_loader, desc="Predicting (Traditional)")
    
    with torch.no_grad():
        for batch in pbar:
            story_ids = batch['story_id']
            narrative_chunks = batch['narrative_chunks'].to(device)
            backstory = batch['backstory'].to(device)
            chunk_mask = batch['chunk_mask'].to(device)
            
            # Forward pass
            logits, _ = model(narrative_chunks, backstory, chunk_mask)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long().squeeze()
            
            for story_id, pred in zip(story_ids, preds):
                predictions_dict[story_id] = pred.item()
    
    return predictions_dict


def main(args):
    """Main prediction function with Pathway integration"""
    # Apply performance optimizations early
    try:
        apply_optimizations(threads=args.threads, interop_threads=args.interop_threads)
    except Exception:
        pass

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    print("\n" + "=" * 70)
    print("PATHWAY-ENHANCED NARRATIVE CONSISTENCY PREDICTION")
    print("Track B - Incremental Streaming Processing")
    print("=" * 70)
    
    print("\nInitializing components...")
    
    # Initialize text processing
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
    
    # Load model
    print(f"\nLoading model from {args.model_path}...")
    model = ConsistencyClassifier(
        embedding_dim=encoder.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)
    
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("✓ Model loaded successfully")
    
    # Choose prediction method
    if args.use_pathway:
        print("\n🚀 Using Pathway for incremental prediction")
        
        success = predict_with_pathway(
            model=model,
            encoder=encoder,
            chunker=chunker,
            test_csv=args.test_csv,
            output_csv=args.output_csv,
            device=device
        )
        
        if success:
            print(f"\n✓ Pathway prediction completed!")
            print(f"Results saved to: {args.output_csv}")
            
            # Display results
            try:
                results_df = pd.read_csv(args.output_csv)
                print(f"\nTotal predictions: {len(results_df)}")
                print(f"Consistent: {(results_df['prediction'] == 1).sum()}")
                print(f"Inconsistent: {(results_df['prediction'] == 0).sum()}")
                
                print("\nSample predictions:")
                print(results_df.head(10).to_string(index=False))
            except:
                pass
            
            return
    
    # Fallback to traditional method
    print("\n📊 Using traditional batch prediction")
    
    from utils import NarrativeDataset
    
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
    
    # Generate predictions
    print("\nGenerating predictions...")
    predictions_dict = predict_traditional(model, test_loader, device)
    
    # Create results dataframe
    results_df = pd.DataFrame([
        {'story_id': story_id, 'prediction': pred}
        for story_id, pred in predictions_dict.items()
    ])
    
    # Sort by story_id
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
    parser = argparse.ArgumentParser(
        description='Generate predictions with Pathway integration'
    )
    
    # Data arguments
    parser.add_argument('--test_csv', type=str, default='test.csv')
    parser.add_argument('--novels_dir', type=str, default='novels')
    
    # Model arguments
    parser.add_argument('--model_path', type=str, 
                       default='checkpoints/best_model.pt')
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.3)
    
    # Processing arguments
    parser.add_argument('--chunk_size', type=int, default=1000)
    parser.add_argument('--overlap', type=int, default=200)
    parser.add_argument('--chunk_strategy', type=str, default='sliding')
    parser.add_argument('--max_chunks', type=int, default=100)
    parser.add_argument('--encoder_model', type=str, 
                       default='all-MiniLM-L6-v2')
    parser.add_argument('--encoder_batch_size', type=int, default=64,
                       help='Batch size for sentence encoder')
    parser.add_argument('--cache_dir', type=str, default='.cache/embeddings',
                       help='Directory to cache computed embeddings')
    
    # Prediction arguments
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--use_pathway', action='store_true',
                       help='Use Pathway for incremental streaming prediction')
    parser.add_argument('--use_hybrid', action='store_true',
                       help='Use hybrid reasoning (neural + rules)')
    parser.add_argument('--threads', type=int, default=4,
                       help='Max compute threads for BLAS/torch')
    parser.add_argument('--interop_threads', type=int, default=1,
                       help='PyTorch inter-op threads')
    
    # Output arguments
    parser.add_argument('--output_csv', type=str, default='results.csv')
    
    args = parser.parse_args()
    
    main(args)
