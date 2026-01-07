#!/usr/bin/env python3
"""
Demo script - Test the model components without training
Shows how the system processes data
"""

import torch
import numpy as np
from text_processing import NarrativeChunker, SemanticEncoder, ChunkProcessor
from narrative_memory import NarrativeMemory, MemoryUpdater
from model import ConsistencyClassifier

def demo():
    print("=" * 70)
    print("NARRATIVE CONSISTENCY MODEL - COMPONENT DEMO")
    print("=" * 70)
    
    # Sample data
    sample_novel = """
    Captain Grant was a brave sailor who loved the sea. He had three children
    and a loyal crew. One day, while exploring uncharted waters, his ship
    encountered a terrible storm. The captain made the difficult decision to
    save his crew by ordering them to abandon ship. He stayed behind to secure
    important documents. The ship sank, and the captain was lost at sea.
    His family searched for him for years, following mysterious clues.
    """ * 50  # Repeat to simulate longer text
    
    sample_backstory = """
    As a young man, Captain Grant served in the navy where he learned navigation
    and leadership. He married his childhood sweetheart and had three children.
    His passion for exploration led him to command his own vessel.
    """
    
    print("\n1. TEXT PROCESSING")
    print("-" * 70)
    
    # Initialize chunker
    chunker = NarrativeChunker(chunk_size=100, overlap=20)
    chunks = chunker.chunk_text(sample_novel)
    print(f"✓ Chunked novel into {len(chunks)} segments")
    print(f"  First chunk preview: {chunks[0][0][:100]}...")
    
    # Initialize encoder (this will download the model on first run)
    print("\n2. SEMANTIC ENCODING")
    print("-" * 70)
    print("  Loading sentence transformer model...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder = SemanticEncoder(device=device)
    print(f"✓ Encoder loaded (device: {device})")
    print(f"  Embedding dimension: {encoder.embedding_dim}")
    
    # Encode chunks
    chunk_texts = [c[0] for c in chunks[:5]]  # First 5 chunks
    embeddings = encoder.encode(chunk_texts, show_progress=True)
    print(f"✓ Encoded {len(chunk_texts)} chunks")
    print(f"  Embedding shape: {embeddings.shape}")
    
    # Encode backstory
    backstory_emb = encoder.encode_single(sample_backstory)
    print(f"✓ Encoded backstory")
    print(f"  Backstory embedding shape: {backstory_emb.shape}")
    
    print("\n3. NARRATIVE MEMORY")
    print("-" * 70)
    
    # Initialize memory
    memory = NarrativeMemory(embedding_dim=encoder.embedding_dim)
    print(f"✓ Initialized narrative memory")
    
    # Update memory with chunks
    for i, emb in enumerate(embeddings):
        memory.update({
            'events': [emb.cpu().numpy()],
            'temporal_marker': emb.cpu().numpy()
        })
    
    state = memory.get_state_summary()
    print(f"✓ Memory state after processing:")
    for key, value in state.items():
        print(f"    {key}: {value}")
    
    print("\n4. NEURAL MODEL")
    print("-" * 70)
    
    # Initialize model
    model = ConsistencyClassifier(
        embedding_dim=encoder.embedding_dim,
        hidden_dim=256,
        num_layers=2,
        dropout=0.3
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model initialized")
    print(f"  Total parameters: {total_params:,}")
    
    # Test forward pass
    model.eval()
    with torch.no_grad():
        # Prepare dummy batch
        batch_size = 1
        num_chunks = len(embeddings)
        
        # Stack embeddings
        narrative_chunks = embeddings.unsqueeze(0)  # [1, num_chunks, embed_dim]
        backstory_batch = backstory_emb.unsqueeze(0)  # [1, embed_dim]
        chunk_mask = torch.ones(batch_size, num_chunks, dtype=torch.bool)
        
        # Forward pass
        logits, attention_weights = model(narrative_chunks, backstory_batch, chunk_mask)
        prob = torch.sigmoid(logits)
        prediction = (prob > 0.5).long().item()
    
    print(f"✓ Model forward pass successful")
    print(f"  Logit: {logits.item():.4f}")
    print(f"  Probability: {prob.item():.4f}")
    print(f"  Prediction: {prediction} ({'Consistent' if prediction == 1 else 'Inconsistent'})")
    print(f"  (Note: Untrained model, random prediction expected)")
    
    print("\n5. CONSISTENCY CHECKING")
    print("-" * 70)
    
    from narrative_memory import BackstoryConsistencyChecker
    
    checker = BackstoryConsistencyChecker(encoder.embedding_dim)
    is_consistent, confidence, evidence = checker.check_consistency(
        backstory_emb.cpu().numpy(),
        memory
    )
    
    print(f"✓ Rule-based consistency check:")
    print(f"  Consistent: {is_consistent}")
    print(f"  Confidence: {confidence:.4f}")
    print(f"  Evidence: {evidence}")
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETE!")
    print("=" * 70)
    print("\nAll components working correctly! ✓")
    print("\nNext steps:")
    print("  1. Create 'novels/' directory with .txt files")
    print("  2. Run: python train.py --epochs 5 (quick test)")
    print("  3. Run: python predict.py")
    print("\n")

if __name__ == '__main__':
    try:
        demo()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        print("\nMake sure dependencies are installed:")
        print("  pip install -r requirements.txt")
        import traceback
        traceback.print_exc()
