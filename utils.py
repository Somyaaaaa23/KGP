"""
Utility functions and data loading
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import os
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
import numpy as np


class NarrativeDataset(Dataset):
    """
    Dataset for narrative consistency task
    """
    
    def __init__(self, 
                 csv_path: str,
                 novels_dir: str = 'novels',
                 processor: Optional[object] = None,
                 max_chunks: int = 100,
                 is_test: bool = False):
        """
        Args:
            csv_path: Path to train.csv or test.csv
            novels_dir: Directory containing novel files
            processor: ChunkProcessor instance
            max_chunks: Maximum chunks per narrative
            is_test: Whether this is test data (no labels)
        """
        self.df = pd.read_csv(csv_path)
        self.novels_dir = novels_dir
        self.processor = processor
        self.max_chunks = max_chunks
        self.is_test = is_test
        
        # Cache for processed data
        self.cache = {}
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Returns:
            {
                'story_id': str,
                'narrative_chunks': tensor [num_chunks, embedding_dim],
                'backstory': tensor [embedding_dim],
                'character': str,
                'label': int (if not test),
                'chunk_mask': tensor [num_chunks]
            }
        """
        if idx in self.cache:
            return self.cache[idx]
        
        row = self.df.iloc[idx]
        
        # Load novel text
        novel_text = self._load_novel(row['book_name'])
        
        # Get character and backstory
        character = row['char'] if pd.notna(row.get('char')) else None
        backstory = row['content']
        
        # Process if processor available
        if self.processor is not None:
            # Process narrative
            chunk_embeddings, metadata = self.processor.process_narrative(
                novel_text, character
            )
            
            # Process backstory
            backstory_embedding = self.processor.process_backstory(backstory)
            
            # Truncate/pad chunks
            num_chunks = chunk_embeddings.shape[0]
            if num_chunks > self.max_chunks:
                chunk_embeddings = chunk_embeddings[:self.max_chunks]
                chunk_mask = torch.ones(self.max_chunks, dtype=torch.bool)
            else:
                # Pad
                padding = torch.zeros(
                    self.max_chunks - num_chunks,
                    chunk_embeddings.shape[1]
                )
                chunk_embeddings = torch.cat([chunk_embeddings, padding], dim=0)
                chunk_mask = torch.cat([
                    torch.ones(num_chunks, dtype=torch.bool),
                    torch.zeros(self.max_chunks - num_chunks, dtype=torch.bool)
                ])
        else:
            # Placeholder if no processor
            chunk_embeddings = None
            backstory_embedding = None
            chunk_mask = None
        
        item = {
            'story_id': row['id'],
            'narrative_chunks': chunk_embeddings,
            'backstory': backstory_embedding,
            'character': character,
            'chunk_mask': chunk_mask,
            'novel_text': novel_text  # Keep for memory building
        }
        
        if not self.is_test:
            # Map label: 'consistent' -> 1, 'contradict' -> 0
            label = 1 if row['label'] == 'consistent' else 0
            item['label'] = label
        
        return item
    
    def _load_novel(self, book_name: str) -> str:
        """Load novel text from file"""
        # Try common patterns
        possible_paths = [
            os.path.join(self.novels_dir, f"{book_name}.txt"),
            os.path.join(self.novels_dir, book_name),
            f"{book_name}.txt",
            book_name
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
        
        # If not found, return placeholder
        print(f"Warning: Novel not found for {book_name}")
        return f"[Novel text for {book_name} not available]"


def create_dataloaders(train_csv: str,
                       processor: object,
                       batch_size: int = 4,
                       max_chunks: int = 100,
                       novels_dir: str = 'novels',
                       val_split: float = 0.2) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders
    
    Args:
        train_csv: Path to training CSV
        processor: ChunkProcessor instance
        batch_size: Batch size
        max_chunks: Max chunks per narrative
        novels_dir: Directory with novels
        val_split: Validation split ratio
    
    Returns:
        (train_loader, val_loader)
    """
    # Load full dataset
    full_dataset = NarrativeDataset(
        train_csv,
        novels_dir=novels_dir,
        processor=processor,
        max_chunks=max_chunks,
        is_test=False
    )
    
    # Stratified split to maintain class balance
    df = pd.read_csv(train_csv)
    labels = (df['label'] == 'consistent').astype(int).values
    
    train_idx, val_idx = train_test_split(
        range(len(full_dataset)),
        test_size=val_split,
        stratify=labels,
        random_state=42
    )
    
    train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
    val_dataset = torch.utils.data.Subset(full_dataset, val_idx)
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 to avoid pickling issues
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    return train_loader, val_loader


def create_test_loader(test_csv: str,
                       processor: object,
                       batch_size: int = 4,
                       max_chunks: int = 100,
                       novels_dir: str = 'novels') -> DataLoader:
    """
    Create test dataloader
    """
    test_dataset = NarrativeDataset(
        test_csv,
        novels_dir=novels_dir,
        processor=processor,
        max_chunks=max_chunks,
        is_test=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    return test_loader


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function for batching
    Handles variable-length sequences
    """
    # Stack tensors
    story_ids = [item['story_id'] for item in batch]
    
    narrative_chunks = torch.stack([item['narrative_chunks'] for item in batch])
    backstories = torch.stack([item['backstory'] for item in batch])
    chunk_masks = torch.stack([item['chunk_mask'] for item in batch])
    
    characters = [item['character'] for item in batch]
    novel_texts = [item['novel_text'] for item in batch]
    
    result = {
        'story_id': story_ids,
        'narrative_chunks': narrative_chunks,
        'backstory': backstories,
        'chunk_mask': chunk_masks,
        'character': characters,
        'novel_text': novel_texts
    }
    
    # Add labels if available
    if 'label' in batch[0]:
        labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
        result['label'] = labels
    
    return result


class MetricsTracker:
    """Track training and evaluation metrics"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.correct = 0
        self.total = 0
        self.loss_sum = 0.0
        self.loss_count = 0
        self.predictions = []
        self.targets = []
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, loss: float = None):
        """Update metrics with batch results"""
        preds = predictions.cpu().numpy()
        targs = targets.cpu().numpy()
        
        self.predictions.extend(preds)
        self.targets.extend(targs)
        
        self.correct += (preds == targs).sum()
        self.total += len(preds)
        
        if loss is not None:
            self.loss_sum += loss
            self.loss_count += 1
    
    def get_metrics(self) -> Dict[str, float]:
        """Get current metrics"""
        accuracy = self.correct / self.total if self.total > 0 else 0.0
        avg_loss = self.loss_sum / self.loss_count if self.loss_count > 0 else 0.0
        
        # Compute precision, recall, F1
        preds = np.array(self.predictions)
        targs = np.array(self.targets)
        
        tp = ((preds == 1) & (targs == 1)).sum()
        fp = ((preds == 1) & (targs == 0)).sum()
        fn = ((preds == 0) & (targs == 1)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'loss': avg_loss,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }


def save_checkpoint(model: torch.nn.Module,
                   optimizer: torch.optim.Optimizer,
                   epoch: int,
                   metrics: Dict,
                   path: str):
    """Save model checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(model: torch.nn.Module,
                   optimizer: torch.optim.Optimizer,
                   path: str,
                   device: str = 'cpu') -> int:
    """Load model checkpoint"""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    metrics = checkpoint.get('metrics', {})
    
    print(f"Checkpoint loaded from {path}")
    print(f"Epoch: {epoch}, Metrics: {metrics}")
    
    return epoch
