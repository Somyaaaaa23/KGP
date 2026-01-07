"""
Text chunking and encoding modules
Handles narrative segmentation and semantic encoding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
from sentence_transformers import SentenceTransformer
import re
import os
import hashlib
import numpy as np


class NarrativeChunker:
    """
    Splits novel into ordered chunks while preserving temporal structure
    """
    
    def __init__(self, 
                 chunk_size: int = 1000,
                 overlap: int = 200,
                 strategy: str = 'sliding'):
        """
        Args:
            chunk_size: Number of words per chunk
            overlap: Overlapping words between chunks
            strategy: 'sliding' or 'chapter' based chunking
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.strategy = strategy
    
    def chunk_text(self, text: str) -> List[Tuple[str, int]]:
        """
        Chunk text into ordered segments
        
        Returns:
            List of (chunk_text, position_index) tuples
        """
        if self.strategy == 'chapter':
            return self._chunk_by_chapters(text)
        else:
            return self._chunk_sliding_window(text)
    
    def _chunk_sliding_window(self, text: str) -> List[Tuple[str, int]]:
        """Sliding window chunking"""
        words = text.split()
        chunks = []
        
        start = 0
        position = 0
        
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk = ' '.join(words[start:end])
            chunks.append((chunk, position))
            
            start += (self.chunk_size - self.overlap)
            position += 1
        
        return chunks
    
    def _chunk_by_chapters(self, text: str) -> List[Tuple[str, int]]:
        """Chapter-based chunking (if chapters are detectable)"""
        # Common chapter patterns
        chapter_pattern = r'(Chapter\s+\d+|CHAPTER\s+[IVXLCDM]+|\n\n[IVXLCDM]+\.\s+)'
        
        splits = re.split(chapter_pattern, text)
        chunks = []
        
        current_chunk = ""
        position = 0
        
        for i, segment in enumerate(splits):
            if re.match(chapter_pattern, segment):
                if current_chunk:
                    chunks.append((current_chunk.strip(), position))
                    position += 1
                current_chunk = segment
            else:
                current_chunk += segment
        
        if current_chunk:
            chunks.append((current_chunk.strip(), position))
        
        # If no chapters found, fall back to sliding window
        if len(chunks) <= 1:
            return self._chunk_sliding_window(text)
        
        return chunks
    
    def extract_character_mentions(self, text: str, 
                                   character_names: List[str]) -> List[str]:
        """
        Extract mentioned characters from chunk
        
        Args:
            text: Chunk text
            character_names: List of possible character names
        
        Returns:
            List of mentioned characters
        """
        mentioned = []
        text_lower = text.lower()
        
        for name in character_names:
            if name.lower() in text_lower:
                mentioned.append(name)
        
        return mentioned


class SemanticEncoder:
    """
    Encodes text chunks into semantic embeddings
    Uses pre-trained sentence transformers
    """
    
    def __init__(self, 
                 model_name: str = 'all-MiniLM-L6-v2',
                 device: str = None):
        """
        Args:
            model_name: HuggingFace model name
            device: 'cuda', 'cpu', or None (auto-detect)
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.device = device
        self.model = SentenceTransformer(model_name)
        self.model.to(device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
    
    def encode(self, texts: List[str], 
               batch_size: int = 32,
               show_progress: bool = False,
               normalize_embeddings: bool = False) -> torch.Tensor:
        """
        Encode list of texts into embeddings
        
        Args:
            texts: List of text strings
            batch_size: Batch size for encoding
            show_progress: Show progress bar
        
        Returns:
            Tensor of shape [num_texts, embedding_dim]
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_tensor=True,
            device=self.device
        )
        if normalize_embeddings:
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings
    
    def encode_single(self, text: str) -> torch.Tensor:
        """Encode single text"""
        return self.encode([text], batch_size=1)[0]


class ChunkProcessor:
    """
    Combines chunking and encoding with character extraction
    """
    
    def __init__(self, 
                 chunker: NarrativeChunker,
                 encoder: SemanticEncoder,
                 cache_dir: Optional[str] = None,
                 encoder_batch_size: int = 32):
        self.chunker = chunker
        self.encoder = encoder
        self.cache_dir = cache_dir
        self.encoder_batch_size = encoder_batch_size
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
    
    def process_narrative(self, 
                         text: str,
                         character_name: str = None,
                         cache_key: Optional[str] = None) -> Tuple[torch.Tensor, List[Dict]]:
        """
        Process full narrative into encoded chunks with metadata
        
        Args:
            text: Full novel text
            character_name: Main character to track
        
        Returns:
            (embeddings_tensor, metadata_list)
        """
        # Chunk the text
        chunks = self.chunker.chunk_text(text)
        
        # Extract texts and positions
        chunk_texts = [chunk[0] for chunk in chunks]
        positions = [chunk[1] for chunk in chunks]
        
        # Attempt to load from cache
        embeddings = None
        if self.cache_dir is not None:
            if cache_key is None:
                hasher = hashlib.md5()
                hasher.update(str(self.chunker.chunk_size).encode())
                hasher.update(str(self.chunker.overlap).encode())
                hasher.update(self.chunker.strategy.encode())
                # Include content length and a prefix for stability
                prefix = text[:20000]
                hasher.update(str(len(text)).encode())
                hasher.update(prefix.encode(errors='ignore'))
                cache_key = hasher.hexdigest()
            cache_path = os.path.join(self.cache_dir, f"emb_{cache_key}.npz")
            if os.path.exists(cache_path):
                try:
                    data = np.load(cache_path)
                    arr = torch.from_numpy(data['embeddings'])
                    if arr.shape[0] == len(chunk_texts):
                        embeddings = arr
                except Exception:
                    embeddings = None

        # Encode all chunks if cache miss
        if embeddings is None:
            embeddings = self.encoder.encode(
                chunk_texts, 
                batch_size=self.encoder_batch_size,
                show_progress=True
            )
            # Save to cache
            if self.cache_dir is not None:
                try:
                    np.savez_compressed(
                        os.path.join(self.cache_dir, f"emb_{cache_key}.npz"),
                        embeddings=embeddings.cpu().numpy()
                    )
                except Exception:
                    pass
        
        # Build metadata
        metadata = []
        for i, (text, pos) in enumerate(chunks):
            meta = {
                'position': pos,
                'text': text,
                'length': len(text.split())
            }
            
            # Track character mentions
            if character_name:
                chars = self.chunker.extract_character_mentions(
                    text, [character_name]
                )
                meta['characters'] = chars
            
            metadata.append(meta)
        
        return embeddings, metadata
    
    def process_backstory(self, backstory: str) -> torch.Tensor:
        """Encode backstory into single embedding"""
        return self.encoder.encode_single(backstory)
