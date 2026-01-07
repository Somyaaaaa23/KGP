"""
Persistent Narrative Memory - Core BDH-inspired component
Maintains structured narrative state across the full novel
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional


class NarrativeMemory:
    """
    Persistent memory that tracks:
    - Character traits and beliefs
    - Key events and causality chains
    - Constraints and ruled-out possibilities
    - Temporal consistency
    """
    
    def __init__(self, embedding_dim: int = 768):
        self.embedding_dim = embedding_dim
        self.reset()
    
    def reset(self):
        """Reset memory for new narrative"""
        self.character_traits = {}  # character -> trait embeddings
        self.events = []  # ordered list of event embeddings
        self.constraints = []  # hard constraints (facts)
        self.beliefs = {}  # soft beliefs with confidence scores
        self.causal_chains = []  # cause -> effect relationships
        self.temporal_markers = []  # time-ordered narrative markers
        
    def update(self, chunk_info: Dict):
        """
        Incremental update from new chunk
        Only updates affected beliefs/constraints (sparse updates)
        """
        # Extract character mentions and traits
        if 'character_embeddings' in chunk_info:
            for char, embedding in chunk_info['character_embeddings'].items():
                if char not in self.character_traits:
                    self.character_traits[char] = []
                self.character_traits[char].append(embedding)
        
        # Add events
        if 'events' in chunk_info:
            self.events.extend(chunk_info['events'])
        
        # Update constraints
        if 'constraints' in chunk_info:
            self.constraints.extend(chunk_info['constraints'])
        
        # Update beliefs with selective override
        if 'beliefs' in chunk_info:
            for key, (value, confidence) in chunk_info['beliefs'].items():
                if key not in self.beliefs or confidence > self.beliefs[key][1]:
                    self.beliefs[key] = (value, confidence)
        
        # Add causal relationships
        if 'causal_chains' in chunk_info:
            self.causal_chains.extend(chunk_info['causal_chains'])
        
        # Track temporal progression
        if 'temporal_marker' in chunk_info:
            self.temporal_markers.append(chunk_info['temporal_marker'])
    
    def get_character_representation(self, character: str) -> Optional[np.ndarray]:
        """Get aggregated representation for a character"""
        if character not in self.character_traits:
            return None
        # Average all trait embeddings
        return np.mean(self.character_traits[character], axis=0)
    
    def check_constraint_violation(self, constraint_embedding: np.ndarray, 
                                   threshold: float = 0.85) -> bool:
        """Check if new constraint violates existing ones"""
        if not self.constraints:
            return False
        
        # Check cosine similarity with existing constraints
        for existing in self.constraints:
            similarity = np.dot(constraint_embedding, existing) / (
                np.linalg.norm(constraint_embedding) * np.linalg.norm(existing)
            )
            # High negative similarity indicates contradiction
            if similarity < -threshold:
                return True
        return False
    
    def get_state_summary(self) -> Dict:
        """Get current narrative state for comparison"""
        return {
            'num_characters': len(self.character_traits),
            'num_events': len(self.events),
            'num_constraints': len(self.constraints),
            'num_beliefs': len(self.beliefs),
            'causal_chain_length': len(self.causal_chains),
            'temporal_depth': len(self.temporal_markers)
        }


class MemoryUpdater(nn.Module):
    """
    Neural component that decides what to update in memory
    Implements selective, incremental belief updates
    """
    
    def __init__(self, embedding_dim: int = 768, hidden_dim: int = 512):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Attention mechanism for identifying important information
        self.importance_scorer = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Character trait extractor
        self.char_extractor = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        # Event encoder
        self.event_encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        # Constraint detector
        self.constraint_detector = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, chunk_embedding: torch.Tensor, 
                character_mentions: Optional[List[str]] = None) -> Dict:
        """
        Process chunk and extract memory updates
        
        Args:
            chunk_embedding: [batch_size, embedding_dim]
            character_mentions: List of character names mentioned
        
        Returns:
            Dictionary of memory updates
        """
        batch_size = chunk_embedding.shape[0]
        
        # Score importance
        importance = self.importance_scorer(chunk_embedding)
        
        # Extract character traits
        char_features = {}
        if character_mentions:
            char_embedding = self.char_extractor(chunk_embedding)
            for char in character_mentions:
                char_features[char] = char_embedding.detach().cpu().numpy()
        
        # Encode events
        event_embedding = self.event_encoder(chunk_embedding)
        events = [event_embedding[i].detach().cpu().numpy() 
                  for i in range(batch_size)]
        
        # Detect constraints
        is_constraint = self.constraint_detector(chunk_embedding)
        constraints = []
        for i in range(batch_size):
            if is_constraint[i] > 0.7:  # threshold for constraint
                constraints.append(chunk_embedding[i].detach().cpu().numpy())
        
        return {
            'importance': importance.detach().cpu().numpy(),
            'character_embeddings': char_features,
            'events': events,
            'constraints': constraints,
            'temporal_marker': chunk_embedding.mean(dim=0).detach().cpu().numpy()
        }


class BackstoryConsistencyChecker:
    """
    Compares backstory against narrative memory
    Identifies logical contradictions, violated constraints, etc.
    """
    
    def __init__(self, embedding_dim: int = 768):
        self.embedding_dim = embedding_dim
    
    def check_consistency(self, 
                         backstory_embedding: np.ndarray,
                         narrative_memory: NarrativeMemory,
                         character_name: Optional[str] = None) -> Tuple[bool, float, Dict]:
        """
        Check if backstory is consistent with narrative memory
        
        Returns:
            (is_consistent, confidence_score, evidence_dict)
        """
        evidence = {
            'constraint_violations': 0,
            'character_mismatch': False,
            'causal_incompatibility': 0,
            'temporal_inconsistency': False
        }
        
        # Check 1: Hard constraint violations
        if narrative_memory.check_constraint_violation(backstory_embedding):
            evidence['constraint_violations'] += 1
        
        # Check 2: Character trait consistency
        if character_name and character_name in narrative_memory.character_traits:
            char_rep = narrative_memory.get_character_representation(character_name)
            if char_rep is not None:
                similarity = np.dot(backstory_embedding, char_rep) / (
                    np.linalg.norm(backstory_embedding) * np.linalg.norm(char_rep)
                )
                if similarity < 0.3:  # Low similarity = mismatch
                    evidence['character_mismatch'] = True
        
        # Check 3: Event consistency
        if narrative_memory.events:
            event_similarities = []
            for event in narrative_memory.events:
                sim = np.dot(backstory_embedding, event) / (
                    np.linalg.norm(backstory_embedding) * np.linalg.norm(event) + 1e-8
                )
                event_similarities.append(sim)
            
            # Check for strong contradictions
            if min(event_similarities) < -0.5:
                evidence['causal_incompatibility'] += 1
        
        # Compute final consistency score
        violation_count = (
            evidence['constraint_violations'] +
            evidence['character_mismatch'] +
            evidence['causal_incompatibility']
        )
        
        # Lower violations = higher consistency
        confidence = max(0.0, 1.0 - (violation_count * 0.25))
        is_consistent = violation_count < 2
        
        return is_consistent, confidence, evidence
