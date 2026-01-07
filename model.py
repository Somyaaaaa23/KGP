"""
Consistency classifier and reasoning model
Combines narrative memory with learned classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np


class ConsistencyClassifier(nn.Module):
    """
    Neural classifier for backstory consistency
    Takes narrative memory state + backstory embedding -> binary prediction
    """
    
    def __init__(self, 
                 embedding_dim: int = 768,
                 hidden_dim: int = 512,
                 num_layers: int = 3,
                 dropout: float = 0.3):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        # Narrative aggregator - summarizes full narrative
        self.narrative_aggregator = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        
        # Attention mechanism for important chunks
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Backstory encoder
        self.backstory_encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Comparison network
        combined_dim = (hidden_dim * 2) + hidden_dim  # narrative + backstory
        
        layers = []
        current_dim = combined_dim
        
        for i in range(num_layers):
            next_dim = hidden_dim // (2 ** i)
            layers.extend([
                nn.Linear(current_dim, next_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.BatchNorm1d(next_dim)
            ])
            current_dim = next_dim
        
        # Final binary classifier
        layers.append(nn.Linear(current_dim, 1))
        
        self.classifier = nn.Sequential(*layers)
    
    def forward(self, 
                narrative_chunks: torch.Tensor,
                backstory_embedding: torch.Tensor,
                chunk_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            narrative_chunks: [batch_size, num_chunks, embedding_dim]
            backstory_embedding: [batch_size, embedding_dim]
            chunk_mask: [batch_size, num_chunks] - mask for padding
        
        Returns:
            (logits, attention_weights)
        """
        batch_size = narrative_chunks.shape[0]
        
        # Process narrative with LSTM
        narrative_encoded, (h_n, c_n) = self.narrative_aggregator(narrative_chunks)
        # narrative_encoded: [batch_size, num_chunks, hidden_dim*2]
        
        # Apply attention to find important chunks
        attn_output, attn_weights = self.attention(
            narrative_encoded,
            narrative_encoded,
            narrative_encoded,
            key_padding_mask=~chunk_mask if chunk_mask is not None else None
        )
        
        # Aggregate narrative representation (mean pooling over chunks)
        if chunk_mask is not None:
            mask_expanded = chunk_mask.unsqueeze(-1).float()
            narrative_rep = (attn_output * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            narrative_rep = attn_output.mean(dim=1)
        
        # Encode backstory
        backstory_rep = self.backstory_encoder(backstory_embedding)
        
        # Combine representations
        combined = torch.cat([narrative_rep, backstory_rep], dim=-1)
        
        # Classify
        logits = self.classifier(combined)
        
        return logits, {'attention_weights': attn_weights}
    
    def predict(self, 
                narrative_chunks: torch.Tensor,
                backstory_embedding: torch.Tensor,
                chunk_mask: Optional[torch.Tensor] = None) -> Tuple[int, float]:
        """
        Make binary prediction
        
        Returns:
            (prediction, confidence)
        """
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(narrative_chunks, backstory_embedding, chunk_mask)
            probs = torch.sigmoid(logits)
            prediction = (probs > 0.5).long().item()
            confidence = probs.item() if prediction == 1 else (1 - probs.item())
        
        return prediction, confidence


class HybridReasoningModel:
    """
    Combines rule-based consistency checking with neural classification
    Implements Track B reasoning approach
    """
    
    def __init__(self, 
                 classifier: ConsistencyClassifier,
                 embedding_dim: int = 768,
                 use_rules: bool = True,
                 rule_weight: float = 0.3):
        """
        Args:
            classifier: Neural classifier
            embedding_dim: Embedding dimension
            use_rules: Whether to use rule-based checking
            rule_weight: Weight for rule-based score (0-1)
        """
        self.classifier = classifier
        self.embedding_dim = embedding_dim
        self.use_rules = use_rules
        self.rule_weight = rule_weight
        self.neural_weight = 1.0 - rule_weight
    
    def predict(self,
                narrative_chunks: torch.Tensor,
                backstory_embedding: torch.Tensor,
                narrative_memory: Optional[object] = None,
                character_name: Optional[str] = None,
                chunk_mask: Optional[torch.Tensor] = None) -> Tuple[int, float, Dict]:
        """
        Hybrid prediction combining neural and rule-based reasoning
        
        Returns:
            (prediction, confidence, evidence)
        """
        # Neural prediction
        neural_pred, neural_conf = self.classifier.predict(
            narrative_chunks, backstory_embedding, chunk_mask
        )
        
        evidence = {
            'neural_prediction': neural_pred,
            'neural_confidence': neural_conf,
            'rule_based_score': None,
            'final_method': 'neural'
        }
        
        # Rule-based checking if available
        if self.use_rules and narrative_memory is not None:
            from narrative_memory import BackstoryConsistencyChecker
            
            checker = BackstoryConsistencyChecker(self.embedding_dim)
            backstory_np = backstory_embedding.cpu().numpy()
            
            rule_consistent, rule_conf, rule_evidence = checker.check_consistency(
                backstory_np,
                narrative_memory,
                character_name
            )
            
            rule_pred = 1 if rule_consistent else 0
            
            evidence['rule_prediction'] = rule_pred
            evidence['rule_confidence'] = rule_conf
            evidence['rule_evidence'] = rule_evidence
            evidence['final_method'] = 'hybrid'
            
            # Combine scores
            combined_score = (
                self.neural_weight * neural_conf * neural_pred +
                self.rule_weight * rule_conf * rule_pred
            )
            
            final_pred = 1 if combined_score > 0.5 else 0
            final_conf = combined_score if final_pred == 1 else (1 - combined_score)
            
            return final_pred, final_conf, evidence
        
        return neural_pred, neural_conf, evidence


class FocalLoss(nn.Module):
    """
    Focal loss for handling class imbalance
    Focuses training on hard examples
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [batch_size, 1] raw logits
            targets: [batch_size] binary labels
        
        Returns:
            Focal loss value
        """
        probs = torch.sigmoid(logits).squeeze()
        targets = targets.float()
        
        # Binary cross entropy
        bce = F.binary_cross_entropy_with_logits(
            logits.squeeze(), 
            targets, 
            reduction='none'
        )
        
        # Focal term
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_term = (1 - p_t) ** self.gamma
        
        # Alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        loss = alpha_t * focal_term * bce
        
        return loss.mean()
