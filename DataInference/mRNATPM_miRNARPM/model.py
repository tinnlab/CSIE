"""
Bidirectional mRNA-miRNA Pathway Transformer Model

Architecture:
    Input Expression (genes) + Modality Token
        ↓
    Pathway Encoder (genes → pathways)
        ↓
    Add Source Modality Embedding
        ↓
    Transformer Encoder (self-attention over pathways)
        ↓
    Add Target Modality Embedding
        ↓
    Transformer Decoder
        ↓
    Pathway Decoder (pathways → genes)
        ↓
    Output Expression (genes)

Key Features:
- Single unified model handles both mRNA→miRNA and miRNA→mRNA
- Pathway-based compression using KEGG pathways
- Modality embeddings explicitly encode input/output types
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def load_kegg_pathways(
    pathway_file: str,
    gene_list: List[str]
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Load KEGG pathway data and create pathway-gene matrix.
    
    Args:
        pathway_file: Path to JSON file with pathway data
        gene_list: List of gene names in the dataset (must match order in data)
    
    Returns:
        pathway_gene_matrix: Binary matrix (n_pathways, n_genes)
        pathway_names: List of pathway IDs
        matched_genes: List of genes that appear in pathways
    """
    logger.info(f"Loading KEGG pathways from: {pathway_file}")
    
    # Load pathway data
    with open(pathway_file, 'r') as f:
        pathway_data = json.load(f)
    
    pathway_names = list(pathway_data.keys())
    n_pathways = len(pathway_names)
    n_genes = len(gene_list)
    
    logger.info(f"  Total pathways in file: {n_pathways}")
    logger.info(f"  Total genes in dataset: {n_genes}")
    
    # Create gene to index mapping
    gene_to_idx = {gene: idx for idx, gene in enumerate(gene_list)}
    
    # Initialize pathway-gene matrix
    pathway_gene_matrix = np.zeros((n_pathways, n_genes), dtype=np.float32)
    
    # Fill matrix
    matched_genes_set = set()
    pathway_sizes = []
    
    for pathway_idx, (pathway_id, genes_in_pathway) in enumerate(pathway_data.items()):
        for gene in genes_in_pathway:
            if gene in gene_to_idx:
                gene_idx = gene_to_idx[gene]
                pathway_gene_matrix[pathway_idx, gene_idx] = 1.0
                matched_genes_set.add(gene)
        
        pathway_sizes.append(pathway_gene_matrix[pathway_idx].sum())
    
    matched_genes = sorted(list(matched_genes_set))
    
    # Statistics
    logger.info(f"  Genes matched to pathways: {len(matched_genes)} / {n_genes} ({len(matched_genes)/n_genes*100:.1f}%)")
    logger.info(f"  Average genes per pathway: {np.mean(pathway_sizes):.1f}")
    logger.info(f"  Pathway size range: [{int(np.min(pathway_sizes))}, {int(np.max(pathway_sizes))}]")
    
    # Remove empty pathways (no genes matched)
    non_empty_pathways = pathway_gene_matrix.sum(axis=1) > 0
    pathway_gene_matrix = pathway_gene_matrix[non_empty_pathways]
    pathway_names = [name for i, name in enumerate(pathway_names) if non_empty_pathways[i]]
    
    logger.info(f"  Final pathways (non-empty): {len(pathway_names)}")
    
    return pathway_gene_matrix, pathway_names, matched_genes


class PathwayEncoder(nn.Module):
    """
    Encodes gene expression to pathway activities.
    Maps from gene space (n_genes) to pathway space (n_pathways).
    
    Uses biologically-informed linear layer where weights represent
    gene-pathway membership initialized from KEGG pathways.
    """
    
    def __init__(
        self,
        n_genes: int,
        n_pathways: int,
        hidden_dim: int,
        dropout: float = 0.1,
        pathway_gene_matrix: Optional[np.ndarray] = None
    ):
        """
        Args:
            n_genes: Number of genes
            n_pathways: Number of pathways
            hidden_dim: Hidden dimension for pathway representations
            dropout: Dropout rate
            pathway_gene_matrix: Optional (n_pathways, n_genes) binary matrix
        """
        super().__init__()
        
        self.n_genes = n_genes
        self.n_pathways = n_pathways
        self.hidden_dim = hidden_dim
        
        # Gene to pathway projection (aggregation)
        self.gene_to_pathway = nn.Linear(n_genes, n_pathways, bias=True)
        
        # Initialize with pathway knowledge if provided
        if pathway_gene_matrix is not None:
            self._initialize_with_pathways(pathway_gene_matrix)
        
        # Pathway feature transformation
        self.pathway_transform = nn.Sequential(
            nn.Linear(n_pathways, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Learnable positional encoding for each pathway
        self.pathway_pos_encoding = nn.Parameter(
            torch.randn(1, n_pathways, hidden_dim) * 0.02
        )
    
    def _initialize_with_pathways(self, pathway_gene_matrix: np.ndarray):
        """Initialize weights with pathway-gene relationships."""
        with torch.no_grad():
            weight = torch.FloatTensor(pathway_gene_matrix)
            # Normalize by pathway size
            pathway_sizes = weight.sum(dim=1, keepdim=True).clamp(min=1)
            weight = weight / pathway_sizes
            self.gene_to_pathway.weight.data = weight
            logger.info("✓ Initialized PathwayEncoder with KEGG pathway knowledge")
    
    def forward(self, gene_expression: torch.Tensor) -> torch.Tensor:
        """
        Args:
            gene_expression: (batch_size, n_genes)
        
        Returns:
            pathway_features: (batch_size, n_pathways, hidden_dim)
        """
        batch_size = gene_expression.size(0)
        
        # Aggregate genes to pathway activities
        pathway_activities = self.gene_to_pathway(gene_expression)  # (B, n_pathways)
        
        # Transform to hidden dimension
        pathway_features = self.pathway_transform(pathway_activities)  # (B, hidden_dim)
        
        # Reshape to sequence format
        pathway_features = pathway_features.unsqueeze(1)  # (B, 1, hidden_dim)
        pathway_features = pathway_features.expand(-1, self.n_pathways, -1)  # (B, n_pathways, hidden_dim)
        
        # Add positional encoding
        pathway_features = pathway_features + self.pathway_pos_encoding
        
        return pathway_features


class PathwayDecoder(nn.Module):
    """
    Decodes pathway activities back to gene expression.
    Maps from pathway space (n_pathways) to gene space (n_genes).
    """
    
    def __init__(
        self,
        n_pathways: int,
        n_genes: int,
        hidden_dim: int,
        dropout: float = 0.1,
        pathway_gene_matrix: Optional[np.ndarray] = None
    ):
        """
        Args:
            n_pathways: Number of pathways
            n_genes: Number of genes
            hidden_dim: Hidden dimension
            dropout: Dropout rate
            pathway_gene_matrix: Optional (n_pathways, n_genes) binary matrix
        """
        super().__init__()
        
        self.n_pathways = n_pathways
        self.n_genes = n_genes
        self.hidden_dim = hidden_dim
        
        # Aggregate pathway features
        self.pathway_aggregate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Pathway to gene projection
        self.pathway_to_gene = nn.Linear(hidden_dim, n_genes, bias=True)
        
        # Optional: Initialize with transposed pathway matrix
        if pathway_gene_matrix is not None:
            self._initialize_with_pathways(pathway_gene_matrix)
    
    def _initialize_with_pathways(self, pathway_gene_matrix: np.ndarray):
        """Initialize decoder with pathway knowledge (transposed)."""
        with torch.no_grad():
            # Transpose: genes should be weighted by their pathway memberships
            weight = torch.FloatTensor(pathway_gene_matrix).T  # (n_genes, n_pathways)
            # Normalize
            gene_pathway_counts = weight.sum(dim=1, keepdim=True).clamp(min=1)
            weight = weight / gene_pathway_counts
            # Note: This initializes the first layer before pathway_to_gene
            # For simplicity, we'll let pathway_to_gene learn freely
            logger.info("✓ PathwayDecoder can use pathway structure (optional)")
    
    def forward(self, pathway_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pathway_features: (batch_size, n_pathways, hidden_dim)
        
        Returns:
            gene_expression: (batch_size, n_genes)
        """
        # Aggregate across pathways
        # Mean pooling over pathway dimension
        pathway_agg = pathway_features.mean(dim=1)  # (B, hidden_dim)
        
        # Transform
        pathway_agg = self.pathway_aggregate(pathway_agg)  # (B, hidden_dim)
        
        # Project to gene space
        gene_expression = self.pathway_to_gene(pathway_agg)  # (B, n_genes)
        
        return gene_expression


class ModalityEmbedding(nn.Module):
    """
    Learnable embeddings that indicate modality type (mRNA vs miRNA).
    """
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        
        # Two modality types: 0 = mRNA, 1 = miRNA
        self.modality_embed = nn.Embedding(2, hidden_dim)
        
        # Initialize with small values
        nn.init.normal_(self.modality_embed.weight, mean=0, std=0.02)
    
    def forward(self, modality_ids: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Args:
            modality_ids: (batch_size,) - 0 for mRNA, 1 for miRNA
            seq_len: Sequence length (n_pathways)
        
        Returns:
            modality_features: (batch_size, seq_len, hidden_dim)
        """
        # Get embeddings
        modality_emb = self.modality_embed(modality_ids)  # (B, hidden_dim)
        
        # Expand to sequence length
        modality_emb = modality_emb.unsqueeze(1)  # (B, 1, hidden_dim)
        modality_emb = modality_emb.expand(-1, seq_len, -1)  # (B, seq_len, hidden_dim)
        
        return modality_emb


class TransformerEncoderLayer(nn.Module):
    """
    Standard transformer encoder layer with multi-head self-attention.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, hidden_dim)
            mask: Optional attention mask
        
        Returns:
            output: (batch_size, seq_len, hidden_dim)
        """
        # Self-attention with residual
        attn_out, _ = self.self_attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward with residual
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)
        
        return x


class TransformerDecoderLayer(nn.Module):
    """
    Standard transformer decoder layer with self-attention and cross-attention.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch_size, tgt_seq_len, hidden_dim) - decoder input
            memory: (batch_size, src_seq_len, hidden_dim) - encoder output
            tgt_mask: Optional target mask
            memory_mask: Optional memory mask
        
        Returns:
            output: (batch_size, tgt_seq_len, hidden_dim)
        """
        # Self-attention
        attn_out, _ = self.self_attn(x, x, x, attn_mask=tgt_mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Cross-attention
        attn_out, _ = self.cross_attn(x, memory, memory, attn_mask=memory_mask)
        x = self.norm2(x + self.dropout(attn_out))
        
        # Feed-forward
        ff_out = self.feed_forward(x)
        x = self.norm3(x + ff_out)
        
        return x


class BidirectionalPathwayTransformer(nn.Module):
    """
    Unified bidirectional transformer for mRNA ↔ miRNA translation.
    
    Single model handles both directions using modality embeddings.
    """
    
    def __init__(
        self,
        n_genes: int,
        n_pathways: int,
        hidden_dim: int = 512,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        num_heads: int = 8,
        ff_dim: Optional[int] = None,
        dropout: float = 0.1,
        pathway_gene_matrix: Optional[np.ndarray] = None
    ):
        """
        Args:
            n_genes: Number of genes (common between mRNA and miRNA)
            n_pathways: Number of pathways
            hidden_dim: Hidden dimension
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            num_heads: Number of attention heads
            ff_dim: Feed-forward dimension (default: 4 * hidden_dim)
            dropout: Dropout rate
            pathway_gene_matrix: Optional pathway-gene matrix for initialization
        """
        super().__init__()
        
        self.n_genes = n_genes
        self.n_pathways = n_pathways
        self.hidden_dim = hidden_dim
        
        if ff_dim is None:
            ff_dim = 4 * hidden_dim
        
        # Pathway encoder and decoder
        self.pathway_encoder_module = PathwayEncoder(
            n_genes, n_pathways, hidden_dim, dropout, pathway_gene_matrix
        )
        
        self.pathway_decoder_module = PathwayDecoder(
            n_pathways, n_genes, hidden_dim, dropout, pathway_gene_matrix
        )
        
        # Modality embeddings
        self.source_modality_embed = ModalityEmbedding(hidden_dim)
        self.target_modality_embed = ModalityEmbedding(hidden_dim)
        
        # Transformer encoder
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(hidden_dim, num_heads, ff_dim, dropout)
            for _ in range(num_encoder_layers)
        ])
        
        # Transformer decoder
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(hidden_dim, num_heads, ff_dim, dropout)
            for _ in range(num_decoder_layers)
        ])
        
        logger.info(f"Initialized BidirectionalPathwayTransformer:")
        logger.info(f"  Genes: {n_genes}")
        logger.info(f"  Pathways: {n_pathways}")
        logger.info(f"  Hidden dim: {hidden_dim}")
        logger.info(f"  Encoder layers: {num_encoder_layers}")
        logger.info(f"  Decoder layers: {num_decoder_layers}")
        logger.info(f"  Attention heads: {num_heads}")
        logger.info(f"  Parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def forward(
        self,
        source_expression: torch.Tensor,
        source_modality: int,
        target_modality: int
    ) -> torch.Tensor:
        """
        Forward pass for bidirectional translation.
        
        Args:
            source_expression: (batch_size, n_genes) - input expression
            source_modality: 0 for mRNA, 1 for miRNA
            target_modality: 0 for mRNA, 1 for miRNA
        
        Returns:
            target_expression: (batch_size, n_genes) - predicted expression
        """
        batch_size = source_expression.size(0)
        device = source_expression.device
        
        # 1. Pathway Encoding
        pathway_features = self.pathway_encoder_module(source_expression)  # (B, n_pathways, hidden_dim)
        
        # 2. Add source modality embedding
        source_mod_ids = torch.full((batch_size,), source_modality, dtype=torch.long, device=device)
        source_mod_emb = self.source_modality_embed(source_mod_ids, self.n_pathways)
        encoder_input = pathway_features + source_mod_emb
        
        # 3. Transformer Encoder
        encoder_output = encoder_input
        for layer in self.encoder_layers:
            encoder_output = layer(encoder_output)
        
        # 4. Add target modality embedding for decoder
        target_mod_ids = torch.full((batch_size,), target_modality, dtype=torch.long, device=device)
        target_mod_emb = self.target_modality_embed(target_mod_ids, self.n_pathways)
        decoder_input = encoder_output + target_mod_emb
        
        # 5. Transformer Decoder
        decoder_output = decoder_input
        for layer in self.decoder_layers:
            decoder_output = layer(decoder_output, encoder_output)
        
        # 6. Pathway Decoding
        target_expression = self.pathway_decoder_module(decoder_output)  # (B, n_genes)
        
        return target_expression
    
    def mrna_to_mirna(self, mrna_expression: torch.Tensor) -> torch.Tensor:
        """Convenience method: mRNA → miRNA."""
        return self.forward(mrna_expression, source_modality=0, target_modality=1)
    
    def mirna_to_mrna(self, mirna_expression: torch.Tensor) -> torch.Tensor:
        """Convenience method: miRNA → mRNA."""
        return self.forward(mirna_expression, source_modality=1, target_modality=0)


# Example usage
# if __name__ == "__main__":
#     # Test model creation
#     print("="*80)
#     print("TESTING MODEL")
#     print("="*80)
    
#     # Load pathway data
#     pathway_file = "/data/daotran/Cancer_Subtyping/BiB_Submission/Data/NewKEGGgs.json"
    
#     # Dummy gene list (in practice, this comes from your data loader)
#     dummy_genes = [f"GENE{i}" for i in range(8000)]
    
#     # Load pathways
#     pathway_matrix, pathway_names, matched_genes = load_kegg_pathways(
#         pathway_file, dummy_genes
#     )
    
#     print(f"\nCreating model...")
#     model = BidirectionalPathwayTransformer(
#         n_genes=len(dummy_genes),
#         n_pathways=len(pathway_names),
#         hidden_dim=512,
#         num_encoder_layers=6,
#         num_decoder_layers=6,
#         num_heads=8,
#         dropout=0.1,
#         pathway_gene_matrix=pathway_matrix
#     )
    
#     # Test forward pass
#     print(f"\nTesting forward pass...")
#     batch_size = 32
#     mrna_input = torch.randn(batch_size, len(dummy_genes))
    
#     # mRNA → miRNA
#     mirna_output = model.mrna_to_mirna(mrna_input)
#     print(f"✓ mRNA → miRNA: {mrna_input.shape} → {mirna_output.shape}")
    
#     # miRNA → mRNA
#     mrna_output = model.mirna_to_mrna(mirna_output)
#     print(f"✓ miRNA → mRNA: {mirna_output.shape} → {mrna_output.shape}")
    
#     print(f"\n✓ Model test complete!")

