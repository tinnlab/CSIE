"""
Loss Functions for Bidirectional mRNA-miRNAiso Transformer

Three main loss components:
1. Reconstruction Loss: MSE between predicted and actual expression
2. Cycle Consistency Loss: mRNA→miRNAiso→mRNA should equal original mRNA
3. Pathway Coherence Loss: Genes in same pathway should co-vary consistently

Total Loss = λ1 * Reconstruction + λ2 * Cycle + λ3 * Pathway
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class ReconstructionLoss(nn.Module):
    """
    Basic reconstruction loss between predicted and actual expression.
    Uses MSE (Mean Squared Error).
    """
    
    def __init__(self, reduction: str = 'mean'):
        """
        Args:
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.mse = nn.MSELoss(reduction=reduction)
    
    def forward(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            predicted: (batch_size, n_genes) - model prediction
            target: (batch_size, n_genes) - ground truth
        
        Returns:
            loss: Scalar tensor
        """
        return self.mse(predicted, target)


class CycleConsistencyLoss(nn.Module):
    """
    Cycle consistency loss ensures that:
    - mRNA → miRNAiso → mRNA ≈ original mRNA
    - miRNAiso → mRNA → miRNAiso ≈ original miRNAiso
    
    This encourages the model to learn invertible mappings.
    """
    
    def __init__(self, reduction: str = 'mean'):
        """
        Args:
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.mse = nn.MSELoss(reduction=reduction)
    
    def forward(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            original: (batch_size, n_genes) - original expression
            reconstructed: (batch_size, n_genes) - expression after cycle
        
        Returns:
            loss: Scalar tensor
        """
        return self.mse(reconstructed, original)


class PathwayCoherenceLoss(nn.Module):
    """
    Pathway coherence loss encourages genes in the same pathway to have
    coherent expression patterns.
    
    For each pathway, we compute:
    1. The variance of gene expressions within that pathway
    2. We want low variance (genes move together)
    
    Alternative formulation:
    - Compute pairwise correlations within pathways
    - Encourage high positive correlations
    """
    
    def __init__(
        self,
        pathway_gene_matrix: np.ndarray,
        min_pathway_size: int = 5,
        use_correlation: bool = True
    ):
        """
        Args:
            pathway_gene_matrix: (n_pathways, n_genes) binary matrix
            min_pathway_size: Minimum genes in pathway to compute loss
            use_correlation: If True, use correlation-based loss; else variance-based
        """
        super().__init__()
        
        self.min_pathway_size = min_pathway_size
        self.use_correlation = use_correlation
        
        # Convert to tensor and register as buffer (not trainable)
        self.register_buffer(
            'pathway_gene_matrix',
            torch.FloatTensor(pathway_gene_matrix)
        )
        
        # Precompute pathway sizes
        pathway_sizes = pathway_gene_matrix.sum(axis=1)
        valid_pathways = pathway_sizes >= min_pathway_size
        
        self.register_buffer(
            'valid_pathways',
            torch.BoolTensor(valid_pathways)
        )
        
        n_valid = valid_pathways.sum()
        logger.info(f"PathwayCoherenceLoss: {n_valid} pathways with ≥{min_pathway_size} genes")
    
    def forward(
        self,
        expression: torch.Tensor,
        predicted: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute pathway coherence loss.
        
        We want the predicted expression to maintain the same pathway-level
        co-expression patterns as the original expression.
        
        Args:
            expression: (batch_size, n_genes) - original expression
            predicted: (batch_size, n_genes) - predicted expression
        
        Returns:
            loss: Scalar tensor
        """
        if self.use_correlation:
            return self._correlation_based_loss(expression, predicted)
        else:
            return self._variance_based_loss(expression, predicted)
    
    def _variance_based_loss(
        self,
        expression: torch.Tensor,
        predicted: torch.Tensor
    ) -> torch.Tensor:
        """
        Variance-based: penalize high variance within pathways.
        Genes in same pathway should have similar expression changes.
        """
        batch_size = expression.size(0)
        device = expression.device
        
        # Compute expression difference (change from input to output)
        expr_diff = predicted - expression  # (B, n_genes)
        
        total_loss = 0.0
        n_valid = 0
        
        # For each valid pathway
        for pathway_idx in range(self.pathway_gene_matrix.size(0)):
            if not self.valid_pathways[pathway_idx]:
                continue
            
            # Get genes in this pathway
            pathway_mask = self.pathway_gene_matrix[pathway_idx].bool()  # (n_genes,)
            pathway_expr_diff = expr_diff[:, pathway_mask]  # (B, n_genes_in_pathway)
            
            # Compute variance across genes within pathway (for each sample)
            pathway_var = pathway_expr_diff.var(dim=1)  # (B,)
            
            # Add to total loss
            total_loss += pathway_var.mean()
            n_valid += 1
        
        # Average over pathways
        if n_valid > 0:
            return total_loss / n_valid
        else:
            return torch.tensor(0.0, device=device)
    
    def _correlation_based_loss(
        self,
        expression: torch.Tensor,
        predicted: torch.Tensor
    ) -> torch.Tensor:
        """
        Correlation-based: maintain within-pathway correlations.
        
        For each pathway, compute correlation matrix of genes in that pathway
        for both original and predicted expression. The correlation patterns
        should be similar.
        """
        batch_size = expression.size(0)
        device = expression.device
        
        total_loss = 0.0
        n_valid = 0
        
        # For each valid pathway
        for pathway_idx in range(self.pathway_gene_matrix.size(0)):
            if not self.valid_pathways[pathway_idx]:
                continue
            
            # Get genes in this pathway
            pathway_mask = self.pathway_gene_matrix[pathway_idx].bool()  # (n_genes,)
            
            # Extract pathway genes
            orig_pathway = expression[:, pathway_mask]  # (B, n_pathway_genes)
            pred_pathway = predicted[:, pathway_mask]  # (B, n_pathway_genes)
            
            # Compute pathway-level statistics (mean expression across genes)
            # This measures if the overall pathway activity is preserved
            orig_pathway_mean = orig_pathway.mean(dim=1)  # (B,)
            pred_pathway_mean = pred_pathway.mean(dim=1)  # (B,)
            
            # MSE between pathway means
            pathway_loss = F.mse_loss(pred_pathway_mean, orig_pathway_mean)
            
            total_loss += pathway_loss
            n_valid += 1
        
        # Average over pathways
        if n_valid > 0:
            return total_loss / n_valid
        else:
            return torch.tensor(0.0, device=device)


class BidirectionalTransformerLoss(nn.Module):
    """
    Combined loss for bidirectional transformer training.
    
    Total Loss = λ1 * Reconstruction + λ2 * Cycle + λ3 * Pathway
    """
    
    def __init__(
        self,
        pathway_gene_matrix: Optional[np.ndarray] = None,
        lambda_reconstruction: float = 1.0,
        lambda_cycle: float = 0.1,
        lambda_pathway: float = 0.01,
        min_pathway_size: int = 5,
        use_correlation: bool = True
    ):
        """
        Args:
            pathway_gene_matrix: Optional pathway-gene matrix for pathway loss
            lambda_reconstruction: Weight for reconstruction loss
            lambda_cycle: Weight for cycle consistency loss
            lambda_pathway: Weight for pathway coherence loss
            min_pathway_size: Minimum pathway size for coherence loss
            use_correlation: Whether to use correlation-based pathway loss
        """
        super().__init__()
        
        self.lambda_reconstruction = lambda_reconstruction
        self.lambda_cycle = lambda_cycle
        self.lambda_pathway = lambda_pathway
        
        # Loss components
        self.reconstruction_loss = ReconstructionLoss()
        self.cycle_loss = CycleConsistencyLoss()
        
        if pathway_gene_matrix is not None and lambda_pathway > 0:
            self.pathway_loss = PathwayCoherenceLoss(
                pathway_gene_matrix,
                min_pathway_size,
                use_correlation
            )
            self.use_pathway_loss = True
        else:
            self.pathway_loss = None
            self.use_pathway_loss = False
        
        logger.info(f"Initialized BidirectionalTransformerLoss:")
        logger.info(f"  λ_reconstruction: {lambda_reconstruction}")
        logger.info(f"  λ_cycle: {lambda_cycle}")
        logger.info(f"  λ_pathway: {lambda_pathway}")
        logger.info(f"  Use pathway loss: {self.use_pathway_loss}")
    
    def forward(
        self,
        model: nn.Module,
        mrna_batch: torch.Tensor,
        mirna_batch: torch.Tensor,
        compute_cycle: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss for a batch.
        
        Args:
            model: The bidirectional transformer model
            mrna_batch: (batch_size, n_genes) - mRNA expression
            mirna_batch: (batch_size, n_genes) - miRNAiso expression
            compute_cycle: Whether to compute cycle consistency loss
        
        Returns:
            total_loss: Combined loss scalar
            loss_dict: Dictionary with individual loss components
        """
        device = mrna_batch.device
        
        # Initialize loss dictionary
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=device)
        
        # ===== 1. Forward Direction: mRNA → miRNAiso =====
        mirna_pred = model.mrna_to_mirna(mrna_batch)
        
        # Reconstruction loss (mRNA → miRNAiso)
        recon_loss_forward = self.reconstruction_loss(mirna_pred, mirna_batch)
        loss_dict['recon_mrna2mirna'] = recon_loss_forward.item()
        total_loss += self.lambda_reconstruction * recon_loss_forward
        
        # Pathway coherence loss (mRNA → miRNAiso)
        if self.use_pathway_loss:
            pathway_loss_forward = self.pathway_loss(mrna_batch, mirna_pred)
            loss_dict['pathway_mrna2mirna'] = pathway_loss_forward.item()
            total_loss += self.lambda_pathway * pathway_loss_forward
        
        # ===== 2. Backward Direction: miRNAiso → mRNA =====
        mrna_pred = model.mirna_to_mrna(mirna_batch)
        
        # Reconstruction loss (miRNAiso → mRNA)
        recon_loss_backward = self.reconstruction_loss(mrna_pred, mrna_batch)
        loss_dict['recon_mirna2mrna'] = recon_loss_backward.item()
        total_loss += self.lambda_reconstruction * recon_loss_backward
        
        # Pathway coherence loss (miRNAiso → mRNA)
        if self.use_pathway_loss:
            pathway_loss_backward = self.pathway_loss(mirna_batch, mrna_pred)
            loss_dict['pathway_mirna2mrna'] = pathway_loss_backward.item()
            total_loss += self.lambda_pathway * pathway_loss_backward
        
        # ===== 3. Cycle Consistency =====
        if compute_cycle and self.lambda_cycle > 0:
            # Forward cycle: mRNA → miRNAiso → mRNA
            mrna_reconstructed = model.mirna_to_mrna(mirna_pred)
            cycle_loss_forward = self.cycle_loss(mrna_batch, mrna_reconstructed)
            loss_dict['cycle_mrna'] = cycle_loss_forward.item()
            total_loss += self.lambda_cycle * cycle_loss_forward
            
            # Backward cycle: miRNAiso → mRNA → miRNAiso
            mirna_reconstructed = model.mrna_to_mirna(mrna_pred)
            cycle_loss_backward = self.cycle_loss(mirna_batch, mirna_reconstructed)
            loss_dict['cycle_mirna'] = cycle_loss_backward.item()
            total_loss += self.lambda_cycle * cycle_loss_backward
        
        # ===== 4. Average reconstruction loss for logging =====
        loss_dict['recon_total'] = (recon_loss_forward.item() + recon_loss_backward.item()) / 2
        
        if self.use_pathway_loss:
            loss_dict['pathway_total'] = (pathway_loss_forward.item() + pathway_loss_backward.item()) / 2
        
        if compute_cycle and self.lambda_cycle > 0:
            loss_dict['cycle_total'] = (cycle_loss_forward.item() + cycle_loss_backward.item()) / 2
        
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict


# # Example usage
# if __name__ == "__main__":
#     print("="*80)
#     print("TESTING LOSS FUNCTIONS")
#     print("="*80)
    
#     # Create dummy data
#     batch_size = 32
#     n_genes = 1000
#     n_pathways = 50
    
#     # Dummy pathway matrix
#     pathway_matrix = np.random.rand(n_pathways, n_genes) > 0.9
#     pathway_matrix = pathway_matrix.astype(np.float32)
    
#     # Test individual losses
#     print("\n1. Testing ReconstructionLoss...")
#     recon_loss = ReconstructionLoss()
#     pred = torch.randn(batch_size, n_genes)
#     target = torch.randn(batch_size, n_genes)
#     loss = recon_loss(pred, target)
#     print(f"   Loss: {loss.item():.4f}")
    
#     print("\n2. Testing CycleConsistencyLoss...")
#     cycle_loss = CycleConsistencyLoss()
#     original = torch.randn(batch_size, n_genes)
#     reconstructed = original + torch.randn(batch_size, n_genes) * 0.1
#     loss = cycle_loss(original, reconstructed)
#     print(f"   Loss: {loss.item():.4f}")
    
#     print("\n3. Testing PathwayCoherenceLoss...")
#     pathway_loss = PathwayCoherenceLoss(pathway_matrix, min_pathway_size=5)
#     expression = torch.randn(batch_size, n_genes)
#     predicted = expression + torch.randn(batch_size, n_genes) * 0.1
#     loss = pathway_loss(expression, predicted)
#     print(f"   Loss: {loss.item():.4f}")
    
#     print("\n4. Testing Combined Loss...")
#     combined_loss = BidirectionalTransformerLoss(
#         pathway_gene_matrix=pathway_matrix,
#         lambda_reconstruction=1.0,
#         lambda_cycle=0.1,
#         lambda_pathway=0.01
#     )
    
#     # Need a dummy model
#     class DummyModel(nn.Module):
#         def __init__(self):
#             super().__init__()
#             self.linear = nn.Linear(n_genes, n_genes)
        
#         def mrna_to_mirna(self, x):
#             return self.linear(x)
        
#         def mirna_to_mrna(self, x):
#             return self.linear(x)
    
#     model = DummyModel()
#     mrna = torch.randn(batch_size, n_genes)
#     mirna = torch.randn(batch_size, n_genes)
    
#     total_loss, loss_dict = combined_loss(model, mrna, mirna, compute_cycle=True)
    
#     print(f"   Total Loss: {total_loss.item():.4f}")
#     print(f"   Loss breakdown:")
#     for key, value in loss_dict.items():
#         print(f"     {key}: {value:.4f}")
    
#     print("\n✓ All loss functions working correctly!")

