"""
Training Script for Bidirectional mRNA-miRNAiso Transformer

Features:
- Bidirectional training (both mRNA→miRNAiso and miRNAiso→mRNA)
- Validation with early stopping
- Checkpointing (best model and regular intervals)
- Learning rate scheduling
- Comprehensive logging (TensorBoard/console)
- Gradient clipping
- Mixed precision training (optional)
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data import TCGADataLoader, create_dataloaders
from model import BidirectionalPathwayTransformer, load_kegg_pathways
from losses import BidirectionalTransformerLoss

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Trainer:
    """
    Trainer class for bidirectional pathway transformer.
    """
    
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        data_loader: TCGADataLoader,
        config: Dict,
        device: torch.device,
        checkpoint_dir: str = "./checkpoints",
        log_dir: str = "./logs"
    ):
        """
        Args:
            model: BidirectionalPathwayTransformer model
            criterion: BidirectionalTransformerLoss
            optimizer: Optimizer
            train_loader: Training data loader
            val_loader: Validation data loader
            data_loader: TCGADataLoader instance (for scalers and gene list)
            config: Configuration dictionary
            device: Device to train on
            checkpoint_dir: Directory to save checkpoints
            log_dir: Directory for tensorboard logs
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.data_loader = data_loader
        self.config = config
        self.device = device
        
        # Directories
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        
        # Learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        
        # Gradient scaler for mixed precision (optional)
        self.use_amp = config.get('use_amp', False)
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        logger.info(f"Trainer initialized")
        logger.info(f"  Device: {device}")
        logger.info(f"  Checkpoint dir: {checkpoint_dir}")
        logger.info(f"  Log dir: {log_dir}")
        logger.info(f"  Mixed precision: {self.use_amp}")
    
    def _create_scheduler(self):
        """Create learning rate scheduler."""
        scheduler_type = self.config.get('scheduler', 'reduce_on_plateau')
        
        if scheduler_type == 'reduce_on_plateau':
            return ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5
            )
        elif scheduler_type == 'cosine':
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['epochs'],
                eta_min=1e-6
            )
        else:
            return None
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary with average training losses
        """
        self.model.train()
        epoch_losses = {
            'total': [],
            'recon_total': [],
            'cycle_total': [],
            'pathway_total': []
        }
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1}")
        
        for batch_idx, (mrna_batch, mirna_batch) in enumerate(pbar):
            # Move to device
            mrna_batch = mrna_batch.to(self.device)
            mirna_batch = mirna_batch.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    loss, loss_dict = self.criterion(
                        self.model,
                        mrna_batch,
                        mirna_batch,
                        compute_cycle=True
                    )
                
                # Backward pass with scaling
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.get('grad_clip', 0) > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['grad_clip']
                    )
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard training
                loss, loss_dict = self.criterion(
                    self.model,
                    mrna_batch,
                    mirna_batch,
                    compute_cycle=True
                )
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.config.get('grad_clip', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['grad_clip']
                    )
                
                # Optimizer step
                self.optimizer.step()
            
            # Log losses
            for key in epoch_losses.keys():
                if key in loss_dict:
                    epoch_losses[key].append(loss_dict[key])
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss_dict['total']:.4f}",
                'recon': f"{loss_dict['recon_total']:.4f}"
            })
            
            # Log to tensorboard
            if self.global_step % self.config.get('log_interval', 10) == 0:
                for key, value in loss_dict.items():
                    self.writer.add_scalar(f'train/{key}', value, self.global_step)
                
                # Log learning rate
                current_lr = self.optimizer.param_groups[0]['lr']
                self.writer.add_scalar('train/lr', current_lr, self.global_step)
            
            self.global_step += 1
        
        # Compute average losses
        avg_losses = {key: np.mean(values) for key, values in epoch_losses.items()}
        
        return avg_losses
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Validate the model.
        
        Returns:
            Dictionary with average validation losses
        """
        self.model.eval()
        epoch_losses = {
            'total': [],
            'recon_total': [],
            'cycle_total': [],
            'pathway_total': []
        }
        
        for mrna_batch, mirna_batch in tqdm(self.val_loader, desc="Validation"):
            # Move to device
            mrna_batch = mrna_batch.to(self.device)
            mirna_batch = mirna_batch.to(self.device)
            
            # Forward pass
            loss, loss_dict = self.criterion(
                self.model,
                mrna_batch,
                mirna_batch,
                compute_cycle=True
            )
            
            # Log losses
            for key in epoch_losses.keys():
                if key in loss_dict:
                    epoch_losses[key].append(loss_dict[key])
        
        # Compute average losses
        avg_losses = {key: np.mean(values) for key, values in epoch_losses.items()}
        
        return avg_losses
    
    @torch.no_grad()
    def evaluate_metrics(self) -> Dict[str, float]:
        """
        Evaluate additional metrics (correlations, MSE, etc.).
        
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        
        all_mrna = []
        all_mirna = []
        all_mrna_pred = []
        all_mirna_pred = []
        
        for mrna_batch, mirna_batch in self.val_loader:
            mrna_batch = mrna_batch.to(self.device)
            mirna_batch = mirna_batch.to(self.device)
            
            # Predictions
            mirna_pred = self.model.mrna_to_mirna(mrna_batch)
            mrna_pred = self.model.mirna_to_mrna(mirna_batch)
            
            # Store
            all_mrna.append(mrna_batch.cpu())
            all_mirna.append(mirna_batch.cpu())
            all_mrna_pred.append(mrna_pred.cpu())
            all_mirna_pred.append(mirna_pred.cpu())
        
        # Concatenate
        all_mrna = torch.cat(all_mrna, dim=0).numpy()
        all_mirna = torch.cat(all_mirna, dim=0).numpy()
        all_mrna_pred = torch.cat(all_mrna_pred, dim=0).numpy()
        all_mirna_pred = torch.cat(all_mirna_pred, dim=0).numpy()
        
        # Compute metrics
        metrics = {}
        
        # Pearson correlation (sample-wise average)
        from scipy.stats import pearsonr
        
        # mRNA → miRNAiso correlation
        corrs_m2mi = []
        for i in range(len(all_mirna)):
            corr, _ = pearsonr(all_mirna[i], all_mirna_pred[i])
            if not np.isnan(corr):
                corrs_m2mi.append(corr)
        metrics['corr_mrna2mirna'] = np.mean(corrs_m2mi)
        
        # miRNAiso → mRNA correlation
        corrs_mi2m = []
        for i in range(len(all_mrna)):
            corr, _ = pearsonr(all_mrna[i], all_mrna_pred[i])
            if not np.isnan(corr):
                corrs_mi2m.append(corr)
        metrics['corr_mirna2mrna'] = np.mean(corrs_mi2m)
        
        # MSE
        metrics['mse_mrna2mirna'] = np.mean((all_mirna - all_mirna_pred) ** 2)
        metrics['mse_mirna2mrna'] = np.mean((all_mrna - all_mrna_pred) ** 2)
        
        return metrics
    
    def save_checkpoint(self, is_best: bool = False, filename: Optional[str] = None):
        """
        Save model checkpoint.
        
        Args:
            is_best: Whether this is the best model so far
            filename: Optional custom filename
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            # NEW: Save scalers and gene list for inference
            'mrna_scaler_mean': self.data_loader.mrna_scaler.mean_,
            'mrna_scaler_scale': self.data_loader.mrna_scaler.scale_,
            'mirna_scaler_mean': self.data_loader.mirna_scaler.mean_,
            'mirna_scaler_scale': self.data_loader.mirna_scaler.scale_,
            'common_genes': self.data_loader.common_genes
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # if filename is None:
        #     filename = f"checkpoint_epoch_{self.current_epoch}.pt"
        
        # filepath = self.checkpoint_dir / filename
        # torch.save(checkpoint, filepath)
        # logger.info(f"Saved checkpoint: {filepath}")
        
        # Save best model separately
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model: {best_path}")
    
    def load_checkpoint(self, filepath: str):
        """Load checkpoint from file."""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        logger.info(f"Loaded checkpoint from: {filepath}")
        logger.info(f"  Resuming from epoch {self.current_epoch}")
    
    def train(self):
        """
        Main training loop.
        """
        logger.info("="*80)
        logger.info("STARTING TRAINING")
        logger.info("="*80)
        logger.info(f"Training for {self.config['epochs']} epochs")
        logger.info(f"Train batches: {len(self.train_loader)}")
        logger.info(f"Val batches: {len(self.val_loader)}")
        
        start_time = time.time()
        
        for epoch in range(self.current_epoch, self.config['epochs']):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # Train
            train_losses = self.train_epoch()
            
            # Validate
            val_losses = self.validate()
            
            # Evaluate metrics
            if epoch % self.config.get('eval_interval', 5) == 0:
                metrics = self.evaluate_metrics()
                logger.info(f"Metrics - Corr(mRNA→miRNAiso): {metrics['corr_mrna2mirna']:.4f}, "
                           f"Corr(miRNAiso→mRNA): {metrics['corr_mirna2mrna']:.4f}")
                
                # Log to tensorboard
                for key, value in metrics.items():
                    self.writer.add_scalar(f'val/{key}', value, epoch)
            
            # Log to tensorboard
            for key, value in val_losses.items():
                self.writer.add_scalar(f'val/{key}', value, epoch)
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_losses['total'])
                else:
                    self.scheduler.step()
            
            # Print epoch summary
            epoch_time = time.time() - epoch_start_time
            logger.info(f"\nEpoch {epoch+1}/{self.config['epochs']} ({epoch_time:.1f}s)")
            logger.info(f"  Train Loss: {train_losses['total']:.4f} "
                       f"(Recon: {train_losses['recon_total']:.4f}, "
                       f"Cycle: {train_losses.get('cycle_total', 0):.4f})")
            logger.info(f"  Val Loss:   {val_losses['total']:.4f} "
                       f"(Recon: {val_losses['recon_total']:.4f}, "
                       f"Cycle: {val_losses.get('cycle_total', 0):.4f})")
            
            # Check for improvement
            is_best = val_losses['total'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_losses['total']
                self.epochs_without_improvement = 0
                logger.info(f"  ✓ New best validation loss: {self.best_val_loss:.4f}")
            else:
                self.epochs_without_improvement += 1
                logger.info(f"  No improvement for {self.epochs_without_improvement} epochs")
            
            # Save best model only
            if is_best:
                self.save_checkpoint(is_best=True)
            
            # Early stopping
            if self.epochs_without_improvement >= self.config.get('patience', 20):
                logger.info(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
        
        # Training complete
        total_time = time.time() - start_time
        logger.info("\n" + "="*80)
        logger.info("TRAINING COMPLETE")
        logger.info("="*80)
        logger.info(f"Total training time: {total_time/3600:.2f} hours")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        
        # Save final results
        results = {
            'best_val_loss': self.best_val_loss,
            'total_epochs': self.current_epoch + 1,
            'config': self.config
        }
        
        results_path = self.checkpoint_dir / "final_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"\nSaved final results to: {results_path}")
        
        self.writer.close()
    
    


def main(config_dict=None):
    """
    Main training function.
    
    Args:
        config_dict: Optional dictionary with configuration parameters.
                     If None, uses default configuration.
    """
    # Use provided config or create default
    if config_dict is None:
        config = {
            # Data
            'data_root': "/data/daotran/Cancer_Subtyping/BiB_Submission/Data/TCGA_csv",
            'pathway_file': "/data/daotran/Cancer_Subtyping/BiB_Submission/Data/NewKEGGgs.json",
            'batch_size': 32,
            'num_workers': 4,
            
            # Model
            'hidden_dim': 512,
            'num_encoder_layers': 6,
            'num_decoder_layers': 6,
            'num_heads': 8,
            'dropout': 0.1,
            
            # Loss
            'lambda_reconstruction': 1.0,
            'lambda_cycle': 0.1,
            'lambda_pathway': 0.01,
            
            # Training
            'epochs': 100,
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'grad_clip': 1.0,
            'scheduler': 'reduce_on_plateau',
            'patience': 20,
            
            # Logging
            'log_interval': 10,
            'eval_interval': 5,
            
            # Other
            'use_amp': False,  # Mixed precision
            'seed': 42
        }
    else:
        config = config_dict
    
    # Set seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    # Device
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpu_id = 1  # Use GPU 0
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load data
    logger.info("\n" + "="*80)
    logger.info("LOADING DATA")
    logger.info("="*80)
    
    data_loader = TCGADataLoader(
        data_root=config['data_root'],
        min_samples_per_cancer=10,
        normalize=True,
        log_transform=True,
        handle_missing='zero'  # Default: replace missing with 0
    )
    
    mrna_data, mirna_data = data_loader.load_data()
    
    # Split
    mrna_train, mirna_train, mrna_val, mirna_val = \
        data_loader.get_train_val_split(
            val_size=0.2,
            random_state=config['seed']
        )
    
    # Normalize
    mrna_train, mirna_train, mrna_val, mirna_val = \
        data_loader.normalize_data(
            mrna_train, mrna_val,
            mirna_train, mirna_val
        )
    
    np.save("./checkpoints/mrna_train_normalized.npy", mrna_train)

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        mrna_train, mirna_train,
        mrna_val, mirna_val,
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )
    
    # Load pathways
    logger.info("\n" + "="*80)
    logger.info("LOADING PATHWAYS")
    logger.info("="*80)
    
    pathway_matrix, pathway_names, matched_genes = load_kegg_pathways(
        config['pathway_file'],
        data_loader.common_genes
    )
    
    # Create model
    logger.info("\n" + "="*80)
    logger.info("CREATING MODEL")
    logger.info("="*80)
    
    model = BidirectionalPathwayTransformer(
        n_genes=len(data_loader.common_genes),
        n_pathways=len(pathway_names),
        hidden_dim=config['hidden_dim'],
        num_encoder_layers=config['num_encoder_layers'],
        num_decoder_layers=config['num_decoder_layers'],
        num_heads=config['num_heads'],
        dropout=config['dropout'],
        pathway_gene_matrix=pathway_matrix
    ).to(device)
    
    # Create loss
    criterion = BidirectionalTransformerLoss(
        pathway_gene_matrix=pathway_matrix,
        lambda_reconstruction=config['lambda_reconstruction'],
        lambda_cycle=config['lambda_cycle'],
        lambda_pathway=config['lambda_pathway']
    )
    
    # Create optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        data_loader=data_loader,
        config=config,
        device=device,
        checkpoint_dir="./checkpoints",
        log_dir="./logs"
    )
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()