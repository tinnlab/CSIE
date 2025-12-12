"""
Inference Script for Bidirectional mRNA-miRNA Transformer

Generate predictions using a trained model on new data.

Features:
- Load trained model from checkpoint
- Predict mRNA → miRNA or miRNA → mRNA
- Inverse transform to original scale
- Save predictions to CSV
- Compute metrics if ground truth available
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Union
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from tqdm import tqdm

from data import TCGADataLoader, PairedExpressionDataset
from model import BidirectionalPathwayTransformer, load_kegg_pathways

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Predictor:
    """
    Predictor class for inference with trained model.
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[torch.device] = None
    ):
        """
        Args:
            checkpoint_path: Path to trained model checkpoint
            device: Device to run inference on (default: auto-detect)
        """
        self.checkpoint_path = Path(checkpoint_path)
        
        if device is None:
            # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.device = torch.device(f'cuda:{2}' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        logger.info(f"Using device: {self.device}")
        
        # Will be loaded
        self.model = None
        self.config = None
        self.data_loader = None
        
        # Load checkpoint
        self._load_checkpoint()
    
    def _load_checkpoint(self):
        """Load model and configuration from checkpoint."""
        logger.info(f"Loading checkpoint from: {self.checkpoint_path}")
        
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        self.config = checkpoint['config']
        
        logger.info(f"  Epoch: {checkpoint['epoch']}")
        logger.info(f"  Best val loss: {checkpoint.get('best_val_loss', 'N/A')}")
        
        # NEW: Load scalers and gene list from checkpoint (no data loading needed!)
        if 'common_genes' in checkpoint:
            logger.info("\n✓ Loading gene list and scalers from checkpoint (fast!)")
            
            common_genes = checkpoint['common_genes']
            
            # Reconstruct scalers from saved parameters
            from sklearn.preprocessing import StandardScaler
            
            mrna_scaler = StandardScaler()
            mrna_scaler.mean_ = checkpoint['mrna_scaler_mean']
            mrna_scaler.scale_ = checkpoint['mrna_scaler_scale']
            mrna_scaler.n_features_in_ = len(common_genes)
            
            mirna_scaler = StandardScaler()
            mirna_scaler.mean_ = checkpoint['mirna_scaler_mean']
            mirna_scaler.scale_ = checkpoint['mirna_scaler_scale']
            mirna_scaler.n_features_in_ = len(common_genes)
            
            # Create minimal data_loader object just to hold scalers and genes
            class MinimalDataLoader:
                pass
            
            self.data_loader = MinimalDataLoader()
            self.data_loader.common_genes = common_genes
            self.data_loader.mrna_scaler = mrna_scaler
            self.data_loader.mirna_scaler = mirna_scaler
            
            logger.info(f"  Genes: {len(common_genes)}")
            
        else:
            # OLD CHECKPOINT: Fallback to loading data (will be slow)
            logger.warning("\n⚠️  Old checkpoint format detected - loading data (slow)...")
            logger.warning("   Re-save your model to speed up inference in the future.")
            
            self.data_loader = TCGADataLoader(
                data_root=self.config['data_root'],
                min_samples_per_cancer=10,
                normalize=False,
                log_transform=True,
                handle_missing='zero'
            )
            
            logger.info("Loading data to extract gene list...")
            mrna_data, mirna_data = self.data_loader.load_data()
            
            # Get splits to fit scalers
            mrna_train, mirna_train, mrna_val, mirna_val, mrna_test, mirna_test = \
                self.data_loader.get_train_val_split(
            val_size=0.2,
                    random_state=self.config['seed']
                )
            
            # Fit scalers
            self.data_loader.normalize_data(
                mrna_train, mrna_val, mrna_test,
                mirna_train, mirna_val, mirna_test
            )
            
            logger.info(f"  Genes: {len(self.data_loader.common_genes)}")
        
        # Load pathways
        logger.info("\nLoading pathway information...")
        pathway_matrix, pathway_names, _ = load_kegg_pathways(
            self.config['pathway_file'],
            self.data_loader.common_genes
        )
        
        # Create model
        logger.info("\nCreating model...")
        self.model = BidirectionalPathwayTransformer(
            n_genes=len(self.data_loader.common_genes),
            n_pathways=len(pathway_names),
            hidden_dim=self.config['hidden_dim'],
            num_encoder_layers=self.config['num_encoder_layers'],
            num_decoder_layers=self.config['num_decoder_layers'],
            num_heads=self.config['num_heads'],
            dropout=self.config['dropout'],
            pathway_gene_matrix=pathway_matrix
        ).to(self.device)
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        logger.info("✓ Model loaded successfully")
    
    @torch.no_grad()
    def predict_mrna_to_mirna(
        self,
        mrna_input: np.ndarray,
        inverse_transform: bool = True
    ) -> np.ndarray:
        """
        Predict miRNA from mRNA.
        
        Args:
            mrna_input: (n_samples, n_genes) - mRNA expression (original scale)
            inverse_transform: Whether to transform back to original scale
        
        Returns:
            mirna_pred: (n_samples, n_genes) - Predicted miRNA expression
        """
        # Normalize input
        mrna_normalized = self.data_loader.mrna_scaler.transform(mrna_input)
        
        # Convert to tensor
        mrna_tensor = torch.FloatTensor(mrna_normalized).to(self.device)
        
        # Predict
        mirna_pred_normalized = self.model.mrna_to_mirna(mrna_tensor)
        
        # Convert back to numpy
        mirna_pred_normalized = mirna_pred_normalized.cpu().numpy()
        
        # Inverse transform if requested
        if inverse_transform:
            mirna_pred = self.data_loader.mirna_scaler.inverse_transform(mirna_pred_normalized)
        else:
            mirna_pred = mirna_pred_normalized
        
        return mirna_pred
    
    @torch.no_grad()
    def predict_mirna_to_mrna(
        self,
        mirna_input: np.ndarray,
        inverse_transform: bool = True
    ) -> np.ndarray:
        """
        Predict mRNA from miRNA.
        
        Args:
            mirna_input: (n_samples, n_genes) - miRNA expression (original scale)
            inverse_transform: Whether to transform back to original scale
        
        Returns:
            mrna_pred: (n_samples, n_genes) - Predicted mRNA expression
        """
        # Normalize input
        mirna_normalized = self.data_loader.mirna_scaler.transform(mirna_input)
        
        # Convert to tensor
        mirna_tensor = torch.FloatTensor(mirna_normalized).to(self.device)
        
        # Predict
        mrna_pred_normalized = self.model.mirna_to_mrna(mirna_tensor)
        
        # Convert back to numpy
        mrna_pred_normalized = mrna_pred_normalized.cpu().numpy()
        
        # Inverse transform if requested
        if inverse_transform:
            mrna_pred = self.data_loader.mrna_scaler.inverse_transform(mrna_pred_normalized)
        else:
            mrna_pred = mrna_pred_normalized
        
        return mrna_pred
    
    def cdf_matching(self, data: np.ndarray, reference_data: np.ndarray) -> np.ndarray:
        """
        Transform data using cumulative distribution function matching.
        Performs matching separately for each gene (column).
        
        Args:
            data: Data to transform (n_samples, n_genes)
            reference_data: Reference distribution (n_ref_samples, n_genes)
        
        Returns:
            Transformed data with same shape as input
        """
        n_samples, n_genes = data.shape
        transformed = np.zeros_like(data)
        
        # Process each gene separately
        for gene_idx in range(n_genes):
            gene_data = data[:, gene_idx]
            gene_ref = reference_data[:, gene_idx]
            
            # Sort data and get ranks
            data_ranks = np.argsort(np.argsort(gene_data))
            
            # Get corresponding values from reference distribution
            ref_sorted = np.sort(gene_ref)
            n_ref = len(ref_sorted)
            
            # Map ranks to reference values
            # Scale ranks to reference distribution size
            scaled_ranks = (data_ranks * (n_ref - 1) / (len(data_ranks) - 1)).astype(int)
            scaled_ranks = np.clip(scaled_ranks, 0, n_ref - 1)
            
            transformed[:, gene_idx] = ref_sorted[scaled_ranks]
        
        return transformed


    def preprocess_new_data(
        self,
        new_data: pd.DataFrame,
        modality: str = 'mrna',
        log_transform: bool = True,
        normalize_setting: str = 'independent'
    ) -> np.ndarray:
        """
        Preprocess new data to match training data format.
        
        Handles:
        - Replacing NA/NaN values with 0
        - Aligning genes (keeping only training genes, filling missing with 0)
        - Log transformation (if needed)
        - Z-score normalization with three different settings
        
        Args:
            new_data: DataFrame with samples as rows, genes as columns
            modality: 'mrna' or 'mirna' (determines which scaler to use)
            log_transform: Whether to apply log1p transformation
            normalize_setting: Normalization method
                - 'use_train': Use training statistics for z-score normalization
                - 'independent': Use new data's own statistics, then apply CDF matching
                - 'no_norm': Skip normalization entirely
        
        Returns:
            Preprocessed array ready for model input (normalized/transformed, aligned)
        """
        logger.info("Preprocessing new data...")
        logger.info(f"  Input shape: {new_data.shape}")
        logger.info(f"  Modality: {modality}")
        logger.info(f"  Normalize setting: {normalize_setting}")
        
        # Validate normalize_setting parameter
        valid_settings = ['use_train', 'independent', 'no_norm']
        if normalize_setting not in valid_settings:
            raise ValueError(f"normalize_setting must be one of {valid_settings}, got: {normalize_setting}")
        
        # Step 1: Replace NA/NaN with 0
        data_filled = new_data.fillna(0)
        n_na_filled = new_data.isna().sum().sum()
        if n_na_filled > 0:
            logger.info(f"  ✓ Filled {n_na_filled} NA/NaN values with 0")
        
        # Step 2: Align genes with training data
        training_genes = self.data_loader.common_genes
        current_genes = data_filled.columns.tolist()
        
        # Find common genes
        common_genes = set(training_genes).intersection(set(current_genes))
        logger.info(f"  ✓ Found {len(common_genes)} / {len(training_genes)} training genes in new data")
        
        # Create aligned dataframe
        aligned_data = pd.DataFrame(
            0.0,  # Fill with 0 for missing genes
            index=data_filled.index,
            columns=training_genes
        )
        
        # Fill in available genes
        for gene in common_genes:
            aligned_data[gene] = data_filled[gene]
        
        missing_genes = len(training_genes) - len(common_genes)
        if missing_genes > 0:
            logger.info(f"  ✓ Filled {missing_genes} missing genes with 0")
        
        # Step 3: Log transformation (if needed)
        if log_transform:
            # Check if data seems to be in log scale already
            data_max = aligned_data.values.max()
            data_min = aligned_data.values.min()
            if data_min >= 0 and data_max > 100:
                aligned_data = np.log2(aligned_data + 1)
                logger.info(f"  ✓ Applied log2 transformation")
            else:
                logger.info(f"  ⚠ Data max value ({data_max:.2f}) suggests already log-transformed, skipping")
        
        # Step 4: Normalization based on normalize_setting
        if normalize_setting == 'no_norm':
            # No normalization
            logger.info(f"  ✓ Skipping normalization (no_norm)")
            final_data = aligned_data.values
            
        elif normalize_setting == 'use_train':
            # Use training statistics for z-score normalization
            if modality.lower() == 'mrna':
                scaler = self.data_loader.mrna_scaler
            elif modality.lower() == 'mirna':
                scaler = self.data_loader.mirna_scaler
            else:
                raise ValueError(f"Unknown modality: {modality}. Use 'mrna' or 'mirna'")
            
            final_data = scaler.transform(aligned_data.values)
            logger.info(f"  ✓ Applied z-score normalization using training statistics")
            
        elif normalize_setting == 'independent':
            # Independent z-score normalization followed by CDF matching
            logger.info(f"  ✓ Performing independent normalization with CDF matching...")
            
            # Step 4a: Z-score normalize using new data's own statistics
            data_mean = aligned_data.values.mean(axis=0, keepdims=True)
            data_std = aligned_data.values.std(axis=0, keepdims=True)
            # Avoid division by zero
            data_std[data_std == 0] = 1.0
            
            normalized_data = (aligned_data.values - data_mean) / data_std
            logger.info(f"    - Applied z-score normalization using new data statistics")
            
            # Step 4b: Load training data for CDF matching
            checkpoint_dir = Path("./checkpoints")
            train_data_path = checkpoint_dir / "mrna_train_normalized.npy"
            
            if not train_data_path.exists():
                logger.warning(f"    ⚠ Training data not found at {train_data_path}")
                logger.warning(f"    ⚠ Falling back to normalization without CDF matching")
                final_data = normalized_data
            else:
                # Load training data
                train_data = np.load(train_data_path)
                logger.info(f"    - Loaded training data: {train_data.shape}")
                
                # Step 4c: Apply CDF matching
                final_data = self.cdf_matching(normalized_data, train_data)
                logger.info(f"    - Applied CDF matching to training distribution")
        
        logger.info(f"  Final shape: {final_data.shape}")
        logger.info(f"  Final range: [{final_data.min():.4f}, {final_data.max():.4f}]")
        
        return final_data


    @torch.no_grad()
    def predict_new_data(
        self,
        new_data: pd.DataFrame,
        source_modality: str = 'mrna',
        target_modality: str = 'mirna',
        log_transform: bool = True,
        normalize_setting: str = 'independent',
        return_dataframe: bool = True
    ) -> Union[np.ndarray, pd.DataFrame]:
        """
        Predict on completely new data (convenience method).
        
        This method handles all preprocessing automatically:
        1. Fill NA with 0
        2. Align genes
        3. Log transform
        4. Normalize (based on normalize_setting)
        5. Predict
        6. Inverse transform
        
        Args:
            new_data: DataFrame with expression data (samples × genes)
            source_modality: 'mrna' or 'mirna'
            target_modality: 'mrna' or 'mirna'
            log_transform: Whether to apply log transformation
            normalize_setting: Normalization method
                - 'use_train': Use training statistics for z-score normalization
                - 'independent': Use new data's own statistics, then apply CDF matching
                - 'no_norm': Skip normalization entirely
            return_dataframe: If True, returns DataFrame; else numpy array
        
        Returns:
            Predicted expression (in original scale)
        """
        # Preprocess
        preprocessed = self.preprocess_new_data(
            new_data,
            modality=source_modality,
            log_transform=log_transform,
            normalize_setting=normalize_setting
        )
        
        # Convert to tensor
        input_tensor = torch.FloatTensor(preprocessed).to(self.device)
        
        # Predict based on direction
        if source_modality.lower() == 'mrna' and target_modality.lower() == 'mirna':
            pred_normalized = self.model.mrna_to_mirna(input_tensor)
            source_scaler = self.data_loader.mrna_scaler
            target_scaler = self.data_loader.mirna_scaler
        elif source_modality.lower() == 'mirna' and target_modality.lower() == 'mrna':
            pred_normalized = self.model.mirna_to_mrna(input_tensor)
            source_scaler = self.data_loader.mirna_scaler
            target_scaler = self.data_loader.mrna_scaler
        else:
            raise ValueError(f"Invalid modality pair: {source_modality} → {target_modality}")
        
        # Convert to numpy
        pred_normalized = pred_normalized.cpu().numpy()
        
        # Inverse transform
        pred_original = target_scaler.inverse_transform(pred_normalized)
        preprocessed_revert = source_scaler.inverse_transform(preprocessed)
        
        logger.info(f"✓ Prediction complete: {source_modality} → {target_modality}")
        
        # Return as DataFrame if requested
        if return_dataframe:
            pred_df = pd.DataFrame(
                pred_original,
                index=new_data.index,
                columns=self.data_loader.common_genes
            )
            source_df = pd.DataFrame(
                preprocessed_revert,
                index=new_data.index,
                columns=self.data_loader.common_genes
            )

            return pred_df, source_df
        else:
            return pred_original, preprocessed_revert

    @torch.no_grad()
    def predict_batch(
        self,
        mrna_data: pd.DataFrame,
        mirna_data: pd.DataFrame,
        batch_size: int = 32
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict on a batch of paired samples.
        
        Args:
            mrna_data: DataFrame with mRNA expression
            mirna_data: DataFrame with miRNA expression
            batch_size: Batch size for prediction
        
        Returns:
            Tuple of (mirna_predictions, mrna_predictions)
        """
        # Convert to numpy
        mrna_array = mrna_data.values
        mirna_array = mirna_data.values
        
        n_samples = len(mrna_array)
        
        # Normalize
        mrna_normalized = self.data_loader.mrna_scaler.transform(mrna_array)
        mirna_normalized = self.data_loader.mirna_scaler.transform(mirna_array)
        
        # Create dataset and loader
        dataset = PairedExpressionDataset(mrna_normalized, mirna_normalized)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False
        )
        
        # Predict
        all_mirna_pred = []
        all_mrna_pred = []
        
        for mrna_batch, mirna_batch in tqdm(loader, desc="Predicting"):
            mrna_batch = mrna_batch.to(self.device)
            mirna_batch = mirna_batch.to(self.device)
            
            # mRNA → miRNA
            mirna_pred = self.model.mrna_to_mirna(mrna_batch)
            all_mirna_pred.append(mirna_pred.cpu().numpy())
            
            # miRNA → mRNA
            mrna_pred = self.model.mirna_to_mrna(mirna_batch)
            all_mrna_pred.append(mrna_pred.cpu().numpy())
        
        # Concatenate
        mirna_pred_normalized = np.concatenate(all_mirna_pred, axis=0)
        mrna_pred_normalized = np.concatenate(all_mrna_pred, axis=0)
        
        # Inverse transform
        mirna_pred = self.data_loader.mirna_scaler.inverse_transform(mirna_pred_normalized)
        mrna_pred = self.data_loader.mrna_scaler.inverse_transform(mrna_pred_normalized)
        
        return mirna_pred, mrna_pred
    
    def evaluate_predictions(
        self,
        true_values: np.ndarray,
        predictions: np.ndarray,
        prefix: str = ""
    ) -> Dict[str, float]:
        """
        Compute evaluation metrics.
        
        Args:
            true_values: Ground truth expression
            predictions: Predicted expression
            prefix: Prefix for metric names
        
        Returns:
            Dictionary with metrics
        """
        metrics = {}
        
        # MSE
        mse = np.mean((true_values - predictions) ** 2)
        metrics[f'{prefix}mse'] = mse
        
        # MAE
        mae = np.mean(np.abs(true_values - predictions))
        metrics[f'{prefix}mae'] = mae
        
        # Pearson correlation (sample-wise)
        correlations = []
        for i in range(len(true_values)):
            corr, _ = pearsonr(true_values[i], predictions[i])
            if not np.isnan(corr):
                correlations.append(corr)
        
        metrics[f'{prefix}mean_correlation'] = np.mean(correlations)
        metrics[f'{prefix}median_correlation'] = np.median(correlations)
        metrics[f'{prefix}min_correlation'] = np.min(correlations)
        metrics[f'{prefix}max_correlation'] = np.max(correlations)
        
        # R-squared (global)
        ss_res = np.sum((true_values - predictions) ** 2)
        ss_tot = np.sum((true_values - np.mean(true_values)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        metrics[f'{prefix}r2'] = r2
        
        return metrics
    
    def save_predictions(
        self,
        predictions: np.ndarray,
        sample_ids: List[str],
        gene_names: List[str],
        output_path: str,
        format: str = 'csv'
    ):
        """
        Save predictions to file.
        
        Args:
            predictions: Prediction array (n_samples, n_genes)
            sample_ids: List of sample IDs
            gene_names: List of gene names
            output_path: Output file path
            format: 'csv' or 'npy'
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'csv':
            df = pd.DataFrame(
                predictions,
                index=sample_ids,
                columns=gene_names
            )
            df.to_csv(output_path)
            logger.info(f"Saved predictions to: {output_path}")
        
        elif format == 'npy':
            np.save(output_path, predictions)
            logger.info(f"Saved predictions to: {output_path}")
        
        else:
            raise ValueError(f"Unknown format: {format}")


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description='Run inference with trained model')
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='./checkpoints/best_model.pt',
        help='Path to model checkpoint'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./predictions',
        help='Directory to save predictions'
    )
    
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['train', 'val', 'test', 'all'],
        help='Which data split to predict on'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Batch size for prediction'
    )
    
    args = parser.parse_args()
    
    # Create predictor
    logger.info("="*80)
    logger.info("INFERENCE")
    logger.info("="*80)
    
    predictor = Predictor(args.checkpoint)
    
    # Get data splits
    logger.info("\n" + "="*80)
    logger.info("PREPARING DATA")
    logger.info("="*80)
    
    mrna_train, mirna_train, mrna_val, mirna_val = \
        predictor.data_loader.get_train_val_split(
            val_size=0.2,
            random_state=predictor.config['seed']
        )
    
    # Select split
    if args.split == 'train':
        mrna_data, mirna_data = mrna_train, mirna_train
    elif args.split == 'val':
        mrna_data, mirna_data = mrna_val, mirna_val
    elif args.split == 'test':
        logger.warning("Test split not available (removed from pipeline)")
        logger.info("Using validation split instead...")
        mrna_data, mirna_data = mrna_val, mirna_val
    else:  # all
        mrna_data = pd.concat([mrna_train, mrna_val])
        mirna_data = pd.concat([mirna_train, mirna_val])
    
    logger.info(f"Predicting on {args.split} split: {len(mrna_data)} samples")
    
    # Run predictions
    logger.info("\n" + "="*80)
    logger.info("GENERATING PREDICTIONS")
    logger.info("="*80)
    
    mirna_pred, mrna_pred = predictor.predict_batch(
        mrna_data,
        mirna_data,
        batch_size=args.batch_size
    )
    
    logger.info("✓ Predictions complete")
    
    # Evaluate
    logger.info("\n" + "="*80)
    logger.info("EVALUATION METRICS")
    logger.info("="*80)
    
    # mRNA → miRNA metrics
    logger.info("\nmRNA → miRNA:")
    metrics_m2mi = predictor.evaluate_predictions(
        mirna_data.values,
        mirna_pred,
        prefix='mrna2mirna_'
    )
    for key, value in metrics_m2mi.items():
        logger.info(f"  {key}: {value:.4f}")
    
    # miRNA → mRNA metrics
    logger.info("\nmiRNA → mRNA:")
    metrics_mi2m = predictor.evaluate_predictions(
        mrna_data.values,
        mrna_pred,
        prefix='mirna2mrna_'
    )
    for key, value in metrics_mi2m.items():
        logger.info(f"  {key}: {value:.4f}")
    
    # Save predictions
    logger.info("\n" + "="*80)
    logger.info("SAVING PREDICTIONS")
    logger.info("="*80)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save mRNA → miRNA predictions
    predictor.save_predictions(
        mirna_pred,
        mrna_data.index.tolist(),
        predictor.data_loader.common_genes,
        output_dir / f'{args.split}_mrna2mirna_predictions.csv',
        format='csv'
    )
    
    # Save miRNA → mRNA predictions
    predictor.save_predictions(
        mrna_pred,
        mirna_data.index.tolist(),
        predictor.data_loader.common_genes,
        output_dir / f'{args.split}_mirna2mrna_predictions.csv',
        format='csv'
    )
    
    # Save ground truth for reference
    mrna_data.to_csv(output_dir / f'{args.split}_mrna_true.csv')
    mirna_data.to_csv(output_dir / f'{args.split}_mirna_true.csv')
    
    logger.info(f"  Saved to: {output_dir}")
    
    # Save metrics
    all_metrics = {**metrics_m2mi, **metrics_mi2m}
    metrics_path = output_dir / f'{args.split}_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    logger.info(f"  Metrics saved to: {metrics_path}")
    
    logger.info("\n" + "="*80)
    logger.info("INFERENCE COMPLETE")
    logger.info("="*80)


if __name__ == "__main__":
    main()