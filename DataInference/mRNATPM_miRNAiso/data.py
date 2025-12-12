"""
Data loading and preprocessing for bidirectional mRNA-miRNAiso transformer.

Key points:
- Both mRNA and miRNAiso use gene-level features (same namespace)
- miRNAiso has fewer genes than mRNA (subset)
- We keep only common genes that appear in both mRNA and miRNAiso
- Samples must be aligned between mRNA and miRNAiso (paired data)

This module handles:
1. Loading mRNA and miRNAiso data from multiple TCGA datasets
2. Merging data across cancer types (keeping common features)
3. Aligning to common gene space (intersection of mRNA and miRNAiso genes)
4. Aligning samples between mRNA and miRNAiso
5. Creating PyTorch datasets and dataloaders
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TCGADataLoader:
    """
    Loads and preprocesses paired mRNA-miRNAiso data from TCGA datasets.
    
    Both mRNA and miRNAiso are at gene level, but miRNAiso may have fewer genes.
    Final output will have the same genes (features) for both modalities.
    """
    
    def __init__(
        self,
        data_root: str,
        min_samples_per_cancer: int = 10,
        normalize: bool = True,
        log_transform: bool = True,
        handle_missing: str = 'zero'  # 'drop', 'zero', or 'mean'
    ):
        """
        Args:
            data_root: Root directory containing TCGA-* dataset folders
            min_samples_per_cancer: Minimum paired samples required to include a cancer type
            normalize: Whether to apply standardization (z-score) per gene
            log_transform: Whether to apply log1p transformation
            handle_missing: How to handle missing values ('drop' genes, fill with 'zero', or 'mean')
        """
        self.data_root = Path(data_root)
        self.min_samples_per_cancer = min_samples_per_cancer
        self.normalize = normalize
        self.log_transform = log_transform
        self.handle_missing = handle_missing
        
        # Will be populated after loading
        self.mrna_data: Optional[pd.DataFrame] = None
        self.mirna_data: Optional[pd.DataFrame] = None
        self.cancer_types: List[str] = []
        self.sample_metadata: Optional[pd.DataFrame] = None
        self.common_genes: Optional[List[str]] = None
        
        # Scalers for normalization (fit on training data only)
        self.mrna_scaler: Optional[StandardScaler] = None
        self.mirna_scaler: Optional[StandardScaler] = None
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Main method to load and process all data.
        
        Returns:
            Tuple of (mRNA_dataframe, miRNAiso_dataframe) with:
            - Same samples (rows aligned)
            - Same genes/features (columns aligned)
            - Shape: (n_samples, n_common_genes)
        """
        logger.info(f"Loading data from: {self.data_root}")
        logger.info("="*80)
        
        # Step 1: Find all TCGA dataset folders
        tcga_folders = self._find_tcga_folders()
        logger.info(f"Found {len(tcga_folders)} TCGA dataset folders: {[f.name for f in tcga_folders]}")
        
        # Step 2: Load individual cancer type data
        mrna_dict, mirna_dict = self._load_cancer_type_data(tcga_folders)
        
        # Step 3: Merge across cancer types (keeping only common features within each modality)
        logger.info("\n" + "="*80)
        logger.info("MERGING CANCER TYPES")
        logger.info("="*80)
        self.mrna_data = self._merge_cancer_types(mrna_dict, data_type="mRNA")
        self.mirna_data = self._merge_cancer_types(mirna_dict, data_type="miRNAiso")
        
        logger.info(f"\nAfter merging across cancer types:")
        logger.info(f"  mRNA shape: {self.mrna_data.shape}")
        logger.info(f"  miRNAiso shape: {self.mirna_data.shape}")
        
        # Step 4: Align samples (keep only common samples between mRNA and miRNAiso)
        logger.info("\n" + "="*80)
        logger.info("ALIGNING SAMPLES")
        logger.info("="*80)
        self.mrna_data, self.mirna_data = self._align_samples(self.mrna_data, self.mirna_data)
        
        # Step 5: Align features (keep only common GENES between mRNA and miRNAiso)
        logger.info("\n" + "="*80)
        logger.info("ALIGNING GENES (FEATURES)")
        logger.info("="*80)
        self.mrna_data, self.mirna_data = self._align_genes(self.mrna_data, self.mirna_data)
        
        # Step 6: Create metadata
        self._create_sample_metadata()
        
        # Step 6.5: Remove cancer types with too few samples for stratification
        self._filter_small_cancer_types(min_samples=2)
        
        # Step 7: Handle missing values
        if self.handle_missing != 'drop':
            logger.info(f"\nHandling missing values with strategy: {self.handle_missing}")
            self.mrna_data = self._handle_missing_values(self.mrna_data, self.handle_missing)
            self.mirna_data = self._handle_missing_values(self.mirna_data, self.handle_missing)
        
        # Step 8: Log transformation
        if self.log_transform:
            logger.info("\nApplying log1p transformation...")
            self.mrna_data = self._log_transform(self.mrna_data)
            self.mirna_data = self._log_transform(self.mirna_data)
        
        # Print final summary
        logger.info("\n" + "="*80)
        logger.info("FINAL DATA SUMMARY")
        logger.info("="*80)
        self._print_data_summary()
        
        return self.mrna_data, self.mirna_data
    
    def _find_tcga_folders(self) -> List[Path]:
        """Find all TCGA-* folders in data_root."""
        if not self.data_root.exists():
            raise ValueError(f"Data root does not exist: {self.data_root}")
        
        folders = [f for f in self.data_root.iterdir() 
                  if f.is_dir() and f.name.startswith('TCGA-')]
        
        if not folders:
            raise ValueError(f"No TCGA-* folders found in {self.data_root}")
        
        return sorted(folders)
    
    def _load_cancer_type_data(
        self, 
        tcga_folders: List[Path]
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
        """
        Load mRNA and miRNAiso data from each cancer type folder.
        
        Returns:
            Tuple of (mrna_dict, mirna_dict) where keys are cancer types
        """
        logger.info("\n" + "="*80)
        logger.info("LOADING INDIVIDUAL CANCER TYPES")
        logger.info("="*80)
        
        mrna_dict = {}
        mirna_dict = {}
        
        for folder in tcga_folders:
            cancer_type = folder.name
            mrna_path = folder / "mRNATPM.csv"
            mirna_path = folder / "miRNAiso.csv"
            
            # Check if both files exist
            if not mrna_path.exists():
                logger.warning(f"⚠️  Skipping {cancer_type}: mRNATPM.csv not found")
                continue
            if not mirna_path.exists():
                logger.warning(f"⚠️  Skipping {cancer_type}: miRNAiso.csv not found")
                continue
            
            try:
                # Load data (assuming samples as rows, genes/features as columns)
                mrna_df = pd.read_csv(mrna_path, index_col=0)
                mirna_df = pd.read_csv(mirna_path, index_col=0)
                
                logger.info(f"\n{cancer_type}:")
                logger.info(f"  Loaded mRNA: {mrna_df.shape[0]} samples × {mrna_df.shape[1]} genes")
                logger.info(f"  Loaded miRNAiso: {mirna_df.shape[0]} samples × {mirna_df.shape[1]} genes")
                
                # Find common samples for this cancer type
                common_samples = mrna_df.index.intersection(mirna_df.index)
                
                if len(common_samples) < self.min_samples_per_cancer:
                    logger.warning(
                        f"  ⚠️  Skipping: only {len(common_samples)} paired samples "
                        f"(minimum: {self.min_samples_per_cancer})"
                    )
                    continue
                
                # Keep only common samples for this cancer type
                mrna_df = mrna_df.loc[common_samples]
                mirna_df = mirna_df.loc[common_samples]
                
                mrna_dict[cancer_type] = mrna_df
                mirna_dict[cancer_type] = mirna_df
                
                logger.info(f"  ✓ Kept {len(common_samples)} paired samples")
                
                self.cancer_types.append(cancer_type)
                
            except Exception as e:
                logger.error(f"  ❌ Error loading {cancer_type}: {e}")
                continue
        
        if not mrna_dict:
            raise ValueError("No valid cancer type data found!")
        
        logger.info(f"\n✓ Successfully loaded {len(mrna_dict)} cancer types")
        
        return mrna_dict, mirna_dict
    
    def _merge_cancer_types(
        self, 
        data_dict: Dict[str, pd.DataFrame],
        data_type: str
    ) -> pd.DataFrame:
        """
        Merge multiple cancer types, keeping only genes that appear in ALL cancer types.
        Also tracks which cancer type each sample belongs to.
        
        Args:
            data_dict: Dictionary of {cancer_type: dataframe}
            data_type: "mRNA" or "miRNAiso" (for logging)
        
        Returns:
            Merged dataframe with common genes across all cancer types
        """
        logger.info(f"\nMerging {data_type} data across {len(data_dict)} cancer types...")
        
        # Get common genes across ALL cancer types
        common_genes = set(data_dict[self.cancer_types[0]].columns)
        for cancer_type in self.cancer_types[1:]:
            common_genes = common_genes.intersection(data_dict[cancer_type].columns)
        
        common_genes = sorted(list(common_genes))
        
        logger.info(f"  Common genes across all cancer types: {len(common_genes)}")
        
        if len(common_genes) == 0:
            raise ValueError(f"No common genes found across cancer types for {data_type}!")
        
        # Keep only common genes and concatenate
        # IMPORTANT: Track cancer type for each sample
        dfs_to_merge = []
        sample_to_cancer = {}  # Store mapping
        
        for cancer_type in self.cancer_types:
            df = data_dict[cancer_type][common_genes]
            dfs_to_merge.append(df)
            
            # Track which cancer type each sample belongs to
            for sample_id in df.index:
                sample_to_cancer[sample_id] = cancer_type
        
        merged_df = pd.concat(dfs_to_merge, axis=0)
        
        # Store the mapping (will use in metadata creation)
        if not hasattr(self, '_sample_to_cancer_map'):
            self._sample_to_cancer_map = sample_to_cancer
        
        logger.info(f"  Final {data_type} shape: {merged_df.shape}")
        
        return merged_df
    
    def _align_samples(
        self, 
        mrna_df: pd.DataFrame, 
        mirna_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Keep only samples that appear in both mRNA and miRNAiso datasets.
        
        Args:
            mrna_df: mRNA dataframe
            mirna_df: miRNAiso dataframe
        
        Returns:
            Tuple of aligned (mRNA_df, miRNAiso_df)
        """
        logger.info(f"Before alignment:")
        logger.info(f"  mRNA samples: {len(mrna_df)}")
        logger.info(f"  miRNAiso samples: {len(mirna_df)}")
        
        # Find common samples
        common_samples = sorted(list(mrna_df.index.intersection(mirna_df.index)))
        
        if len(common_samples) == 0:
            raise ValueError("No common samples found between mRNA and miRNAiso!")
        
        # Align both dataframes to common samples
        mrna_aligned = mrna_df.loc[common_samples]
        mirna_aligned = mirna_df.loc[common_samples]
        
        logger.info(f"After alignment:")
        logger.info(f"  Common samples: {len(common_samples)}")
        logger.info(f"  Removed from mRNA: {len(mrna_df) - len(common_samples)}")
        logger.info(f"  Removed from miRNAiso: {len(mirna_df) - len(common_samples)}")
        
        return mrna_aligned, mirna_aligned
    
    def _align_genes(
        self, 
        mrna_df: pd.DataFrame, 
        mirna_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Keep only genes that appear in both mRNA and miRNAiso datasets.
        This is the key step since miRNAiso has fewer genes than mRNA.
        
        Args:
            mrna_df: mRNA dataframe
            mirna_df: miRNAiso dataframe
        
        Returns:
            Tuple of gene-aligned (mRNA_df, miRNAiso_df) with same columns
        """
        logger.info(f"Before gene alignment:")
        logger.info(f"  mRNA genes: {len(mrna_df.columns)}")
        logger.info(f"  miRNAiso genes: {len(mirna_df.columns)}")
        
        # Find common genes (intersection)
        common_genes = sorted(list(set(mrna_df.columns).intersection(set(mirna_df.columns))))
        
        if len(common_genes) == 0:
            raise ValueError("No common genes found between mRNA and miRNAiso!")
        
        self.common_genes = common_genes
        
        # Keep only common genes for both
        mrna_aligned = mrna_df[common_genes]
        mirna_aligned = mirna_df[common_genes]
        
        logger.info(f"After gene alignment:")
        logger.info(f"  Common genes: {len(common_genes)}")
        logger.info(f"  Removed from mRNA: {len(mrna_df.columns) - len(common_genes)}")
        logger.info(f"  Removed from miRNAiso: {len(mirna_df.columns) - len(common_genes)}")
        
        return mrna_aligned, mirna_aligned
    
    def _create_sample_metadata(self):
        """Create metadata dataframe with cancer type labels for each sample."""
        sample_ids = self.mrna_data.index.tolist()
        
        # Use the mapping created during merge (from subfolder names)
        cancer_labels = [self._sample_to_cancer_map.get(sid, 'Unknown') for sid in sample_ids]
        
        self.sample_metadata = pd.DataFrame({
            'sample_id': sample_ids,
            'cancer_type': cancer_labels
        })
        
        logger.info(f"\nCreated metadata for {len(sample_ids)} samples")
        logger.info(f"Cancer type distribution:")
        for cancer_type, count in self.sample_metadata['cancer_type'].value_counts().items():
            logger.info(f"  {cancer_type}: {count} samples")
    
    def _filter_small_cancer_types(self, min_samples: int = 2):
        """
        Remove cancer types with too few samples for stratified splitting.
        This is necessary after gene alignment may have reduced sample counts.
        
        Args:
            min_samples: Minimum samples required per cancer type (default: 2 for stratification)
        """
        logger.info("\n" + "="*80)
        logger.info("FILTERING SMALL CANCER TYPES")
        logger.info("="*80)
        
        cancer_counts = self.sample_metadata['cancer_type'].value_counts()
        small_cancer_types = cancer_counts[cancer_counts < min_samples].index.tolist()
        
        if small_cancer_types:
            logger.info(f"Removing cancer types with <{min_samples} samples:")
            for cancer_type in small_cancer_types:
                count = cancer_counts[cancer_type]
                logger.info(f"  ❌ {cancer_type}: {count} sample(s)")
            
            # Get samples to keep
            samples_to_keep = self.sample_metadata[
                ~self.sample_metadata['cancer_type'].isin(small_cancer_types)
            ]['sample_id'].tolist()
            
            # Filter data
            self.mrna_data = self.mrna_data.loc[samples_to_keep]
            self.mirna_data = self.mirna_data.loc[samples_to_keep]
            self.sample_metadata = self.sample_metadata[
                self.sample_metadata['sample_id'].isin(samples_to_keep)
            ]
            
            # Update cancer types list
            self.cancer_types = [ct for ct in self.cancer_types if ct not in small_cancer_types]
            
            logger.info(f"\n✓ Removed {len(small_cancer_types)} cancer type(s)")
            logger.info(f"✓ Kept {len(self.cancer_types)} cancer types with ≥{min_samples} samples")
            logger.info(f"✓ Final sample count: {len(self.mrna_data)}")
        else:
            logger.info(f"✓ All cancer types have ≥{min_samples} samples")
    
    def _handle_missing_values(self, df: pd.DataFrame, strategy: str) -> pd.DataFrame:
        """Handle missing values in the dataframe."""
        if df.isnull().sum().sum() == 0:
            logger.info(f"  No missing values found")
            return df
        
        missing_count = df.isnull().sum().sum()
        logger.info(f"  Found {missing_count} missing values")
        
        if strategy == 'zero':
            df = df.fillna(0)
            logger.info(f"  Filled with zeros")
        elif strategy == 'mean':
            df = df.fillna(df.mean())
            logger.info(f"  Filled with column means")
        # 'drop' is handled by keeping only genes without missing values
        
        return df
    
    def _log_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply log2(x + 1) transformation if data meets criteria:
        - min >= 0 (no negative values)
        - max > 100 (high dynamic range indicating raw counts/TPM)
        
        This handles skewed distributions common in RNA-seq data.
        """
        data_min = df.min().min()
        data_max = df.max().max()
        
        logger.info(f"  Data range before transform: [{data_min:.4f}, {data_max:.4f}]")
        
        # Check if transformation criteria are met
        if data_min >= 0 and data_max > 100:
            logger.info(f"  ✓ Applying log2(x + 1) transformation (criteria met: min≥0, max>100)")
            df_transformed = np.log2(df + 1)
        else:
            logger.info(f"  ⚠️  Skipping log transformation (criteria not met)")
            logger.info(f"     Current: min={data_min:.2f}, max={data_max:.2f}")
            logger.info(f"     Required: min≥0 AND max>100")
            df_transformed = df
        
        logger.info(f"  Data range after transform: [{df_transformed.min().min():.4f}, {df_transformed.max().max():.4f}]")
        
        return df_transformed
    
    def _print_data_summary(self):
        """Print comprehensive summary of loaded data."""
        logger.info(f"Total samples: {len(self.mrna_data)}")
        logger.info(f"Total genes (features): {len(self.common_genes)}")
        logger.info(f"Cancer types included: {len(self.cancer_types)}")
        logger.info(f"  {', '.join(self.cancer_types)}")
        
        logger.info(f"\nmRNA data:")
        logger.info(f"  Shape: {self.mrna_data.shape}")
        logger.info(f"  Value range: [{self.mrna_data.min().min():.4f}, {self.mrna_data.max().max():.4f}]")
        logger.info(f"  Mean: {self.mrna_data.mean().mean():.4f}")
        logger.info(f"  Std: {self.mrna_data.std().mean():.4f}")
        
        logger.info(f"\nmiRNAiso data:")
        logger.info(f"  Shape: {self.mirna_data.shape}")
        logger.info(f"  Value range: [{self.mirna_data.min().min():.4f}, {self.mirna_data.max().max():.4f}]")
        logger.info(f"  Mean: {self.mirna_data.mean().mean():.4f}")
        logger.info(f"  Std: {self.mirna_data.std().mean():.4f}")
    
    def get_train_val_split(
        self,
        val_size: float = 0.2,
        random_state: int = 42,
        stratify_by_cancer: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train/val sets.
        
        Args:
            val_size: Proportion of data for validation
            random_state: Random seed for reproducibility
            stratify_by_cancer: Whether to stratify split by cancer type
        
        Returns:
            Tuple of (mrna_train, mirna_train, mrna_val, mirna_val)
        """
        if self.mrna_data is None or self.mirna_data is None:
            raise ValueError("Data not loaded! Call load_data() first.")
        
        # Get indices
        indices = np.arange(len(self.mrna_data))
        
        # Stratification labels
        stratify = self.sample_metadata['cancer_type'].values if stratify_by_cancer else None
        
        # Split: train vs val
        train_idx, val_idx = train_test_split(
            indices,
            test_size=val_size,
            random_state=random_state,
            stratify=stratify
        )
        
        # Split data
        mrna_train = self.mrna_data.iloc[train_idx]
        mirna_train = self.mirna_data.iloc[train_idx]
        
        mrna_val = self.mrna_data.iloc[val_idx]
        mirna_val = self.mirna_data.iloc[val_idx]
        
        logger.info("\n" + "="*80)
        logger.info("DATA SPLIT")
        logger.info("="*80)
        logger.info(f"Train: {len(train_idx)} samples ({len(train_idx)/len(indices)*100:.1f}%)")
        logger.info(f"Val:   {len(val_idx)} samples ({len(val_idx)/len(indices)*100:.1f}%)")
        
        if stratify_by_cancer:
            logger.info("\nCancer type distribution:")
            for split_name, split_idx in [("Train", train_idx), ("Val", val_idx)]:
                cancer_dist = self.sample_metadata.iloc[split_idx]['cancer_type'].value_counts()
                logger.info(f"  {split_name}: {dict(cancer_dist)}")
        
        return mrna_train, mirna_train, mrna_val, mirna_val
    
    def normalize_data(
        self,
        mrna_train: pd.DataFrame,
        mrna_val: pd.DataFrame,
        mirna_train: pd.DataFrame,
        mirna_val: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Normalize data using z-score standardization.
        Fit scalers on training data only, then transform all splits.
        
        Args:
            Train/val dataframes for mRNA and miRNAiso
        
        Returns:
            Normalized train/val dataframes
        """
        logger.info("\n" + "="*80)
        logger.info("NORMALIZATION (Z-SCORE)")
        logger.info("="*80)
        
        # Fit scalers on training data
        self.mrna_scaler = StandardScaler()
        self.mirna_scaler = StandardScaler()
        
        mrna_train_norm = pd.DataFrame(
            self.mrna_scaler.fit_transform(mrna_train),
            index=mrna_train.index,
            columns=mrna_train.columns
        )
        
        mirna_train_norm = pd.DataFrame(
            self.mirna_scaler.fit_transform(mirna_train),
            index=mirna_train.index,
            columns=mirna_train.columns
        )
        
        logger.info("✓ Fitted scalers on training data")
        
        # Transform validation and test data
        mrna_val_norm = pd.DataFrame(
            self.mrna_scaler.transform(mrna_val),
            index=mrna_val.index,
            columns=mrna_val.columns
        )
        
        mirna_val_norm = pd.DataFrame(
            self.mirna_scaler.transform(mirna_val),
            index=mirna_val.index,
            columns=mirna_val.columns
        )
        
        logger.info("✓ Normalized validation data")
        logger.info(f"\nmRNA - Train mean: {mrna_train_norm.mean().mean():.4f}, std: {mrna_train_norm.std().mean():.4f}")
        logger.info(f"miRNAiso - Train mean: {mirna_train_norm.mean().mean():.4f}, std: {mirna_train_norm.std().mean():.4f}")
        
        return mrna_train_norm, mirna_train_norm, mrna_val_norm, mirna_val_norm


class PairedExpressionDataset(Dataset):
    """
    PyTorch Dataset for paired mRNA-miRNAiso expression data.
    Supports bidirectional training.
    """
    
    def __init__(
        self,
        mrna_data: Union[pd.DataFrame, np.ndarray],
        mirna_data: Union[pd.DataFrame, np.ndarray]
    ):
        """
        Args:
            mrna_data: mRNA expression data (samples × genes)
            mirna_data: miRNAiso expression data (samples × genes, same gene space)
        """
        # Convert to numpy if needed
        if isinstance(mrna_data, pd.DataFrame):
            self.mrna = torch.FloatTensor(mrna_data.values)
        else:
            self.mrna = torch.FloatTensor(mrna_data)
        
        if isinstance(mirna_data, pd.DataFrame):
            self.mirna = torch.FloatTensor(mirna_data.values)
        else:
            self.mirna = torch.FloatTensor(mirna_data)
        
        assert len(self.mrna) == len(self.mirna), "mRNA and miRNAiso must have same number of samples"
        assert self.mrna.shape[1] == self.mirna.shape[1], "mRNA and miRNAiso must have same number of features (genes)"
    
    def __len__(self) -> int:
        return len(self.mrna)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            Tuple of (mRNA_expression, miRNAiso_expression) for the same sample
        """
        return self.mrna[idx], self.mirna[idx]


def create_dataloaders(
    mrna_train: pd.DataFrame,
    mirna_train: pd.DataFrame,
    mrna_val: pd.DataFrame,
    mirna_val: pd.DataFrame,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for train/val splits.
    
    Args:
        Train/val dataframes for mRNA and miRNAiso
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_dataset = PairedExpressionDataset(mrna_train, mirna_train)
    val_dataset = PairedExpressionDataset(mrna_val, mirna_val)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # Drop last incomplete batch for stable training
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    logger.info("\n" + "="*80)
    logger.info("DATALOADERS CREATED")
    logger.info("="*80)
    logger.info(f"Train batches: {len(train_loader)} (batch_size={batch_size})")
    logger.info(f"Val batches:   {len(val_loader)}")
    
    return train_loader, val_loader


## Example usage
# if __name__ == "__main__":
#     # Example of how to use this module
#     data_root = "/data/daotran/Cancer_Subtyping/BiB_Submission/Data/TCGA_csv"
    
#     # Initialize data loader
#     loader = TCGADataLoader(
#         data_root=data_root,
#         min_samples_per_cancer=10,
#         normalize=False,  # Will normalize after split
#         log_transform=True,
#         handle_missing='zero'
#     )
    
#     # Load all data
#     mrna_data, mirna_data = loader.load_data()
    
#     # Split into train/val/test
#     mrna_train, mirna_train, mrna_val, mirna_val, mrna_test, mirna_test = \
#         loader.get_train_val_test_split(
#             val_size=0.15,
#             test_size=0.15,
#             random_state=42,
#             stratify_by_cancer=True
#         )
    
#     # Normalize (fit on train, transform all)
#     mrna_train, mirna_train, mrna_val, mirna_val, mrna_test, mirna_test = \
#         loader.normalize_data(
#             mrna_train, mrna_val, mrna_test,
#             mirna_train, mirna_val, mirna_test
#         )
    
#     # Create dataloaders
#     train_loader, val_loader, test_loader = create_dataloaders(
#         mrna_train, mirna_train,
#         mrna_val, mirna_val,
#         mrna_test, mirna_test,
#         batch_size=32,
#         num_workers=4
#     )
    
#     # Test a batch
#     print("\n" + "="*80)
#     print("TESTING DATA LOADING")
#     print("="*80)
#     for mrna_batch, mirna_batch in train_loader:
#         print(f"mRNA batch shape: {mrna_batch.shape}")
#         print(f"miRNAiso batch shape: {mirna_batch.shape}")
#         print(f"mRNA range: [{mrna_batch.min():.4f}, {mrna_batch.max():.4f}]")
#         print(f"miRNAiso range: [{mirna_batch.min():.4f}, {mirna_batch.max():.4f}]")
#         break
    
#     print("\n✓ Data loading pipeline complete!")