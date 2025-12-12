"""
Configuration file for Bidirectional mRNATPM-miRNAiso Transformer

Centralized configuration for all hyperparameters and settings.
"""

from dataclasses import dataclass, asdict
from typing import Optional
import json


@dataclass
class DataConfig:
    """Data loading and preprocessing configuration."""
    data_root: str = "/data/daotran/Cancer_Subtyping/BiB_Submission/Data/TCGA_csv"
    pathway_file: str = "/data/daotran/Cancer_Subtyping/BiB_Submission/Data/NewKEGGgs.json"
    min_samples_per_cancer: int = 10
    val_size: float = 0.2  # Updated from 0.15
    batch_size: int = 32
    num_workers: int = 4
    log_transform: bool = True
    normalize: bool = True
    handle_missing: str = 'zero'  # Changed default from 'drop' to 'zero'
    stratify_by_cancer: bool = True


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    hidden_dim: int = 512
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    num_heads: int = 8
    ff_dim: Optional[int] = None  # Default: 4 * hidden_dim
    dropout: float = 0.1


@dataclass
class LossConfig:
    """Loss function configuration."""
    lambda_reconstruction: float = 1.0
    lambda_cycle: float = 0.1
    lambda_pathway: float = 0.01
    min_pathway_size: int = 5
    use_correlation: bool = True  # For pathway coherence loss


@dataclass
class TrainingConfig:
    """Training loop configuration."""
    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    scheduler: str = 'reduce_on_plateau'  # 'reduce_on_plateau', 'cosine', or None
    patience: int = 20  # For early stopping
    use_amp: bool = False  # Mixed precision training
    seed: int = 42


@dataclass
class LoggingConfig:
    """Logging and checkpointing configuration."""
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    log_interval: int = 10  # Log every N batches
    eval_interval: int = 5  # Evaluate metrics every N epochs


@dataclass
class Config:
    """Main configuration class combining all sub-configs."""
    data: DataConfig
    model: ModelConfig
    loss: LossConfig
    training: TrainingConfig
    logging: LoggingConfig
    
    @classmethod
    def default(cls):
        """Create default configuration."""
        return cls(
            data=DataConfig(),
            model=ModelConfig(),
            loss=LossConfig(),
            training=TrainingConfig(),
            logging=LoggingConfig()
        )
    
    def to_dict(self):
        """Convert config to dictionary."""
        return {
            'data': asdict(self.data),
            'model': asdict(self.model),
            'loss': asdict(self.loss),
            'training': asdict(self.training),
            'logging': asdict(self.logging)
        }
    
    def save(self, filepath: str):
        """Save configuration to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str):
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        return cls(
            data=DataConfig(**config_dict['data']),
            model=ModelConfig(**config_dict['model']),
            loss=LossConfig(**config_dict['loss']),
            training=TrainingConfig(**config_dict['training']),
            logging=LoggingConfig(**config_dict['logging'])
        )


# Predefined configurations for different scenarios

# def get_quick_test_config():
#     """Configuration for quick testing (small model, few epochs)."""
#     config = Config.default()
#     config.model.hidden_dim = 256
#     config.model.num_encoder_layers = 2
#     config.model.num_decoder_layers = 2
#     config.training.epochs = 10
#     config.training.patience = 5
#     config.data.batch_size = 64
#     return config

def get_quick_test_config():
    """Configuration for quick testing (small model, few epochs)."""
    config = Config.default()
    config.model.hidden_dim = 256
    config.model.num_encoder_layers = 1
    config.model.num_decoder_layers = 1
    config.model.num_heads = 2
    config.training.epochs = 100
    config.training.patience = 20
    config.data.batch_size = 64
    return config


def get_full_training_config():
    """Configuration for full training (production settings)."""
    config = Config.default()
    config.model.hidden_dim = 512
    config.model.num_encoder_layers = 6
    config.model.num_decoder_layers = 6
    config.training.epochs = 100
    config.training.patience = 20
    config.data.batch_size = 32
    return config


def get_large_model_config():
    """Configuration for large model (more capacity)."""
    config = Config.default()
    config.model.hidden_dim = 768
    config.model.num_encoder_layers = 8
    config.model.num_decoder_layers = 8
    config.model.num_heads = 12
    config.training.epochs = 150
    config.training.learning_rate = 5e-5
    config.data.batch_size = 16
    return config



# # Example usage
# if __name__ == "__main__":
#     # Create default config
#     config = Config.default()
    
#     # Print config
#     print("Default Configuration:")
#     print("=" * 80)
#     import json
#     print(json.dumps(config.to_dict(), indent=2))
    
#     # Save config
#     config.save("config_default.json")
#     print("\n✓ Saved default config to: config_default.json")
    
#     # Create quick test config
#     test_config = get_quick_test_config()
#     test_config.save("config_quick_test.json")
#     print("✓ Saved quick test config to: config_quick_test.json")
    
#     # Create full training config
#     full_config = get_full_training_config()
#     full_config.save("config_full_training.json")
#     print("✓ Saved full training config to: config_full_training.json")
    
#     # Load config
#     loaded_config = Config.load("config_default.json")
#     print("\n✓ Successfully loaded config from file")


