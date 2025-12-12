#!/usr/bin/env python3
"""
Simple script to launch training with different configurations.

Usage:
    python run.py                    # Default config
    python run.py --quick            # Quick test config
    python run.py --large            # Large model config
    python run.py --config my.json   # Custom config file
"""

import argparse
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import Config, get_quick_test_config, get_full_training_config, get_large_model_config
from train import main as train_main


def config_to_dict(config):
    """
    Convert Config object to dictionary for train.py
    """
    return {
        # Data
        'data_root': config.data.data_root,
        'pathway_file': config.data.pathway_file,
        'batch_size': config.data.batch_size,
        'num_workers': config.data.num_workers,
        
        # Model
        'hidden_dim': config.model.hidden_dim,
        'num_encoder_layers': config.model.num_encoder_layers,
        'num_decoder_layers': config.model.num_decoder_layers,
        'num_heads': config.model.num_heads,
        'dropout': config.model.dropout,
        
        # Loss
        'lambda_reconstruction': config.loss.lambda_reconstruction,
        'lambda_cycle': config.loss.lambda_cycle,
        'lambda_pathway': config.loss.lambda_pathway,
        
        # Training
        'epochs': config.training.epochs,
        'learning_rate': config.training.learning_rate,
        'weight_decay': config.training.weight_decay,
        'grad_clip': config.training.grad_clip,
        'scheduler': config.training.scheduler,
        'patience': config.training.patience,
        'use_amp': config.training.use_amp,
        'seed': config.training.seed,
        
        # Logging
        'log_interval': config.logging.log_interval,
        'eval_interval': config.logging.eval_interval,
    }


def parse_args():
    parser = argparse.ArgumentParser(description='Train Bidirectional mRNA-miRNA Transformer')
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config JSON file'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Use quick test configuration (small model, few epochs)'
    )
    
    parser.add_argument(
        '--large',
        action='store_true',
        help='Use large model configuration (more capacity)'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint to resume training from'
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load configuration
    if args.config:
        print(f"Loading configuration from: {args.config}")
        config = Config.load(args.config)
    elif args.quick:
        print("Using quick test configuration")
        config = get_quick_test_config()
    elif args.large:
        print("Using large model configuration")
        config = get_large_model_config()
    else:
        print("Using default configuration")
        config = get_full_training_config()
    
    # Display configuration
    print("\n" + "="*80)
    print("CONFIGURATION")
    print("="*80)
    print(f"Model: hidden_dim={config.model.hidden_dim}, "
          f"layers={config.model.num_encoder_layers}+{config.model.num_decoder_layers}")
    print(f"Training: epochs={config.training.epochs}, "
          f"lr={config.training.learning_rate}, "
          f"batch_size={config.data.batch_size}")
    print(f"Loss: λ_recon={config.loss.lambda_reconstruction}, "
          f"λ_cycle={config.loss.lambda_cycle}, "
          f"λ_pathway={config.loss.lambda_pathway}")
    print("="*80 + "\n")
    
    # Save config for reference
    config_save_path = Path(config.logging.checkpoint_dir) / "config.json"
    config_save_path.parent.mkdir(parents=True, exist_ok=True)
    config.save(str(config_save_path))
    print(f"Configuration saved to: {config_save_path}\n")
    
    # Convert config to dictionary for train.py
    config_dict = config_to_dict(config)
    
    # Start training
    try:
        train_main(config_dict)  # Pass config to train_main!
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()