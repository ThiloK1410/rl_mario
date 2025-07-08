#!/usr/bin/env python3
"""
Quick start script for running the optimized Mario RL training pipeline.
This script provides an easy way to start training with optimal settings.
"""

import sys
import os
import argparse
import torch

def check_environment():
    """Check if the environment is properly set up."""
    print("Checking environment...")
    
    # Check Python version
    if sys.version_info < (3, 7):
        print(f"❌ Python {sys.version_info.major}.{sys.version_info.minor} detected. Python 3.7+ required.")
        return False
    else:
        print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor}")
    
    # Check PyTorch
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"  - GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("⚠️  CUDA not available - training will be slow")
    except ImportError:
        print("❌ PyTorch not installed")
        return False
    
    # Check other dependencies
    required_packages = ['numpy', 'pandas', 'gym']
    missing = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"❌ {package} not installed")
            missing.append(package)
    
    if missing:
        print(f"\nInstall missing packages with: pip install {' '.join(missing)}")
        return False
    
    # Check for TorchRL (optional but recommended)
    try:
        import torchrl
        print(f"✓ TorchRL {torchrl.__version__} (Prioritized Replay enabled)")
    except ImportError:
        print("⚠️  TorchRL not installed - using standard replay buffer")
        print("   Install with: pip install torchrl")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Run optimized Mario RL training')
    parser.add_argument('--config', type=str, default='optimized',
                       choices=['optimized', 'original'],
                       help='Configuration to use (default: optimized)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs to train (default: from config)')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--processes', type=int, default=None,
                       help='Number of collector processes (default: from config)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Training batch size (default: from config)')
    parser.add_argument('--no-cuda', action='store_true',
                       help='Disable CUDA even if available')
    parser.add_argument('--profile', action='store_true',
                       help='Enable performance profiling')
    
    args = parser.parse_args()
    
    print("="*80)
    print("Mario RL Optimized Training Pipeline")
    print("="*80)
    
    # Check environment
    if not check_environment():
        print("\n❌ Environment check failed. Please install missing dependencies.")
        sys.exit(1)
    
    print("\n✓ Environment check passed!")
    
    # Set up configuration
    if args.config == 'optimized':
        print("\nUsing optimized configuration for maximum performance")
        # Import optimized config
        import config_optimized as config
        
        # Override config values if specified
        if args.epochs is not None:
            config.NUM_EPOCHS = args.epochs
            print(f"  - Training for {args.epochs} epochs")
        if args.processes is not None:
            config.NUM_PROCESSES = args.processes
            print(f"  - Using {args.processes} collector processes")
        if args.batch_size is not None:
            config.BATCH_SIZE = args.batch_size
            print(f"  - Batch size: {args.batch_size}")
            
        # Show effective batch size
        print(f"  - Effective batch size: {config.BATCH_SIZE * config.GRADIENT_ACCUMULATION_STEPS}")
        print(f"  - Buffer size: {config.BUFFER_SIZE:,}")
        print(f"  - Model update frequency: every {config.MODEL_UPDATE_FREQUENCY} epochs")
        
    else:
        print("\nUsing original configuration")
        import config
        
        if args.epochs is not None:
            config.NUM_EPOCHS = args.epochs
    
    # Disable CUDA if requested
    if args.no_cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        print("\n⚠️  CUDA disabled by user")
    
    # Enable profiling if requested
    if args.profile:
        print("\n📊 Performance profiling enabled")
        os.environ['ENABLE_PROFILING'] = '1'
    
    # Import and run the appropriate training script
    print(f"\n🚀 Starting training with {args.config} configuration...")
    print("-"*80)
    
    try:
        if args.config == 'optimized':
            # Set checkpoint if provided
            if args.checkpoint:
                os.environ['RESUME_CHECKPOINT'] = args.checkpoint
                print(f"Resuming from checkpoint: {args.checkpoint}")
            
            from mario_rl_optimized import main as train_main
        else:
            from mario_rl import main as train_main
        
        # Run training
        train_main()
        
        print("\n✓ Training completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()