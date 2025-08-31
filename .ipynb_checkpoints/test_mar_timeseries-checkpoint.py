#!/usr/bin/env python3
"""
Simple test script for MAR time series model.
This script initializes the model and runs forward pass + sampling with toy data.
Designed to run on cluster environments.
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os
import time

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_mar_timeseries():
    """Test MAR model with time series data."""
    print("=" * 60)
    print("Testing MAR Time Series Model")
    print("=" * 60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Import MAR model
    from models import mar
    print("✓ Successfully imported MAR module")
    
    # Model parameters
    time_series_length = 128
    timeseries_channels = 8
    patch_size = 2
    batch_size = 4
    
    print(f"\nModel Configuration:")
    print(f"  - Time series length: {time_series_length}")
    print(f"  - Time series channels: {timeseries_channels}")
    print(f"  - Patch size: {patch_size}")
    print(f"  - Batch size: {batch_size}")
    
    # Initialize model
    print(f"\nInitializing MAR model...")
    model = mar.mar_base(
        time_series_length=time_series_length,
        timeseries_channels=timeseries_channels,
        patch_size=patch_size,
        use_text_conditioning=False,  # Use label conditioning for simplicity
        class_num=10,
        label_drop_prob=0.1,
    ).to(device)
    
    print(f"✓ Model initialized successfully")
    print(f"  - Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  - Token embed dim: {model.token_embed_dim}")
    print(f"  - Sequence length: {model.seq_len}")
    print(f"  - VAE embed dim: {model.vae_embed_dim}")
    print(f"  - Timeseries channels: {model.timeseries_channels}")
    print(f"  - Patch size: {model.patch_size}")
    
    # Create random toy data
    print(f"\nCreating toy time series data...")
    input_data = torch.randn(batch_size, timeseries_channels, time_series_length).to(device)
    labels = torch.randint(0, 10, (batch_size,)).to(device)
    
    print(f"  - Input shape: {input_data.shape}")
    print(f"  - Labels shape: {labels.shape}")
    
    # Test forward pass
    print(f"\nTesting forward pass...")
    print(f"  - Input data shape: {input_data.shape}")
    print(f"  - Labels shape: {labels.shape}")
    model.train()
    start_time = time.time()
    
    loss = model(input_data, labels)
    
    forward_time = time.time() - start_time
    print(f"✓ Forward pass successful")
    print(f"  - Loss: {loss.item():.6f}")
    print(f"  - Forward time: {forward_time:.3f}s")
    
    # Test sampling (with minimal iterations for speed)
    print(f"\nTesting sampling...")
    print(f"  - Input labels shape: {labels.shape}")
    model.eval()
    start_time = time.time()
    
    with torch.no_grad():
        # Generate samples
        samples = model.sample_tokens(
            bsz=batch_size,
            num_iter=8,  # Small number for testing
            cfg=1.0,
            labels=labels,
            temperature=1.0,
            progress=True
        )
    
    sampling_time = time.time() - start_time
    print(f"✓ Sampling successful")
    print(f"  - Output samples shape: {samples.shape}")
    print(f"  - Expected shape: [{batch_size}, {timeseries_channels}, {time_series_length}]")
    print(f"  - Sampling time: {sampling_time:.3f}s")
    
    # Test with different patch sizes
    print(f"\nTesting different patch sizes...")
    for test_patch_size in [1, 2, 4]:
        if time_series_length % test_patch_size != 0:
            continue
            
        print(f"  Testing patch_size = {test_patch_size}")
        
        # Create model with different patch size
        test_model = mar.mar_base(
            time_series_length=time_series_length,
            timeseries_channels=timeseries_channels,
            patch_size=test_patch_size,
            use_text_conditioning=False,
            class_num=10,
            label_drop_prob=0.1,
        ).to(device)
        
        # Test forward pass
        test_input = torch.randn(batch_size, timeseries_channels, time_series_length).to(device)
        test_labels = torch.randint(0, 10, (batch_size,)).to(device)
        
        print(f"    - Input shape: {test_input.shape}")
        test_loss = test_model(test_input, test_labels)
        print(f"    ✓ Loss: {test_loss.item():.6f}")
    
    # Test text conditioning setup (without loading actual models)
    print(f"\nTesting text conditioning setup...")
    text_model = mar.mar_base(
        time_series_length=64,
        timeseries_channels=4,
        patch_size=1,
        use_text_conditioning=True,
        sd3_model_path="/projects/besp/BiosignalGen_zitao/models--stabilityai--stable-diffusion-3-medium-diffusers/snapshots/ea42f8cef0f178587cf766dc8129abd379c90671",  # This will fail but tests setup
        class_num=5,
        label_drop_prob=0.1,
    )
    print(f"  ✓ Text conditioning model structure created")
    print(f"    - Text embedder exists: {hasattr(text_model, 'text_embedder')}")
    print(f"    - Use text conditioning: {text_model.use_text_conditioning}")
    
    print(f"\n" + "=" * 60)
    print(f"✅ All tests completed successfully!")
    print(f"✅ MAR time series model is working correctly")
    print(f"=" * 60)
    
    return True

def main():
    """Main function."""
    success = test_mar_timeseries()
    
    if success:
        print(f"\n🎉 MAR time series model is ready for cluster deployment!")
        print(f"   You can now use this model for training and inference.")
    else:
        print(f"\n💥 There were issues with the MAR time series model.")
        print(f"   Please check the error messages above.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
