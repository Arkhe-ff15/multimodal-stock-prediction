#!/usr/bin/env python3
"""
LSTM Training Diagnostic Script
==============================
This will help identify why LSTM training completes instantly
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from datetime import datetime
import sys
from pathlib import Path

# Add src directory
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def diagnostic_test():
    """Comprehensive diagnostic test for LSTM training issues"""
    
    print("🔍 LSTM TRAINING DIAGNOSTIC TEST")
    print("=" * 50)
    
    # Test 1: Data Loading
    print("\n1️⃣ TESTING DATA LOADING...")
    try:
        from models import EnhancedDataLoader
        data_loader = EnhancedDataLoader()
        baseline_dataset = data_loader.load_dataset('baseline')
        
        train_data = baseline_dataset['splits']['train']
        val_data = baseline_dataset['splits']['val']
        
        print(f"✅ Data loaded successfully")
        print(f"   📊 Train shape: {train_data.shape}")
        print(f"   📊 Val shape: {val_data.shape}")
        print(f"   📊 Features: {len(baseline_dataset['selected_features'])}")
        
        # Check data quality
        target_train = train_data['target_5'].dropna()
        target_val = val_data['target_5'].dropna()
        
        print(f"   🎯 Train targets: {len(target_train)} valid, variance: {target_train.var():.6f}")
        print(f"   🎯 Val targets: {len(target_val)} valid, variance: {target_val.var():.6f}")
        
        if target_train.var() < 0.000001:
            print("❌ PROBLEM: Training targets have near-zero variance!")
            print("   This will cause instant convergence!")
            
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return
    
    # Test 2: Feature Selection
    print("\n2️⃣ TESTING FEATURE SELECTION...")
    try:
        feature_analysis = baseline_dataset['feature_analysis']
        available_features = feature_analysis['available_features']
        
        # Build feature list like in training
        feature_cols = []
        for category in ['price_volume_features', 'technical_features', 'time_features', 'lag_features']:
            category_features = feature_analysis.get(category, [])
            for feature in category_features:
                if feature not in ['stock_id', 'symbol', 'date'] and 'target_' not in feature:
                    feature_cols.append(feature)
        
        if not feature_cols:
            exclude_patterns = ['stock_id', 'symbol', 'date', 'target_']
            feature_cols = [f for f in available_features 
                          if not any(pattern in f for pattern in exclude_patterns)]
        
        final_feature_cols = [col for col in feature_cols if col in train_data.columns]
        
        print(f"✅ Feature selection successful")
        print(f"   📊 Selected features: {len(final_feature_cols)}")
        print(f"   📝 Examples: {final_feature_cols[:5]}")
        
        if len(final_feature_cols) < 3:
            print("❌ PROBLEM: Too few features for training!")
            
    except Exception as e:
        print(f"❌ Feature selection failed: {e}")
        return
    
    # Test 3: Dataset Creation
    print("\n3️⃣ TESTING DATASET CREATION...")
    try:
        from models import EnhancedLSTMDataset
        
        train_dataset = EnhancedLSTMDataset(
            train_data, final_feature_cols, 'target_5', sequence_length=30
        )
        val_dataset = EnhancedLSTMDataset(
            val_data, final_feature_cols, 'target_5', sequence_length=30
        )
        
        print(f"✅ Datasets created successfully")
        print(f"   📊 Train sequences: {len(train_dataset)}")
        print(f"   📊 Val sequences: {len(val_dataset)}")
        
        if len(train_dataset) == 0:
            print("❌ PROBLEM: No training sequences created!")
            return
            
        # Test a sample
        sample_x, sample_y = train_dataset[0]
        print(f"   🔍 Sample input shape: {sample_x.shape}")
        print(f"   🔍 Sample target: {sample_y.item():.6f}")
        
    except Exception as e:
        print(f"❌ Dataset creation failed: {e}")
        return
    
    # Test 4: DataLoader Creation
    print("\n4️⃣ TESTING DATALOADER CREATION...")
    try:
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        print(f"✅ DataLoaders created successfully")
        print(f"   📊 Train batches: {len(train_loader)}")
        print(f"   📊 Val batches: {len(val_loader)}")
        
        # Test batch loading
        batch_x, batch_y = next(iter(train_loader))
        print(f"   🔍 Batch input shape: {batch_x.shape}")
        print(f"   🔍 Batch targets shape: {batch_y.shape}")
        print(f"   🔍 Target range: {batch_y.min().item():.4f} to {batch_y.max().item():.4f}")
        
    except Exception as e:
        print(f"❌ DataLoader creation failed: {e}")
        return
    
    # Test 5: Model Creation
    print("\n5️⃣ TESTING MODEL CREATION...")
    try:
        from models import EnhancedLSTMModel, EnhancedLSTMTrainer
        
        model = EnhancedLSTMModel(
            input_size=len(final_feature_cols),
            hidden_size=128,
            num_layers=2,
            dropout=0.2,
            use_attention=True
        )
        
        lstm_trainer = EnhancedLSTMTrainer(
            model, learning_rate=0.001, weight_decay=0.0001, model_name="LSTM_Test"
        )
        
        print(f"✅ Model created successfully")
        
        # Test forward pass
        test_output = lstm_trainer(batch_x)
        print(f"   🔍 Model output shape: {test_output.shape}")
        print(f"   🔍 Output range: {test_output.min().item():.4f} to {test_output.max().item():.4f}")
        
        # Test loss calculation
        loss = lstm_trainer.criterion(test_output, batch_y)
        print(f"   🔍 Initial loss: {loss.item():.6f}")
        
        if loss.item() < 0.0001:
            print("❌ PROBLEM: Loss is extremely small from start!")
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return
    
    # Test 6: Training Configuration
    print("\n6️⃣ TESTING TRAINING CONFIGURATION...")
    try:
        from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
        from pytorch_lightning.loggers import TensorBoardLogger
        
        # Check callbacks
        early_stop = EarlyStopping(
            monitor='val_loss', patience=20, mode='min', verbose=True
        )
        print(f"✅ Early stopping: patience={early_stop.patience}")
        
        # Check trainer config
        trainer = pl.Trainer(
            max_epochs=100,
            accelerator="auto",
            devices="auto",
            enable_progress_bar=True,
            deterministic=True,
            log_every_n_steps=50,
            check_val_every_n_epoch=1,
            callbacks=[early_stop]
        )
        
        print(f"✅ Trainer configured: max_epochs={trainer.max_epochs}")
        
    except Exception as e:
        print(f"❌ Training configuration failed: {e}")
        return
    
    # Test 7: Single Training Step
    print("\n7️⃣ TESTING SINGLE TRAINING STEP...")
    try:
        # Manual training step
        lstm_trainer.train()
        optimizer = torch.optim.AdamW(lstm_trainer.parameters(), lr=0.001)
        
        initial_loss = None
        for i, (batch_x, batch_y) in enumerate(train_loader):
            if i >= 3:  # Test first 3 batches only
                break
                
            optimizer.zero_grad()
            output = lstm_trainer(batch_x)
            loss = lstm_trainer.criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            
            if initial_loss is None:
                initial_loss = loss.item()
            
            print(f"   🔍 Batch {i}: loss={loss.item():.6f}")
        
        final_loss = loss.item()
        loss_change = abs(final_loss - initial_loss)
        
        print(f"✅ Manual training test completed")
        print(f"   📈 Loss change: {loss_change:.6f}")
        
        if loss_change < 0.000001:
            print("❌ PROBLEM: Loss is not changing during training!")
        
    except Exception as e:
        print(f"❌ Manual training test failed: {e}")
        return
    
    print("\n" + "=" * 50)
    print("🎯 DIAGNOSTIC SUMMARY:")
    print("If all tests passed, the issue might be in:")
    print("1. PyTorch Lightning trainer callbacks")
    print("2. Early stopping triggering immediately") 
    print("3. Learning rate scheduler issues")
    print("4. Validation data problems")
    print("=" * 50)

if __name__ == "__main__":
    diagnostic_test()