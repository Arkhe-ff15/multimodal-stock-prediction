#!/usr/bin/env python3
"""
Pre-Training Data Validation Script
===================================
Run this before training to catch potential issues early
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

def validate_data_preparation():
    """Validate that data preparation completed successfully"""
    
    print("🔍 PRE-TRAINING DATA VALIDATION")
    print("=" * 50)
    
    issues = []
    warnings = []
    
    # Check required directories
    required_dirs = [
        Path("data/model_ready"),
        Path("data/scalers"), 
        Path("results/data_prep")
    ]
    
    for dir_path in required_dirs:
        if not dir_path.exists():
            issues.append(f"Missing directory: {dir_path}")
        else:
            print(f"✅ Directory exists: {dir_path}")
    
    # Check required files
    required_files = [
        "data/model_ready/baseline_train.csv",
        "data/model_ready/baseline_val.csv",
        "data/model_ready/baseline_test.csv",
        "data/model_ready/enhanced_train.csv", 
        "data/model_ready/enhanced_val.csv",
        "data/model_ready/enhanced_test.csv"
    ]
    
    for file_path in required_files:
        if not Path(file_path).exists():
            issues.append(f"Missing file: {file_path}")
        else:
            # Validate file content
            try:
                df = pd.read_csv(file_path)
                print(f"✅ File valid: {file_path} ({df.shape})")
                
                # Check essential columns
                essential_cols = ['stock_id', 'symbol', 'date', 'target_5']
                missing_cols = [col for col in essential_cols if col not in df.columns]
                if missing_cols:
                    issues.append(f"Missing columns in {file_path}: {missing_cols}")
                
                # Check data quality
                if df.empty:
                    issues.append(f"Empty file: {file_path}")
                elif len(df) < 100:
                    warnings.append(f"Very small dataset: {file_path} ({len(df)} rows)")
                
                # Check target coverage
                target_coverage = df['target_5'].notna().mean()
                if target_coverage < 0.5:
                    warnings.append(f"Low target coverage in {file_path}: {target_coverage:.1%}")
                
            except Exception as e:
                issues.append(f"Cannot read {file_path}: {e}")
    
    # Check feature metadata
    for dataset_type in ['baseline', 'enhanced']:
        features_file = f"results/data_prep/{dataset_type}_selected_features.json"
        if Path(features_file).exists():
            try:
                with open(features_file, 'r') as f:
                    features = json.load(f)
                print(f"✅ Features metadata: {dataset_type} ({len(features)} features)")
                
                if len(features) < 5:
                    warnings.append(f"Very few features in {dataset_type}: {len(features)}")
                
            except Exception as e:
                issues.append(f"Cannot read features metadata {features_file}: {e}")
        else:
            issues.append(f"Missing features metadata: {features_file}")
    
    # Check for temporal decay features (novel methodology)
    enhanced_features_file = "results/data_prep/enhanced_selected_features.json"
    if Path(enhanced_features_file).exists():
        try:
            with open(enhanced_features_file, 'r') as f:
                enhanced_features = json.load(f)
            
            decay_features = [f for f in enhanced_features if 'decay' in f.lower() and 'sentiment' in f.lower()]
            if decay_features:
                print(f"🔬 Temporal decay methodology detected: {len(decay_features)} features")
            else:
                warnings.append("No temporal decay features found - novel methodology may not be preserved")
                
        except:
            pass
    
    # Memory check
    import psutil
    memory = psutil.virtual_memory()
    if memory.available < 2 * 1024**3:  # Less than 2GB available
        warnings.append(f"Low available memory: {memory.available / 1024**3:.1f}GB")
    
    # Summary
    print(f"\n📊 VALIDATION SUMMARY:")
    print(f"   ✅ Checks passed: {len(required_files) + len(required_dirs) - len(issues)}")
    print(f"   ❌ Issues found: {len(issues)}")
    print(f"   ⚠️ Warnings: {len(warnings)}")
    
    if issues:
        print(f"\n❌ CRITICAL ISSUES:")
        for issue in issues:
            print(f"   • {issue}")
        print(f"\n🔧 FIX: Run data preparation first:")
        print(f"   python src/data_prep.py --regenerate-all")
        return False
    
    if warnings:
        print(f"\n⚠️ WARNINGS:")
        for warning in warnings:
            print(f"   • {warning}")
        print(f"\n📝 These warnings may affect training but are not critical")
    
    if not issues:
        print(f"\n🎉 DATA VALIDATION PASSED!")
        print(f"✅ Ready for model training")
        return True
    
    return False

if __name__ == "__main__":
    validate_data_preparation()