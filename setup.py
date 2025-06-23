#!/usr/bin/env python3
"""
Direct Fix Script for models.py Horizon Issues
=============================================
Fixes the 2 critical bugs for [5, 22, 90] horizon support
"""

import os
import shutil
from pathlib import Path
from datetime import datetime

def fix_models_py():
    """Fix the 2 critical horizon bugs in models.py"""
    
    print("🔧 FIXING MODELS.PY HORIZON ISSUES")
    print("=" * 50)
    
    # Paths
    models_path = Path("src/models.py")
    backup_path = Path(f"src/models_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py")
    
    # Check if file exists
    if not models_path.exists():
        print(f"❌ models.py not found at: {models_path}")
        return False
    
    # Create backup
    print(f"💾 Creating backup: {backup_path}")
    shutil.copy2(models_path, backup_path)
    
    # Read the file
    print(f"📖 Reading {models_path}")
    with open(models_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Track changes
    changes_made = []
    original_content = content
    
    # ===== FIX 1: LSTM validation dataset target =====
    print("🔧 Fix 1: LSTM validation dataset target consistency...")
    
    # Find and replace the hardcoded 'target_5' in val_dataset creation
    old_val_dataset = """val_dataset = EnhancedLSTMDataset(
                    dataset['splits']['val'], feature_cols, 'target_5', sequence_length=30
                )"""
    
    new_val_dataset = """val_dataset = EnhancedLSTMDataset(
                    dataset['splits']['val'], feature_cols, target_col, sequence_length=30
                )"""
    
    if old_val_dataset in content:
        content = content.replace(old_val_dataset, new_val_dataset)
        changes_made.append("✅ Fixed LSTM validation dataset target")
        print("   ✅ Fixed LSTM validation dataset target")
    else:
        # Try alternative format
        alt_old = "dataset['splits']['val'], feature_cols, 'target_5', sequence_length=30"
        alt_new = "dataset['splits']['val'], feature_cols, target_col, sequence_length=30"
        
        if alt_old in content:
            content = content.replace(alt_old, alt_new)
            changes_made.append("✅ Fixed LSTM validation dataset target (alternative format)")
            print("   ✅ Fixed LSTM validation dataset target (alternative format)")
        else:
            print("   ⚠️ Could not find exact LSTM validation target pattern")
    
    # ===== FIX 2: Validation function hardcoded target_5 =====
    print("🔧 Fix 2: Validation function dynamic target...")
    
    # Find the _validate_dataset_integrity function and fix the hardcoded target_5
    old_validation_block = """        # Check for required columns
        # AFTER:
        base_required = ['stock_id', 'symbol', 'date'] 
        target_cols = [col for col in splits['train'].columns if col.startswith('target_')]
        if not target_cols:
            raise ModelTrainingError("No target columns found")
        required_cols = base_required + [target_cols[0]]  # Use any available target
        missing_cols = [col for col in required_cols if col not in splits['train'].columns]
        if missing_cols:
            raise ModelTrainingError(f"Missing required columns: {missing_cols}")
        
        # Enhanced data quality checks
        for split_name, split_df in splits.items():
            # Check for empty splits
            if split_df.empty:
                raise ModelTrainingError(f"{split_name} split is empty")
            
            # Check target coverage
            target_coverage = split_df['target_5'].notna().mean()
            if target_coverage < 0.7:
                logger.warning(f"⚠️ Low target coverage in {split_name}: {target_coverage:.1%}")"""
    
    new_validation_block = """        # Check for required columns
        base_required = ['stock_id', 'symbol', 'date'] 
        target_cols = [col for col in splits['train'].columns if col.startswith('target_')]
        if not target_cols:
            raise ModelTrainingError("No target columns found")
        
        primary_target = target_cols[0]  # Use first available target
        required_cols = base_required + [primary_target]
        missing_cols = [col for col in required_cols if col not in splits['train'].columns]
        if missing_cols:
            raise ModelTrainingError(f"Missing required columns: {missing_cols}")
        
        # Enhanced data quality checks
        for split_name, split_df in splits.items():
            # Check for empty splits
            if split_df.empty:
                raise ModelTrainingError(f"{split_name} split is empty")
            
            # Check target coverage using dynamic target
            target_coverage = split_df[primary_target].notna().mean()
            if target_coverage < 0.7:
                logger.warning(f"⚠️ Low target coverage in {split_name} for {primary_target}: {target_coverage:.1%}")"""
    
    if old_validation_block in content:
        content = content.replace(old_validation_block, new_validation_block)
        changes_made.append("✅ Fixed validation function dynamic target")
        print("   ✅ Fixed validation function dynamic target")
    else:
        # Try to find just the problematic line
        old_target_coverage = "target_coverage = split_df['target_5'].notna().mean()"
        new_target_coverage = """# Use first available target column
            available_targets = [col for col in split_df.columns if col.startswith('target_')]
            primary_target = available_targets[0] if available_targets else 'target_5'
            target_coverage = split_df[primary_target].notna().mean()"""
        
        if old_target_coverage in content:
            content = content.replace(old_target_coverage, new_target_coverage)
            changes_made.append("✅ Fixed target coverage line")
            print("   ✅ Fixed target coverage line")
        else:
            print("   ⚠️ Could not find exact validation target pattern")
    
    # ===== FIX 3: Add logging for target selection =====
    print("🔧 Fix 3: Add target selection logging...")
    
    # Find the logging section and add target info
    old_logging = """                logger.info(f"   📊 Training sequences: {len(train_dataset):,}")
                logger.info(f"   📊 Validation sequences: {len(val_dataset):,}")"""
    
    new_logging = """                logger.info(f"   📊 Training sequences: {len(train_dataset):,}")
                logger.info(f"   📊 Validation sequences: {len(val_dataset):,}")
                logger.info(f"   🎯 Using target column: {target_col}")"""
    
    if old_logging in content:
        content = content.replace(old_logging, new_logging)
        changes_made.append("✅ Added target selection logging")
        print("   ✅ Added target selection logging")
    
    # ===== Write the fixed file =====
    if changes_made:
        print(f"\n💾 Writing fixed file...")
        with open(models_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ FIXES APPLIED SUCCESSFULLY!")
        print(f"📋 Changes made:")
        for change in changes_made:
            print(f"   {change}")
        print(f"💾 Backup saved as: {backup_path}")
        
        return True
    else:
        print(f"⚠️ No changes were made - patterns not found")
        print(f"💡 You may need to apply fixes manually")
        return False

def fix_config_yaml():
    """Fix config.yaml horizons"""
    print(f"\n🔧 FIXING CONFIG.YAML HORIZONS")
    print("-" * 30)
    
    config_path = Path("config.yaml")
    if not config_path.exists():
        print(f"❌ config.yaml not found")
        return False
    
    # Read config
    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Create backup
    backup_path = Path(f"config_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml")
    shutil.copy2(config_path, backup_path)
    
    changes = []
    
    # Fix target_horizons
    if "target_horizons: [1, 5, 10, 22, 44]" in content:
        content = content.replace(
            "target_horizons: [1, 5, 10, 22, 44]",
            "target_horizons: [5, 22, 90]"
        )
        changes.append("✅ Fixed target_horizons to [5, 22, 90]")
    
    # Fix primary_horizon
    if "primary_horizon: 22" in content:
        content = content.replace(
            "primary_horizon: 22",
            "primary_horizon: 5"
        )
        changes.append("✅ Fixed primary_horizon to 5")
    
    if changes:
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ Config fixes applied:")
        for change in changes:
            print(f"   {change}")
        print(f"💾 Backup: {backup_path}")
        return True
    else:
        print(f"⚠️ No config changes needed")
        return False

def verify_fixes():
    """Verify the fixes were applied correctly"""
    print(f"\n🔍 VERIFYING FIXES")
    print("-" * 20)
    
    models_path = Path("src/models.py")
    if not models_path.exists():
        print(f"❌ models.py not found")
        return False
    
    with open(models_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    checks = [
        ("target_col in LSTM validation", "feature_cols, target_col, sequence_length=30" in content),
        ("Dynamic target in validation", "primary_target = target_cols[0]" in content or "primary_target =" in content),
        ("Target logging added", "Using target column:" in content),
    ]
    
    all_good = True
    for check_name, check_result in checks:
        status = "✅" if check_result else "❌"
        print(f"   {status} {check_name}")
        if not check_result:
            all_good = False
    
    return all_good

def main():
    """Main execution"""
    print("🔧 AUTOMATIC HORIZON FIXES FOR [5, 22, 90]")
    print("=" * 60)
    
    # Fix models.py
    models_fixed = fix_models_py()
    
    # Fix config.yaml
    config_fixed = fix_config_yaml()
    
    # Verify fixes
    verification_passed = verify_fixes()
    
    # Summary
    print(f"\n🎯 SUMMARY")
    print("=" * 20)
    print(f"Models.py: {'✅ Fixed' if models_fixed else '❌ Issues'}")
    print(f"Config.yaml: {'✅ Fixed' if config_fixed else '❌ Issues'}")
    print(f"Verification: {'✅ Passed' if verification_passed else '❌ Failed'}")
    
    if models_fixed and verification_passed:
        print(f"\n🎉 ALL HORIZON FIXES APPLIED SUCCESSFULLY!")
        print(f"✅ Your code now supports [5, 22, 90] horizons")
        print(f"🚀 Ready to run: python src/models.py")
    else:
        print(f"\n⚠️ SOME FIXES MAY NEED MANUAL INTERVENTION")
        print(f"💡 Check the patterns that couldn't be automatically fixed")
    
    return models_fixed and verification_passed

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)