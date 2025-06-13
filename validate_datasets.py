#!/usr/bin/env python3
"""
Dataset Validation Script
========================

Quick script to validate dataset symbol columns and structure.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path("src").absolute()))

from data import load_dataset, DatasetType, list_all_datasets

def validate_datasets():
    """Validate all datasets"""
    print("🔍 DATASET VALIDATION REPORT")
    print("=" * 50)
    
    for dataset_type in DatasetType:
        try:
            data = load_dataset(dataset_type)
            
            print(f"\n📊 {dataset_type.value.upper()} DATASET:")
            print(f"   Shape: {data.shape}")
            
            if 'symbol' in data.columns:
                symbols = data['symbol'].unique()
                symbol_counts = data['symbol'].value_counts()
                print(f"   Symbols: {list(symbols)}")
                print(f"   Distribution: {dict(symbol_counts)}")
            else:
                print("   ❌ NO SYMBOL COLUMN!")
            
            if 'date' in data.columns:
                date_range = f"{data['date'].min().date()} to {data['date'].max().date()}"
                print(f"   Date range: {date_range}")
            elif isinstance(data.index, pd.DatetimeIndex):
                date_range = f"{data.index.min().date()} to {data.index.max().date()}"
                print(f"   Date range (index): {date_range}")
            
            # Check for key columns
            key_columns = ['target_5', 'close', 'volume']
            missing_key_cols = [col for col in key_columns if col not in data.columns]
            if missing_key_cols:
                print(f"   ⚠️ Missing key columns: {missing_key_cols}")
            else:
                print("   ✅ All key columns present")
            
        except FileNotFoundError:
            print(f"\n📊 {dataset_type.value.upper()} DATASET: Not found")
        except Exception as e:
            print(f"\n📊 {dataset_type.value.upper()} DATASET: Error - {e}")
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    validate_datasets()
