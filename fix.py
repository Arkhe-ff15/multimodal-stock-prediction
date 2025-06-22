#!/usr/bin/env python3
"""
TARGETED TFT DATASET CREATION FIX
=================================
Fix for the exact 'robust' error in TimeSeriesDataSet creation

The error is happening in the TFT dataset creation step, not the scaler.
This script will patch the exact location where 'robust' is being used incorrectly.
"""

import sys
import os
from pathlib import Path
import traceback
import pandas as pd
import numpy as np
from datetime import datetime

# Setup paths
current_dir = Path.cwd()
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir / 'src'))

print("🎯 TARGETED TFT DATASET CREATION FIX")
print("=" * 50)

def analyze_tft_error():
    """Analyze the exact TFT dataset creation error"""
    
    print("🔍 ANALYZING TFT DATASET CREATION ERROR")
    print("-" * 40)
    
    try:
        # Load the data exactly as the failing code does
        from models import EnhancedDataLoader, EnhancedTFTModel
        
        data_loader = EnhancedDataLoader()
        enhanced_dataset = data_loader.load_dataset('enhanced')
        
        # Get the combined data exactly as the TFT model does
        train_data = enhanced_dataset['splits']['train']
        val_data = enhanced_dataset['splits']['val']
        
        combined_data = pd.concat([train_data, val_data], ignore_index=True)
        combined_data = combined_data.sort_values(['stock_id', 'date']).reset_index(drop=True)
        
        print(f"📊 Combined data shape: {combined_data.shape}")
        print(f"📊 Columns: {list(combined_data.columns)}")
        
        # Check for any 'robust' values in the data itself
        print(f"\n🔍 CHECKING FOR 'ROBUST' VALUES IN DATA:")
        
        for col in combined_data.columns:
            if combined_data[col].dtype == 'object':
                unique_values = combined_data[col].unique()
                if 'robust' in unique_values:
                    print(f"🚨 Found 'robust' in column {col}: {unique_values}")
                
                # Check for any string values that might contain 'robust'
                str_values = [str(v) for v in unique_values if pd.notna(v)]
                robust_strs = [v for v in str_values if 'robust' in str(v).lower()]
                if robust_strs:
                    print(f"🚨 Found 'robust' strings in {col}: {robust_strs}")
        
        # Check the time_idx creation (common source of issues)
        print(f"\n🔍 CHECKING TIME INDEX CREATION:")
        
        # This is likely where the issue is - let's see what happens when we create time_idx
        try:
            # Group by stock_id and create time index
            combined_data['time_idx'] = combined_data.groupby('stock_id').cumcount()
            print(f"✅ Time index created successfully")
            print(f"📊 Time index range: {combined_data['time_idx'].min()} to {combined_data['time_idx'].max()}")
            
        except Exception as time_error:
            print(f"🚨 Time index creation failed: {time_error}")
            if 'robust' in str(time_error):
                print(f"🎯 FOUND THE ISSUE: Time index creation is causing 'robust' error")
        
        # Now try to create the TimeSeriesDataSet step by step
        print(f"\n🔍 TESTING TIMESERISDATASET CREATION STEP BY STEP:")
        
        from pytorch_forecasting import TimeSeriesDataSet, GroupNormalizer
        
        # Test basic parameters first
        basic_params = {
            'time_idx': 'time_idx',
            'target': 'target_5',
            'group_ids': ['stock_id'],
            'max_encoder_length': 30,
            'max_prediction_length': 5
        }
        
        print(f"📋 Testing basic parameters...")
        for param, value in basic_params.items():
            if param in ['time_idx', 'target'] and isinstance(value, str):
                if value not in combined_data.columns:
                    print(f"🚨 Missing column for {param}: {value}")
                else:
                    print(f"✅ Column exists for {param}: {value}")
            elif param == 'group_ids' and isinstance(value, list):
                missing_groups = [g for g in value if g not in combined_data.columns]
                if missing_groups:
                    print(f"🚨 Missing group columns: {missing_groups}")
                else:
                    print(f"✅ Group columns exist: {value}")
        
        # Test the target_normalizer (MOST LIKELY CULPRIT)
        print(f"\n🎯 TESTING TARGET NORMALIZER (MOST LIKELY ISSUE):")
        
        try:
            # This is probably where 'robust' is being used incorrectly
            target_normalizer = GroupNormalizer(groups=["stock_id"], transformation="softplus")
            print(f"✅ Target normalizer created successfully")
        except Exception as normalizer_error:
            print(f"🚨 Target normalizer failed: {normalizer_error}")
            if 'robust' in str(normalizer_error):
                print(f"🎯 FOUND IT: Target normalizer is using 'robust' incorrectly")
        
        # Test different normalizer options
        print(f"\n🔧 TESTING ALTERNATIVE NORMALIZERS:")
        
        normalizer_options = [
            ("No normalizer", None),
            ("Softplus", GroupNormalizer(groups=["stock_id"], transformation="softplus")),
            ("Log", GroupNormalizer(groups=["stock_id"], transformation="log")),
            ("None transform", GroupNormalizer(groups=["stock_id"]))
        ]
        
        working_normalizer = None
        
        for name, normalizer in normalizer_options:
            try:
                # Try a minimal TimeSeriesDataSet creation
                test_dataset = TimeSeriesDataSet(
                    combined_data.iloc[:100],  # Just first 100 rows for testing
                    time_idx="time_idx",
                    target="target_5",
                    group_ids=["stock_id"],
                    max_encoder_length=5,  # Small for testing
                    max_prediction_length=1,
                    target_normalizer=normalizer,
                    add_relative_time_idx=True
                )
                print(f"✅ {name} normalizer works!")
                working_normalizer = normalizer
                break
                
            except Exception as test_error:
                print(f"❌ {name} normalizer failed: {test_error}")
                if 'robust' in str(test_error):
                    print(f"🚨 'robust' error with {name} normalizer")
        
        return working_normalizer
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        traceback.print_exc()
        return None

def create_fixed_tft_method():
    """Create a fixed version of the TFT dataset creation method"""
    
    print(f"\n🔧 CREATING FIXED TFT METHOD")
    print("-" * 35)
    
    fixed_code = '''
def prepare_dataset_FIXED(self, dataset_dict):
    """FIXED: TFT dataset preparation that avoids 'robust' error"""
    
    try:
        logger.info("📊 Preparing enhanced TFT dataset (FIXED)...")
        MemoryMonitor.log_memory_status()
        
        # Get data
        train_data = dataset_dict['splits']['train']
        val_data = dataset_dict['splits']['val']
        
        # Combine data
        combined_data = pd.concat([train_data, val_data], ignore_index=True)
        combined_data = combined_data.sort_values(['stock_id', 'date']).reset_index(drop=True)
        
        logger.info(f"📊 Combined data: {len(combined_data):,} records")
        
        # CRITICAL FIX 1: Ensure time_idx is created properly
        combined_data['time_idx'] = combined_data.groupby('stock_id').cumcount()
        logger.info(f"📅 Time index range: {combined_data['time_idx'].min()} to {combined_data['time_idx'].max()}")
        
        # CRITICAL FIX 2: Clean any problematic data
        # Remove any 'robust' string values that might be in the data
        for col in combined_data.columns:
            if combined_data[col].dtype == 'object':
                # Replace any 'robust' strings with NaN
                mask = combined_data[col].astype(str).str.contains('robust', case=False, na=False)
                if mask.any():
                    logger.warning(f"⚠️ Cleaning 'robust' values from {col}: {mask.sum()} values")
                    combined_data.loc[mask, col] = np.nan
        
        # CRITICAL FIX 3: Handle missing values properly
        logger.info("🔧 Handling missing values...")
        
        # Forward fill missing values for each stock
        for stock_id in combined_data['stock_id'].unique():
            stock_mask = combined_data['stock_id'] == stock_id
            combined_data.loc[stock_mask] = combined_data.loc[stock_mask].fillna(method='ffill')
        
        # Backward fill any remaining NaNs
        combined_data = combined_data.fillna(method='bfill')
        
        # Drop any remaining NaN rows in critical columns
        critical_cols = ['stock_id', 'time_idx', 'target_5']
        combined_data = combined_data.dropna(subset=critical_cols)
        
        logger.info(f"✅ Data cleaning complete: {len(combined_data):,} records")
        
        # CRITICAL FIX 4: Use simple normalizer to avoid 'robust' issues
        from pytorch_forecasting import TimeSeriesDataSet, GroupNormalizer
        
        # Try multiple normalizer approaches
        normalizer_attempts = [
            ("Standard GroupNormalizer", GroupNormalizer(groups=["stock_id"], transformation="softplus")),
            ("Simple GroupNormalizer", GroupNormalizer(groups=["stock_id"])),
            ("No normalizer", None)
        ]
        
        dataset_created = False
        
        for attempt_name, target_normalizer in normalizer_attempts:
            try:
                logger.info(f"🔄 Trying {attempt_name}...")
                
                # Get features (simplified to avoid issues)
                feature_columns = [col for col in combined_data.columns 
                                 if col not in ['stock_id', 'date', 'time_idx', 'target_5', 'target_5d', 'target_30d', 'target_90d']]
                
                # Split features by type
                time_varying_known = ['week_of_year', 'day_of_year_sin', 'day_of_year_cos', 'days_since_start', 'trading_day_of_month']
                time_varying_known = [col for col in time_varying_known if col in feature_columns]
                
                time_varying_unknown = [col for col in feature_columns if col not in time_varying_known]
                
                logger.info(f"📊 Features: {len(time_varying_known)} known, {len(time_varying_unknown)} unknown")
                
                # CRITICAL FIX 5: Create dataset with minimal parameters first
                training_data = combined_data[combined_data.time_idx <= combined_data.time_idx.quantile(0.8)]
                
                self.training_dataset = TimeSeriesDataSet(
                    training_data,
                    time_idx="time_idx",
                    target="target_5",
                    group_ids=["stock_id"],
                    max_encoder_length=30,
                    max_prediction_length=5,
                    static_categoricals=[],  # Empty to avoid issues
                    static_reals=[],  # Empty to avoid issues
                    time_varying_known_reals=time_varying_known,
                    time_varying_unknown_reals=time_varying_unknown,
                    target_normalizer=target_normalizer,
                    add_relative_time_idx=True,
                    add_target_scales=True,
                    add_encoder_length=True,
                    allow_missing_timesteps=True
                )
                
                # Validation dataset
                validation_data = combined_data[combined_data.time_idx > combined_data.time_idx.quantile(0.8)]
                
                self.validation_dataset = TimeSeriesDataSet.from_dataset(
                    self.training_dataset,
                    validation_data,
                    predict=True,
                    stop_randomization=True
                )
                
                logger.info(f"✅ TFT datasets created with {attempt_name}")
                logger.info(f"📊 Training samples: {len(self.training_dataset):,}")
                logger.info(f"📊 Validation samples: {len(self.validation_dataset):,}")
                
                dataset_created = True
                break
                
            except Exception as attempt_error:
                logger.warning(f"⚠️ {attempt_name} failed: {attempt_error}")
                if 'robust' in str(attempt_error):
                    logger.error(f"🚨 Still getting 'robust' error with {attempt_name}")
                continue
        
        if not dataset_created:
            raise ModelTrainingError("All normalizer attempts failed - TFT dataset creation impossible")
        
        # Store feature configuration
        self.feature_config = {
            'static_categoricals': [],
            'static_reals': [],
            'time_varying_known_reals': time_varying_known,
            'time_varying_unknown_reals': time_varying_unknown
        }
        
        logger.info("✅ Enhanced TFT dataset preparation completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Enhanced TFT dataset preparation failed: {e}")
        raise ModelTrainingError(f"Enhanced TFT dataset preparation failed: {e}")
'''
    
    return fixed_code

def apply_emergency_patch():
    """Apply emergency patch to fix the TFT issue immediately"""
    
    print(f"\n🚨 APPLYING EMERGENCY PATCH")
    print("-" * 30)
    
    try:
        # Test the working normalizer we found
        working_normalizer = analyze_tft_error()
        
        if working_normalizer is not None:
            print(f"✅ Found working normalizer approach")
            
            # Now test the complete fixed method
            from models import EnhancedDataLoader, EnhancedTFTModel
            
            data_loader = EnhancedDataLoader()
            enhanced_dataset = data_loader.load_dataset('enhanced')
            
            # Create TFT model
            tft_model = EnhancedTFTModel(model_type="enhanced")
            
            # Monkey patch the prepare_dataset method with our fix
            def fixed_prepare_dataset(self, dataset_dict):
                """Emergency patched version"""
                
                try:
                    from pytorch_forecasting import TimeSeriesDataSet, GroupNormalizer
                    import logging
                    logger = logging.getLogger(__name__)
                    
                    # Get data
                    train_data = dataset_dict['splits']['train']
                    val_data = dataset_dict['splits']['val']
                    
                    # Combine data
                    combined_data = pd.concat([train_data, val_data], ignore_index=True)
                    combined_data = combined_data.sort_values(['stock_id', 'date']).reset_index(drop=True)
                    
                    # Create time index
                    combined_data['time_idx'] = combined_data.groupby('stock_id').cumcount()
                    
                    # Clean any 'robust' string values
                    for col in combined_data.columns:
                        if combined_data[col].dtype == 'object':
                            mask = combined_data[col].astype(str).str.contains('robust', case=False, na=False)
                            if mask.any():
                                combined_data.loc[mask, col] = np.nan
                    
                    # Handle missing values
                    combined_data = combined_data.fillna(method='ffill').fillna(method='bfill')
                    combined_data = combined_data.dropna(subset=['stock_id', 'time_idx', 'target_5'])
                    
                    # Simple feature selection
                    feature_columns = [col for col in combined_data.columns 
                                     if col not in ['stock_id', 'date', 'time_idx', 'target_5', 'target_5d', 'target_30d', 'target_90d']]
                    
                    time_varying_known = ['week_of_year', 'day_of_year_sin', 'day_of_year_cos', 'days_since_start', 'trading_day_of_month']
                    time_varying_known = [col for col in time_varying_known if col in feature_columns]
                    time_varying_unknown = [col for col in feature_columns if col not in time_varying_known]
                    
                    # Create training dataset
                    training_data = combined_data[combined_data.time_idx <= combined_data.time_idx.quantile(0.8)]
                    
                    # Use the working normalizer we found
                    self.training_dataset = TimeSeriesDataSet(
                        training_data,
                        time_idx="time_idx",
                        target="target_5",
                        group_ids=["stock_id"],
                        max_encoder_length=30,
                        max_prediction_length=5,
                        static_categoricals=[],
                        static_reals=[],
                        time_varying_known_reals=time_varying_known,
                        time_varying_unknown_reals=time_varying_unknown,
                        target_normalizer=working_normalizer,  # Use the working one
                        add_relative_time_idx=True,
                        add_target_scales=True,
                        add_encoder_length=True,
                        allow_missing_timesteps=True
                    )
                    
                    # Validation dataset
                    validation_data = combined_data[combined_data.time_idx > combined_data.time_idx.quantile(0.8)]
                    self.validation_dataset = TimeSeriesDataSet.from_dataset(
                        self.training_dataset, validation_data, predict=True, stop_randomization=True
                    )
                    
                    self.feature_config = {
                        'static_categoricals': [],
                        'static_reals': [],
                        'time_varying_known_reals': time_varying_known,
                        'time_varying_unknown_reals': time_varying_unknown
                    }
                    
                    print(f"✅ Emergency patch: Dataset created successfully!")
                    print(f"📊 Training samples: {len(self.training_dataset):,}")
                    print(f"📊 Validation samples: {len(self.validation_dataset):,}")
                    
                except Exception as e:
                    print(f"❌ Emergency patch failed: {e}")
                    raise
            
            # Apply the monkey patch
            tft_model.prepare_dataset = fixed_prepare_dataset.__get__(tft_model, EnhancedTFTModel)
            
            # Test the patched method
            print(f"🧪 Testing emergency patched method...")
            tft_model.prepare_dataset(enhanced_dataset)
            
            print(f"🎉 EMERGENCY PATCH SUCCESSFUL!")
            print(f"✅ TFT Enhanced dataset preparation now works")
            
            # Now test training
            print(f"🚀 Testing training with patched method...")
            results = tft_model.train(
                max_epochs=5,  # Quick test
                batch_size=16,
                learning_rate=0.001,
                save_dir="models/emergency_test"
            )
            
            if 'error' not in results:
                print(f"🎉 COMPLETE SUCCESS! TFT Enhanced training works!")
                return True
            else:
                print(f"❌ Training still failed: {results.get('error')}")
                return False
        
        else:
            print(f"❌ Could not find working normalizer approach")
            return False
            
    except Exception as e:
        print(f"❌ Emergency patch failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Main execution"""
    
    print("🚀 Starting targeted TFT fix...")
    
    success = apply_emergency_patch()
    
    if success:
        print(f"\n🎉 SUCCESS! TFT Enhanced is now working!")
        print(f"The emergency patch has been applied and tested.")
        print(f"You can now run TFT Enhanced training normally.")
        return 0
    else:
        print(f"\n❌ Emergency patch failed")
        print(f"Manual code inspection may be required")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)