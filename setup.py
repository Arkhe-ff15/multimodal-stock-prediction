#!/usr/bin/env python3
"""
AUTOMATED FIX APPLICATION SCRIPT
================================

This script automatically applies the critical fixes to your models.py file.
Run with: python apply_fixes.py
"""

import re
import shutil
from pathlib import Path
from datetime import datetime

class ModelFixApplicator:
    def __init__(self, models_file="src/models.py"):
        self.models_file = Path(models_file)
        self.backup_file = self.models_file.with_suffix('.backup.py')
        self.fixes_applied = []
        
    def apply_all_fixes(self):
        """Apply all critical fixes to the models file"""
        print("🔧 AUTOMATED MODEL FIX APPLICATION")
        print("=" * 50)
        
        # Create backup
        if not self.create_backup():
            return False
        
        # Read current file
        with open(self.models_file, 'r') as f:
            content = f.read()
        
        # Apply fixes
        original_content = content
        content = self.fix_lstm_training_step(content)
        content = self.add_scaler_application(content)
        content = self.add_memory_monitoring(content)
        content = self.fix_feature_analysis(content)
        content = self.fix_tft_validation_split(content)
        content = self.add_dataset_validation(content)
        content = self.enhance_config_class(content)
        
        # Write fixed content
        if content != original_content:
            with open(self.models_file, 'w') as f:
                f.write(content)
            print(f"\n✅ Successfully applied {len(self.fixes_applied)} fixes!")
            print(f"📋 Fixes applied: {', '.join(self.fixes_applied)}")
            print(f"💾 Backup saved to: {self.backup_file}")
        else:
            print("⚠️ No changes needed - file appears to be already fixed")
        
        return True
    
    def create_backup(self):
        """Create backup of original file"""
        try:
            shutil.copy2(self.models_file, self.backup_file)
            print(f"✅ Backup created: {self.backup_file}")
            return True
        except Exception as e:
            print(f"❌ Failed to create backup: {e}")
            return False
    
    def fix_lstm_training_step(self, content):
        """Fix LSTM training_step method"""
        # Pattern to find the broken training_step
        pattern = r'def training_step\(self, batch, batch_idx\):\s*loss = self\.model\.training_step\(batch, batch_idx\)'
        
        replacement = '''def training_step(self, batch, batch_idx):
        """Fixed training step"""
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)
        
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # Enhanced memory management
        if (batch_idx % self.config.memory_cleanup_frequency == 0 or 
            MemoryMonitor.should_cleanup(self.config.memory_threshold_percent)):
            MemoryMonitor.cleanup_memory()
            if batch_idx % 100 == 0:
                MemoryMonitor.log_memory_status()
        
        return loss'''
        
        if re.search(pattern, content):
            content = re.sub(pattern, replacement, content)
            self.fixes_applied.append("LSTM_training_step")
            print("✅ Fixed LSTM training_step")
        
        return content
    
    def add_scaler_application(self, content):
        """Add scaler application method"""
        # Check if method already exists
        if '_apply_scaler_to_splits' in content:
            print("ℹ️ Scaler application method already exists")
            return content
        
        # Find CompleteDataLoader class
        class_pattern = r'class CompleteDataLoader:.*?(?=\n    def load_dataset)'
        
        scaler_method = '''
    
    def _apply_scaler_to_splits(self, splits: Dict[str, pd.DataFrame], 
                               scaler: Optional[object], 
                               feature_cols: List[str]) -> Dict[str, pd.DataFrame]:
        """Apply scaler to feature columns in all splits"""
        if scaler is None or not feature_cols:
            logger.warning("No scaler or features to scale")
            return splits
        
        for split_name, df in splits.items():
            existing_features = [f for f in feature_cols if f in df.columns]
            numeric_features = []
            
            for feature in existing_features:
                if df[feature].dtype in ['float64', 'float32', 'int64', 'int32']:
                    numeric_features.append(feature)
            
            if numeric_features:
                try:
                    scaled_values = scaler.transform(df[numeric_features])
                    df.loc[:, numeric_features] = scaled_values
                    logger.info(f"   ✅ Scaled {len(numeric_features)} features in {split_name}")
                except Exception as e:
                    logger.error(f"Failed to scale {split_name}: {e}")
                    raise
        
        return splits
'''
        
        # Insert after class definition
        insertion_point = "class CompleteDataLoader:"
        if insertion_point in content:
            # Find the __init__ method end
            init_end = content.find("logger.info(\"✅ Directory structure validation passed\")", 
                                  content.find(insertion_point))
            if init_end > 0:
                # Insert after the __init__ method
                insert_pos = content.find("\n", init_end) + 1
                content = content[:insert_pos] + scaler_method + content[insert_pos:]
                self.fixes_applied.append("scaler_application")
                print("✅ Added scaler application method")
        
        # Also fix load_dataset to use scaler
        load_dataset_pattern = r'(feature_analysis = self\._analyze_financial_features.*?\n)'
        replacement = r'\1            # Apply scaler\n            if scaler and feature_metadata:\n                splits = self._apply_scaler_to_splits(splits, scaler, feature_metadata)\n\n'
        
        if not "self._apply_scaler_to_splits" in content:
            content = re.sub(load_dataset_pattern, replacement, content)
        
        return content
    
    def add_memory_monitoring(self, content):
        """Add enhanced memory monitoring"""
        # Check if should_cleanup exists
        if 'should_cleanup' in content:
            print("ℹ️ Memory monitoring already enhanced")
            return content
        
        # Add should_cleanup method
        memory_method = '''
    
    @staticmethod
    def should_cleanup(threshold_percent: float = 80.0) -> bool:
        """Check if memory cleanup is needed"""
        stats = MemoryMonitor.get_memory_usage()
        return (stats['percent'] > threshold_percent or 
                stats.get('gpu_percent', 0) > threshold_percent)
'''
        
        # Find MemoryMonitor class
        monitor_class = "class MemoryMonitor:"
        if monitor_class in content:
            # Find cleanup_memory method
            cleanup_end = content.find("logger.warning(f\"⚠️ Memory cleanup failed: {e}\")", 
                                     content.find(monitor_class))
            if cleanup_end > 0:
                insert_pos = content.find("\n", cleanup_end) + 1
                content = content[:insert_pos] + memory_method + content[insert_pos:]
                self.fixes_applied.append("memory_monitoring")
                print("✅ Added enhanced memory monitoring")
        
        return content
    
    def fix_feature_analysis(self, content):
        """Fix feature analysis to use sets"""
        # Check if already using sets
        if "analysis = {\n            'identifier_features': set()," in content:
            print("ℹ️ Feature analysis already uses sets")
            return content
        
        # Replace initialization
        old_init = """analysis = {
            'identifier_features': [],
            'target_features': [],"""
        
        new_init = """analysis = {
            'identifier_features': set(),
            'target_features': set(),"""
        
        if old_init in content:
            content = content.replace(old_init, new_init)
            
            # Also replace list operations with set operations
            content = re.sub(r'analysis\[(.*?)\]\.append\((.*?)\)', 
                           r'analysis[\1].add(\2)', content)
            
            # Add conversion to lists at the end
            conversion_code = '''
        
        # Convert sets to sorted lists
        for key in analysis.keys():
            if key != 'available_features' and isinstance(analysis[key], set):
                analysis[key] = sorted(list(analysis[key]))
'''
            
            # Find the return statement in _analyze_financial_features
            return_pattern = r'(\n        return analysis)'
            content = re.sub(return_pattern, conversion_code + r'\1', content)
            
            self.fixes_applied.append("feature_analysis_sets")
            print("✅ Fixed feature analysis to use sets")
        
        return content
    
    def fix_tft_validation_split(self, content):
        """Fix TFT validation split calculation"""
        # Look for the problematic validation split code
        old_pattern = r'if pd\.isna\(val_start_idx\):\s*val_start_idx = int\(combined_data\[\'time_idx\'\]\.max\(\) \* 0\.8\)'
        
        new_code = '''if pd.isna(val_start_idx):
            # Fallback with proper temporal calculation
            train_end_date = train_data['date'].max()
            val_start_idx = combined_data[combined_data['date'] > train_end_date]['time_idx'].min()
            
            if pd.isna(val_start_idx):
                # Last resort: use date-based split
                total_days = (combined_data['date'].max() - combined_data['date'].min()).days
                val_start_date_calc = combined_data['date'].min() + timedelta(days=int(total_days * 0.8))
                val_start_idx = combined_data[combined_data['date'] >= val_start_date_calc]['time_idx'].min()
                
                if pd.isna(val_start_idx):
                    raise ValueError("Cannot determine validation split index")'''
        
        if re.search(old_pattern, content):
            content = re.sub(old_pattern, new_code, content)
            self.fixes_applied.append("tft_validation_split")
            print("✅ Fixed TFT validation split calculation")
        
        return content
    
    def add_dataset_validation(self, content):
        """Add enhanced dataset validation"""
        # Fix _load_data_splits method
        old_pattern = r'if splits\[split\]\.empty:\s*raise ValueError\(f"Empty {split} split"\)'
        
        new_validation = '''if splits[split].empty:
                raise ValueError(f"Empty {split} split in {file_path}")
            
            # Validate required columns
            required_cols = ['date', 'symbol']
            missing_cols = [col for col in required_cols if col not in splits[split].columns]
            if missing_cols:
                raise ValueError(f"Missing required columns in {split}: {missing_cols}")
            
            # Check for target columns
            target_cols = [col for col in splits[split].columns if col.startswith('target_')]
            if not target_cols:
                raise ValueError(f"No target columns found in {split}")'''
        
        if old_pattern in content and "# Check for target columns" not in content:
            content = re.sub(old_pattern, new_validation, content)
            self.fixes_applied.append("dataset_validation")
            print("✅ Added enhanced dataset validation")
        
        return content
    
    def enhance_config_class(self, content):
        """Enhance configuration class"""
        # Check if already enhanced
        if "min_lstm_features: int = 10" in content:
            print("ℹ️ Configuration already enhanced")
            return content
        
        # Add new configuration parameters
        config_pattern = r'(# Financial parameters\s*quantiles: List\[float\] = None)'
        
        new_params = '''# Feature requirements
    min_lstm_features: int = 10
    min_tft_baseline_features: int = 15
    min_tft_enhanced_features: int = 20
    min_temporal_decay_features: int = 5
    
    # Memory management
    memory_cleanup_frequency: int = 20
    memory_threshold_percent: float = 80.0
    batch_size_reduction_factor: float = 0.5
    
    # Financial parameters
    quantiles: List[float] = None'''
        
        if re.search(config_pattern, content):
            content = re.sub(config_pattern, new_params, content)
            self.fixes_applied.append("enhanced_config")
            print("✅ Enhanced configuration class")
        
        return content

def main():
    """Apply all fixes"""
    print("🚀 Financial Model Fix Application Tool")
    print("=" * 50)
    
    # Check if models.py exists
    if not Path("src/models.py").exists():
        print("❌ Error: src/models.py not found!")
        print("Please run this script from the project root directory.")
        return 1
    
    # Apply fixes
    applicator = ModelFixApplicator()
    if applicator.apply_all_fixes():
        print("\n✅ All fixes applied successfully!")
        print("\n📋 Next steps:")
        print("1. Review the changes in src/models.py")
        print("2. Run the test script: python test_fixes.py")
        print("3. If tests pass, proceed with training")
        
        # Run validation test
        print("\n🧪 Running validation tests...")
        import subprocess
        try:
            result = subprocess.run(["python", "test_fixes.py"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Validation tests passed!")
            else:
                print("⚠️ Some validation tests failed. Check test_results.txt")
        except:
            print("ℹ️ Run test_fixes.py manually to validate")
        
        return 0
    else:
        print("\n❌ Fix application failed!")
        return 1

if __name__ == "__main__":
    exit(main())