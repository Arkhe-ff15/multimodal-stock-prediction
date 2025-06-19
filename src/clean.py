#!/usr/bin/env python3
import sys
import os
from pathlib import Path

# Add src directory to Python path so we can import config_reader
script_dir = Path(__file__).parent
if 'src' in str(script_dir):
    # Running from src directory
    sys.path.insert(0, str(script_dir))
else:
    # Running from project root
    sys.path.insert(0, str(script_dir / 'src'))


"""
clean.py - Dataset Cleaning Utility
==================================

🧹 COMPREHENSIVE DATA CLEANING UTILITY:
1. Fixes column naming issues (dots, duplicates, special characters)
2. Validates data integrity and consistency
3. Removes corrupted or invalid entries
4. Optimizes dataset for model compatibility (LSTM, TFT, etc.)
5. Creates backup before cleaning
6. Provides detailed cleaning report

USAGE:
    python clean.py                    # Clean main combined dataset
    python clean.py --file custom.csv # Clean specific file
    python clean.py --validate-only   # Just validate, don't clean
    python clean.py --force           # Skip confirmation prompts
"""

import pandas as pd
import numpy as np
import argparse
import logging
from pathlib import Path
from datetime import datetime
import warnings
import shutil
import json
from typing import Dict, List, Tuple, Any, Optional
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# Standard paths
DATA_DIR = "data/processed"
BACKUP_DIR = "data/backups"
MAIN_DATASET = f"{DATA_DIR}/combined_dataset.csv"

class DatasetCleaner:
    """Comprehensive dataset cleaning utility"""
    
    def __init__(self, file_path: str = MAIN_DATASET, create_backup: bool = True):
        self.file_path = Path(file_path)
        self.create_backup = create_backup
        self.backup_path = None
        
        # Cleaning statistics
        self.cleaning_stats = {
            'original_shape': None,
            'final_shape': None,
            'columns_renamed': 0,
            'duplicates_removed': 0,
            'invalid_values_fixed': 0,
            'rows_removed': 0,
            'issues_found': [],
            'issues_fixed': []
        }
        
        # Validation rules
        self.validation_rules = {
            'required_columns': ['stock_id', 'symbol', 'date', 'open', 'high', 'low', 'close', 'volume', 'target_5'],
            'numeric_columns': ['open', 'high', 'low', 'close', 'volume'],
            'date_columns': ['date'],
            'forbidden_characters': ['.', ' ', '-', '+', '*', '/', '\\', '|', '(', ')', '[', ']'],
            'max_nan_percentage': 0.5,  # 50% max NaN values per column
            'min_rows_per_symbol': 100
        }
    
    def clean_dataset(self, validate_only: bool = False, force: bool = False) -> bool:
        """
        Main cleaning function
        
        Args:
            validate_only: Only validate, don't actually clean
            force: Skip confirmation prompts
            
        Returns:
            True if successful, False otherwise
        """
        logger.info("🧹 STARTING DATASET CLEANING")
        logger.info("=" * 60)
        logger.info(f"📁 File: {self.file_path}")
        logger.info(f"🔍 Mode: {'Validation Only' if validate_only else 'Full Clean'}")
        
        try:
            # Step 1: Load and validate file exists
            if not self.file_path.exists():
                logger.error(f"❌ File not found: {self.file_path}")
                return False
            
            # Step 2: Load dataset
            logger.info("📥 Loading dataset...")
            data = pd.read_csv(self.file_path)
            self.cleaning_stats['original_shape'] = data.shape
            logger.info(f"   📊 Original shape: {data.shape}")
            
            # Step 3: Validate dataset
            validation_results = self._validate_dataset(data)
            
            if validate_only:
                self._print_validation_report(validation_results)
                return len(validation_results['errors']) == 0
            
            # Step 4: Check if cleaning is needed
            if len(validation_results['errors']) == 0 and len(validation_results['warnings']) == 0:
                logger.info("✅ Dataset is already clean!")
                return True
            
            # Step 5: Show what will be cleaned
            self._print_cleaning_preview(validation_results)
            
            # Step 6: Confirmation (unless forced)
            if not force:
                response = input("\n🤔 Proceed with cleaning? (y/N): ").strip().lower()
                if response not in ['y', 'yes']:
                    logger.info("❌ Cleaning cancelled by user")
                    return False
            
            # Step 7: Create backup
            if self.create_backup:
                self.backup_path = self._create_backup()
            
            # Step 8: Perform cleaning
            logger.info("\n🧹 Performing dataset cleaning...")
            cleaned_data = self._perform_cleaning(data, validation_results)
            
            # Step 9: Final validation
            final_validation = self._validate_dataset(cleaned_data)
            
            # Step 10: Save cleaned dataset
            if len(final_validation['errors']) == 0:
                logger.info("💾 Saving cleaned dataset...")
                cleaned_data.to_csv(self.file_path, index=False)
                self.cleaning_stats['final_shape'] = cleaned_data.shape
                
                # Step 11: Generate cleaning report
                self._generate_cleaning_report()
                
                logger.info("✅ DATASET CLEANING COMPLETED SUCCESSFULLY!")
                logger.info(f"   📊 Final shape: {cleaned_data.shape}")
                logger.info(f"   💾 Backup: {self.backup_path}")
                return True
            else:
                logger.error("❌ Cleaning failed - final validation has errors")
                self._print_validation_report(final_validation)
                return False
                
        except Exception as e:
            logger.error(f"❌ Dataset cleaning failed: {e}")
            return False
    
    def _validate_dataset(self, data: pd.DataFrame) -> Dict[str, List[str]]:
        """Comprehensive dataset validation"""
        logger.info("🔍 Validating dataset...")
        
        errors = []
        warnings = []
        info = []
        
        # 1. Check required columns
        missing_required = [col for col in self.validation_rules['required_columns'] if col not in data.columns]
        if missing_required:
            errors.append(f"Missing required columns: {missing_required}")
        
        # 2. Check column naming issues
        problematic_columns = []
        for col in data.columns:
            if any(char in col for char in self.validation_rules['forbidden_characters']):
                problematic_columns.append(col)
        
        if problematic_columns:
            warnings.append(f"Columns with forbidden characters: {problematic_columns[:10]}{'...' if len(problematic_columns) > 10 else ''}")
        
        # 3. Check for duplicate columns
        duplicate_columns = data.columns[data.columns.duplicated()].tolist()
        if duplicate_columns:
            warnings.append(f"Duplicate columns found: {duplicate_columns}")
        
        # 4. Check data types
        for col in self.validation_rules['numeric_columns']:
            if col in data.columns:
                try:
                    pd.to_numeric(data[col], errors='raise')
                except:
                    errors.append(f"Column '{col}' contains non-numeric values")
        
        # 5. Check date columns
        for col in self.validation_rules['date_columns']:
            if col in data.columns:
                try:
                    pd.to_datetime(data[col], errors='raise')
                except:
                    errors.append(f"Column '{col}' contains invalid dates")
        
        # 6. Check for excessive NaN values
        high_nan_columns = []
        for col in data.columns:
            nan_percentage = data[col].isna().mean()
            if nan_percentage > self.validation_rules['max_nan_percentage']:
                high_nan_columns.append(f"{col} ({nan_percentage:.1%})")
        
        if high_nan_columns:
            warnings.append(f"Columns with high NaN percentage: {high_nan_columns[:5]}{'...' if len(high_nan_columns) > 5 else ''}")
        
        # 7. Check symbol data consistency
        if 'symbol' in data.columns:
            symbol_counts = data['symbol'].value_counts()
            low_count_symbols = symbol_counts[symbol_counts < self.validation_rules['min_rows_per_symbol']]
            if len(low_count_symbols) > 0:
                warnings.append(f"Symbols with insufficient data: {list(low_count_symbols.index)}")
        
        # 8. Check for infinite values
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        inf_columns = []
        for col in numeric_cols:
            if np.isinf(data[col]).any():
                inf_columns.append(col)
        
        if inf_columns:
            warnings.append(f"Columns with infinite values: {inf_columns[:5]}{'...' if len(inf_columns) > 5 else ''}")
        
        # 9. Check target variable coverage
        target_cols = [col for col in data.columns if col.startswith('target_')]
        for col in target_cols:
            coverage = data[col].notna().mean()
            if coverage < 0.8:
                warnings.append(f"Low target coverage in {col}: {coverage:.1%}")
            else:
                info.append(f"Good target coverage in {col}: {coverage:.1%}")
        
        # 10. Check data range sanity
        if 'date' in data.columns:
            try:
                date_series = pd.to_datetime(data['date'])
                date_range = (date_series.max() - date_series.min()).days
                if date_range < 365:
                    warnings.append(f"Short date range: {date_range} days")
                else:
                    info.append(f"Date range: {date_range} days ({date_series.min().date()} to {date_series.max().date()})")
            except:
                pass
        
        logger.info(f"   🔍 Validation complete: {len(errors)} errors, {len(warnings)} warnings")
        
        return {
            'errors': errors,
            'warnings': warnings, 
            'info': info
        }
    
    def _perform_cleaning(self, data: pd.DataFrame, validation_results: Dict) -> pd.DataFrame:
        """Perform the actual cleaning operations"""
        cleaned_data = data.copy()
        
        # 1. Fix column names
        logger.info("🔧 Fixing column names...")
        original_columns = list(cleaned_data.columns)
        
        # Clean column names
        cleaned_names = []
        for col in cleaned_data.columns:
            # Replace forbidden characters
            clean_name = col
            for char in self.validation_rules['forbidden_characters']:
                clean_name = clean_name.replace(char, '_')
            
            # Remove consecutive underscores
            while '__' in clean_name:
                clean_name = clean_name.replace('__', '_')
            
            # Remove leading/trailing underscores
            clean_name = clean_name.strip('_')
            
            # Ensure name is not empty
            if not clean_name:
                clean_name = f"unnamed_column_{len(cleaned_names)}"
            
            cleaned_names.append(clean_name)
        
        cleaned_data.columns = cleaned_names
        self.cleaning_stats['columns_renamed'] = sum(1 for old, new in zip(original_columns, cleaned_names) if old != new)
        
        if self.cleaning_stats['columns_renamed'] > 0:
            logger.info(f"   📝 Renamed {self.cleaning_stats['columns_renamed']} columns")
            self.cleaning_stats['issues_fixed'].append('Column names cleaned')
        
        # 2. Remove duplicate columns
        logger.info("🔧 Removing duplicate columns...")
        original_col_count = len(cleaned_data.columns)
        cleaned_data = cleaned_data.loc[:, ~cleaned_data.columns.duplicated()]
        duplicates_removed = original_col_count - len(cleaned_data.columns)
        self.cleaning_stats['duplicates_removed'] = duplicates_removed
        
        if duplicates_removed > 0:
            logger.info(f"   🗑️ Removed {duplicates_removed} duplicate columns")
            self.cleaning_stats['issues_fixed'].append('Duplicate columns removed')
        
        # 3. Fix data types
        logger.info("🔧 Fixing data types...")
        
        # Fix numeric columns
        for col in self.validation_rules['numeric_columns']:
            if col in cleaned_data.columns:
                try:
                    original_dtype = cleaned_data[col].dtype
                    cleaned_data[col] = pd.to_numeric(cleaned_data[col], errors='coerce')
                    if str(original_dtype) != str(cleaned_data[col].dtype):
                        logger.info(f"   🔢 Fixed numeric type for '{col}': {original_dtype} → {cleaned_data[col].dtype}")
                        self.cleaning_stats['issues_fixed'].append(f'Fixed data type for {col}')
                except Exception as e:
                    logger.warning(f"   ⚠️ Could not fix numeric type for '{col}': {e}")
        
        # Fix date columns
        for col in self.validation_rules['date_columns']:
            if col in cleaned_data.columns:
                try:
                    original_dtype = cleaned_data[col].dtype
                    cleaned_data[col] = pd.to_datetime(cleaned_data[col], errors='coerce')
                    if str(original_dtype) != str(cleaned_data[col].dtype):
                        logger.info(f"   📅 Fixed date type for '{col}': {original_dtype} → {cleaned_data[col].dtype}")
                        self.cleaning_stats['issues_fixed'].append(f'Fixed date type for {col}')
                except Exception as e:
                    logger.warning(f"   ⚠️ Could not fix date type for '{col}': {e}")
        
        # 4. Handle infinite values
        logger.info("🔧 Handling infinite values...")
        numeric_cols = cleaned_data.select_dtypes(include=[np.number]).columns
        inf_count = 0
        
        for col in numeric_cols:
            col_inf_count = np.isinf(cleaned_data[col]).sum()
            if col_inf_count > 0:
                cleaned_data[col] = cleaned_data[col].replace([np.inf, -np.inf], np.nan)
                inf_count += col_inf_count
        
        if inf_count > 0:
            logger.info(f"   ♾️ Replaced {inf_count} infinite values with NaN")
            self.cleaning_stats['invalid_values_fixed'] += inf_count
            self.cleaning_stats['issues_fixed'].append('Infinite values replaced')
        
        # 5. Remove rows with all NaN targets
        logger.info("🔧 Cleaning target variables...")
        target_cols = [col for col in cleaned_data.columns if col.startswith('target_')]
        if target_cols:
            original_rows = len(cleaned_data)
            # Remove rows where ALL targets are NaN
            cleaned_data = cleaned_data.dropna(subset=target_cols, how='all')
            rows_removed = original_rows - len(cleaned_data)
            self.cleaning_stats['rows_removed'] += rows_removed
            
            if rows_removed > 0:
                logger.info(f"   🗑️ Removed {rows_removed} rows with no valid targets")
                self.cleaning_stats['issues_fixed'].append('Rows with invalid targets removed')
        
        # 6. Remove symbols with insufficient data
        if 'symbol' in cleaned_data.columns:
            logger.info("🔧 Checking symbol data sufficiency...")
            original_rows = len(cleaned_data)
            symbol_counts = cleaned_data['symbol'].value_counts()
            valid_symbols = symbol_counts[symbol_counts >= self.validation_rules['min_rows_per_symbol']].index
            cleaned_data = cleaned_data[cleaned_data['symbol'].isin(valid_symbols)]
            rows_removed = original_rows - len(cleaned_data)
            
            if rows_removed > 0:
                removed_symbols = set(symbol_counts.index) - set(valid_symbols)
                logger.info(f"   🗑️ Removed {rows_removed} rows from symbols with insufficient data: {list(removed_symbols)}")
                self.cleaning_stats['rows_removed'] += rows_removed
                self.cleaning_stats['issues_fixed'].append('Insufficient symbol data removed')
        
        # 7. Sort data properly
        logger.info("🔧 Sorting data...")
        if 'symbol' in cleaned_data.columns and 'date' in cleaned_data.columns:
            cleaned_data = cleaned_data.sort_values(['symbol', 'date']).reset_index(drop=True)
            self.cleaning_stats['issues_fixed'].append('Data sorted by symbol and date')
        
        # 8. Optimize memory usage
        logger.info("🔧 Optimizing memory usage...")
        
        # Downcast numeric types where possible
        for col in cleaned_data.select_dtypes(include=[np.number]).columns:
            try:
                original_memory = cleaned_data[col].memory_usage(deep=True)
                cleaned_data[col] = pd.to_numeric(cleaned_data[col], downcast='float')
                new_memory = cleaned_data[col].memory_usage(deep=True)
                if new_memory < original_memory:
                    logger.info(f"   💾 Optimized memory for '{col}': {original_memory/1024/1024:.1f}MB → {new_memory/1024/1024:.1f}MB")
            except:
                pass
        
        return cleaned_data
    
    def _create_backup(self) -> str:
        """Create a backup of the original file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = Path(BACKUP_DIR)
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        backup_filename = f"{self.file_path.stem}_before_cleaning_{timestamp}{self.file_path.suffix}"
        backup_path = backup_dir / backup_filename
        
        try:
            shutil.copy2(self.file_path, backup_path)
            logger.info(f"💾 Backup created: {backup_path}")
            return str(backup_path)
        except Exception as e:
            logger.error(f"❌ Failed to create backup: {e}")
            return None
    
    def _print_validation_report(self, results: Dict):
        """Print a formatted validation report"""
        print("\n" + "=" * 60)
        print("📋 DATASET VALIDATION REPORT")
        print("=" * 60)
        
        if results['errors']:
            print("\n❌ ERRORS (Must be fixed):")
            for i, error in enumerate(results['errors'], 1):
                print(f"   {i}. {error}")
        
        if results['warnings']:
            print("\n⚠️ WARNINGS (Should be addressed):")
            for i, warning in enumerate(results['warnings'], 1):
                print(f"   {i}. {warning}")
        
        if results['info']:
            print("\n✅ INFO (Looking good):")
            for i, info in enumerate(results['info'], 1):
                print(f"   {i}. {info}")
        
        if not results['errors'] and not results['warnings']:
            print("\n🎉 DATASET IS CLEAN!")
            print("   No issues found. Dataset is ready for model training.")
        
        print("=" * 60)
    
    def _print_cleaning_preview(self, results: Dict):
        """Print what will be cleaned"""
        print("\n" + "🧹 CLEANING PREVIEW")
        print("-" * 40)
        
        if results['errors']:
            print("❌ The following ERRORS will be fixed:")
            for error in results['errors']:
                print(f"   • {error}")
        
        if results['warnings']:
            print("\n⚠️ The following WARNINGS will be addressed:")
            for warning in results['warnings']:
                print(f"   • {warning}")
        
        print(f"\n📊 Current dataset: {self.cleaning_stats['original_shape']}")
        if self.create_backup:
            print("💾 Backup will be created before cleaning")
    
    def _generate_cleaning_report(self):
        """Generate a detailed cleaning report"""
        report = {
            'cleaning_timestamp': datetime.now().isoformat(),
            'file_path': str(self.file_path),
            'backup_path': self.backup_path,
            'statistics': self.cleaning_stats,
            'cleaning_summary': {
                'original_rows': self.cleaning_stats['original_shape'][0],
                'original_columns': self.cleaning_stats['original_shape'][1],
                'final_rows': self.cleaning_stats['final_shape'][0],
                'final_columns': self.cleaning_stats['final_shape'][1],
                'rows_removed': self.cleaning_stats['rows_removed'],
                'columns_renamed': self.cleaning_stats['columns_renamed'],
                'duplicates_removed': self.cleaning_stats['duplicates_removed'],
                'invalid_values_fixed': self.cleaning_stats['invalid_values_fixed']
            },
            'issues_fixed': self.cleaning_stats['issues_fixed']
        }
        
        # Save report
        report_dir = Path(DATA_DIR)
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"cleaning_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"📋 Cleaning report saved: {report_path}")
        except Exception as e:
            logger.warning(f"⚠️ Could not save cleaning report: {e}")
        
        # Print summary
        print("\n" + "📊 CLEANING SUMMARY")
        print("-" * 40)
        print(f"Original size: {report['cleaning_summary']['original_rows']:,} rows × {report['cleaning_summary']['original_columns']} columns")
        print(f"Final size: {report['cleaning_summary']['final_rows']:,} rows × {report['cleaning_summary']['final_columns']} columns")
        print(f"Rows removed: {report['cleaning_summary']['rows_removed']:,}")
        print(f"Columns renamed: {report['cleaning_summary']['columns_renamed']}")
        print(f"Duplicates removed: {report['cleaning_summary']['duplicates_removed']}")
        print(f"Invalid values fixed: {report['cleaning_summary']['invalid_values_fixed']:,}")
        
        if self.cleaning_stats['issues_fixed']:
            print(f"\nIssues fixed:")
            for issue in self.cleaning_stats['issues_fixed']:
                print(f"   ✅ {issue}")

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(
        description='Dataset Cleaning Utility for ML Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python clean.py                           # Clean main dataset
  python clean.py --validate-only           # Just validate
  python clean.py --file custom.csv         # Clean specific file
  python clean.py --force                   # Skip confirmations
  python clean.py --no-backup               # Don't create backup
        """
    )
    
    parser.add_argument('--file', type=str, default=MAIN_DATASET,
                       help='Path to dataset file (default: combined_dataset.csv)')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only validate dataset, do not clean')
    parser.add_argument('--force', action='store_true',
                       help='Skip confirmation prompts')
    parser.add_argument('--no-backup', action='store_true',
                       help='Skip backup creation')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize cleaner
    cleaner = DatasetCleaner(
        file_path=args.file,
        create_backup=not args.no_backup
    )
    
    # Run cleaning
    success = cleaner.clean_dataset(
        validate_only=args.validate_only,
        force=args.force
    )
    
    if success:
        if args.validate_only:
            print("\n✅ Validation completed successfully!")
        else:
            print("\n✅ Dataset cleaning completed successfully!")
            print("🚀 Your dataset is now ready for model training!")
    else:
        print("\n❌ Dataset cleaning failed!")
        print("💡 Check the logs above for details.")
        exit(1)

if __name__ == "__main__":
    main()