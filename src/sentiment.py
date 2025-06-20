#!/usr/bin/env python3
"""
FIXED SENTIMENT INTEGRATION - PROPER DATASET FLOW
=================================================

FIXED ISSUES:
- ✅ Loads temporal_decay_dataset instead of core dataset
- ✅ Saves to final_enhanced_dataset path instead of overwriting temporal
- ✅ Preserves temporal decay features while adding sentiment
- ✅ Proper dataset flow: core -> temporal_decay -> sentiment_enhanced
- ✅ Creates splits for final enhanced dataset

PIPELINE: data.py → temporal_decay.py → sentiment.py → models.py
"""

import sys
import os
from pathlib import Path

# Add src directory to Python path
script_dir = Path(__file__).parent
if 'src' in str(script_dir):
    sys.path.insert(0, str(script_dir))
else:
    sys.path.insert(0, str(script_dir / 'src'))

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import json
import shutil
import warnings
warnings.filterwarnings('ignore')

# ✅ FIXED: Consistent config integration
from config_reader import load_config, get_data_paths

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProductionSentimentIntegrator:
    """
    Fixed sentiment integration with proper dataset flow
    
    Integration Strategies (in priority order):
    1. FNSPID Daily Sentiment (best - real sentiment data)
    2. Synthetic Sentiment (fallback - always works)
    
    Dataset Flow:
    core_dataset -> temporal_decay_dataset -> final_enhanced_dataset
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize with consistent config approach"""
        try:
            self.config = load_config(config_path)
            self.data_paths = get_data_paths(self.config)
            
            # ✅ FIX: Add final enhanced dataset path
            self.data_paths['final_enhanced_dataset'] = Path("data/processed/final_enhanced_dataset.csv")
            
            self.symbols = self.config['data']['core']['symbols']
            self.target_horizons = self.config['data']['core']['target_horizons']
            self.start_time = datetime.now()
            
            # Integration statistics
            self.stats = {
                'temporal_records': 0,
                'sentiment_records': 0,
                'matched_records': 0,
                'coverage_percentage': 0.0,
                'features_added': 0,
                'strategy_used': 'unknown',
                'processing_time': 0.0
            }
            
            # Create required directories
            for path in [self.data_paths['temporal_decay_dataset'].parent, 
                        self.data_paths['final_enhanced_dataset'].parent]:
                path.mkdir(parents=True, exist_ok=True)
            
            logger.info("🔗 Fixed Sentiment Integrator initialized")
            logger.info(f"   📊 Symbols: {self.symbols}")
            logger.info(f"   🎯 Target horizons: {self.target_horizons}")
            
        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}")
            raise
    
    def analyze_data_availability(self) -> Dict[str, Any]:
        """Analyze what data is available for sentiment integration"""
        logger.info("🔍 Analyzing data availability...")
        
        analysis = {
            'temporal_decay_dataset': {'exists': False, 'records': 0, 'decay_features': []},
            'fnspid_daily_sentiment': {'exists': False, 'records': 0, 'symbols': []},
            'recommended_strategy': 'synthetic'
        }
        
        # ✅ FIX: Check temporal decay dataset (required input)
        if self.data_paths['temporal_decay_dataset'].exists():
            try:
                temporal_sample = pd.read_csv(self.data_paths['temporal_decay_dataset'], nrows=1000)
                decay_features = [col for col in temporal_sample.columns if 'sentiment_decay_' in col]
                
                analysis['temporal_decay_dataset'] = {
                    'exists': True,
                    'records': len(temporal_sample),
                    'decay_features': decay_features,
                    'symbols': temporal_sample['symbol'].unique().tolist() if 'symbol' in temporal_sample.columns else []
                }
                logger.info(f"   ✅ Temporal decay dataset: {len(temporal_sample):,} records (sample)")
                
            except Exception as e:
                logger.error(f"   ❌ Temporal decay dataset unreadable: {e}")
                raise FileNotFoundError("Temporal decay dataset required but not accessible")
        else:
            raise FileNotFoundError(f"Temporal decay dataset not found: {self.data_paths['temporal_decay_dataset']}")
        
        # Check FNSPID sentiment (preferred)
        if self.data_paths['fnspid_daily_sentiment'].exists():
            try:
                fnspid_sample = pd.read_csv(self.data_paths['fnspid_daily_sentiment'], nrows=1000)
                required_cols = ['sentiment_compound', 'confidence', 'symbol', 'date']
                
                if all(col in fnspid_sample.columns for col in required_cols):
                    analysis['fnspid_daily_sentiment'] = {
                        'exists': True,
                        'records': len(fnspid_sample),
                        'symbols': fnspid_sample['symbol'].unique().tolist()
                    }
                    analysis['recommended_strategy'] = 'fnspid_sentiment'
                    logger.info(f"   ✅ FNSPID sentiment: {len(fnspid_sample):,} records (sample)")
                else:
                    logger.warning(f"   ⚠️ FNSPID missing required columns: {required_cols}")
                    
            except Exception as e:
                logger.warning(f"   ⚠️ FNSPID sentiment unreadable: {e}")
        
        # Final strategy decision
        if analysis['recommended_strategy'] == 'synthetic':
            logger.info(f"   🎲 Will use synthetic sentiment (fallback)")
        
        return analysis
    
    def load_sentiment_data(self, strategy: str) -> pd.DataFrame:
        """Load sentiment data based on strategy with robust error handling"""
        logger.info(f"📥 Loading sentiment data (strategy: {strategy})...")
        
        try:
            if strategy == 'fnspid_sentiment':
                return self._load_fnspid_sentiment_data()
            elif strategy == 'synthetic':
                return self._generate_synthetic_sentiment()
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
                
        except Exception as e:
            logger.error(f"❌ Failed to load sentiment data with strategy '{strategy}': {e}")
            
            # Fallback to synthetic if other strategies fail
            if strategy != 'synthetic':
                logger.info("🎲 Falling back to synthetic sentiment...")
                return self._generate_synthetic_sentiment()
            else:
                raise
    
    def _load_fnspid_sentiment_data(self) -> pd.DataFrame:
        """Load FNSPID daily sentiment data"""
        logger.info("   📊 Loading FNSPID sentiment data...")
        
        data = pd.read_csv(self.data_paths['fnspid_daily_sentiment'])
        
        # Validate required columns
        required_cols = ['symbol', 'date', 'sentiment_compound', 'confidence']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"FNSPID data missing columns: {missing_cols}")
        
        # Create sentiment features for integration
        result = data[['symbol', 'date']].copy()
        
        # Add core sentiment features
        result['sentiment_compound'] = data['sentiment_compound']
        result['sentiment_positive'] = data.get('sentiment_positive', data['sentiment_compound'].clip(0, 1))
        result['sentiment_negative'] = data.get('sentiment_negative', (-data['sentiment_compound']).clip(0, 1))
        result['sentiment_confidence'] = data['confidence']
        result['article_count'] = data.get('article_count', 1)
        
        # Add momentum features
        data_sorted = data.sort_values(['symbol', 'date'])
        for window in [3, 7, 14]:
            result[f'sentiment_ma_{window}d'] = (
                data_sorted.groupby('symbol')['sentiment_compound']
                .rolling(window, min_periods=1).mean().values
            )
        
        result['source'] = 'fnspid_real'
        
        logger.info(f"   ✅ Loaded {len(result):,} FNSPID sentiment records")
        return result
    
    def _generate_synthetic_sentiment(self) -> pd.DataFrame:
        """Generate synthetic sentiment data as fallback"""
        logger.info("   🎲 Generating synthetic sentiment data...")
        
        # ✅ FIX: Load temporal decay data to get date/symbol combinations
        temporal_data = pd.read_csv(self.data_paths['temporal_decay_dataset'])
        
        # Get unique symbol-date combinations
        symbol_dates = temporal_data[['symbol', 'date']].drop_duplicates()
        
        # Set seed for reproducibility
        np.random.seed(42)
        
        records = []
        for _, row in symbol_dates.iterrows():
            symbol, date = row['symbol'], row['date']
            
            # Generate base sentiment with symbol-specific bias
            symbol_hash = hash(symbol) % 1000
            symbol_bias = (symbol_hash - 500) / 10000  # Small bias between -0.05 and +0.05
            
            # Add date-based cyclical component
            try:
                date_obj = pd.to_datetime(date)
                day_of_year = date_obj.timetuple().tm_yday
                date_cycle = np.sin(2 * np.pi * day_of_year / 365) * 0.1
            except:
                date_cycle = 0
            
            # Generate base sentiment
            base_sentiment = np.clip(
                np.random.normal(symbol_bias + date_cycle, 0.3), -1, 1
            )
            
            # Create record with sentiment features
            record = {
                'symbol': symbol, 
                'date': date, 
                'source': 'synthetic',
                'sentiment_compound': base_sentiment,
                'sentiment_positive': max(0, base_sentiment),
                'sentiment_negative': max(0, -base_sentiment),
                'sentiment_confidence': np.random.beta(3, 2),
                'article_count': max(1, int(np.random.poisson(2)))
            }
            
            # Add momentum features
            record['sentiment_ma_3d'] = base_sentiment + np.random.normal(0, 0.1)
            record['sentiment_ma_7d'] = base_sentiment + np.random.normal(0, 0.05)
            record['sentiment_ma_14d'] = base_sentiment + np.random.normal(0, 0.03)
            
            records.append(record)
        
        result = pd.DataFrame(records)
        logger.info(f"   ✅ Generated {len(result):,} synthetic sentiment records")
        
        return result
    
    def integrate_sentiment_with_temporal(self, sentiment_data: pd.DataFrame) -> pd.DataFrame:
        """✅ FIX: Integrate sentiment data with temporal decay dataset"""
        logger.info("🔗 Integrating sentiment with temporal decay dataset...")
        
        # ✅ FIX: Load temporal decay dataset instead of core dataset
        temporal_data = pd.read_csv(self.data_paths['temporal_decay_dataset'])
        self.stats['temporal_records'] = len(temporal_data)
        self.stats['sentiment_records'] = len(sentiment_data)
        
        # Create backup of temporal decay dataset
        backup_path = self._create_backup()
        
        # Standardize date formats for merge
        temporal_data['date'] = pd.to_datetime(temporal_data['date']).dt.date
        sentiment_data['date'] = pd.to_datetime(sentiment_data['date']).dt.date
        
        # Merge datasets
        logger.info("   🔄 Merging sentiment with temporal decay data...")
        enhanced_data = temporal_data.merge(
            sentiment_data, 
            on=['symbol', 'date'], 
            how='left'
        )
        
        # Calculate coverage statistics
        sentiment_cols = [col for col in sentiment_data.columns if col not in ['symbol', 'date']]
        if sentiment_cols:
            # Use first sentiment column to calculate coverage
            primary_sentiment_col = sentiment_cols[0]
            matched_mask = enhanced_data[primary_sentiment_col].notna()
            self.stats['matched_records'] = matched_mask.sum()
            self.stats['coverage_percentage'] = (self.stats['matched_records'] / self.stats['temporal_records']) * 100
        
        # Fill missing sentiment values with appropriate defaults
        logger.info("   🔧 Filling missing sentiment values...")
        default_values = self._get_default_sentiment_values()
        
        for col in sentiment_cols:
            if col in enhanced_data.columns:
                enhanced_data[col] = enhanced_data[col].fillna(default_values.get(col, 0.0))
        
        # Add metadata columns
        enhanced_data['sentiment_integration_source'] = sentiment_data['source'].iloc[0] if 'source' in sentiment_data.columns else 'unknown'
        enhanced_data['sentiment_integration_timestamp'] = datetime.now().isoformat()
        
        self.stats['features_added'] = len(enhanced_data.columns) - len(temporal_data.columns)
        
        # Convert dates back to strings for consistency
        enhanced_data['date'] = enhanced_data['date'].astype(str)
        
        logger.info("   ✅ Integration completed successfully!")
        logger.info(f"      📊 Temporal records: {self.stats['temporal_records']:,}")
        logger.info(f"      📊 Sentiment records: {self.stats['sentiment_records']:,}")
        logger.info(f"      📈 Coverage: {self.stats['coverage_percentage']:.1f}%")
        logger.info(f"      🆕 Features added: {self.stats['features_added']}")
        logger.info(f"      💾 Backup created: {backup_path}")
        
        return enhanced_data
    
    def _create_backup(self) -> str:
        """Create backup of temporal decay dataset"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = self.data_paths['temporal_decay_dataset'].parent.parent / "backups"
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            backup_path = backup_dir / f"temporal_decay_dataset_backup_{timestamp}.csv"
            shutil.copy2(self.data_paths['temporal_decay_dataset'], backup_path)
            
            return str(backup_path)
        except Exception as e:
            logger.warning(f"⚠️ Backup creation failed: {e}")
            return "backup_failed"
    
    def _get_default_sentiment_values(self) -> Dict[str, float]:
        """Get default values for missing sentiment features"""
        defaults = {
            'sentiment_compound': 0.0,
            'sentiment_positive': 0.0,
            'sentiment_negative': 0.0,
            'sentiment_confidence': 0.5,
            'article_count': 0,
            'sentiment_ma_3d': 0.0,
            'sentiment_ma_7d': 0.0,
            'sentiment_ma_14d': 0.0,
            'source': 'none',
            'sentiment_integration_source': 'none'
        }
        
        return defaults
    
    def create_enhanced_dataset_splits(self, enhanced_data: pd.DataFrame) -> None:
        """Create splits for the final enhanced dataset"""
        logger.info("🔪 Creating splits for enhanced dataset...")
        
        try:
            # Load existing split information from temporal decay splits
            splits_dir = Path('data/splits')
            
            # Check if temporal decay splits exist
            temporal_train_path = splits_dir / "temporal_decay_enhanced_dataset_train.csv"
            if temporal_train_path.exists():
                logger.info("   📋 Using existing temporal decay split boundaries...")
                
                # Load split boundaries
                train_split = pd.read_csv(temporal_train_path, usecols=['symbol', 'date'])
                val_split = pd.read_csv(splits_dir / "temporal_decay_enhanced_dataset_val.csv", usecols=['symbol', 'date'])
                test_split = pd.read_csv(splits_dir / "temporal_decay_enhanced_dataset_test.csv", usecols=['symbol', 'date'])
                
                # Create split column
                enhanced_data['split'] = 'unknown'
                
                # Merge to assign splits
                for split_name, split_data in [('train', train_split), ('val', val_split), ('test', test_split)]:
                    split_data['split_temp'] = split_name
                    enhanced_data = enhanced_data.merge(
                        split_data[['symbol', 'date', 'split_temp']], 
                        on=['symbol', 'date'], 
                        how='left'
                    )
                    
                    # Update split column
                    mask = enhanced_data['split_temp'] == split_name
                    enhanced_data.loc[mask, 'split'] = split_name
                    enhanced_data.drop('split_temp', axis=1, inplace=True)
                
                # Split the data
                train_data = enhanced_data[enhanced_data['split'] == 'train'].drop('split', axis=1)
                val_data = enhanced_data[enhanced_data['split'] == 'val'].drop('split', axis=1)
                test_data = enhanced_data[enhanced_data['split'] == 'test'].drop('split', axis=1)
                
            else:
                logger.info("   🔪 Creating new temporal splits...")
                # Create new splits using academic splitter
                try:
                    from academic_data_splitter import AcademicDataSplitter
                    splitter = AcademicDataSplitter(train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)
                    train_data, val_data, test_data = splitter.create_temporal_splits(enhanced_data)
                except ImportError:
                    logger.warning("   ⚠️ AcademicDataSplitter not available, using simple date-based split")
                    
                    # Simple date-based split
                    enhanced_data['date'] = pd.to_datetime(enhanced_data['date'])
                    enhanced_data = enhanced_data.sort_values('date')
                    
                    unique_dates = sorted(enhanced_data['date'].unique())
                    n_dates = len(unique_dates)
                    
                    train_end_idx = int(n_dates * 0.7)
                    val_end_idx = int(n_dates * 0.9)
                    
                    train_end_date = unique_dates[train_end_idx - 1]
                    val_end_date = unique_dates[val_end_idx - 1]
                    
                    train_data = enhanced_data[enhanced_data['date'] <= train_end_date].copy()
                    val_data = enhanced_data[(enhanced_data['date'] > train_end_date) & 
                                           (enhanced_data['date'] <= val_end_date)].copy()
                    test_data = enhanced_data[enhanced_data['date'] > val_end_date].copy()
                    
                    # Convert dates back to strings
                    for df in [train_data, val_data, test_data, enhanced_data]:
                        df['date'] = df['date'].astype(str)
            
            # Save splits
            splits_dir.mkdir(parents=True, exist_ok=True)
            
            train_data.to_csv(splits_dir / "final_enhanced_dataset_train.csv", index=False)
            val_data.to_csv(splits_dir / "final_enhanced_dataset_val.csv", index=False)
            test_data.to_csv(splits_dir / "final_enhanced_dataset_test.csv", index=False)
            
            logger.info(f"   ✅ Enhanced dataset splits created:")
            logger.info(f"      🏃 Train: {len(train_data):,} records")
            logger.info(f"      ✋ Val: {len(val_data):,} records")
            logger.info(f"      🧪 Test: {len(test_data):,} records")
            
        except Exception as e:
            logger.warning(f"   ⚠️ Split creation failed: {e}")
    
    def validate_integration(self, enhanced_data: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive validation of integration results"""
        logger.info("🔍 Validating integration results...")
        
        validation = {
            'status': 'success',
            'issues': [],
            'warnings': [],
            'quality_metrics': {},
            'feature_analysis': {},
            'readiness_for_tft': True
        }
        
        try:
            # Check required base columns
            required_base = ['stock_id', 'symbol', 'date', 'close', 'target_5']
            missing_base = [col for col in required_base if col not in enhanced_data.columns]
            if missing_base:
                validation['issues'].append(f"Missing required base columns: {missing_base}")
                validation['readiness_for_tft'] = False
            
            # Check temporal decay features are preserved
            decay_features = [col for col in enhanced_data.columns if 'sentiment_decay_' in col]
            if len(decay_features) == 0:
                validation['warnings'].append("No temporal decay features found - may have been lost in integration")
            
            # Check new sentiment features
            sentiment_features = [col for col in enhanced_data.columns if 
                                col.startswith('sentiment_') and 'decay' not in col]
            if len(sentiment_features) == 0:
                validation['issues'].append("No new sentiment features added")
                validation['readiness_for_tft'] = False
            
            # Quality metrics
            validation['quality_metrics'] = {
                'total_records': len(enhanced_data),
                'unique_symbols': enhanced_data['symbol'].nunique(),
                'date_range': {
                    'start': str(enhanced_data['date'].min()),
                    'end': str(enhanced_data['date'].max())
                },
                'target_coverage': float(enhanced_data['target_5'].notna().mean() * 100),
                'sentiment_coverage': float(enhanced_data[sentiment_features[0]].notna().mean() * 100) if sentiment_features else 0
            }
            
            # Feature analysis
            technical_features = [col for col in enhanced_data.columns if any(
                pattern in col.lower() for pattern in ['ema_', 'sma_', 'rsi_', 'macd', 'bb_', 'atr']
            )]
            time_features = [col for col in enhanced_data.columns if any(
                pattern in col.lower() for pattern in ['year', 'month', 'day', 'time_idx']
            )]
            target_features = [col for col in enhanced_data.columns if col.startswith('target_')]
            
            validation['feature_analysis'] = {
                'total_features': len(enhanced_data.columns),
                'sentiment_features': len(sentiment_features),
                'decay_features': len(decay_features),
                'technical_features': len(technical_features),
                'time_features': len(time_features),
                'target_features': len(target_features)
            }
            
            # Data quality checks
            if enhanced_data['symbol'].nunique() < len(self.symbols) * 0.8:
                validation['warnings'].append(f"Low symbol coverage: {enhanced_data['symbol'].nunique()}/{len(self.symbols)}")
            
            sentiment_coverage = validation['quality_metrics']['sentiment_coverage']
            if sentiment_coverage < 20:
                validation['warnings'].append(f"Low sentiment coverage: {sentiment_coverage:.1f}%")
                validation['status'] = 'warning'
            elif sentiment_coverage < 50:
                validation['warnings'].append(f"Moderate sentiment coverage: {sentiment_coverage:.1f}%")
            
            if validation['issues']:
                validation['status'] = 'issues_found' if validation['readiness_for_tft'] else 'not_ready'
            
            # Log validation results
            logger.info(f"   📊 Validation Status: {validation['status'].upper()}")
            logger.info(f"   🎯 TFT Ready: {'✅' if validation['readiness_for_tft'] else '❌'}")
            
            if validation['issues']:
                logger.warning("   ⚠️ Issues Found:")
                for issue in validation['issues']:
                    logger.warning(f"      • {issue}")
            
            if validation['warnings']:
                logger.info("   💡 Warnings:")
                for warning in validation['warnings']:
                    logger.info(f"      • {warning}")
            
        except Exception as e:
            validation['status'] = 'validation_failed'
            validation['issues'].append(f"Validation error: {e}")
            validation['readiness_for_tft'] = False
            logger.error(f"   ❌ Validation failed: {e}")
        
        return validation
    
    def generate_integration_report(self, validation: Dict[str, Any]) -> str:
        """Generate comprehensive integration report"""
        
        report = {
            'integration_metadata': {
                'timestamp': datetime.now().isoformat(),
                'processing_time_seconds': self.stats['processing_time'],
                'strategy_used': self.stats['strategy_used'],
                'config_symbols': self.symbols,
                'config_horizons': self.target_horizons
            },
            'data_statistics': {
                'temporal_records': self.stats['temporal_records'],
                'sentiment_records': self.stats['sentiment_records'],
                'matched_records': self.stats['matched_records'],
                'coverage_percentage': self.stats['coverage_percentage'],
                'features_added': self.stats['features_added']
            },
            'validation_results': validation,
            'file_paths': {
                'temporal_decay_dataset': str(self.data_paths['temporal_decay_dataset']),
                'final_enhanced_dataset': str(self.data_paths['final_enhanced_dataset'])
            }
        }
        
        # Save report
        report_dir = Path("results/integration")
        report_dir.mkdir(parents=True, exist_ok=True)
        
        report_path = report_dir / f"sentiment_integration_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"📋 Integration report saved: {report_path}")
        return str(report_path)
    
    def run_complete_integration(self) -> Tuple[bool, Dict[str, Any]]:
        """Run complete sentiment integration pipeline"""
        logger.info("🚀 STARTING FIXED SENTIMENT INTEGRATION")
        logger.info("=" * 60)
        
        try:
            start_time = datetime.now()
            
            # Step 1: Analyze data availability
            analysis = self.analyze_data_availability()
            strategy = analysis['recommended_strategy']
            self.stats['strategy_used'] = strategy
            
            logger.info(f"📊 Data Analysis Complete:")
            logger.info(f"   💡 Recommended strategy: {strategy}")
            
            # Step 2: Load sentiment data
            sentiment_data = self.load_sentiment_data(strategy)
            
            # Step 3: ✅ FIX: Integrate with temporal decay dataset
            enhanced_data = self.integrate_sentiment_with_temporal(sentiment_data)
            
            # Step 4: ✅ FIX: Save to final enhanced dataset path
            output_path = self.data_paths['final_enhanced_dataset']
            enhanced_data.to_csv(output_path, index=False)
            
            # Step 5: Create splits for enhanced dataset
            self.create_enhanced_dataset_splits(enhanced_data)
            
            # Step 6: Validate results
            validation = self.validate_integration(enhanced_data)
            
            # Step 7: Calculate final statistics
            self.stats['processing_time'] = (datetime.now() - start_time).total_seconds()
            
            # Step 8: Generate report
            report_path = self.generate_integration_report(validation)
            
            # Success summary
            logger.info("✅ SENTIMENT INTEGRATION COMPLETED SUCCESSFULLY!")
            logger.info("=" * 60)
            logger.info(f"📊 Final Statistics:")
            logger.info(f"   • Strategy: {strategy}")
            logger.info(f"   • Records: {len(enhanced_data):,}")
            logger.info(f"   • Coverage: {self.stats['coverage_percentage']:.1f}%")
            logger.info(f"   • Features added: {self.stats['features_added']}")
            logger.info(f"   • Processing time: {self.stats['processing_time']:.1f}s")
            logger.info(f"   • TFT Ready: {'✅' if validation['readiness_for_tft'] else '❌'}")
            logger.info(f"📁 Enhanced dataset: {output_path}")
            logger.info(f"📋 Report: {report_path}")
            
            return True, {
                'success': True,
                'strategy': strategy,
                'enhanced_dataset_path': str(output_path),
                'temporal_dataset_preserved': str(self.data_paths['temporal_decay_dataset']),
                'records': len(enhanced_data),
                'coverage': self.stats['coverage_percentage'],
                'features_added': self.stats['features_added'],
                'processing_time': self.stats['processing_time'],
                'validation': validation,
                'report_path': report_path,
                'sentiment_features': [col for col in enhanced_data.columns if 
                                     col.startswith('sentiment_') and 'decay' not in col]
            }
            
        except Exception as e:
            logger.error(f"❌ SENTIMENT INTEGRATION FAILED: {e}")
            
            # Try to provide meaningful error context
            error_context = {
                'error': str(e),
                'error_type': type(e).__name__,
                'stage': 'sentiment_integration',
                'strategy_attempted': getattr(self, 'stats', {}).get('strategy_used', 'unknown'),
                'processing_time': (datetime.now() - self.start_time).total_seconds()
            }
            
            return False, error_context

def main():
    """Main execution function for direct script execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Fixed Sentiment Integration')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file path')
    parser.add_argument('--validate-only', action='store_true', help='Only validate existing dataset')
    args = parser.parse_args()
    
    print("🔗 FIXED SENTIMENT INTEGRATION")
    print("=" * 50)
    
    try:
        # Initialize integrator
        integrator = ProductionSentimentIntegrator(args.config)
        
        if args.validate_only:
            # Validation only mode
            if integrator.data_paths['final_enhanced_dataset'].exists():
                enhanced_data = pd.read_csv(integrator.data_paths['final_enhanced_dataset'])
                validation = integrator.validate_integration(enhanced_data)
                
                print(f"\n🔍 VALIDATION RESULTS:")
                print(f"Status: {validation['status'].upper()}")
                print(f"TFT Ready: {'✅' if validation['readiness_for_tft'] else '❌'}")
                
                if validation['issues']:
                    print("Issues:")
                    for issue in validation['issues']:
                        print(f"  • {issue}")
                
                return 0 if validation['readiness_for_tft'] else 1
            else:
                print("❌ No final enhanced dataset found to validate")
                return 1
        
        # Run full integration
        success, result = integrator.run_complete_integration()
        
        if success:
            print(f"\n🎉 SUCCESS!")
            print(f"📊 Strategy: {result['strategy']}")
            print(f"📈 Records: {result['records']:,}")
            print(f"📊 Coverage: {result['coverage']:.1f}%")
            print(f"🆕 Features: {result['features_added']}")
            print(f"⏱️ Time: {result['processing_time']:.1f}s")
            print(f"📁 Final dataset: {result['enhanced_dataset_path']}")
            print(f"📁 Temporal preserved: {result['temporal_dataset_preserved']}")
            
            # Show next steps
            print(f"\n🚀 NEXT STEPS:")
            print("1. ✅ Sentiment integration complete")
            print("2. 🤖 Run: python src/models.py")
            print("3. 📊 Compare baseline vs enhanced TFT performance")
            
            return 0
        else:
            print(f"\n❌ FAILED: {result.get('error', 'Unknown error')}")
            return 1
            
    except Exception as e:
        print(f"\n❌ Integration failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())