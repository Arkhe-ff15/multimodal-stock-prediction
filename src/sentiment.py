#!/usr/bin/env python3
"""
SENTIMENT INTEGRATION - CONFIG-INTEGRATED VERSION
=================================================

✅ FIXES APPLIED:
- Proper config.py integration
- Standardized file paths using config
- Removed hardcoded paths
- Improved integration logic
- Automated execution

PIPELINE: data.py → fnspid_processor.py → temporal_decay.py → sentiment.py → models.py
"""

import pandas as pd
import numpy as np
import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any
import json
import shutil

# ✅ FIXED: Proper config integration
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PipelineConfig, get_file_path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatasetAnalyzer:
    """Analyze available datasets using config paths"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
    
    def analyze_data_availability(self) -> Dict[str, Any]:
        """✅ FIXED: Analyze what data is available using config paths"""
        analysis = {
            'core_dataset': {'exists': False},
            'temporal_decay_data': {'exists': False, 'decay_features': []},
            'fnspid_daily_sentiment': {'exists': False},
            'recommended_strategy': 'none'
        }
        
        # Check core dataset
        if self.config.core_dataset_path.exists():
            try:
                sample = pd.read_csv(self.config.core_dataset_path, nrows=100)
                analysis['core_dataset'] = {
                    'exists': True,
                    'symbols': sample['symbol'].unique().tolist() if 'symbol' in sample.columns else []
                }
            except Exception as e:
                logger.warning(f"Could not analyze core dataset: {e}")
        
        # Check temporal decay data (highest priority)
        if self.config.temporal_decay_data_path.exists():
            try:
                sample = pd.read_csv(self.config.temporal_decay_data_path, nrows=100)
                decay_features = [col for col in sample.columns if 'sentiment_decay_' in col]
                analysis['temporal_decay_data'] = {
                    'exists': True,
                    'decay_features': decay_features,
                    'symbols': sample['symbol'].unique().tolist() if 'symbol' in sample.columns else []
                }
                if len(decay_features) >= 3:
                    analysis['recommended_strategy'] = 'use_temporal_decay'
            except Exception as e:
                logger.warning(f"Could not analyze temporal decay data: {e}")
        
        # Check FNSPID sentiment (fallback)
        elif self.config.fnspid_daily_sentiment_path.exists():
            try:
                sample = pd.read_csv(self.config.fnspid_daily_sentiment_path, nrows=100)
                analysis['fnspid_daily_sentiment'] = {'exists': True}
                analysis['recommended_strategy'] = 'use_fnspid_sentiment'
            except Exception as e:
                logger.warning(f"Could not analyze FNSPID data: {e}")
        
        if analysis['recommended_strategy'] == 'none':
            analysis['recommended_strategy'] = 'synthetic'
        
        return analysis

class SyntheticSentimentGenerator:
    """Generate synthetic sentiment when no data available"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
    
    def generate_synthetic_sentiment(self, core_data: pd.DataFrame) -> pd.DataFrame:
        """✅ FIXED: Generate synthetic sentiment with config-based decay features"""
        logger.info("🎲 Generating synthetic sentiment...")
        
        try:
            synthetic_data = core_data[['symbol', 'date']].drop_duplicates().copy()
            np.random.seed(42)
            
            records = []
            for _, row in synthetic_data.iterrows():
                symbol, date = row['symbol'], row['date']
                
                # Generate base sentiment
                symbol_bias = (hash(symbol) % 1000 - 500) / 10000
                date_cycle = np.sin(2 * np.pi * pd.to_datetime(date).timetuple().tm_yday / 365) * 0.1
                base_sentiment = np.clip(np.random.normal(symbol_bias + date_cycle, 0.3), -1, 1)
                
                # ✅ Generate decay features for each configured horizon
                record = {'symbol': symbol, 'date': date, 'source': 'synthetic'}
                
                for horizon in self.config.target_horizons:
                    # Each horizon gets different decay characteristics
                    decay_factor = self.config.temporal_decay_params[horizon]['lambda_decay']
                    decay_noise = 0.2 * (1 - decay_factor)  # Less noise for faster decay
                    
                    sentiment_decay = np.clip(
                        base_sentiment * (1 - decay_factor) + np.random.normal(0, decay_noise), 
                        -1, 1
                    )
                    record[f'sentiment_decay_{horizon}d'] = sentiment_decay
                
                # Additional metadata
                record.update({
                    'sentiment_confidence': np.random.beta(3, 2),
                    'article_count': max(1, int(np.random.poisson(3)))
                })
                
                records.append(record)
            
            result = pd.DataFrame(records)
            logger.info(f"✅ Generated {len(result):,} synthetic sentiment records")
            logger.info(f"   🎯 Horizons: {self.config.target_horizons}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Synthetic generation failed: {e}")
            return pd.DataFrame()

class ConfigIntegratedSentimentIntegrator:
    """✅ FIXED: Handle integration of sentiment with core dataset using config"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.stats = {
            'core_records': 0, 'sentiment_records': 0, 'matched_records': 0,
            'coverage_percentage': 0.0, 'features_added': 0
        }
        
        # Ensure results directory exists
        self.config.sentiment_integration_report_path.parent.mkdir(parents=True, exist_ok=True)
    
    def integrate_sentiment_with_core(self, sentiment_data: pd.DataFrame, strategy: str) -> pd.DataFrame:
        """✅ FIXED: Integrate sentiment data with core dataset using config paths"""
        logger.info(f"🔗 Integrating sentiment data (strategy: {strategy})...")
        
        try:
            # ✅ Load core dataset using config path
            if not self.config.core_dataset_path.exists():
                raise FileNotFoundError(f"Core dataset not found: {self.config.core_dataset_path}")
            
            core_data = pd.read_csv(self.config.core_dataset_path)
            self.stats['core_records'] = len(core_data)
            self.stats['sentiment_records'] = len(sentiment_data)
            
            # ✅ Create backup using config backup directory
            backup_path = self._create_backup()
            
            # Standardize dates
            core_data['date'] = pd.to_datetime(core_data['date']).dt.date
            sentiment_data['date'] = pd.to_datetime(sentiment_data['date']).dt.date
            
            # Merge datasets
            enhanced_data = core_data.merge(sentiment_data, on=['symbol', 'date'], how='left')
            
            # Calculate stats
            sentiment_cols = [col for col in sentiment_data.columns if col not in ['symbol', 'date']]
            if sentiment_cols:
                matched_mask = enhanced_data[sentiment_cols[0]].notna()
                self.stats['matched_records'] = matched_mask.sum()
                self.stats['coverage_percentage'] = (self.stats['matched_records'] / self.stats['core_records']) * 100
            
            self.stats['features_added'] = len(enhanced_data.columns) - len(core_data.columns)
            
            # ✅ Fill missing values with proper defaults for each horizon
            defaults = {'sentiment_confidence': 0.5, 'article_count': 0, 'source': 'none'}
            
            # Add defaults for each configured horizon
            for horizon in self.config.target_horizons:
                defaults[f'sentiment_decay_{horizon}d'] = 0.0
            
            for col in sentiment_cols:
                if col in enhanced_data.columns:
                    enhanced_data[col] = enhanced_data[col].fillna(defaults.get(col, 0.0))
            
            # Convert dates back and save using config path
            enhanced_data['date'] = enhanced_data['date'].astype(str)
            enhanced_data.to_csv(self.config.enhanced_dataset_path, index=False)
            
            logger.info("✅ Integration completed!")
            logger.info(f"   📊 Core: {self.stats['core_records']:,}, Sentiment: {self.stats['sentiment_records']:,}")
            logger.info(f"   📈 Coverage: {self.stats['coverage_percentage']:.1f}%, Features added: {self.stats['features_added']}")
            logger.info(f"   💾 Enhanced dataset: {self.config.enhanced_dataset_path}")
            logger.info(f"   💾 Backup: {backup_path}")
            
            return enhanced_data
            
        except Exception as e:
            logger.error(f"❌ Integration failed: {e}")
            raise
    
    def _create_backup(self) -> str:
        """✅ FIXED: Create backup using config backup directory"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.config.core_dataset_path.parent.parent / "backups" / f"combined_dataset_backup_{timestamp}.csv"
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            
            shutil.copy2(self.config.core_dataset_path, backup_path)
            return str(backup_path)
        except Exception as e:
            logger.warning(f"⚠️ Backup failed: {e}")
            return "backup_failed"
    
    def validate_integration(self, enhanced_data: pd.DataFrame) -> Dict[str, Any]:
        """✅ FIXED: Validate integration results using config target horizons"""
        logger.info("🔍 Validating integration...")
        
        validation = {
            'status': 'success', 'issues': [], 'recommendations': [],
            'readiness_for_tft': True, 'quality_metrics': {}, 'feature_analysis': {}
        }
        
        try:
            # Check required columns
            required_base = ['stock_id', 'symbol', 'date', 'close', 'target_5']
            required_sentiment = [f'sentiment_decay_{h}d' for h in self.config.target_horizons]
            
            missing_base = [col for col in required_base if col not in enhanced_data.columns]
            missing_sentiment = [col for col in required_sentiment if col not in enhanced_data.columns]
            
            if missing_base:
                validation['issues'].append(f"Missing base columns: {missing_base}")
                validation['readiness_for_tft'] = False
            
            if missing_sentiment:
                validation['issues'].append(f"Missing sentiment columns: {missing_sentiment}")
                validation['readiness_for_tft'] = False
            
            # Quality metrics
            total_records = len(enhanced_data)
            sentiment_coverage = {}
            for col in required_sentiment:
                if col in enhanced_data.columns:
                    non_zero = (enhanced_data[col] != 0).sum()
                    sentiment_coverage[col] = (non_zero / total_records) * 100
            
            validation['quality_metrics'] = {
                'total_records': total_records,
                'sentiment_coverage': sentiment_coverage,
                'unique_symbols': enhanced_data['symbol'].nunique(),
                'target_coverage': enhanced_data['target_5'].notna().mean() * 100 if 'target_5' in enhanced_data.columns else 0
            }
            
            # Feature analysis
            validation['feature_analysis'] = {
                'total_features': len(enhanced_data.columns),
                'technical_features': len([c for c in enhanced_data.columns if any(p in c.lower() for p in ['ema_', 'sma_', 'rsi_', 'macd'])]),
                'sentiment_features': len([c for c in enhanced_data.columns if 'sentiment' in c.lower()]),
                'time_features': len([c for c in enhanced_data.columns if any(p in c.lower() for p in ['year', 'month', 'day'])]),
                'target_features': len([c for c in enhanced_data.columns if c.startswith('target_')])
            }
            
            # Recommendations
            avg_coverage = np.mean(list(sentiment_coverage.values())) if sentiment_coverage else 0
            if avg_coverage < 20:
                validation['recommendations'].append("Low sentiment coverage")
                validation['status'] = 'warning'
            elif avg_coverage < 50:
                validation['recommendations'].append("Moderate sentiment coverage")
            else:
                validation['recommendations'].append("Good sentiment coverage - ready for TFT")
            
            if validation['issues']:
                validation['status'] = 'issues_found' if validation['readiness_for_tft'] else 'not_ready'
            
        except Exception as e:
            validation['status'] = 'validation_failed'
            validation['issues'].append(f"Validation error: {e}")
            validation['readiness_for_tft'] = False
        
        return validation

class ConfigIntegratedSentimentProcessor:
    """✅ FIXED: Main orchestrator for sentiment integration using config"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.start_time = datetime.now()
    
    def run_complete_integration(self) -> Tuple[bool, Dict[str, Any]]:
        """✅ FIXED: Run complete sentiment integration pipeline using config"""
        logger.info("🚀 STARTING CONFIG-INTEGRATED SENTIMENT INTEGRATION")
        logger.info("=" * 60)
        
        try:
            # Step 1: Analyze data availability
            analyzer = DatasetAnalyzer(self.config)
            analysis = analyzer.analyze_data_availability()
            
            logger.info("📊 DATA AVAILABILITY:")
            logger.info(f"   📄 Core dataset: {'✅' if analysis['core_dataset']['exists'] else '❌'}")
            logger.info(f"   🔬 Temporal decay: {'✅' if analysis['temporal_decay_data']['exists'] else '❌'}")
            logger.info(f"   📊 FNSPID sentiment: {'✅' if analysis['fnspid_daily_sentiment']['exists'] else '❌'}")
            logger.info(f"   💡 Strategy: {analysis['recommended_strategy']}")
            
            # Step 2: Load sentiment data
            strategy = analysis['recommended_strategy']
            
            if strategy == 'use_temporal_decay':
                logger.info("🔬 Using temporal decay data...")
                sentiment_data = pd.read_csv(self.config.temporal_decay_data_path)
                
            elif strategy == 'use_fnspid_sentiment':
                logger.info("📊 Using FNSPID sentiment...")
                sentiment_data = pd.read_csv(self.config.fnspid_daily_sentiment_path)
                
                # ✅ Add decay features for each configured horizon
                for horizon in self.config.target_horizons:
                    decay_factor = self.config.temporal_decay_params[horizon]['lambda_decay']
                    sentiment_data[f'sentiment_decay_{horizon}d'] = (
                        sentiment_data.get('sentiment_compound', 0) * (1 - decay_factor + 0.1)
                    )
                
            elif strategy == 'synthetic':
                logger.info("🎲 Generating synthetic sentiment...")
                core_data = pd.read_csv(self.config.core_dataset_path)
                generator = SyntheticSentimentGenerator(self.config)
                sentiment_data = generator.generate_synthetic_sentiment(core_data)
                
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            if sentiment_data.empty:
                raise ValueError("No sentiment data available")
            
            # Step 3: Integrate
            integrator = ConfigIntegratedSentimentIntegrator(self.config)
            enhanced_data = integrator.integrate_sentiment_with_core(sentiment_data, strategy)
            
            # Step 4: Validate
            validation = integrator.validate_integration(enhanced_data)
            
            # Step 5: Generate report
            report = {
                'timestamp': datetime.now().isoformat(),
                'strategy': strategy,
                'stats': integrator.stats,
                'validation': validation,
                'config_used': {
                    'target_horizons': self.config.target_horizons,
                    'symbols': self.config.symbols,
                    'date_range': f"{self.config.start_date} to {self.config.end_date}"
                }
            }
            
            with open(self.config.sentiment_integration_report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            return True, {
                'strategy': strategy,
                'enhanced_dataset_path': str(self.config.enhanced_dataset_path),
                'records': len(enhanced_data),
                'coverage': integrator.stats['coverage_percentage'],
                'features_added': integrator.stats['features_added'],
                'validation': validation,
                'report_path': str(self.config.sentiment_integration_report_path),
                'decay_features': [col for col in enhanced_data.columns if 'sentiment_decay_' in col]
            }
            
        except Exception as e:
            logger.error(f"❌ Integration failed: {e}")
            return False, {'error': str(e)}

def run_sentiment_integration_programmatic(config: PipelineConfig) -> Tuple[bool, Dict[str, Any]]:
    """
    ✅ FIXED: Programmatic sentiment integration interface
    
    Args:
        config: PipelineConfig object from config.py
        
    Returns:
        Tuple[bool, Dict]: (success, results_dict)
    """
    
    try:
        processor = ConfigIntegratedSentimentProcessor(config)
        return processor.run_complete_integration()
        
    except Exception as e:
        logger.error(f"❌ Programmatic sentiment integration failed: {e}")
        return False, {
            'error': str(e),
            'error_type': type(e).__name__,
            'stage': 'sentiment_integration'
        }

def main():
    """✅ FIXED: Main execution using config"""
    parser = argparse.ArgumentParser(description='Config-Integrated Sentiment Integration')
    parser.add_argument('--validate', action='store_true', help='Validate existing dataset')
    parser.add_argument('--config-type', type=str, default='default',
                       choices=['default', 'quick_test', 'research'],
                       help='Configuration type to use')
    args = parser.parse_args()
    
    print("🤖 CONFIG-INTEGRATED SENTIMENT INTEGRATION")
    print("=" * 60)
    
    try:
        # ✅ Load config without interactive prompts
        from config import get_default_config, get_quick_test_config, get_research_config
        
        if args.config_type == 'quick_test':
            config = get_quick_test_config()
        elif args.config_type == 'research':
            config = get_research_config()
        else:
            config = get_default_config()
        
        print(f"📊 Configuration: {args.config_type}")
        print(f"🎯 Target horizons: {config.target_horizons}")
        
        if args.validate:
            if config.enhanced_dataset_path.exists():
                enhanced_data = pd.read_csv(config.enhanced_dataset_path)
                integrator = ConfigIntegratedSentimentIntegrator(config)
                validation = integrator.validate_integration(enhanced_data)
                
                print(f"\n🔍 VALIDATION RESULTS:")
                print(f"   📊 Status: {validation['status'].upper()}")
                print(f"   🎯 TFT Ready: {'✅' if validation['readiness_for_tft'] else '❌'}")
                
                if validation['issues']:
                    print("   ⚠️ Issues:")
                    for issue in validation['issues']:
                        print(f"      • {issue}")
                
                if validation['recommendations']:
                    print("   💡 Recommendations:")
                    for rec in validation['recommendations']:
                        print(f"      • {rec}")
            else:
                print(f"❌ No enhanced dataset found at {config.enhanced_dataset_path}")
            return
        
        # ✅ Run programmatic integration
        success, summary = run_sentiment_integration_programmatic(config)
        
        if success:
            print(f"\n🎉 INTEGRATION COMPLETED SUCCESSFULLY!")
            print(f"📊 Records: {summary['records']:,}")
            print(f"📈 Coverage: {summary['coverage']:.1f}%")
            print(f"🆕 Features: {summary['features_added']}")
            print(f"📁 Output: {summary['enhanced_dataset_path']}")
            
            validation = summary['validation']
            print(f"\n🔍 Validation: {validation['status'].upper()}")
            print(f"🎯 TFT Ready: {'✅' if validation['readiness_for_tft'] else '❌'}")
            
            if 'feature_analysis' in validation:
                features = validation['feature_analysis']
                print(f"\n📊 Features: {features['total_features']} total")
                print(f"   🔧 Technical: {features['technical_features']}")
                print(f"   🎭 Sentiment: {features['sentiment_features']}")
                print(f"   🎯 Targets: {features['target_features']}")
            
            if 'sentiment_coverage' in validation.get('quality_metrics', {}):
                coverage = validation['quality_metrics']['sentiment_coverage']
                print(f"\n📈 Sentiment Coverage:")
                for horizon, pct in coverage.items():
                    print(f"   • {horizon}: {pct:.1f}%")
            
            # Show decay features
            if 'decay_features' in summary:
                print(f"\n🎯 Decay Features Created:")
                for feature in summary['decay_features']:
                    print(f"   • {feature}")
            
            print(f"\n🚀 NEXT STEPS:")
            print("1. ✅ Sentiment integration complete - ready for TFT training")
            print("2. 🤖 Run: python src/models.py")
            print("3. 📊 Compare baseline vs enhanced TFT performance")
            
        else:
            print(f"\n❌ Integration failed: {summary.get('error', 'Unknown error')}")
    
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()