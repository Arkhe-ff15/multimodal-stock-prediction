#!/usr/bin/env python3
"""
TEMPORAL DECAY IMPLEMENTATION - CONFIG-INTEGRATED VERSION
=========================================================

✅ FIXES APPLIED:
- Proper config.py integration
- Removed all interactive prompts
- Standardized file paths using config
- Programmatic execution only
- Fixed input file paths
- Automated parameter configuration

CORE INNOVATION: 
sentiment_weighted = Σ(sentiment_i * exp(-λ_h * age_i)) / Σ(exp(-λ_h * age_i))

Where λ_h varies by forecasting horizon h (5, 30, 90 days)
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime, timedelta
import warnings
import logging
import json
import os
from pathlib import Path
from scipy import stats
import argparse

# ✅ FIXED: Proper config integration
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PipelineConfig, DecayParameters, create_decay_parameters_from_config, get_file_path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SentimentItem:
    """Individual sentiment measurement"""
    date: datetime
    score: float  # [-1, 1] sentiment score
    confidence: float  # [0, 1] confidence score
    article_count: int  # Number of articles aggregated
    source: str  # Data source identifier

class ConfigIntegratedTemporalDecayProcessor:
    """
    ✅ FIXED: Config-integrated temporal decay processor
    
    FIXES:
    - Uses centralized PipelineConfig
    - Standardized file paths from config
    - No interactive prompts
    - Proper error handling
    - Automated execution
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        
        # ✅ Create decay parameters from config
        self.decay_params = create_decay_parameters_from_config(config)
        
        # ✅ Use config paths
        self.input_sentiment_file = config.fnspid_daily_sentiment_path
        self.output_file = config.temporal_decay_data_path
        self.validation_report_file = config.temporal_decay_validation_path
        self.results_dir = config.temporal_decay_validation_path.parent
        
        # Ensure results directory exists
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.trading_calendar = None  # Optional trading calendar
        self.validation_stats = {}
        self.processing_stats = {
            'records_processed': 0,
            'sentiment_items_created': 0,
            'decay_calculations': 0,
            'start_time': datetime.now()
        }
        
        # Validate configuration
        self._validate_parameters()
        
        logger.info(f"🔬 Config-Integrated TemporalDecayProcessor initialized")
        logger.info(f"   🎯 Horizons: {list(self.decay_params.keys())} days")
        logger.info(f"   📈 Decay rates: {[p.lambda_decay for p in self.decay_params.values()]}")
        logger.info(f"   📁 Input: {self.input_sentiment_file}")
        logger.info(f"   📁 Output: {self.output_file}")
        
    def _validate_parameters(self):
        """Validate decay parameters to prevent overfitting"""
        required_horizons = self.config.target_horizons
        
        for horizon in required_horizons:
            if horizon not in self.decay_params:
                raise ValueError(f"Missing decay parameters for horizon {horizon}")
        
        # Check that decay rates follow expected pattern (faster decay for shorter horizons)
        lambdas = [(h, p.lambda_decay) for h, p in self.decay_params.items()]
        lambdas.sort()  # Sort by horizon
        
        for i in range(1, len(lambdas)):
            if lambdas[i][1] > lambdas[i-1][1]:
                warnings.warn(f"Decay rate increases with horizon: {lambdas[i-1]} -> {lambdas[i]}")
                
        logger.info("✅ Decay parameters validated")
    
    def calculate_trading_days(self, start_date: datetime, end_date: datetime) -> int:
        """Calculate number of trading days between dates"""
        if self.trading_calendar is not None:
            return len(self.trading_calendar[(self.trading_calendar >= start_date) & 
                                            (self.trading_calendar <= end_date)])
        else:
            # Approximate: 5/7 ratio for weekdays only (excludes holidays)
            total_days = (end_date - start_date).days
            trading_days = int(total_days * 5/7)
            return max(0, trading_days)
    
    def process_sentiment_for_symbol(self, symbol_data: pd.DataFrame, 
                                   horizon: int) -> pd.DataFrame:
        """ 
        Apply temporal decay to sentiment history for specific symbol and horizon
        
        Args:
            symbol_data: DataFrame with sentiment data for one symbol
            horizon: Forecasting horizon (5, 30, or 90 days)
            
        Returns:
            DataFrame with decay-weighted sentiment features
        """
        if horizon not in self.decay_params:
            raise ValueError(f"Unsupported horizon: {horizon}")
        
        params = self.decay_params[horizon]
        symbol = symbol_data['symbol'].iloc[0] if 'symbol' in symbol_data.columns else 'unknown'
        
        # Sort by date
        symbol_data = symbol_data.sort_values('date').reset_index(drop=True)
        
        # Convert dates to datetime
        symbol_data['date'] = pd.to_datetime(symbol_data['date'])
        
        # Create SentimentItem objects
        sentiment_history = []
        for _, row in symbol_data.iterrows():
            # ✅ Handle different possible column names for sentiment
            sentiment_score = row.get('sentiment_compound', 
                                    row.get('sentiment_score', 
                                           row.get('confidence_weighted_sentiment', 0.0)))
            confidence = row.get('confidence', 
                               row.get('sentiment_confidence', 0.5))
            article_count = row.get('article_count', 1)
            
            sentiment_history.append(SentimentItem(
                date=row['date'].to_pydatetime(),
                score=sentiment_score,
                confidence=confidence,
                article_count=article_count,
                source='fnspid'
            ))
        
        # Process each date for this horizon
        results = []
        for i, current_date in enumerate(symbol_data['date']):
            current_datetime = current_date.to_pydatetime()
            
            # Get sentiment history up to current date
            relevant_history = [s for s in sentiment_history if s.date <= current_datetime]
            
            # Apply temporal decay
            weighted_sentiment, metadata = self.process_sentiment(
                relevant_history, current_datetime, horizon
            )
            
            # Create result record
            result = {
                'symbol': symbol,
                'date': current_date.strftime('%Y-%m-%d'),
                f'sentiment_decay_{horizon}d': weighted_sentiment,
                f'sentiment_weight_sum_{horizon}d': metadata['total_weight'],
                f'sentiment_effective_count_{horizon}d': metadata['filtered_count'],
                f'sentiment_lookback_days_{horizon}d': metadata['effective_lookback'],
                f'sentiment_quality_{horizon}d': metadata['quality'],
                f'sentiment_concentration_{horizon}d': metadata['weight_concentration']
            }
            
            results.append(result)
            self.processing_stats['decay_calculations'] += 1
        
        return pd.DataFrame(results)
    
    def process_sentiment(self, sentiment_history: List[SentimentItem], 
                         current_date: datetime, 
                         horizon: int) -> Tuple[float, Dict]:
        """ 
        Core decay calculation for specific date and horizon
        
        Args:
            sentiment_history: List of historical sentiment measurements
            current_date: Current prediction date
            horizon: Forecasting horizon (5, 30, or 90 days)
            
        Returns:
            Tuple of (weighted_sentiment, metadata)
        """
        params = self.decay_params[horizon]
        
        # Filter sentiment within lookback window
        cutoff_date = current_date - timedelta(days=params.lookback_days)
        recent_sentiment = [s for s in sentiment_history 
                           if s.date >= cutoff_date and s.date <= current_date]
        
        # Quality filtering - prevent overfitting on noisy data
        high_quality_sentiment = [
            s for s in recent_sentiment 
            if (s.confidence >= params.confidence_threshold and 
                s.article_count >= params.min_sentiment_count)
        ]
        
        if len(high_quality_sentiment) == 0:
            return 0.0, {
                'quality': 'insufficient_data', 
                'total_weight': 0.0,
                'filtered_count': 0,
                'raw_count': len(recent_sentiment),
                'effective_lookback': 0,
                'weight_concentration': 0.0,
                'horizon': horizon,
                'lambda_decay': params.lambda_decay
            }
        
        # Apply exponential decay weighting - CORE INNOVATION
        weighted_sentiment = 0.0
        total_weight = 0.0
        weights_list = []
        ages_list = []
        
        for sentiment_item in high_quality_sentiment:
            # Calculate age in trading days
            age_days = self.calculate_trading_days(sentiment_item.date, current_date)
            
            # Exponential decay formula: exp(-λ_h * age)
            weight = np.exp(-params.lambda_decay * age_days)
            
            # Weight by confidence to reduce noise (additional innovation)
            confidence_weight = sentiment_item.confidence
            adjusted_weight = weight * confidence_weight
            
            # Accumulate weighted sentiment
            weighted_sentiment += adjusted_weight * sentiment_item.score
            total_weight += adjusted_weight
            
            weights_list.append(weight)
            ages_list.append(age_days)
        
        # Normalize by total weight
        final_sentiment = weighted_sentiment / total_weight if total_weight > 0 else 0.0
        
        # Calculate weight concentration (how spread out the weights are)
        weight_concentration = np.std(weights_list) if len(weights_list) > 1 else 0.0
        
        # Compile metadata for analysis and validation
        metadata = {
            'quality': 'sufficient',
            'total_weight': float(total_weight),
            'raw_count': len(recent_sentiment),
            'filtered_count': len(high_quality_sentiment),
            'effective_lookback': max(ages_list) if ages_list else 0,
            'weight_concentration': float(weight_concentration),
            'horizon': horizon,
            'lambda_decay': params.lambda_decay,
            'mean_age': np.mean(ages_list) if ages_list else 0,
            'oldest_sentiment_days': max(ages_list) if ages_list else 0
        }
        
        return float(final_sentiment), metadata
    
    def batch_process_all_symbols(self, sentiment_data: pd.DataFrame) -> pd.DataFrame:
        """
        ✅ FIXED: Process temporal decay for all symbols and horizons
        """
        logger.info("🔄 Processing temporal decay for all symbols and horizons...")
        
        all_results = []
        symbols = sentiment_data['symbol'].unique()
        total_symbols = len(symbols)
        
        for i, symbol in enumerate(symbols):
            logger.info(f"   🏢 Processing {symbol} ({i+1}/{total_symbols})...")
            
            # Get data for this symbol
            symbol_data = sentiment_data[sentiment_data['symbol'] == symbol].copy()
            
            # Process each horizon
            symbol_results = None
            for horizon in self.decay_params.keys():
                horizon_results = self.process_sentiment_for_symbol(symbol_data, horizon)
                
                if symbol_results is None:
                    symbol_results = horizon_results[['symbol', 'date']].copy()
                
                # Add horizon-specific columns
                horizon_cols = [col for col in horizon_results.columns 
                              if f'{horizon}d' in col]
                for col in horizon_cols:
                    symbol_results[col] = horizon_results[col]
            
            all_results.append(symbol_results)
            self.processing_stats['records_processed'] += len(symbol_data)
        
        # Combine all results
        final_results = pd.concat(all_results, ignore_index=True)
        
        logger.info(f"✅ Temporal decay processing completed")
        logger.info(f"   📊 Records processed: {self.processing_stats['records_processed']:,}")
        logger.info(f"   🧮 Decay calculations: {self.processing_stats['decay_calculations']:,}")
        logger.info(f"   📈 Final records: {len(final_results):,}")
        
        return final_results
    
    def validate_decay_patterns(self, processed_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate that decay patterns make intuitive sense - prevents overfitting
        """
        logger.info("🔍 Validating temporal decay patterns...")
        
        validation_results = {
            'overall_stats': {},
            'horizon_analysis': {},
            'decay_effectiveness': {},
            'statistical_tests': {},
            'quality_metrics': {}
        }
        
        try:
            # Overall statistics
            validation_results['overall_stats'] = {
                'total_records': len(processed_data),
                'symbols_count': processed_data['symbol'].nunique(),
                'date_range': {
                    'start': processed_data['date'].min(),
                    'end': processed_data['date'].max(),
                    'days': (pd.to_datetime(processed_data['date'].max()) - 
                            pd.to_datetime(processed_data['date'].min())).days
                }
            }
            
            # Analyze each horizon
            for horizon in self.decay_params.keys():
                decay_col = f'sentiment_decay_{horizon}d'
                weight_col = f'sentiment_weight_sum_{horizon}d'
                
                if decay_col in processed_data.columns:
                    decay_values = processed_data[decay_col].dropna()
                    weight_values = processed_data[weight_col].dropna()
                    
                    horizon_stats = {
                        'mean_decay_sentiment': float(decay_values.mean()),
                        'std_decay_sentiment': float(decay_values.std()),
                        'decay_range': [float(decay_values.min()), float(decay_values.max())],
                        'mean_weight_sum': float(weight_values.mean()),
                        'effective_coverage': float((decay_values != 0).mean()),
                        'lambda_parameter': self.decay_params[horizon].lambda_decay
                    }
                    
                    validation_results['horizon_analysis'][horizon] = horizon_stats
            
            # Test decay effectiveness
            horizons = sorted(self.decay_params.keys())
            lambdas = [self.decay_params[h].lambda_decay for h in horizons]
            
            # Monotonicity test
            is_monotonic = all(lambdas[i] >= lambdas[i+1] for i in range(len(lambdas)-1))
            validation_results['statistical_tests']['monotonic_decay_rates'] = is_monotonic
            
            # Correlation test
            correlation_tests = {}
            for horizon in horizons:
                decay_col = f'sentiment_decay_{horizon}d'
                if decay_col in processed_data.columns:
                    sample_data = processed_data.dropna(subset=[decay_col]).copy()
                    sample_data['time_index'] = pd.to_datetime(sample_data['date']).map(lambda x: x.toordinal())
                    
                    if len(sample_data) > 10:
                        correlation, p_value = stats.spearmanr(
                            sample_data['time_index'], 
                            sample_data[decay_col]
                        )
                        correlation_tests[horizon] = {
                            'correlation': float(correlation),
                            'p_value': float(p_value),
                            'significant': p_value < 0.05
                        }
            
            validation_results['statistical_tests']['time_correlations'] = correlation_tests
            
            # Quality metrics
            quality_metrics = {}
            for horizon in horizons:
                quality_col = f'sentiment_quality_{horizon}d'
                if quality_col in processed_data.columns:
                    quality_dist = processed_data[quality_col].value_counts()
                    quality_metrics[horizon] = {
                        'sufficient_data_pct': float(quality_dist.get('sufficient', 0) / len(processed_data) * 100),
                        'insufficient_data_pct': float(quality_dist.get('insufficient_data', 0) / len(processed_data) * 100)
                    }
            
            validation_results['quality_metrics'] = quality_metrics
            
            # Overall validation score
            validation_score = 0
            if validation_results['statistical_tests']['monotonic_decay_rates']:
                validation_score += 25
            
            avg_coverage = np.mean([
                h['effective_coverage'] for h in validation_results['horizon_analysis'].values()
            ])
            validation_score += min(25, avg_coverage * 25)
            
            significant_correlations = sum([
                1 for t in correlation_tests.values() if abs(t['correlation']) > 0.1
            ])
            validation_score += min(25, significant_correlations * 8)
            
            avg_sufficient_data = np.mean([
                q['sufficient_data_pct'] for q in quality_metrics.values()
            ])
            validation_score += min(25, avg_sufficient_data / 4)
            
            validation_results['overall_validation_score'] = validation_score
            
        except Exception as e:
            logger.error(f"❌ Validation failed: {e}")
            validation_results['validation_error'] = str(e)
        
        return validation_results

def load_fnspid_sentiment_data(config: PipelineConfig) -> pd.DataFrame:
    """✅ FIXED: Load FNSPID sentiment data using config paths"""
    logger.info(f"📥 Loading FNSPID sentiment data...")
    
    input_file = config.fnspid_daily_sentiment_path
    
    if not input_file.exists():
        raise FileNotFoundError(f"FNSPID sentiment file not found: {input_file}")
    
    try:
        data = pd.read_csv(input_file)
        
        # Validate required columns
        required_cols = ['symbol', 'date']
        missing_cols = [col for col in required_cols if col not in data.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check for sentiment columns (flexible)
        sentiment_cols = [col for col in data.columns if any(
            pattern in col.lower() for pattern in ['sentiment', 'confidence']
        )]
        
        if not sentiment_cols:
            raise ValueError("No sentiment columns found")
        
        logger.info(f"✅ Loaded sentiment data: {data.shape}")
        logger.info(f"   🏢 Symbols: {data['symbol'].nunique()}")
        logger.info(f"   📅 Date range: {data['date'].min()} to {data['date'].max()}")
        logger.info(f"   🎭 Sentiment columns: {sentiment_cols}")
        
        return data
        
    except Exception as e:
        logger.error(f"❌ Failed to load sentiment data: {e}")
        raise

def run_temporal_decay_processing_programmatic(config: PipelineConfig) -> Tuple[bool, Dict[str, Any]]:
    """
    ✅ FIXED: Programmatic temporal decay processing (no interactive prompts)
    
    Args:
        config: PipelineConfig object from config.py
        
    Returns:
        Tuple[bool, Dict]: (success, results_dict)
    """
    
    try:
        logger.info("🔬 Starting programmatic temporal decay processing")
        
        # Load sentiment data
        sentiment_data = load_fnspid_sentiment_data(config)
        
        # Initialize processor
        processor = ConfigIntegratedTemporalDecayProcessor(config)
        
        # Process temporal decay
        logger.info("🔄 Processing temporal decay...")
        start_time = datetime.now()
        
        processed_data = processor.batch_process_all_symbols(sentiment_data)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        processor.processing_stats['processing_time'] = processing_time
        
        # Validate results
        logger.info("🔍 Validating decay patterns...")
        validation_results = processor.validate_decay_patterns(processed_data)
        
        # Save results
        logger.info("💾 Saving temporal decay results...")
        processed_data.to_csv(processor.output_file, index=False)
        
        # Save validation report
        validation_report = {
            'validation_results': validation_results,
            'processing_stats': processor.processing_stats,
            'timestamp': datetime.now().isoformat(),
            'config_used': {
                'decay_parameters': config.temporal_decay_params,
                'target_horizons': config.target_horizons
            }
        }
        
        with open(processor.validation_report_file, 'w') as f:
            json.dump(validation_report, f, indent=2, default=str)
        
        validation_score = validation_results.get('overall_validation_score', 0)
        
        return True, {
            'status': 'completed',
            'stage': 'temporal_decay',
            'processing_summary': {
                'input_records': len(sentiment_data),
                'output_records': len(processed_data),
                'symbols_processed': processed_data['symbol'].nunique(),
                'processing_time': processing_time,
                'decay_calculations': processor.processing_stats['decay_calculations']
            },
            'validation': {
                'overall_score': validation_score,
                'quality_status': 'excellent' if validation_score >= 75 else 'good' if validation_score >= 50 else 'needs_improvement'
            },
            'output_files': {
                'temporal_decay_data': str(processor.output_file),
                'validation_report': str(processor.validation_report_file)
            },
            'decay_features_created': [col for col in processed_data.columns if 'sentiment_decay_' in col],
            'next_stage_ready': True
        }
        
    except Exception as e:
        logger.error(f"❌ Temporal decay processing failed: {e}")
        return False, {
            'error': str(e),
            'error_type': type(e).__name__,
            'stage': 'temporal_decay',
            'suggestion': 'Check that FNSPID processing completed successfully'
        }

# =============================================================================
# MAIN EXECUTION (NO INTERACTIVE PROMPTS)
# =============================================================================

def main():
    """
    ✅ FIXED: Main execution without interactive prompts
    """
    
    parser = argparse.ArgumentParser(
        description='Temporal Decay Processing - Config Integrated',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--config-type', type=str, default='default',
                       choices=['default', 'quick_test', 'research'],
                       help='Configuration type to use')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only run validation on existing processed data')
    
    args = parser.parse_args()
    
    print("🔬 TEMPORAL DECAY PROCESSING - CONFIG INTEGRATED")
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
        print(f"🎯 Horizons: {config.target_horizons}")
        print(f"📈 Decay rates: {[config.temporal_decay_params[h]['lambda_decay'] for h in config.target_horizons]}")
        
        if args.validate_only:
            # Just validate existing processed data
            if config.temporal_decay_data_path.exists():
                processed_data = pd.read_csv(config.temporal_decay_data_path)
                processor = ConfigIntegratedTemporalDecayProcessor(config)
                validation_results = processor.validate_decay_patterns(processed_data)
                
                print(f"\n🔍 VALIDATION RESULTS:")
                print(f"   📊 Overall Score: {validation_results.get('overall_validation_score', 0):.0f}/100")
            else:
                print(f"❌ No processed data found at {config.temporal_decay_data_path}")
            return
        
        # ✅ Run programmatic processing
        success, results = run_temporal_decay_processing_programmatic(config)
        
        if success:
            print(f"\n✅ TEMPORAL DECAY PROCESSING COMPLETED!")
            print(f"   ⏱️ Processing time: {results['processing_summary']['processing_time']:.1f} seconds")
            print(f"   📊 Output records: {results['processing_summary']['output_records']:,}")
            print(f"   🧮 Decay calculations: {results['processing_summary']['decay_calculations']:,}")
            print(f"   📊 Validation score: {results['validation']['overall_score']:.0f}/100")
            print(f"   📁 Output file: {results['output_files']['temporal_decay_data']}")
            
            # Show decay features created
            decay_features = results['decay_features_created']
            print(f"\n🎯 Decay Features Created ({len(decay_features)}):")
            for feature in decay_features:
                print(f"   • {feature}")
            
            print(f"\n🔄 NEXT STEPS:")
            print("1. ✅ Temporal decay features generated")
            print("2. 🔗 Run sentiment integration:")
            print("   python src/sentiment.py")
            print("3. 🤖 Train enhanced TFT models")
            
        else:
            print(f"\n❌ Temporal decay processing failed: {results['error']}")
            print(f"💡 Suggestion: {results.get('suggestion', 'Check logs for details')}")
        
    except Exception as e:
        print(f"\n❌ Execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()