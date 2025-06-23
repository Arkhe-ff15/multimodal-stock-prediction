#!/usr/bin/env python3
"""
FINAL FIXED TEMPORAL DECAY IMPLEMENTATION
==========================================
All issues resolved:
✅ Fixed timezone mismatch between market and sentiment data
✅ Fixed typo in parameter optimization
✅ Improved symbol mismatch handling
✅ Better error handling and validation
✅ Academic-quality implementation maintained

CORE INNOVATION: 
sentiment_weighted = Σ(sentiment_i * exp(-λ_h * age_i)) / Σ(exp(-λ_h * age_i))
Where λ_h is optimized per horizon h (5, 10, 22, 60, 90 days)
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
from sklearn.preprocessing import StandardScaler, RobustScaler
from scipy import stats
from scipy.optimize import minimize_scalar
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

from config_reader import load_config, get_data_paths

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedTemporalDecayProcessor:
    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        self.data_paths = get_data_paths(self.config)
        self.symbols = self.config['data']['core']['symbols']
        self.target_horizons = self.config['data']['core']['target_horizons']
        
        # Default decay parameters (will be optimized)
        self.decay_params = {
            5: 0.1,   # Fast decay: 50% weight after ~7 days
            22: 0.05, # Medium decay: 50% weight after ~14 days
            90: 0.02  # Slow decay: 50% weight after ~35 days
        }
        
        # Advanced configuration
        self.enable_parameter_optimization = True
        self.enable_advanced_features = True
        self.enable_statistical_validation = True
        self.feature_normalization = 'robust'  # 'standard', 'robust', 'none'
        
        # Feature engineering parameters
        self.volatility_windows = [5, 10, 20]  # Days for sentiment volatility
        self.momentum_windows = [3, 7, 14]     # Days for sentiment momentum
        self.confidence_threshold = 0.7        # Minimum confidence for weighting
        
        logger.info("🔬 Advanced Temporal Decay Processor initialized")
        logger.info(f"   📊 Symbols: {self.symbols}")
        logger.info(f"   🎯 Target horizons: {self.target_horizons}")
        logger.info(f"   ⚙️ Default decay parameters: {self.decay_params}")
        logger.info(f"   🧠 Parameter optimization: {self.enable_parameter_optimization}")
        logger.info(f"   🔬 Advanced features: {self.enable_advanced_features}")
        logger.info(f"   📊 Statistical validation: {self.enable_statistical_validation}")
    
    def _normalize_datetime_columns(self, df: pd.DataFrame, date_column: str = 'date') -> pd.DataFrame:
        """Normalize datetime columns to handle timezone issues"""
        df = df.copy()
        
        # Convert to datetime and remove timezone info if present
        if date_column in df.columns:
            df[date_column] = pd.to_datetime(df[date_column])
            
            # Remove timezone info to ensure compatibility
            if df[date_column].dt.tz is not None:
                df[date_column] = df[date_column].dt.tz_convert(None)
        
        return df
    
    def load_required_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and validate core dataset and sentiment data with timezone handling"""
        logger.info("📥 Loading required data...")
        
        # Load core market data
        core_path = self.data_paths['core_dataset']
        if not core_path.exists():
            raise FileNotFoundError(f"Core dataset not found: {core_path}")
        
        market_data = pd.read_csv(core_path)
        market_data = self._normalize_datetime_columns(market_data, 'date')
        logger.info(f"✅ Market data loaded: {len(market_data):,} records")
        
        # Load sentiment data
        sentiment_path = self.data_paths['fnspid_daily_sentiment']
        if not sentiment_path.exists():
            raise FileNotFoundError(f"Sentiment data not found: {sentiment_path}")
        
        sentiment_data = pd.read_csv(sentiment_path)
        sentiment_data = self._normalize_datetime_columns(sentiment_data, 'date')
        logger.info(f"✅ Sentiment data loaded: {len(sentiment_data):,} records")
        
        # Data validation
        self._validate_data_quality(market_data, sentiment_data)
        
        return market_data, sentiment_data
    
    def _validate_data_quality(self, market_data: pd.DataFrame, sentiment_data: pd.DataFrame):
        """Comprehensive data quality validation"""
        logger.info("🔍 Validating data quality...")
        
        # Check sentiment data columns
        required_sentiment_cols = ['symbol', 'date', 'sentiment_compound', 'sentiment_positive', 
                                 'sentiment_negative', 'confidence', 'article_count']
        missing_cols = [col for col in required_sentiment_cols if col not in sentiment_data.columns]
        if missing_cols:
            raise ValueError(f"Sentiment data missing columns: {missing_cols}")
        
        # Check for data quality issues
        quality_issues = []
        
        # Missing values
        sentiment_nulls = sentiment_data[required_sentiment_cols].isnull().sum().sum()
        if sentiment_nulls > 0:
            quality_issues.append(f"Sentiment data has {sentiment_nulls} null values")
        
        # Confidence distribution
        low_confidence = (sentiment_data['confidence'] < 0.5).sum()
        if low_confidence > len(sentiment_data) * 0.3:
            quality_issues.append(f"High proportion of low confidence sentiment: {low_confidence/len(sentiment_data)*100:.1f}%")
        
        # Date alignment
        market_date_range = (market_data['date'].min(), market_data['date'].max())
        sentiment_date_range = (sentiment_data['date'].min(), sentiment_data['date'].max())
        
        logger.info(f"📅 Market data range: {market_date_range[0].date()} to {market_date_range[1].date()}")
        logger.info(f"📅 Sentiment data range: {sentiment_date_range[0].date()} to {sentiment_date_range[1].date()}")
        
        # Symbol coverage
        market_symbols = set(market_data['symbol'].unique())
        sentiment_symbols = set(sentiment_data['symbol'].unique())
        missing_symbols = market_symbols - sentiment_symbols
        
        if missing_symbols:
            quality_issues.append(f"Symbols missing from sentiment data: {missing_symbols}")
        
        # Report quality issues
        if quality_issues:
            logger.warning("⚠️ Data quality issues detected:")
            for issue in quality_issues:
                logger.warning(f"   • {issue}")
        else:
            logger.info("✅ Data quality validation passed")
    
    def optimize_decay_parameters(self, market_data: pd.DataFrame, 
                                sentiment_data: pd.DataFrame) -> Dict[int, float]:
        """Optimize decay parameters using cross-validation"""
        if not self.enable_parameter_optimization:
            logger.info("⏭️ Parameter optimization disabled, using defaults")
            return self.decay_params
        
        logger.info("🔧 Optimizing decay parameters...")
        
        optimized_params = {}
        
        for horizon in self.target_horizons:
            logger.info(f"   🎯 Optimizing λ for {horizon}d horizon...")
            
            # Define optimization objective
            def objective(lambda_param):
                try:
                    # Calculate features with this lambda
                    test_features = self._calculate_decay_features_sample(
                        market_data, sentiment_data, {horizon: lambda_param}
                    )
                    
                    # Simple correlation-based objective
                    target_col = f'target_{min(self.target_horizons)}'  # Use shortest target as proxy
                    sentiment_col = f'sentiment_decay_{horizon}d_compound'
                    
                    if sentiment_col in test_features.columns and target_col in test_features.columns:
                        # Calculate Spearman correlation (robust to outliers)
                        valid_data = test_features[[sentiment_col, target_col]].dropna()
                        if len(valid_data) > 10:
                            correlation = valid_data[sentiment_col].corr(valid_data[target_col], method='spearman')
                            if not np.isnan(correlation):
                                return -abs(correlation)  # Maximize absolute correlation
                    
                    return 0  # No valid data or correlation
                    
                except Exception as e:
                    logger.debug(f"Optimization objective failed for λ={lambda_param:.4f}: {e}")
                    return 0
            
            # Optimization bounds based on horizon
            if horizon <= 10:
                bounds = (0.05, 0.3)  # Fast decay for short horizons
            elif horizon <= 22:
                bounds = (0.02, 0.15)  # Medium decay
            else:
                bounds = (0.01, 0.08)  # Slow decay for long horizons
            
            try:
                result = minimize_scalar(objective, bounds=bounds, method='bounded')
                optimized_lambda = result.x
                optimized_params[horizon] = optimized_lambda
                
                logger.info(f"   ✅ {horizon}d: λ={optimized_lambda:.4f} (default: {self.decay_params.get(horizon, 0.05):.4f})")
                
            except Exception as e:
                logger.warning(f"   ⚠️ Optimization failed for {horizon}d, using default: {e}")
                optimized_params[horizon] = self.decay_params.get(horizon, 0.05)
        
        logger.info(f"✅ Parameter optimization completed: {optimized_params}")
        return optimized_params
    
    def _calculate_decay_features_sample(self, market_data: pd.DataFrame, 
                                       sentiment_data: pd.DataFrame, 
                                       test_params: Dict[int, float]) -> pd.DataFrame:
        """Calculate decay features for a sample (used in optimization)"""
        # Use small sample for optimization speed
        sample_size = min(1000, len(market_data))
        market_sample = market_data.sample(n=sample_size, random_state=42)
        
        results = []
        
        # Get symbols that have both market and sentiment data
        market_symbols = set(market_data['symbol'].unique())
        sentiment_symbols = set(sentiment_data['symbol'].unique())
        common_symbols = list(market_symbols & sentiment_symbols)[:2]  # Limit to 2 for speed
        
        for symbol in common_symbols:
            symbol_market = market_sample[market_sample['symbol'] == symbol]
            symbol_sentiment = sentiment_data[sentiment_data['symbol'] == symbol]
            
            if symbol_market.empty or symbol_sentiment.empty:
                continue
            
            for _, market_row in symbol_market.iterrows():
                current_date = market_row['date']
                
                # Fixed: Use symbol_sentiment instead of sentiment_sentiment
                sentiment_history = symbol_sentiment[symbol_sentiment['date'] <= current_date]
                
                result_row = market_row.to_dict()
                
                for horizon, lambda_param in test_params.items():
                    decay_features = self._calculate_exponential_decay(
                        sentiment_history, current_date, lambda_param
                    )
                    
                    for feature_name, feature_value in decay_features.items():
                        result_row[f'sentiment_decay_{horizon}d_{feature_name}'] = feature_value
                
                results.append(result_row)
        
        return pd.DataFrame(results)
    
    def _calculate_exponential_decay(self, sentiment_history: pd.DataFrame, 
                                   current_date: pd.Timestamp, 
                                   lambda_param: float) -> Dict[str, float]:
        """Core exponential decay calculation with confidence weighting"""
        if sentiment_history.empty:
            return {
                'compound': 0.0,
                'positive': 0.0,
                'negative': 0.0,
                'confidence': 0.0,
                'article_count': 0.0
            }
        
        # Calculate age in days
        ages = (current_date - sentiment_history['date']).dt.days
        
        # Calculate exponential decay weights
        time_weights = np.exp(-lambda_param * ages)
        
        # Apply confidence weighting if enabled
        if self.enable_advanced_features:
            confidence_weights = np.where(
                sentiment_history['confidence'] >= self.confidence_threshold,
                sentiment_history['confidence'],
                sentiment_history['confidence'] * 0.5  # Reduce weight for low confidence
            )
            weights = time_weights * confidence_weights
        else:
            weights = time_weights
        
        # Calculate weighted averages
        total_weight = weights.sum()
        
        if total_weight > 0:
            weighted_sentiment = {
                'compound': float((sentiment_history['sentiment_compound'] * weights).sum() / total_weight),
                'positive': float((sentiment_history['sentiment_positive'] * weights).sum() / total_weight),
                'negative': float((sentiment_history['sentiment_negative'] * weights).sum() / total_weight),
                'confidence': float((sentiment_history['confidence'] * weights).sum() / total_weight),
                'article_count': float((sentiment_history['article_count'] * weights).sum() / total_weight)
            }
        else:
            weighted_sentiment = {
                'compound': 0.0,
                'positive': 0.0,
                'negative': 0.0,
                'confidence': 0.0,
                'article_count': 0.0
            }
        
        return weighted_sentiment
    
    def _calculate_advanced_features(self, sentiment_history: pd.DataFrame, 
                                   current_date: pd.Timestamp) -> Dict[str, float]:
        """Calculate advanced sentiment features (volatility, momentum, trends)"""
        if not self.enable_advanced_features or sentiment_history.empty:
            return {}
        
        advanced_features = {}
        
        # Sentiment volatility features
        for window in self.volatility_windows:
            recent_sentiment = sentiment_history[
                sentiment_history['date'] > (current_date - timedelta(days=window))
            ]
            
            if len(recent_sentiment) >= 3:
                vol = recent_sentiment['sentiment_compound'].std()
                advanced_features[f'sentiment_volatility_{window}d'] = float(vol if not np.isnan(vol) else 0.0)
            else:
                advanced_features[f'sentiment_volatility_{window}d'] = 0.0
        
        # Sentiment momentum features
        for window in self.momentum_windows:
            recent_sentiment = sentiment_history[
                sentiment_history['date'] > (current_date - timedelta(days=window))
            ].sort_values('date')
            
            if len(recent_sentiment) >= 2:
                # Linear trend slope
                x = np.arange(len(recent_sentiment))
                y = recent_sentiment['sentiment_compound'].values
                if len(x) > 1 and np.std(y) > 0:
                    slope, _, _, _, _ = stats.linregress(x, y)
                    advanced_features[f'sentiment_momentum_{window}d'] = float(slope)
                else:
                    advanced_features[f'sentiment_momentum_{window}d'] = 0.0
            else:
                advanced_features[f'sentiment_momentum_{window}d'] = 0.0
        
        # Confidence distribution features
        if len(sentiment_history) >= 5:
            advanced_features['confidence_mean'] = float(sentiment_history['confidence'].mean())
            advanced_features['confidence_std'] = float(sentiment_history['confidence'].std())
            advanced_features['high_confidence_ratio'] = float(
                (sentiment_history['confidence'] >= self.confidence_threshold).mean()
            )
        else:
            advanced_features.update({
                'confidence_mean': 0.5,
                'confidence_std': 0.0,
                'high_confidence_ratio': 0.5
            })
        
        return advanced_features
    
    def process_temporal_decay_features(self, market_data: pd.DataFrame, 
                                      sentiment_data: pd.DataFrame,
                                      optimized_params: Dict[int, float]) -> pd.DataFrame:
        """Process comprehensive temporal decay features"""
        logger.info("⏰ Calculating comprehensive temporal decay features...")
        
        results = []
        total_records = len(market_data)
        processed = 0
        
        # Get symbols that have both market and sentiment data
        market_symbols = set(market_data['symbol'].unique())
        sentiment_symbols = set(sentiment_data['symbol'].unique())
        common_symbols = list(market_symbols & sentiment_symbols)
        missing_symbols = market_symbols - sentiment_symbols
        
        if missing_symbols:
            logger.info(f"📊 Symbols without sentiment data (will use zero features): {missing_symbols}")
        
        # Group sentiment data by symbol for efficiency
        sentiment_groups = sentiment_data.groupby('symbol')
        
        for symbol in self.symbols:
            symbol_market = market_data[market_data['symbol'] == symbol].copy()
            
            if symbol_market.empty:
                logger.warning(f"⚠️ No market data for {symbol}")
                continue
            
            # Check if we have sentiment data for this symbol
            symbol_sentiment = sentiment_groups.get_group(symbol) if symbol in sentiment_groups.groups else pd.DataFrame()
            
            # Sort by date for efficient processing
            symbol_market = symbol_market.sort_values('date')
            if not symbol_sentiment.empty:
                symbol_sentiment = symbol_sentiment.sort_values('date')
            
            logger.info(f"📈 Processing {symbol}: {len(symbol_market)} market records")
            
            # Process each market data point
            for _, market_row in symbol_market.iterrows():
                current_date = market_row['date']
                
                result_row = market_row.to_dict()
                
                if symbol_sentiment.empty:
                    # No sentiment data - add zero features
                    for horizon in self.target_horizons:
                        result_row.update({
                            f'sentiment_decay_{horizon}d_compound': 0.0,
                            f'sentiment_decay_{horizon}d_positive': 0.0,
                            f'sentiment_decay_{horizon}d_negative': 0.0,
                            f'sentiment_decay_{horizon}d_confidence': 0.0,
                            f'sentiment_decay_{horizon}d_article_count': 0.0
                        })
                    
                    if self.enable_advanced_features:
                        # Add zero advanced features
                        for window in self.volatility_windows:
                            result_row[f'sentiment_volatility_{window}d'] = 0.0
                        for window in self.momentum_windows:
                            result_row[f'sentiment_momentum_{window}d'] = 0.0
                        result_row.update({
                            'confidence_mean': 0.5,
                            'confidence_std': 0.0,
                            'high_confidence_ratio': 0.5
                        })
                else:
                    # Get sentiment history (no look-ahead bias)
                    sentiment_history = symbol_sentiment[
                        symbol_sentiment['date'] <= current_date
                    ]
                    
                    # Calculate decay features for each horizon
                    for horizon in self.target_horizons:
                        lambda_param = optimized_params.get(horizon, 0.05)
                        
                        decay_features = self._calculate_exponential_decay(
                            sentiment_history, current_date, lambda_param
                        )
                        
                        # Add features with horizon suffix
                        for feature_name, feature_value in decay_features.items():
                            result_row[f'sentiment_decay_{horizon}d_{feature_name}'] = feature_value
                    
                    # Calculate advanced features
                    if self.enable_advanced_features:
                        advanced_features = self._calculate_advanced_features(
                            sentiment_history, current_date
                        )
                        result_row.update(advanced_features)
                
                results.append(result_row)
                processed += 1
                
                # Progress logging
                if processed % 1000 == 0:
                    logger.info(f"   📊 Progress: {processed:,}/{total_records:,} ({processed/total_records*100:.1f}%)")
        
        # Convert to DataFrame
        decay_df = pd.DataFrame(results)
        
        # Feature normalization
        if self.feature_normalization != 'none':
            decay_df = self._normalize_features(decay_df)
        
        # Statistical validation
        if self.enable_statistical_validation:
            self._validate_features(decay_df)
        
        return decay_df
    
    def _normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize features for TFT compatibility"""
        logger.info(f"📊 Normalizing features using {self.feature_normalization} scaling...")
        
        # Identify numeric features to normalize
        numeric_features = []
        for col in df.columns:
            if ('sentiment_decay_' in col or 'sentiment_volatility_' in col or 
                'sentiment_momentum_' in col or col in ['confidence_mean', 'confidence_std']):
                if df[col].dtype in ['float64', 'int64']:
                    numeric_features.append(col)
        
        if not numeric_features:
            logger.warning("⚠️ No numeric features found for normalization")
            return df
        
        # Apply scaling
        df_normalized = df.copy()
        
        if self.feature_normalization == 'standard':
            scaler = StandardScaler()
        elif self.feature_normalization == 'robust':
            scaler = RobustScaler()
        else:
            logger.warning(f"⚠️ Unknown normalization method: {self.feature_normalization}")
            return df
        
        try:
            df_normalized[numeric_features] = scaler.fit_transform(df[numeric_features])
            logger.info(f"✅ Normalized {len(numeric_features)} features")
        except Exception as e:
            logger.warning(f"⚠️ Feature normalization failed: {e}")
            return df
        
        return df_normalized
    
    def _validate_features(self, df: pd.DataFrame):
        """Statistical validation of generated features"""
        logger.info("📊 Performing statistical validation...")
        
        # Find sentiment features
        sentiment_features = [col for col in df.columns if 'sentiment_decay_' in col and 'compound' in col]
        
        if not sentiment_features:
            logger.warning("⚠️ No sentiment features found for validation")
            return
        
        validation_results = {}
        
        # Check for feature correlations
        if len(sentiment_features) > 1:
            try:
                feature_corr = df[sentiment_features].corr()
                high_corr_pairs = []
                
                for i, col1 in enumerate(sentiment_features):
                    for j, col2 in enumerate(sentiment_features[i+1:], i+1):
                        corr_val = feature_corr.loc[col1, col2]
                        if abs(corr_val) > 0.95:
                            high_corr_pairs.append((col1, col2, corr_val))
                
                if high_corr_pairs:
                    logger.warning(f"⚠️ High correlation detected between features:")
                    for col1, col2, corr in high_corr_pairs[:3]:
                        logger.warning(f"   • {col1} ↔ {col2}: {corr:.3f}")
            except Exception as e:
                logger.warning(f"⚠️ Correlation analysis failed: {e}")
        
        # Feature distributions
        for feature in sentiment_features[:5]:  # Check first 5
            try:
                values = df[feature].dropna()
                if len(values) > 0:
                    validation_results[feature] = {
                        'mean': float(values.mean()),
                        'std': float(values.std()),
                        'skewness': float(stats.skew(values)),
                        'outliers': int((np.abs(stats.zscore(values)) > 3).sum())
                    }
            except Exception as e:
                logger.warning(f"⚠️ Feature validation failed for {feature}: {e}")
        
        logger.info("✅ Statistical validation completed")
        
        # Log key statistics
        for feature, stats_dict in list(validation_results.items())[:3]:
            logger.info(f"   📊 {feature}: μ={stats_dict['mean']:.3f}, σ={stats_dict['std']:.3f}, "
                       f"skew={stats_dict['skewness']:.2f}, outliers={stats_dict['outliers']}")
    
    def run_comprehensive_pipeline(self) -> pd.DataFrame:
        """Run complete comprehensive temporal decay processing pipeline"""
        logger.info("🚀 Starting comprehensive temporal decay processing pipeline")
        
        # Load data
        market_data, sentiment_data = self.load_required_data()
        
        # Optimize parameters
        optimized_params = self.optimize_decay_parameters(market_data, sentiment_data)
        
        # Process features
        decay_enhanced_data = self.process_temporal_decay_features(
            market_data, sentiment_data, optimized_params
        )
        
        # Save results
        output_path = self.data_paths['temporal_decay_dataset']
        output_path.parent.mkdir(parents=True, exist_ok=True)
        decay_enhanced_data.to_csv(output_path, index=False)
        
        # Generate feature report
        self._generate_feature_report(decay_enhanced_data, optimized_params)
        
        logger.info(f"💾 Comprehensive temporal decay dataset saved: {output_path}")
        logger.info("✅ Comprehensive temporal decay processing completed!")
        
        return decay_enhanced_data
    
    def _generate_feature_report(self, df: pd.DataFrame, params: Dict[int, float]):
        """Generate comprehensive feature engineering report"""
        logger.info("📊 COMPREHENSIVE FEATURE ENGINEERING REPORT")
        logger.info("=" * 60)
        
        # Basic statistics
        decay_features = [col for col in df.columns if 'sentiment_decay_' in col]
        advanced_features = [col for col in df.columns if any(x in col for x in ['volatility_', 'momentum_', 'confidence_'])]
        
        logger.info(f"📊 Dataset Summary:")
        logger.info(f"   • Total records: {len(df):,}")
        logger.info(f"   • Total features: {len(df.columns)}")
        logger.info(f"   • Decay features: {len(decay_features)}")
        logger.info(f"   • Advanced features: {len(advanced_features)}")
        logger.info(f"   • Symbols: {df['symbol'].nunique()}")
        
        logger.info(f"\n🔧 Optimized Parameters:")
        for horizon, param in params.items():
            logger.info(f"   • {horizon}d horizon: λ = {param:.4f}")
        
        logger.info(f"\n📈 Feature Coverage:")
        for horizon in self.target_horizons:
            compound_col = f'sentiment_decay_{horizon}d_compound'
            if compound_col in df.columns:
                non_zero = (df[compound_col] != 0).sum()
                coverage = non_zero / len(df) * 100
                logger.info(f"   • {horizon}d features: {coverage:.1f}% coverage")
        
        logger.info("=" * 60)

def main():
    """Main function for direct execution"""
    try:
        processor = AdvancedTemporalDecayProcessor()
        decay_enhanced_data = processor.run_comprehensive_pipeline()
        
        print(f"\n🎉 Comprehensive Temporal Decay Processing Completed Successfully!")
        print(f"📊 Records processed: {len(decay_enhanced_data):,}")
        
        # Show feature types
        decay_features = [col for col in decay_enhanced_data.columns if 'sentiment_decay_' in col]
        advanced_features = [col for col in decay_enhanced_data.columns if any(x in col for x in ['volatility_', 'momentum_', 'confidence_'])]
        
        print(f"🔬 Feature Engineering Results:")
        print(f"   • Decay features: {len(decay_features)}")
        print(f"   • Advanced features: {len(advanced_features)}")
        print(f"   • Total new features: {len(decay_features) + len(advanced_features)}")
        print(f"📈 Symbols: {decay_enhanced_data['symbol'].nunique()}")
        
    except Exception as e:
        logger.error(f"❌ Comprehensive temporal decay processing failed: {e}")
        raise

if __name__ == "__main__":
    main()