#!/usr/bin/env python3
"""
DATA_PREP.PY - Comprehensive Data Preparation Pipeline
====================================================

Final data preparation pipeline that handles:
1. Correlation fixes and feature selection
2. Normalization and scaling  
3. Train/validation/test splits (temporal)
4. Missing value treatment
5. Outlier handling
6. Feature engineering final touches
7. Model-ready dataset export

Processes both:
- combined_dataset.csv (baseline technical data)
- final_enhanced_dataset.csv (enhanced with sentiment)

Usage:
    python src/data_prep.py
    python src/data_prep.py --baseline-only
    python src/data_prep.py --enhanced-only
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging
from datetime import datetime
import argparse
import json
import shutil
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.impute import SimpleImputer, KNNImputer
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# Paths
DATA_DIR = Path("data/processed")
SPLITS_DIR = Path("data/splits")
MODELS_DIR = Path("data/model_ready")
SCALERS_DIR = Path("data/scalers")
REPORTS_DIR = Path("results/data_prep")

# Ensure directories exist
for dir_path in [SPLITS_DIR, MODELS_DIR, SCALERS_DIR, REPORTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

class ComprehensiveDataPreparator:
    """
    Comprehensive data preparation pipeline for ML model training
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._get_default_config()
        self.scalers = {}
        self.feature_selectors = {}
        self.preprocessing_stats = {}
        
    def _get_default_config(self) -> Dict:
        """Get default configuration for data preparation"""
        return {
            'correlation_threshold': 0.95,
            'feature_selection': {
                'method': 'mutual_info',  # 'correlation', 'mutual_info', 'f_regression'
                'k_best': 50,  # Top K features to select
                'min_target_correlation': 0.01
            },
            'scaling': {
                'method': 'robust',  # 'standard', 'robust', 'minmax'
                'feature_range': (0, 1)  # For MinMaxScaler
            },
            'outlier_treatment': {
                'method': 'iqr',  # 'iqr', 'zscore', 'percentile'
                'iqr_multiplier': 1.5,
                'zscore_threshold': 3.0,
                'percentile_range': (1, 99)
            },
            'missing_values': {
                'method': 'knn',  # 'mean', 'median', 'knn', 'forward_fill'
                'knn_neighbors': 5
            },
            'splits': {
                'train_ratio': 0.7,
                'val_ratio': 0.2,
                'test_ratio': 0.1,
                'method': 'temporal'  # 'temporal', 'random'
            },
            'target_columns': ['target_5', 'target_30', 'target_90'],
            'identifier_columns': ['stock_id', 'symbol', 'date'],
            'exclude_from_scaling': ['target_5_direction']
        }
    
    def prepare_datasets(self, 
                        baseline_path: str = "data/processed/combined_dataset.csv",
                        enhanced_path: str = "data/processed/final_enhanced_dataset.csv",
                        process_baseline: bool = True,
                        process_enhanced: bool = True) -> Dict[str, str]:
        """
        Main function to prepare both baseline and enhanced datasets
        """
        logger.info("🚀 STARTING COMPREHENSIVE DATA PREPARATION")
        logger.info("=" * 70)
        
        results = {}
        
        try:
            # Process baseline dataset (technical data only)
            if process_baseline and Path(baseline_path).exists():
                logger.info(f"📊 Processing baseline dataset: {baseline_path}")
                baseline_results = self._process_single_dataset(
                    baseline_path, 
                    dataset_type="baseline",
                    output_prefix="baseline"
                )
                results['baseline'] = baseline_results
                logger.info("✅ Baseline dataset processing complete!")
            
            # Process enhanced dataset (with sentiment)
            if process_enhanced and Path(enhanced_path).exists():
                logger.info(f"📊 Processing enhanced dataset: {enhanced_path}")
                enhanced_results = self._process_single_dataset(
                    enhanced_path, 
                    dataset_type="enhanced",
                    output_prefix="enhanced"
                )
                results['enhanced'] = enhanced_results
                logger.info("✅ Enhanced dataset processing complete!")
            
            # Generate comparison report
            if len(results) > 1:
                self._generate_comparison_report(results)
            
            logger.info("🎉 COMPREHENSIVE DATA PREPARATION COMPLETE!")
            return results
            
        except Exception as e:
            logger.error(f"❌ Data preparation failed: {e}")
            raise
    
    def _process_single_dataset(self, 
                               input_path: str, 
                               dataset_type: str,
                               output_prefix: str) -> Dict[str, str]:
        """Process a single dataset through the complete pipeline"""
        
        logger.info(f"📥 Loading {dataset_type} dataset...")
        df = pd.read_csv(input_path)
        original_shape = df.shape
        logger.info(f"   📊 Original shape: {original_shape}")
        
        # Store preprocessing stats
        self.preprocessing_stats[dataset_type] = {
            'original_shape': original_shape,
            'steps_applied': []
        }
        
        # Step 1: Data Quality Checks
        logger.info("🔍 Step 1: Data quality checks...")
        df = self._quality_checks(df, dataset_type)
        
        # Step 2: Handle Missing Values
        logger.info("🔧 Step 2: Handling missing values...")
        df = self._handle_missing_values(df, dataset_type)
        
        # Step 3: Fix Correlation Issues
        logger.info("🔧 Step 3: Fixing correlation issues...")
        df = self._fix_correlations(df, dataset_type)
        
        # Step 4: Outlier Treatment
        logger.info("🔧 Step 4: Outlier treatment...")
        df = self._handle_outliers(df, dataset_type)
        
        # Step 5: Feature Engineering Final Touches
        logger.info("🔧 Step 5: Final feature engineering...")
        df = self._final_feature_engineering(df, dataset_type)
        
        # Step 6: Feature Selection
        logger.info("🎯 Step 6: Feature selection...")
        df = self._feature_selection(df, dataset_type)
        
        # Step 7: Create Train/Val/Test Splits
        logger.info("✂️ Step 7: Creating data splits...")
        splits = self._create_splits(df, dataset_type)
        
        # Step 8: Scale Features
        logger.info("📊 Step 8: Scaling features...")
        scaled_splits = self._scale_features(splits, dataset_type)
        
        # Step 9: Save Processed Data
        logger.info("💾 Step 9: Saving processed data...")
        output_paths = self._save_processed_data(scaled_splits, output_prefix, dataset_type)
        
        # Step 10: Generate Dataset Report
        logger.info("📋 Step 10: Generating report...")
        self._generate_dataset_report(df, scaled_splits, dataset_type, output_prefix)
        
        final_shape = df.shape
        logger.info(f"   📊 Final shape: {original_shape} → {final_shape}")
        
        return output_paths
    
    def _quality_checks(self, df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
        """Comprehensive data quality checks and basic fixes"""
        
        # Check for duplicate rows
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            df = df.drop_duplicates()
            logger.info(f"   🗑️ Removed {duplicates} duplicate rows")
            self.preprocessing_stats[dataset_type]['steps_applied'].append(f"Removed {duplicates} duplicates")
        
        # Check date column
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            logger.info(f"   📅 Date range: {df['date'].min()} to {df['date'].max()}")
        
        # Check for completely empty columns
        empty_cols = df.columns[df.isna().all()].tolist()
        if empty_cols:
            df = df.drop(columns=empty_cols)
            logger.info(f"   🗑️ Removed {len(empty_cols)} empty columns: {empty_cols}")
        
        # Check data types
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        logger.info(f"   📊 Numeric columns: {len(numeric_cols)}")
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
        """Handle missing values using configured method with proper data type conversion"""
        
        method = self.config['missing_values']['method']
        
        # Separate different column types
        identifier_cols = [col for col in self.config['identifier_columns'] if col in df.columns]
        target_cols = [col for col in self.config['target_columns'] if col in df.columns]
        feature_cols = [col for col in df.columns if col not in identifier_cols + target_cols]
        
        # Count missing values before
        missing_before = df[feature_cols].isna().sum().sum()
        logger.info(f"   📊 Missing values before treatment: {missing_before}")
        
        if missing_before > 0 or df[feature_cols].dtypes.apply(lambda x: x == 'object').any():
            # STEP 1: Handle string representations of missing values and convert to numeric
            logger.info("   🔧 Converting feature columns to numeric and handling string nulls...")
            
            # Define string representations of missing values
            null_strings = ['none', 'None', 'NONE', 'null', 'NULL', 'na', 'NA', 'n/a', 'N/A', 
                            'nan', 'NaN', 'NAN', '', ' ', 'missing', 'MISSING']
            
            for col in feature_cols:
                try:
                    # First, replace string representations of null with actual NaN
                    if df[col].dtype == 'object':
                        df[col] = df[col].replace(null_strings, np.nan)
                    
                    # Convert to numeric, forcing errors to NaN
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                except Exception as e:
                    logger.warning(f"   ⚠️ Could not convert column {col} to numeric: {e}")
                    # If conversion fails, drop the column
                    if col in feature_cols:
                        feature_cols.remove(col)
                        logger.warning(f"   🗑️ Dropping problematic column: {col}")
            
            # STEP 2: Check for infinite values and replace with NaN
            logger.info("   🔧 Handling infinite values...")
            for col in feature_cols:
                if col in df.columns:
                    inf_mask = np.isinf(df[col])
                    if inf_mask.any():
                        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                        logger.info(f"   ♾️ Replaced {inf_mask.sum()} infinite values in {col}")
            
            # STEP 3: Recount missing values after cleanup
            missing_after_cleanup = df[feature_cols].isna().sum().sum()
            logger.info(f"   📊 Missing values after cleanup: {missing_after_cleanup}")
            
            # STEP 4: Apply imputation method
            if missing_after_cleanup > 0:
                # Ensure we only work with columns that have numeric data
                numeric_feature_cols = []
                for col in feature_cols:
                    if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                        numeric_feature_cols.append(col)
                
                if not numeric_feature_cols:
                    logger.warning("   ⚠️ No numeric feature columns found for imputation")
                    return df
                
                logger.info(f"   🔧 Applying {method} imputation to {len(numeric_feature_cols)} numeric columns...")
                
                if method == 'knn':
                    # Use KNN imputer for features
                    try:
                        knn_imputer = KNNImputer(n_neighbors=self.config['missing_values']['knn_neighbors'])
                        
                        # Create a copy of the numeric data for imputation
                        numeric_data = df[numeric_feature_cols].copy()
                        
                        # Apply imputation
                        imputed_data = knn_imputer.fit_transform(numeric_data)
                        
                        # Put the imputed data back
                        df[numeric_feature_cols] = imputed_data
                        
                    except Exception as e:
                        logger.error(f"   ❌ KNN imputation failed: {e}")
                        logger.info("   🔄 Falling back to median imputation...")
                        
                        # Fallback to median imputation
                        for col in numeric_feature_cols:
                            if df[col].isna().any():
                                median_value = df[col].median()
                                df[col].fillna(median_value, inplace=True)
                    
                elif method == 'forward_fill':
                    # Forward fill by symbol (for time series)
                    if 'symbol' in df.columns:
                        df[numeric_feature_cols] = df.groupby('symbol')[numeric_feature_cols].fillna(method='ffill')
                        df[numeric_feature_cols] = df[numeric_feature_cols].fillna(method='bfill')
                    else:
                        df[numeric_feature_cols] = df[numeric_feature_cols].fillna(method='ffill')
                        df[numeric_feature_cols] = df[numeric_feature_cols].fillna(method='bfill')
                        
                else:  # mean or median
                    try:
                        imputer = SimpleImputer(strategy=method)
                        df[numeric_feature_cols] = imputer.fit_transform(df[numeric_feature_cols])
                    except Exception as e:
                        logger.error(f"   ❌ {method} imputation failed: {e}")
                        logger.info("   🔄 Falling back to manual median imputation...")
                        
                        # Manual fallback
                        for col in numeric_feature_cols:
                            if df[col].isna().any():
                                fill_value = df[col].median() if method == 'median' else df[col].mean()
                                df[col].fillna(fill_value, inplace=True)
                
                # STEP 5: Final validation and cleanup
                remaining_missing = df[numeric_feature_cols].isna().sum().sum()
                logger.info(f"   ✅ Missing values after treatment: {remaining_missing}")
                
                # If there are still missing values, fill with 0
                if remaining_missing > 0:
                    logger.info(f"   🔧 Filling remaining {remaining_missing} missing values with 0...")
                    df[numeric_feature_cols] = df[numeric_feature_cols].fillna(0)
                
                # Track improvement
                improvement = missing_before - remaining_missing
                self.preprocessing_stats[dataset_type]['steps_applied'].append(
                    f"Imputed {improvement} missing values using {method} (with data type conversion)"
                )
            
            else:
                logger.info("   ✅ No missing values found after data type conversion")
        
        return df
    
    def _fix_correlations(self, df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
        """Fix high correlation issues between features while preserving sentiment features"""
        
        # Find numeric columns excluding identifiers and targets
        identifier_cols = [col for col in self.config['identifier_columns'] if col in df.columns]
        target_cols = [col for col in self.config['target_columns'] if col in df.columns]
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col not in identifier_cols + target_cols]
        
        if len(feature_cols) < 2:
            logger.info("   ✅ Insufficient features for correlation analysis")
            return df
        
        # CRITICAL FIX: Separate sentiment features from technical features
        sentiment_features = [col for col in feature_cols if any(
            pattern in col.lower() for pattern in [
                'sentiment_decay_', 'sentiment_compound', 'sentiment_positive', 
                'sentiment_negative', 'sentiment_confidence', 'sentiment_ma_',
                'confidence_mean', 'confidence_std', 'high_confidence_ratio',
                'sentiment_volatility_', 'sentiment_momentum_'
            ]
        )]
        
        technical_features = [col for col in feature_cols if col not in sentiment_features]
        
        logger.info(f"   📊 Feature analysis: {len(technical_features)} technical, {len(sentiment_features)} sentiment")
        
        # Calculate correlation matrix
        corr_matrix = df[feature_cols].corr().abs()
        
        # Different thresholds for different feature types
        technical_threshold = self.config['correlation_threshold']  # 0.95
        sentiment_threshold = 0.98  # Higher threshold for sentiment features (preserve them)
        
        # Find highly correlated pairs
        high_corr_pairs = []
        
        for i, col1 in enumerate(feature_cols):
            for j, col2 in enumerate(feature_cols[i+1:], i+1):
                corr_value = corr_matrix.loc[col1, col2]
                
                # Determine threshold based on feature types
                if col1 in sentiment_features and col2 in sentiment_features:
                    threshold = sentiment_threshold  # More lenient for sentiment-sentiment pairs
                elif col1 in sentiment_features or col2 in sentiment_features:
                    threshold = 0.97  # Moderate for sentiment-technical pairs
                else:
                    threshold = technical_threshold  # Strict for technical-technical pairs
                
                if corr_value > threshold:
                    high_corr_pairs.append((col1, col2, corr_value, threshold))
        
        if high_corr_pairs:
            logger.info(f"   ⚠️ Found {len(high_corr_pairs)} high correlation pairs")
            
            # Strategy: Remove features with lower target correlation, but protect sentiment features
            target_col = 'target_5' if 'target_5' in df.columns else target_cols[0] if target_cols else None
            features_to_remove = set()
            
            if target_col:
                target_corr = df[feature_cols + [target_col]].corr()[target_col].abs()
                
                for col1, col2, corr_val, threshold_used in high_corr_pairs:
                    # PROTECTION LOGIC: Prioritize keeping sentiment features
                    col1_is_sentiment = col1 in sentiment_features
                    col2_is_sentiment = col2 in sentiment_features
                    
                    if col1_is_sentiment and col2_is_sentiment:
                        # Both are sentiment features - keep the one with higher target correlation
                        if target_corr[col1] >= target_corr[col2]:
                            features_to_remove.add(col2)
                            logger.info(f"   🗑️ Removing sentiment {col2} (target_corr: {target_corr[col2]:.3f} < {target_corr[col1]:.3f})")
                        else:
                            features_to_remove.add(col1)
                            logger.info(f"   🗑️ Removing sentiment {col1} (target_corr: {target_corr[col1]:.3f} < {target_corr[col2]:.3f})")
                    
                    elif col1_is_sentiment and not col2_is_sentiment:
                        # col1 is sentiment, col2 is technical - prefer keeping sentiment unless much worse
                        if target_corr[col2] > target_corr[col1] + 0.02:  # 2% buffer for sentiment
                            features_to_remove.add(col1)
                            logger.info(f"   🗑️ Removing sentiment {col1} (much lower target_corr: {target_corr[col1]:.3f} vs {target_corr[col2]:.3f})")
                        else:
                            features_to_remove.add(col2)
                            logger.info(f"   🗑️ Removing technical {col2} (protecting sentiment)")
                    
                    elif not col1_is_sentiment and col2_is_sentiment:
                        # col2 is sentiment, col1 is technical - prefer keeping sentiment unless much worse
                        if target_corr[col1] > target_corr[col2] + 0.02:  # 2% buffer for sentiment
                            features_to_remove.add(col2)
                            logger.info(f"   🗑️ Removing sentiment {col2} (much lower target_corr: {target_corr[col2]:.3f} vs {target_corr[col1]:.3f})")
                        else:
                            features_to_remove.add(col1)
                            logger.info(f"   🗑️ Removing technical {col1} (protecting sentiment)")
                    
                    else:
                        # Both are technical features - use standard logic
                        if target_corr[col1] >= target_corr[col2]:
                            features_to_remove.add(col2)
                            logger.info(f"   🗑️ Removing technical {col2} (target_corr: {target_corr[col2]:.3f} < {target_corr[col1]:.3f})")
                        else:
                            features_to_remove.add(col1)
                            logger.info(f"   🗑️ Removing technical {col1} (target_corr: {target_corr[col1]:.3f} < {target_corr[col2]:.3f})")
            
            else:
                # Fallback: remove the second feature in each pair, but still protect sentiment
                for col1, col2, corr_val, threshold_used in high_corr_pairs:
                    if col1 in sentiment_features and col2 not in sentiment_features:
                        features_to_remove.add(col2)  # Remove technical, keep sentiment
                        logger.info(f"   🗑️ Removing technical {col2} (protecting sentiment)")
                    elif col2 in sentiment_features and col1 not in sentiment_features:
                        features_to_remove.add(col1)  # Remove technical, keep sentiment
                        logger.info(f"   🗑️ Removing technical {col1} (protecting sentiment)")
                    else:
                        features_to_remove.add(col2)  # Default behavior
                        logger.info(f"   🗑️ Removing {col2} (correlation: {corr_val:.3f})")
            
            # Apply removals
            if features_to_remove:
                sentiment_removed = len([f for f in features_to_remove if f in sentiment_features])
                technical_removed = len([f for f in features_to_remove if f in technical_features])
                
                df = df.drop(columns=list(features_to_remove))
                logger.info(f"   ✅ Removed {len(features_to_remove)} highly correlated features")
                logger.info(f"       📊 Technical removed: {technical_removed}")
                logger.info(f"       🎭 Sentiment removed: {sentiment_removed}")
                
                # Warning if too many sentiment features removed
                if sentiment_removed > len(sentiment_features) * 0.3:
                    logger.warning(f"   ⚠️ HIGH SENTIMENT LOSS: {sentiment_removed}/{len(sentiment_features)} sentiment features removed!")
                
                self.preprocessing_stats[dataset_type]['steps_applied'].append(
                    f"Removed {len(features_to_remove)} correlated features (protected sentiment)"
                )
        else:
            logger.info(f"   ✅ No high correlations found")
        
        return df
    
    def _handle_outliers(self, df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
        """Handle outliers using configured method"""
        
        method = self.config['outlier_treatment']['method']
        
        # Get numeric feature columns
        identifier_cols = [col for col in self.config['identifier_columns'] if col in df.columns]
        target_cols = [col for col in self.config['target_columns'] if col in df.columns]
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col not in identifier_cols + target_cols]
        
        outliers_treated = 0
        
        for col in feature_cols:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                multiplier = self.config['outlier_treatment']['iqr_multiplier']
                lower_bound = Q1 - multiplier * IQR
                upper_bound = Q3 + multiplier * IQR
                
            elif method == 'zscore':
                threshold = self.config['outlier_treatment']['zscore_threshold']
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                lower_bound = df[col].mean() - threshold * df[col].std()
                upper_bound = df[col].mean() + threshold * df[col].std()
                
            elif method == 'percentile':
                p_range = self.config['outlier_treatment']['percentile_range']
                lower_bound = df[col].quantile(p_range[0] / 100)
                upper_bound = df[col].quantile(p_range[1] / 100)
            
            # Count and clip outliers
            outliers_before = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            if outliers_before > 0:
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                outliers_treated += outliers_before
        
        if outliers_treated > 0:
            logger.info(f"   ✅ Treated {outliers_treated} outliers using {method} method")
            self.preprocessing_stats[dataset_type]['steps_applied'].append(f"Treated {outliers_treated} outliers")
        else:
            logger.info(f"   ✅ No significant outliers found")
        
        return df
    
    def _final_feature_engineering(self, df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
        """Final feature engineering touches"""
        
        original_cols = len(df.columns)
        
        # Add interaction features between sentiment and technical indicators
        if dataset_type == 'enhanced' and any('sentiment' in col for col in df.columns):
            sentiment_cols = [col for col in df.columns if 'sentiment' in col]
            technical_cols = ['rsi_14', 'macd_line', 'bb_position', 'volume_ratio']
            
            new_features = 0
            for sent_col in sentiment_cols[:2]:  # Limit to prevent explosion
                for tech_col in technical_cols:
                    if tech_col in df.columns:
                        interaction_col = f"{sent_col.split('_')[1]}_{tech_col}_interaction"
                        df[interaction_col] = df[sent_col] * df[tech_col]
                        new_features += 1
            
            if new_features > 0:
                logger.info(f"   🔧 Added {new_features} sentiment-technical interaction features")
                self.preprocessing_stats[dataset_type]['steps_applied'].append(f"Added {new_features} interaction features")
        
        # Add momentum features
        if 'returns' in df.columns:
            # Rolling momentum features
            for window in [3, 7, 14]:
                col_name = f'momentum_{window}d'
                if col_name not in df.columns:
                    df[col_name] = df.groupby('symbol')['returns'].rolling(window).mean().reset_index(0, drop=True)
        
        # Add volatility features
        if 'returns' in df.columns:
            for window in [5, 20]:
                col_name = f'volatility_{window}d'
                if col_name not in df.columns:
                    df[col_name] = df.groupby('symbol')['returns'].rolling(window).std().reset_index(0, drop=True)
        
        new_cols = len(df.columns) - original_cols
        if new_cols > 0:
            logger.info(f"   🔧 Added {new_cols} final engineering features")
        
        # Fill any new NaN values
        df = df.fillna(method='ffill').fillna(0)
        
        return df
    
    def _feature_selection(self, df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
        """Select best features using configured method"""
        
        method = self.config['feature_selection']['method']
        k_best = self.config['feature_selection']['k_best']
        
        # Get feature and target columns
        identifier_cols = [col for col in self.config['identifier_columns'] if col in df.columns]
        target_cols = [col for col in self.config['target_columns'] if col in df.columns]
        feature_cols = [col for col in df.columns if col not in identifier_cols + target_cols]
        
        if len(feature_cols) <= k_best:
            logger.info(f"   ✅ Feature count ({len(feature_cols)}) <= k_best ({k_best}), skipping selection")
            return df
        
        # Use primary target for feature selection
        target_col = 'target_5' if 'target_5' in target_cols else target_cols[0]
        
        # Prepare data for feature selection
        X = df[feature_cols].fillna(0)
        y = df[target_col].fillna(0)
        
        # Remove rows where target is NaN
        valid_mask = ~y.isna()
        X = X[valid_mask]
        y = y[valid_mask]
        
        if len(X) == 0:
            logger.warning(f"   ⚠️ No valid target values for feature selection")
            return df
        
        # Apply feature selection method
        if method == 'correlation':
            # Select features with highest target correlation
            correlations = X.corrwith(y).abs().sort_values(ascending=False)
            selected_features = correlations.head(k_best).index.tolist()
            
        elif method == 'mutual_info':
            # Use mutual information
            selector = SelectKBest(score_func=mutual_info_regression, k=min(k_best, len(feature_cols)))
            selector.fit(X, y)
            selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]
            
        elif method == 'f_regression':
            # Use F-test
            selector = SelectKBest(score_func=f_regression, k=min(k_best, len(feature_cols)))
            selector.fit(X, y)
            selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]
        
        # Keep identifier, target, and selected feature columns
        final_columns = identifier_cols + target_cols + selected_features
        df_selected = df[final_columns]
        
        logger.info(f"   🎯 Selected {len(selected_features)} features using {method}")
        logger.info(f"   📊 Dataset shape: {df.shape} → {df_selected.shape}")
        
        # Store feature selector for later use
        self.feature_selectors[dataset_type] = selected_features
        self.preprocessing_stats[dataset_type]['steps_applied'].append(f"Selected {len(selected_features)} features using {method}")
        
        return df_selected
    
    def _create_splits(self, df: pd.DataFrame, dataset_type: str) -> Dict[str, pd.DataFrame]:
        """Create train/validation/test splits"""
        
        if self.config['splits']['method'] == 'temporal':
            # Sort by date for temporal splits
            df = df.sort_values('date')
            
            # Calculate split indices
            n_total = len(df)
            train_size = int(n_total * self.config['splits']['train_ratio'])
            val_size = int(n_total * self.config['splits']['val_ratio'])
            
            # Create splits
            train_df = df.iloc[:train_size].copy()
            val_df = df.iloc[train_size:train_size + val_size].copy()
            test_df = df.iloc[train_size + val_size:].copy()
            
            logger.info(f"   📊 Temporal splits created:")
            logger.info(f"      🏃 Train: {len(train_df)} ({len(train_df)/n_total:.1%})")
            logger.info(f"      ✋ Val: {len(val_df)} ({len(val_df)/n_total:.1%})")
            logger.info(f"      🧪 Test: {len(test_df)} ({len(test_df)/n_total:.1%})")
            
            if 'date' in df.columns:
                logger.info(f"      📅 Train period: {train_df['date'].min()} to {train_df['date'].max()}")
                logger.info(f"      📅 Val period: {val_df['date'].min()} to {val_df['date'].max()}")
                logger.info(f"      📅 Test period: {test_df['date'].min()} to {test_df['date'].max()}")
        
        else:
            # Random splits (not recommended for time series)
            from sklearn.model_selection import train_test_split
            
            train_val_df, test_df = train_test_split(df, test_size=self.config['splits']['test_ratio'], random_state=42)
            train_df, val_df = train_test_split(train_val_df, test_size=self.config['splits']['val_ratio']/(1-self.config['splits']['test_ratio']), random_state=42)
            
            logger.info(f"   📊 Random splits created:")
            logger.info(f"      🏃 Train: {len(train_df)}")
            logger.info(f"      ✋ Val: {len(val_df)}")
            logger.info(f"      🧪 Test: {len(test_df)}")
        
        splits = {
            'train': train_df,
            'val': val_df,
            'test': test_df
        }
        
        self.preprocessing_stats[dataset_type]['split_sizes'] = {
            'train': len(train_df),
            'val': len(val_df),
            'test': len(test_df)
        }
        
        return splits
    
    def _scale_features(self, splits: Dict[str, pd.DataFrame], dataset_type: str) -> Dict[str, pd.DataFrame]:
        """Scale features using configured method"""
        
        method = self.config['scaling']['method']
        
        # Get feature columns (exclude identifiers and targets)
        identifier_cols = [col for col in self.config['identifier_columns'] if col in splits['train'].columns]
        target_cols = [col for col in self.config['target_columns'] if col in splits['train'].columns]
        exclude_cols = self.config['exclude_from_scaling']
        
        all_cols = splits['train'].columns.tolist()
        feature_cols = [col for col in all_cols if col not in identifier_cols + target_cols + exclude_cols]
        
        # Initialize scaler
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler(feature_range=self.config['scaling']['feature_range'])
        else:
            logger.warning(f"   ⚠️ Unknown scaling method: {method}, using standard")
            scaler = StandardScaler()
        
        # Fit scaler on training data
        scaler.fit(splits['train'][feature_cols])
        
        # Apply scaling to all splits
        scaled_splits = {}
        for split_name, split_df in splits.items():
            scaled_df = split_df.copy()
            scaled_df[feature_cols] = scaler.transform(split_df[feature_cols])
            scaled_splits[split_name] = scaled_df
        
        # Store scaler for later use
        self.scalers[dataset_type] = scaler
        
        logger.info(f"   📊 Scaled {len(feature_cols)} features using {method} scaling")
        self.preprocessing_stats[dataset_type]['steps_applied'].append(f"Scaled {len(feature_cols)} features using {method}")
        
        return scaled_splits
    
    def _save_processed_data(self, splits: Dict[str, pd.DataFrame], prefix: str, dataset_type: str) -> Dict[str, str]:
        """Save processed datasets and scalers"""
        
        output_paths = {}
        
        # Save splits
        for split_name, split_df in splits.items():
            output_path = MODELS_DIR / f"{prefix}_{split_name}.csv"
            split_df.to_csv(output_path, index=False)
            output_paths[f"{split_name}_path"] = str(output_path)
            logger.info(f"   💾 Saved {split_name} split: {output_path}")
        
        # Save full processed dataset
        full_df = pd.concat([splits['train'], splits['val'], splits['test']], ignore_index=True)
        full_path = MODELS_DIR / f"{prefix}_processed.csv"
        full_df.to_csv(full_path, index=False)
        output_paths['full_path'] = str(full_path)
        
        # Save scaler
        if dataset_type in self.scalers:
            scaler_path = SCALERS_DIR / f"{prefix}_scaler.joblib"
            joblib.dump(self.scalers[dataset_type], scaler_path)
            output_paths['scaler_path'] = str(scaler_path)
            logger.info(f"   💾 Saved scaler: {scaler_path}")
        
        # Save feature list
        if dataset_type in self.feature_selectors:
            features_path = REPORTS_DIR / f"{prefix}_selected_features.json"
            with open(features_path, 'w') as f:
                json.dump(self.feature_selectors[dataset_type], f, indent=2)
            output_paths['features_path'] = str(features_path)
        
        return output_paths
    
    def _generate_dataset_report(self, df: pd.DataFrame, splits: Dict[str, pd.DataFrame], dataset_type: str, prefix: str):
        """Generate comprehensive dataset report"""
        
        report = {
            'dataset_type': dataset_type,
            'timestamp': datetime.now().isoformat(),
            'preprocessing_config': self.config,
            'original_stats': self.preprocessing_stats[dataset_type],
            'final_stats': {
                'shape': df.shape,
                'features': len([col for col in df.columns if col not in self.config['identifier_columns'] + self.config['target_columns']]),
                'targets': len([col for col in df.columns if col in self.config['target_columns']]),
                'data_coverage': float(df.notna().mean().mean()),
                'date_range': {
                    'start': str(df['date'].min()) if 'date' in df.columns else None,
                    'end': str(df['date'].max()) if 'date' in df.columns else None
                }
            },
            'split_stats': {
                split_name: {
                    'size': len(split_df),
                    'target_coverage': float(split_df[self.config['target_columns'][0]].notna().mean()) if self.config['target_columns'][0] in split_df.columns else 0,
                    'feature_stats': {
                        'mean': float(split_df.select_dtypes(include=[np.number]).mean().mean()),
                        'std': float(split_df.select_dtypes(include=[np.number]).std().mean())
                    }
                }
                for split_name, split_df in splits.items()
            }
        }
        
        # Save report
        report_path = REPORTS_DIR / f"{prefix}_preparation_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"   📋 Dataset report saved: {report_path}")
    
    def _generate_comparison_report(self, results: Dict[str, Dict]):
        """Generate comparison report between baseline and enhanced datasets"""
        
        comparison = {
            'timestamp': datetime.now().isoformat(),
            'comparison_type': 'baseline_vs_enhanced',
            'datasets': results,
            'key_differences': {}
        }
        
        # Add comparison metrics if both datasets were processed
        if 'baseline' in results and 'enhanced' in results:
            baseline_stats = self.preprocessing_stats.get('baseline', {})
            enhanced_stats = self.preprocessing_stats.get('enhanced', {})
            
            comparison['key_differences'] = {
                'feature_count_difference': enhanced_stats.get('original_shape', (0, 0))[1] - baseline_stats.get('original_shape', (0, 0))[1],
                'processing_steps_baseline': len(baseline_stats.get('steps_applied', [])),
                'processing_steps_enhanced': len(enhanced_stats.get('steps_applied', [])),
            }
        
        # Save comparison report
        comparison_path = REPORTS_DIR / f"baseline_vs_enhanced_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(comparison_path, 'w') as f:
            json.dump(comparison, f, indent=2, default=str)
        
        logger.info(f"   📊 Comparison report saved: {comparison_path}")

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Comprehensive Data Preparation Pipeline')
    parser.add_argument('--baseline-only', action='store_true', help='Process only baseline dataset')
    parser.add_argument('--enhanced-only', action='store_true', help='Process only enhanced dataset')
    parser.add_argument('--config', type=str, help='Path to custom configuration file')
    
    args = parser.parse_args()
    
    # Load configuration
    config = None
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Initialize data preparator
    preparator = ComprehensiveDataPreparator(config)
    
    # Determine which datasets to process
    process_baseline = not args.enhanced_only
    process_enhanced = not args.baseline_only
    
    # Run data preparation
    results = preparator.prepare_datasets(
        process_baseline=process_baseline,
        process_enhanced=process_enhanced
    )
    
    # Print summary
    print("\n🎉 DATA PREPARATION COMPLETE!")
    print("=" * 50)
    for dataset_type, paths in results.items():
        print(f"\n📊 {dataset_type.upper()} DATASET:")
        for key, path in paths.items():
            if key.endswith('_path'):
                print(f"   📁 {key.replace('_path', '').title()}: {path}")
    
    print(f"\n📁 All outputs saved in:")
    print(f"   📊 Model-ready data: {MODELS_DIR}")
    print(f"   📈 Scalers: {SCALERS_DIR}")
    print(f"   📋 Reports: {REPORTS_DIR}")
    
    print(f"\n🚀 READY FOR MODEL TRAINING!")
    print(f"   python src/models.py")

if __name__ == "__main__":
    main()