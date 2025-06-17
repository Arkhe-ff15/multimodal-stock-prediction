#!/usr/bin/env python3
"""
DATA STANDARDS - Universal Pipeline Data Interface
=================================================

Fixes: Date format inconsistencies, column name variations, type mismatches
Result: Standardized data format across all pipeline stages
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class PipelineDataStandards:
    """Universal data format standards for the entire pipeline"""
    
    # Standard column names
    REQUIRED_COLUMNS = {
        'symbol': str,
        'date': str,  # Always string in YYYY-MM-DD format
        'close': float,
        'target_5': float
    }
    
    STANDARD_COLUMNS = {
        # Core stock data
        'stock_id': str,
        'symbol': str,
        'date': str,
        'open': float,
        'high': float,
        'low': float,
        'close': float,
        'volume': float,
        
        # Technical indicators (examples)
        'ema_5': float,
        'ema_10': float,
        'ema_20': float,
        'rsi_14': float,
        'macd_line': float,
        'bb_upper': float,
        'bb_lower': float,
        
        # Time features
        'year': int,
        'month': int,
        'day': int,
        'time_idx': int,
        
        # Target variables
        'target_5': float,
        'target_30': float,
        'target_90': float,
        
        # Sentiment features (if present)
        'sentiment_decay_5d': float,
        'sentiment_decay_30d': float,
        'sentiment_decay_90d': float,
        'sentiment_confidence': float,
        'article_count': int
    }
    
    # Standard date format
    DATE_FORMAT = '%Y-%m-%d'
    
    # Feature groups for model training
    FEATURE_GROUPS = {
        'core_stock': ['open', 'high', 'low', 'close', 'volume'],
        'technical_indicators': [
            'ema_5', 'ema_10', 'ema_20', 'rsi_14', 'macd_line', 
            'bb_upper', 'bb_lower', 'atr', 'vwap'
        ],
        'time_features': ['year', 'month', 'day', 'time_idx'],
        'sentiment_features': [
            'sentiment_decay_5d', 'sentiment_decay_30d', 'sentiment_decay_90d',
            'sentiment_confidence', 'article_count'
        ],
        'target_variables': ['target_5', 'target_30', 'target_90']
    }

class DataValidator:
    """Validates data against pipeline standards"""
    
    @staticmethod
    def validate_pipeline_data(data: pd.DataFrame, stage: str) -> Dict[str, Union[bool, List[str]]]:
        """
        Validate data format for specific pipeline stage
        
        Args:
            data: DataFrame to validate
            stage: Pipeline stage ('core', 'sentiment', 'model_ready')
            
        Returns:
            Validation results dictionary
        """
        logger.info(f"🔍 Validating data for stage: {stage}")
        
        validation = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'info': []
        }
        
        try:
            # Check basic requirements
            if data.empty:
                validation['errors'].append("DataFrame is empty")
                validation['is_valid'] = False
                return validation
            
            # Check required columns for stage
            required_cols = DataValidator._get_required_columns_for_stage(stage)
            missing_cols = [col for col in required_cols if col not in data.columns]
            
            if missing_cols:
                validation['errors'].append(f"Missing required columns: {missing_cols}")
                validation['is_valid'] = False
            
            # Validate data types
            type_errors = DataValidator._validate_data_types(data)
            validation['errors'].extend(type_errors)
            
            # Validate date format
            date_errors = DataValidator._validate_date_format(data)
            validation['errors'].extend(date_errors)
            
            # Stage-specific validations
            stage_errors = DataValidator._validate_stage_specific(data, stage)
            validation['errors'].extend(stage_errors)
            
            # Calculate data quality metrics
            quality_info = DataValidator._calculate_data_quality(data)
            validation['info'].extend(quality_info)
            
            if validation['errors']:
                validation['is_valid'] = False
            
        except Exception as e:
            validation['errors'].append(f"Validation failed: {e}")
            validation['is_valid'] = False
        
        # Log validation results
        if validation['is_valid']:
            logger.info("✅ Data validation passed")
        else:
            logger.error(f"❌ Data validation failed: {validation['errors']}")
        
        return validation
    
    @staticmethod
    def _get_required_columns_for_stage(stage: str) -> List[str]:
        """Get required columns for specific pipeline stage"""
        stage_requirements = {
            'core': ['symbol', 'date', 'close'],
            'sentiment': ['symbol', 'date', 'sentiment_compound'],
            'temporal_decay': ['symbol', 'date', 'sentiment_decay_5d'],
            'model_ready': ['symbol', 'date', 'close', 'target_5', 'time_idx']
        }
        return stage_requirements.get(stage, ['symbol', 'date'])
    
    @staticmethod
    def _validate_data_types(data: pd.DataFrame) -> List[str]:
        """Validate data types match standards"""
        errors = []
        
        for col in data.columns:
            if col in PipelineDataStandards.STANDARD_COLUMNS:
                expected_type = PipelineDataStandards.STANDARD_COLUMNS[col]
                
                if expected_type == str:
                    if data[col].dtype not in ['object', 'string']:
                        errors.append(f"Column {col} should be string, got {data[col].dtype}")
                elif expected_type == float:
                    if not pd.api.types.is_numeric_dtype(data[col]):
                        errors.append(f"Column {col} should be numeric, got {data[col].dtype}")
                elif expected_type == int:
                    if not pd.api.types.is_integer_dtype(data[col]):
                        errors.append(f"Column {col} should be integer, got {data[col].dtype}")
        
        return errors
    
    @staticmethod
    def _validate_date_format(data: pd.DataFrame) -> List[str]:
        """Validate date column format"""
        errors = []
        
        if 'date' in data.columns:
            try:
                # Try to parse dates
                parsed_dates = pd.to_datetime(data['date'])
                
                # Check if timezone-naive
                if hasattr(parsed_dates.dt, 'tz') and parsed_dates.dt.tz is not None:
                    errors.append("Date column should be timezone-naive")
                
                # Check format consistency (sample check)
                sample_dates = data['date'].dropna().head(10)
                for date_str in sample_dates:
                    try:
                        datetime.strptime(str(date_str), PipelineDataStandards.DATE_FORMAT)
                    except ValueError:
                        errors.append(f"Date format inconsistent. Expected {PipelineDataStandards.DATE_FORMAT}, got {date_str}")
                        break
                        
            except Exception as e:
                errors.append(f"Date parsing failed: {e}")
        
        return errors
    
    @staticmethod
    def _validate_stage_specific(data: pd.DataFrame, stage: str) -> List[str]:
        """Stage-specific validation rules"""
        errors = []
        
        if stage == 'model_ready':
            # Check for time_idx
            if 'time_idx' in data.columns:
                if data['time_idx'].isnull().any():
                    errors.append("time_idx contains null values")
                
                # Check time_idx is sequential within symbols
                for symbol in data['symbol'].unique()[:5]:  # Sample check
                    symbol_data = data[data['symbol'] == symbol].sort_values('date')
                    time_idx_diff = symbol_data['time_idx'].diff().dropna()
                    if not (time_idx_diff == 1).all():
                        errors.append(f"time_idx not sequential for symbol {symbol}")
                        break
            
            # Check target coverage
            target_cols = [col for col in data.columns if col.startswith('target_')]
            for target_col in target_cols:
                coverage = data[target_col].notna().mean()
                if coverage < 0.5:
                    errors.append(f"Low target coverage for {target_col}: {coverage:.1%}")
        
        elif stage == 'sentiment':
            # Check sentiment value ranges
            sentiment_cols = [col for col in data.columns if 'sentiment' in col.lower()]
            for col in sentiment_cols:
                if 'decay' in col or 'compound' in col:
                    values = data[col].dropna()
                    if len(values) > 0 and (values.min() < -2 or values.max() > 2):
                        errors.append(f"Sentiment values out of expected range for {col}")
        
        return errors
    
    @staticmethod
    def _calculate_data_quality(data: pd.DataFrame) -> List[str]:
        """Calculate data quality metrics"""
        info = []
        
        # Overall coverage
        overall_coverage = data.notna().mean().mean()
        info.append(f"Overall data coverage: {overall_coverage:.1%}")
        
        # Symbol distribution
        symbol_count = data['symbol'].nunique() if 'symbol' in data.columns else 0
        info.append(f"Unique symbols: {symbol_count}")
        
        # Date range
        if 'date' in data.columns:
            try:
                date_min = data['date'].min()
                date_max = data['date'].max()
                info.append(f"Date range: {date_min} to {date_max}")
            except:
                pass
        
        return info

class DataStandardizer:
    """Standardizes data format across pipeline stages"""
    
    @staticmethod
    def standardize_data(data: pd.DataFrame, target_stage: str) -> pd.DataFrame:
        """
        Standardize data format for target pipeline stage
        
        Args:
            data: Input DataFrame
            target_stage: Target pipeline stage
            
        Returns:
            Standardized DataFrame
        """
        logger.info(f"🔧 Standardizing data for stage: {target_stage}")
        
        standardized_data = data.copy()
        
        try:
            # Standardize date format
            standardized_data = DataStandardizer._standardize_dates(standardized_data)
            
            # Standardize column names
            standardized_data = DataStandardizer._standardize_column_names(standardized_data)
            
            # Standardize data types
            standardized_data = DataStandardizer._standardize_data_types(standardized_data)
            
            # Stage-specific standardization
            standardized_data = DataStandardizer._standardize_for_stage(standardized_data, target_stage)
            
            logger.info(f"✅ Data standardization completed for {target_stage}")
            
        except Exception as e:
            logger.error(f"❌ Data standardization failed: {e}")
            raise
        
        return standardized_data
    
    @staticmethod
    def _standardize_dates(data: pd.DataFrame) -> pd.DataFrame:
        """Standardize date column format"""
        if 'date' in data.columns:
            # Convert to datetime then to standard string format
            data['date'] = pd.to_datetime(data['date'])
            
            # Remove timezone if present
            if hasattr(data['date'].dt, 'tz') and data['date'].dt.tz is not None:
                data['date'] = data['date'].dt.tz_localize(None)
            
            # Convert to standard string format
            data['date'] = data['date'].dt.strftime(PipelineDataStandards.DATE_FORMAT)
        
        return data
    
    @staticmethod
    def _standardize_column_names(data: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names using common mappings"""
        column_mappings = {
            # Sentiment column variations
            'sentiment_compound': 'sentiment_compound',
            'sentiment_score': 'sentiment_compound',
            'confidence': 'sentiment_confidence',
            'sentiment_confidence': 'sentiment_confidence',
            
            # Target variations
            'target_5d': 'target_5',
            'target_30d': 'target_30',
            'target_90d': 'target_90',
            
            # Stock data
            'Stock_symbol': 'symbol',
            'Date': 'date'
        }
        
        # Apply mappings
        for old_name, new_name in column_mappings.items():
            if old_name in data.columns and new_name not in data.columns:
                data = data.rename(columns={old_name: new_name})
        
        return data
    
    @staticmethod
    def _standardize_data_types(data: pd.DataFrame) -> pd.DataFrame:
        """Standardize data types according to standards"""
        for col in data.columns:
            if col in PipelineDataStandards.STANDARD_COLUMNS:
                expected_type = PipelineDataStandards.STANDARD_COLUMNS[col]
                
                try:
                    if expected_type == float:
                        data[col] = pd.to_numeric(data[col], errors='coerce')
                    elif expected_type == int:
                        data[col] = pd.to_numeric(data[col], errors='coerce').astype('Int64')
                    elif expected_type == str:
                        data[col] = data[col].astype(str)
                except Exception as e:
                    logger.warning(f"Could not convert {col} to {expected_type}: {e}")
        
        return data
    
    @staticmethod
    def _standardize_for_stage(data: pd.DataFrame, stage: str) -> pd.DataFrame:
        """Apply stage-specific standardization"""
        if stage == 'model_ready':
            # Ensure time_idx exists and is sequential per symbol
            if 'time_idx' not in data.columns:
                data = data.sort_values(['symbol', 'date'])
                data['time_idx'] = data.groupby('symbol').cumcount()
            
            # Ensure required model columns exist
            required_model_cols = ['stock_id', 'symbol', 'date', 'close', 'target_5', 'time_idx']
            for col in required_model_cols:
                if col not in data.columns:
                    if col == 'stock_id' and 'symbol' in data.columns:
                        # Generate stock_id from symbol
                        symbol_to_id = {
                            symbol: f"stock_{idx:04d}" 
                            for idx, symbol in enumerate(sorted(data['symbol'].unique()), 1)
                        }
                        data['stock_id'] = data['symbol'].map(symbol_to_id)
        
        elif stage == 'sentiment':
            # Ensure sentiment columns are in expected range
            sentiment_cols = [col for col in data.columns if 'sentiment_decay_' in col]
            for col in sentiment_cols:
                data[col] = data[col].clip(-1, 1)  # Clip to expected range
        
        return data

# Convenience functions for pipeline integration
def validate_and_standardize(data: pd.DataFrame, stage: str) -> Tuple[pd.DataFrame, bool]:
    """
    Validate and standardize data in one step
    
    Returns:
        (standardized_data, is_valid)
    """
    # Standardize first
    standardized_data = DataStandardizer.standardize_data(data, stage)
    
    # Then validate
    validation = DataValidator.validate_pipeline_data(standardized_data, stage)
    
    return standardized_data, validation['is_valid']

def get_feature_columns_for_model(data: pd.DataFrame, model_type: str) -> Dict[str, List[str]]:
    """
    Get appropriate feature columns for specific model type
    
    Args:
        data: Standardized DataFrame
        model_type: 'baseline' or 'enhanced'
        
    Returns:
        Dictionary with feature column groups
    """
    available_cols = data.columns.tolist()
    
    # Get feature groups that are actually available
    feature_groups = {}
    for group_name, columns in PipelineDataStandards.FEATURE_GROUPS.items():
        available_features = [col for col in columns if col in available_cols]
        if available_features:
            feature_groups[group_name] = available_features
    
    # Return appropriate features based on model type
    if model_type == 'baseline':
        # Only technical features
        model_features = []
        for group in ['core_stock', 'technical_indicators', 'time_features']:
            model_features.extend(feature_groups.get(group, []))
        return {'features': model_features}
    
    elif model_type == 'enhanced':
        # All features including sentiment
        all_features = []
        for group in ['core_stock', 'technical_indicators', 'time_features', 'sentiment_features']:
            all_features.extend(feature_groups.get(group, []))
        return {'features': all_features}
    
    else:
        return feature_groups

if __name__ == "__main__":
    # Example usage
    
    # Create sample data
    sample_data = pd.DataFrame({
        'symbol': ['AAPL', 'AAPL', 'MSFT', 'MSFT'],
        'Date': ['2024-01-01', '2024-01-02', '2024-01-01', '2024-01-02'],  # Note capital D
        'close': [150.0, 152.0, 300.0, 305.0],
        'target_5d': [0.01, -0.02, 0.015, 0.005],  # Note 5d suffix
        'sentiment_score': [0.1, -0.2, 0.3, 0.0]  # Note different name
    })
    
    print("🔍 Original data:")
    print(sample_data.head())
    print(f"Columns: {sample_data.columns.tolist()}")
    
    # Standardize for model stage
    standardized_data, is_valid = validate_and_standardize(sample_data, 'model_ready')
    
    print("\n🔧 Standardized data:")
    print(standardized_data.head())
    print(f"Columns: {standardized_data.columns.tolist()}")
    print(f"Valid: {is_valid}")