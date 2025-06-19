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
Comprehensive Data Standards Module for Sentiment TFT Pipeline
Provides consistent validation, standardization, and quality checks across ALL pipeline stages
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, date
import warnings
from pathlib import Path

logger = logging.getLogger(__name__)

# =====================================================
# PIPELINE SCHEMA DEFINITIONS
# =====================================================

class PipelineSchemas:
    """Defines expected data schemas for all pipeline stages"""
    
    # Raw Data Schemas
    RAW_FNSPID = {
        'required_columns': ['Date', 'Article_title', 'Stock_symbol'],
        'optional_columns': ['Url', 'Publisher', 'Author', 'Article'],
        'data_types': {
            'Date': 'datetime',
            'Article_title': 'string',
            'Stock_symbol': 'string'
        }
    }
    
    RAW_MARKET_DATA = {
        'required_columns': ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume'],
        'data_types': {
            'date': 'datetime',
            'symbol': 'string',
            'open': 'float',
            'high': 'float', 
            'low': 'float',
            'close': 'float',
            'volume': 'int'
        }
    }
    
    # Processed Data Schemas
    SENTIMENT_ANALYSIS = {
        'required_columns': ['date', 'symbol', 'headline', 'sentiment_compound', 'confidence'],
        'optional_columns': ['sentiment_label', 'article_count'],
        'data_types': {
            'date': 'datetime',
            'symbol': 'string',
            'headline': 'string',
            'sentiment_compound': 'float',
            'confidence': 'float'
        },
        'value_ranges': {
            'sentiment_compound': (-1.0, 1.0),
            'confidence': (0.0, 1.0)
        }
    }
    
    DAILY_SENTIMENT = {
        'required_columns': ['date', 'symbol', 'sentiment_compound', 'confidence', 'article_count'],
        'data_types': {
            'date': 'datetime',
            'symbol': 'string',
            'sentiment_compound': 'float',
            'confidence': 'float',
            'article_count': 'int'
        },
        'value_ranges': {
            'sentiment_compound': (-1.0, 1.0),
            'confidence': (0.0, 1.0),
            'article_count': (1, None)
        }
    }
    
    # Feature Engineering Schemas
    TEMPORAL_FEATURES = {
        'required_columns': ['date', 'symbol', 'sentiment_compound', 'sentiment_decay', 'price_return'],
        'optional_columns': ['volume_change', 'volatility', 'moving_avg_sentiment'],
        'data_types': {
            'date': 'datetime',
            'symbol': 'string',
            'sentiment_compound': 'float',
            'sentiment_decay': 'float',
            'price_return': 'float'
        }
    }
    
    # Model Training Schemas
    MODEL_TRAINING = {
        'required_columns': ['date', 'symbol', 'target', 'sentiment_features'],
        'data_types': {
            'date': 'datetime',
            'symbol': 'string',
            'target': 'float'
        }
    }
    
    # Model Output Schemas
    MODEL_PREDICTIONS = {
        'required_columns': ['date', 'symbol', 'prediction', 'confidence_interval'],
        'data_types': {
            'date': 'datetime',
            'symbol': 'string',
            'prediction': 'float',
            'confidence_interval': 'float'
        }
    }


# =====================================================
# COMPREHENSIVE DATA VALIDATOR
# =====================================================

class DataValidator:
    """Validates data format and quality for all pipeline stages"""
    
    @staticmethod
    def validate_pipeline_data(data: pd.DataFrame, 
                             stage: str, 
                             schema: Dict = None,
                             custom_checks: List = None) -> Tuple[bool, Dict]:
        """
        Comprehensive validation for any pipeline stage
        
        Args:
            data: DataFrame to validate
            stage: Pipeline stage ('raw_fnspid', 'sentiment_analysis', 'daily_sentiment', etc.)
            schema: Optional custom schema (overrides default)
            custom_checks: Additional validation functions
            
        Returns:
            tuple: (success: bool, validation_info: dict)
        """
        try:
            if data is None or len(data) == 0:
                return False, {'error': 'Empty or null data', 'stage': stage}
            
            # Get schema for stage
            schema = schema or DataValidator._get_schema_for_stage(stage)
            if not schema:
                logger.warning(f"No schema defined for stage: {stage}")
                return True, {'stage': stage, 'warning': 'No schema validation available'}
            
            validation_results = {
                'stage': stage,
                'rows': len(data),
                'columns': list(data.columns),
                'validation_passed': True,
                'errors': [],
                'warnings': [],
                'info': {}
            }
            
            # 1. Required Columns Check
            missing_required = DataValidator._check_required_columns(data, schema)
            if missing_required:
                validation_results['validation_passed'] = False
                validation_results['errors'].append(f"Missing required columns: {missing_required}")
            
            # 2. Data Types Check
            type_errors = DataValidator._check_data_types(data, schema)
            if type_errors:
                validation_results['warnings'].extend(type_errors)
            
            # 3. Value Ranges Check
            range_errors = DataValidator._check_value_ranges(data, schema)
            if range_errors:
                validation_results['warnings'].extend(range_errors)
            
            # 4. Data Quality Checks
            quality_issues = DataValidator._check_data_quality(data, stage)
            if quality_issues:
                validation_results['warnings'].extend(quality_issues)
            
            # 5. Custom Checks
            if custom_checks:
                for check_func in custom_checks:
                    try:
                        check_result = check_func(data)
                        if not check_result[0]:
                            validation_results['warnings'].append(check_result[1])
                    except Exception as e:
                        validation_results['warnings'].append(f"Custom check failed: {str(e)}")
            
            # 6. Generate Summary Info
            validation_results['info'] = DataValidator._generate_data_summary(data, stage)
            
            return validation_results['validation_passed'], validation_results
            
        except Exception as e:
            return False, {'error': f'Validation failed: {str(e)}', 'stage': stage}
    
    @staticmethod
    def _get_schema_for_stage(stage: str) -> Dict:
        """Get schema definition for pipeline stage"""
        schema_mapping = {
            'raw_fnspid': PipelineSchemas.RAW_FNSPID,
            'fnspid': PipelineSchemas.RAW_FNSPID,
            'raw_market_data': PipelineSchemas.RAW_MARKET_DATA,
            'sentiment_analysis': PipelineSchemas.SENTIMENT_ANALYSIS,
            'daily_sentiment': PipelineSchemas.DAILY_SENTIMENT,
            'temporal_features': PipelineSchemas.TEMPORAL_FEATURES,
            'model_training': PipelineSchemas.MODEL_TRAINING,
            'model_predictions': PipelineSchemas.MODEL_PREDICTIONS
        }
        return schema_mapping.get(stage)
    
    @staticmethod
    def _check_required_columns(data: pd.DataFrame, schema: Dict) -> List[str]:
        """Check for missing required columns"""
        required = schema.get('required_columns', [])
        missing = [col for col in required if col not in data.columns]
        return missing
    
    @staticmethod
    def _check_data_types(data: pd.DataFrame, schema: Dict) -> List[str]:
        """Check data types match schema"""
        type_issues = []
        expected_types = schema.get('data_types', {})
        
        for col, expected_type in expected_types.items():
            if col in data.columns:
                actual_dtype = str(data[col].dtype)
                
                # Type mapping checks
                if expected_type == 'datetime' and not pd.api.types.is_datetime64_any_dtype(data[col]):
                    type_issues.append(f"Column '{col}' should be datetime, got {actual_dtype}")
                elif expected_type == 'string' and not pd.api.types.is_string_dtype(data[col]) and not pd.api.types.is_object_dtype(data[col]):
                    type_issues.append(f"Column '{col}' should be string, got {actual_dtype}")
                elif expected_type == 'float' and not pd.api.types.is_numeric_dtype(data[col]):
                    type_issues.append(f"Column '{col}' should be numeric, got {actual_dtype}")
                elif expected_type == 'int' and not pd.api.types.is_integer_dtype(data[col]):
                    type_issues.append(f"Column '{col}' should be integer, got {actual_dtype}")
        
        return type_issues
    
    @staticmethod
    def _check_value_ranges(data: pd.DataFrame, schema: Dict) -> List[str]:
        """Check values are within expected ranges"""
        range_issues = []
        value_ranges = schema.get('value_ranges', {})
        
        for col, (min_val, max_val) in value_ranges.items():
            if col in data.columns:
                col_data = data[col].dropna()
                if len(col_data) > 0:
                    actual_min = col_data.min()
                    actual_max = col_data.max()
                    
                    if min_val is not None and actual_min < min_val:
                        range_issues.append(f"Column '{col}' has values below minimum {min_val}: {actual_min}")
                    if max_val is not None and actual_max > max_val:
                        range_issues.append(f"Column '{col}' has values above maximum {max_val}: {actual_max}")
        
        return range_issues
    
    @staticmethod
    def _check_data_quality(data: pd.DataFrame, stage: str) -> List[str]:
        """Check general data quality issues"""
        quality_issues = []
        
        # Check for excessive null values
        null_percentages = data.isnull().sum() / len(data) * 100
        high_null_cols = null_percentages[null_percentages > 50].index.tolist()
        if high_null_cols:
            quality_issues.append(f"High null percentages in columns: {high_null_cols}")
        
        # Check for duplicate rows
        duplicate_count = data.duplicated().sum()
        if duplicate_count > 0:
            quality_issues.append(f"Found {duplicate_count} duplicate rows")
        
        # Stage-specific quality checks
        if stage in ['sentiment_analysis', 'daily_sentiment']:
            if 'sentiment_compound' in data.columns:
                extreme_sentiment = data['sentiment_compound'].abs() > 0.95
                if extreme_sentiment.sum() > len(data) * 0.1:  # More than 10% extreme values
                    quality_issues.append("High percentage of extreme sentiment values (>0.95)")
        
        return quality_issues
    
    @staticmethod
    def _generate_data_summary(data: pd.DataFrame, stage: str) -> Dict:
        """Generate summary information about the data"""
        summary = {
            'row_count': len(data),
            'column_count': len(data.columns),
            'null_counts': data.isnull().sum().to_dict(),
            'memory_usage_mb': data.memory_usage(deep=True).sum() / 1024 / 1024
        }
        
        # Add stage-specific summary info
        if 'date' in data.columns:
            try:
                date_col = pd.to_datetime(data['date'])
                summary['date_range'] = {
                    'start': date_col.min().strftime('%Y-%m-%d'),
                    'end': date_col.max().strftime('%Y-%m-%d'),
                    'unique_dates': date_col.nunique()
                }
            except:
                pass
        
        if 'symbol' in data.columns:
            summary['symbols'] = {
                'unique_count': data['symbol'].nunique(),
                'symbols': sorted(data['symbol'].unique())
            }
        
        if 'sentiment_compound' in data.columns:
            summary['sentiment_stats'] = {
                'mean': float(data['sentiment_compound'].mean()),
                'std': float(data['sentiment_compound'].std()),
                'min': float(data['sentiment_compound'].min()),
                'max': float(data['sentiment_compound'].max())
            }
        
        return summary


# =====================================================
# COMPREHENSIVE DATA STANDARDIZER
# =====================================================

class DataStandardizer:
    """Standardizes data formats across all pipeline components"""
    
    def __init__(self):
        self.column_mappings = self._initialize_column_mappings()
        self.date_formats = ['%Y-%m-%d', '%Y/%m/%d', '%m/%d/%Y', '%d/%m/%Y']
    
    def _initialize_column_mappings(self) -> Dict:
        """Initialize standard column name mappings"""
        return {
            # Stock symbol standardization
            'stock_symbol': 'symbol',
            'Stock_symbol': 'symbol',
            'ticker': 'symbol',
            'Symbol': 'symbol',
            'SYMBOL': 'symbol',
            'stock': 'symbol',
            
            # Date standardization
            'Date': 'date',
            'DATE': 'date',
            'timestamp': 'date',
            'time': 'date',
            
            # Price data standardization
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'Adj Close': 'adj_close',
            
            # Text data standardization
            'Article_title': 'headline',
            'article_title': 'headline',
            'title': 'headline',
            'Title': 'headline',
            'news_title': 'headline',
            'Article': 'article_text',
            'article': 'article_text',
            'content': 'article_text',
            
            # Sentiment standardization
            'sentiment': 'sentiment_compound',
            'sentiment_score': 'sentiment_compound',
            'compound': 'sentiment_compound'
        }
    
    def standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names across pipeline"""
        df = df.copy()
        return df.rename(columns=self.column_mappings)
    
    def standardize_dates(self, df: pd.DataFrame, date_column: str = 'date') -> pd.DataFrame:
        """Standardize date formats to YYYY-MM-DD"""
        df = df.copy()
        
        if date_column not in df.columns:
            logger.warning(f"Date column '{date_column}' not found in data")
            return df
        
        try:
            # Try pandas automatic parsing first
            df[date_column] = pd.to_datetime(df[date_column])
        except:
            # Try specific formats
            for fmt in self.date_formats:
                try:
                    df[date_column] = pd.to_datetime(df[date_column], format=fmt)
                    break
                except:
                    continue
            else:
                logger.warning(f"Could not parse dates in column '{date_column}'")
                return df
        
        # Convert to standard string format
        df[date_column] = df[date_column].dt.strftime('%Y-%m-%d')
        return df
    
    def standardize_symbols(self, df: pd.DataFrame, symbol_column: str = 'symbol') -> pd.DataFrame:
        """Standardize stock symbol format"""
        df = df.copy()
        
        if symbol_column not in df.columns:
            return df
        
        # Clean and standardize symbols
        df[symbol_column] = (df[symbol_column]
                           .astype(str)
                           .str.strip()
                           .str.upper()
                           .str.replace(r'[^A-Z]', '', regex=True))
        
        # Remove empty symbols
        df = df[df[symbol_column].str.len() > 0]
        
        return df
    
    def standardize_text(self, df: pd.DataFrame, text_columns: List[str] = None) -> pd.DataFrame:
        """Standardize text data (headlines, articles)"""
        df = df.copy()
        
        if text_columns is None:
            text_columns = [col for col in df.columns if col in ['headline', 'article_text', 'content']]
        
        for col in text_columns:
            if col in df.columns:
                df[col] = (df[col]
                          .astype(str)
                          .str.strip()
                          .str.replace(r'\s+', ' ', regex=True)  # Normalize whitespace
                          .str.replace(r'[^\w\s.,!?;:-]', '', regex=True))  # Remove special chars
                
                # Remove empty text
                df = df[df[col].str.len() > 0]
        
        return df
    
    def standardize_sentiment_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize sentiment analysis output"""
        df = df.copy()
        
        # Apply column standardization
        df = self.standardize_column_names(df)
        
        # Ensure required columns exist
        required_cols = ['date', 'symbol', 'sentiment_compound']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required sentiment columns: {missing}")
        
        # Standardize dates
        df = self.standardize_dates(df, 'date')
        
        # Standardize symbols
        df = self.standardize_symbols(df, 'symbol')
        
        # Standardize sentiment values
        if 'sentiment_compound' in df.columns:
            df['sentiment_compound'] = pd.to_numeric(df['sentiment_compound'], errors='coerce')
            df['sentiment_compound'] = df['sentiment_compound'].clip(-1, 1)
        
        if 'confidence' in df.columns:
            df['confidence'] = pd.to_numeric(df['confidence'], errors='coerce')
            df['confidence'] = df['confidence'].clip(0, 1)
        
        # Remove rows with null sentiment
        df = df.dropna(subset=['sentiment_compound'])
        
        return df
    
    def standardize_market_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize market/price data"""
        df = df.copy()
        
        # Apply column standardization
        df = self.standardize_column_names(df)
        
        # Standardize dates and symbols
        df = self.standardize_dates(df, 'date')
        df = self.standardize_symbols(df, 'symbol')
        
        # Standardize price columns
        price_columns = ['open', 'high', 'low', 'close', 'adj_close']
        for col in price_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df = df[df[col] > 0]  # Remove negative prices
        
        # Standardize volume
        if 'volume' in df.columns:
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
            df = df[df['volume'] >= 0]  # Remove negative volume
        
        return df
    
    def standardize_fnspid_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize FNSPID dataset format"""
        df = df.copy()
        
        # Apply column standardization
        df = self.standardize_column_names(df)
        
        # Ensure core columns exist
        required_cols = ['date', 'symbol', 'headline']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required FNSPID columns: {missing}")
        
        # Standardize each component
        df = self.standardize_dates(df, 'date')
        df = self.standardize_symbols(df, 'symbol')
        df = self.standardize_text(df, ['headline', 'article_text'])
        
        return df
    
    def standardize_pipeline_stage(self, df: pd.DataFrame, stage: str) -> pd.DataFrame:
        """Auto-standardize data based on pipeline stage"""
        
        stage_standardizers = {
            'raw_fnspid': self.standardize_fnspid_data,
            'fnspid': self.standardize_fnspid_data,
            'raw_market_data': self.standardize_market_data,
            'sentiment_analysis': self.standardize_sentiment_data,
            'daily_sentiment': self.standardize_sentiment_data,
            'temporal_features': self.standardize_sentiment_data,  # Uses similar format
        }
        
        standardizer_func = stage_standardizers.get(stage)
        if standardizer_func:
            return standardizer_func(df)
        else:
            # Apply basic standardization
            df = self.standardize_column_names(df)
            if 'date' in df.columns:
                df = self.standardize_dates(df)
            if 'symbol' in df.columns:
                df = self.standardize_symbols(df)
            return df


# =====================================================
# PIPELINE DATA QUALITY MANAGER
# =====================================================

class DataQualityManager:
    """Manages data quality across entire pipeline"""
    
    def __init__(self):
        self.validator = DataValidator()
        self.standardizer = DataStandardizer()
        self.quality_log = []
    
    def process_stage(self, 
                     data: pd.DataFrame, 
                     stage: str,
                     standardize: bool = True,
                     validate: bool = True) -> Tuple[bool, pd.DataFrame, Dict]:
        """
        Complete data processing for a pipeline stage
        
        Args:
            data: Input data
            stage: Pipeline stage name
            standardize: Whether to apply standardization
            validate: Whether to run validation
            
        Returns:
            tuple: (success, processed_data, quality_report)
        """
        
        quality_report = {
            'stage': stage,
            'input_rows': len(data),
            'operations': [],
            'issues': [],
            'success': True
        }
        
        try:
            processed_data = data.copy()
            
            # 1. Standardization
            if standardize:
                processed_data = self.standardizer.standardize_pipeline_stage(processed_data, stage)
                quality_report['operations'].append('standardization')
                quality_report['standardized_rows'] = len(processed_data)
            
            # 2. Validation
            if validate:
                validation_success, validation_info = self.validator.validate_pipeline_data(
                    processed_data, stage
                )
                quality_report['operations'].append('validation')
                quality_report['validation'] = validation_info
                
                if not validation_success:
                    quality_report['success'] = False
                    quality_report['issues'].extend(validation_info.get('errors', []))
            
            # 3. Log quality metrics
            self.quality_log.append(quality_report)
            
            quality_report['output_rows'] = len(processed_data)
            
            return quality_report['success'], processed_data, quality_report
            
        except Exception as e:
            quality_report['success'] = False
            quality_report['issues'].append(f"Processing failed: {str(e)}")
            logger.error(f"Data quality processing failed for stage {stage}: {str(e)}")
            return False, data, quality_report
    
    def get_pipeline_quality_summary(self) -> Dict:
        """Get summary of data quality across entire pipeline"""
        if not self.quality_log:
            return {'message': 'No quality data available'}
        
        summary = {
            'stages_processed': len(self.quality_log),
            'overall_success': all(log['success'] for log in self.quality_log),
            'total_issues': sum(len(log['issues']) for log in self.quality_log),
            'stage_details': {}
        }
        
        for log in self.quality_log:
            summary['stage_details'][log['stage']] = {
                'success': log['success'],
                'input_rows': log['input_rows'],
                'output_rows': log.get('output_rows', log['input_rows']),
                'issues_count': len(log['issues']),
                'operations': log['operations']
            }
        
        return summary


# =====================================================
# CONVENIENCE FUNCTIONS
# =====================================================

def validate_fnspid_format(df: pd.DataFrame) -> Tuple[bool, Dict]:
    """Legacy function for backward compatibility"""
    return DataValidator.validate_pipeline_data(df, stage='fnspid')

def standardize_data(df: pd.DataFrame, format_type: str = 'auto') -> pd.DataFrame:
    """Main standardization function"""
    standardizer = DataStandardizer()
    
    if format_type == 'auto':
        # Auto-detect format based on columns
        if 'Stock_symbol' in df.columns:
            format_type = 'fnspid'
        elif 'sentiment_compound' in df.columns:
            format_type = 'sentiment_analysis'
        elif all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            format_type = 'raw_market_data'
    
    return standardizer.standardize_pipeline_stage(df, format_type)

def validate_and_standardize(df: pd.DataFrame, stage: str) -> Tuple[bool, pd.DataFrame, Dict]:
    """One-stop function for validation and standardization"""
    quality_manager = DataQualityManager()
    return quality_manager.process_stage(df, stage, standardize=True, validate=True)


# =====================================================
# EXPORTS
# =====================================================

__all__ = [
    'PipelineSchemas',
    'DataValidator', 
    'DataStandardizer',
    'DataQualityManager',
    'validate_fnspid_format',
    'standardize_data',
    'validate_and_standardize'
]