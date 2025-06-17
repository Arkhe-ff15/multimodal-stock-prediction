#!/usr/bin/env python3
"""
CENTRALIZED PIPELINE CONFIGURATION
==================================

Central configuration management for the entire temporal decay sentiment-enhanced TFT pipeline.
All configuration parameters, paths, and defaults are defined here to ensure consistency
across all pipeline stages.

Author: Research Team
Version: 1.0 (Production Ready)
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from pathlib import Path
import logging

# Set up logging
logger = logging.getLogger(__name__)

# =============================================================================
# PATH CONSTANTS - Centralized path management
# =============================================================================

# Base directories
BASE_DIR = Path(__file__).parent.parent  # Project root
DATA_DIR = BASE_DIR / "data"
SRC_DIR = BASE_DIR / "src"
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Data subdirectories
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
BACKUP_DATA_DIR = DATA_DIR / "backups"
CACHE_DATA_DIR = DATA_DIR / "cache"

# Results subdirectories
MODELS_CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
EVALUATION_RESULTS_DIR = RESULTS_DIR / "evaluation"
PIPELINE_REPORTS_DIR = RESULTS_DIR / "pipeline_reports"
TEMPORAL_DECAY_RESULTS_DIR = RESULTS_DIR / "temporal_decay"
SENTIMENT_INTEGRATION_RESULTS_DIR = RESULTS_DIR / "sentiment_integration"

# Logs subdirectories
TRAINING_LOGS_DIR = LOGS_DIR / "training"
PIPELINE_LOGS_DIR = LOGS_DIR / "pipeline"

# =============================================================================
# FILE NAME CONSTANTS - Standardized file naming
# =============================================================================

# Input files
FNSPID_RAW_FILE = "nasdaq_external_data.csv"  # FIXED: was "nasdaq_exteral_data.csv"

# Pipeline stage outputs (standardized naming)
CORE_DATASET_FILE = "combined_dataset.csv"
FNSPID_FILTERED_ARTICLES_FILE = "fnspid_filtered_articles.csv"
FNSPID_ARTICLE_SENTIMENT_FILE = "fnspid_article_sentiment.csv"
FNSPID_DAILY_SENTIMENT_FILE = "fnspid_daily_sentiment.csv"
TEMPORAL_DECAY_DATA_FILE = "sentiment_with_temporal_decay.csv"
ENHANCED_DATASET_FILE = "combined_dataset_with_sentiment.csv"

# Reports and metadata
SYMBOL_MAPPING_FILE = "symbol_mapping.json"
DATA_SUMMARY_FILE = "data_summary.json"
PIPELINE_EXECUTION_REPORT_FILE = "pipeline_execution_report.json"
TEMPORAL_DECAY_VALIDATION_FILE = "temporal_decay_validation.json"
SENTIMENT_INTEGRATION_REPORT_FILE = "sentiment_integration_report.json"

# =============================================================================
# PIPELINE CONFIGURATION
# =============================================================================

@dataclass
class PipelineConfig:
    """
    Main pipeline configuration class
    
    Contains all parameters needed for the complete temporal decay sentiment-enhanced
    TFT pipeline execution. This configuration ensures consistency across all stages.
    """
    
    # ===================
    # CORE PARAMETERS
    # ===================
    
    # Stock symbols to process
    symbols: List[str] = field(default_factory=lambda: [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'JPM', 'JNJ'
    ])
    
    # Date range for analysis
    start_date: str = "2018-01-01"
    end_date: str = "2024-01-31"
    
    # ===================
    # DATA PROCESSING
    # ===================
    
    # FNSPID processing parameters
    fnspid_sample_ratio: float = 0.25  # 25% of FNSPID data for full analysis
    fnspid_chunk_size: int = 15000  # Memory-efficient processing chunk size
    fnspid_min_headline_length: int = 10  # Minimum headline length
    fnspid_max_headline_length: int = 500  # Maximum headline length
    
    # Technical indicators parameters
    technical_indicators_enabled: bool = True
    volume_indicators_enabled: bool = True
    momentum_indicators_enabled: bool = True
    
    # Target horizons for prediction
    target_horizons: List[int] = field(default_factory=lambda: [5, 30, 90])
    
    # ===================
    # TEMPORAL DECAY PARAMETERS
    # ===================
    
    # Horizon-specific decay parameters (core innovation)
    temporal_decay_params: Dict[int, Dict[str, float]] = field(default_factory=lambda: {
        5: {
            'lambda_decay': 0.20,  # Fast decay for short-term predictions
            'lookback_days': 30,
            'min_sentiment_count': 2,
            'confidence_threshold': 0.6
        },
        30: {
            'lambda_decay': 0.10,  # Moderate decay for medium-term predictions
            'lookback_days': 90,
            'min_sentiment_count': 3,
            'confidence_threshold': 0.5
        },
        90: {
            'lambda_decay': 0.05,  # Slow decay for long-term predictions
            'lookback_days': 180,
            'min_sentiment_count': 4,
            'confidence_threshold': 0.5
        }
    })
    
    # ===================
    # MODEL TRAINING
    # ===================
    
    # Training parameters
    max_epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    early_stopping_patience: int = 15
    reduce_lr_patience: int = 8
    
    # Model architecture parameters
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    attention_head_size: int = 4
    
    # TFT-specific parameters
    max_encoder_length: int = 30
    max_prediction_length: int = 5
    min_prediction_length: int = 1
    
    # Data split parameters
    validation_split: float = 0.2
    test_split: float = 0.1
    
    # ===================
    # PIPELINE CONTROL
    # ===================
    
    # Stages to execute
    run_data_collection: bool = True
    run_fnspid_processing: bool = True
    run_temporal_decay: bool = True
    run_sentiment_integration: bool = True
    run_model_training: bool = True
    run_evaluation: bool = True
    
    # Fallback strategies
    use_synthetic_sentiment: bool = False  # If FNSPID data unavailable
    skip_on_errors: bool = False  # Continue pipeline on non-critical errors
    
    # ===================
    # PERFORMANCE & SYSTEM
    # ===================
    
    # Memory management
    enable_garbage_collection: bool = True
    memory_monitoring: bool = True
    
    # Processing optimization
    use_multiprocessing: bool = True
    num_workers: int = 4
    pin_memory: bool = True
    use_mixed_precision: bool = True
    
    # Caching
    enable_caching: bool = True
    cache_validation_hours: int = 24  # Cache validity in hours
    
    # ===================
    # LOGGING & DEBUGGING
    # ===================
    
    # Logging configuration
    log_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR
    enable_file_logging: bool = True
    enable_tensorboard: bool = True
    log_model_architecture: bool = True
    
    # Debug options
    debug_mode: bool = False
    save_intermediate_results: bool = True
    validate_data_at_each_stage: bool = True
    
    # ===================
    # FILE PATHS (Auto-generated)
    # ===================
    
    def __post_init__(self):
        """Initialize derived parameters and create directories"""
        
        # Ensure all directories exist
        self._create_directories()
        
        # Validate configuration
        self._validate_configuration()
        
        # Set up file paths
        self._setup_file_paths()
        
        logger.info("🔧 PipelineConfig initialized successfully")
        logger.info(f"   📊 Symbols: {len(self.symbols)} ({self.symbols[:3]}...)")
        logger.info(f"   📅 Date range: {self.start_date} to {self.end_date}")
        logger.info(f"   🎯 Target horizons: {self.target_horizons}")
    
    def _create_directories(self):
        """Create all required directories"""
        directories = [
            DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, BACKUP_DATA_DIR, CACHE_DATA_DIR,
            RESULTS_DIR, MODELS_DIR, MODELS_CHECKPOINTS_DIR, LOGS_DIR,
            EVALUATION_RESULTS_DIR, PIPELINE_REPORTS_DIR, TEMPORAL_DECAY_RESULTS_DIR,
            SENTIMENT_INTEGRATION_RESULTS_DIR, TRAINING_LOGS_DIR, PIPELINE_LOGS_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _validate_configuration(self):
        """Validate configuration parameters"""
        
        # Validate symbols
        if not self.symbols or len(self.symbols) == 0:
            raise ValueError("At least one symbol must be specified")
        
        # Validate date range
        from datetime import datetime
        start = datetime.strptime(self.start_date, "%Y-%m-%d")
        end = datetime.strptime(self.end_date, "%Y-%m-%d")
        if start >= end:
            raise ValueError("start_date must be before end_date")
        
        # Validate splits
        if self.validation_split + self.test_split >= 1.0:
            raise ValueError("validation_split + test_split must be < 1.0")
        
        # Validate temporal decay parameters
        for horizon, params in self.temporal_decay_params.items():
            if params['lambda_decay'] <= 0 or params['lambda_decay'] > 1:
                raise ValueError(f"lambda_decay for {horizon}d must be in (0, 1]")
            if params['lookback_days'] < horizon:
                logger.warning(f"lookback_days ({params['lookback_days']}) < horizon ({horizon})")
        
        # Validate horizons match target_horizons
        decay_horizons = set(self.temporal_decay_params.keys())
        target_horizons = set(self.target_horizons)
        if decay_horizons != target_horizons:
            logger.warning(f"Decay horizons {decay_horizons} don't match target horizons {target_horizons}")
    
    def _setup_file_paths(self):
        """Set up standardized file paths"""
        
        # Core data files
        self.fnspid_raw_path = RAW_DATA_DIR / FNSPID_RAW_FILE
        self.core_dataset_path = PROCESSED_DATA_DIR / CORE_DATASET_FILE
        self.enhanced_dataset_path = PROCESSED_DATA_DIR / ENHANCED_DATASET_FILE
        
        # FNSPID processing outputs
        self.fnspid_filtered_articles_path = PROCESSED_DATA_DIR / FNSPID_FILTERED_ARTICLES_FILE
        self.fnspid_article_sentiment_path = PROCESSED_DATA_DIR / FNSPID_ARTICLE_SENTIMENT_FILE
        self.fnspid_daily_sentiment_path = PROCESSED_DATA_DIR / FNSPID_DAILY_SENTIMENT_FILE
        
        # Temporal decay outputs
        self.temporal_decay_data_path = PROCESSED_DATA_DIR / TEMPORAL_DECAY_DATA_FILE
        
        # Reports and metadata
        self.symbol_mapping_path = PROCESSED_DATA_DIR / SYMBOL_MAPPING_FILE
        self.data_summary_path = PROCESSED_DATA_DIR / DATA_SUMMARY_FILE
        self.pipeline_report_path = PIPELINE_REPORTS_DIR / PIPELINE_EXECUTION_REPORT_FILE
        self.temporal_decay_validation_path = TEMPORAL_DECAY_RESULTS_DIR / TEMPORAL_DECAY_VALIDATION_FILE
        self.sentiment_integration_report_path = SENTIMENT_INTEGRATION_RESULTS_DIR / SENTIMENT_INTEGRATION_REPORT_FILE
    
    def get_model_config(self, model_name: str = "default") -> 'ModelConfig':
        """Get model configuration based on pipeline config"""
        return ModelConfig(
            name=model_name,
            max_epochs=self.max_epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            attention_head_size=self.attention_head_size,
            max_encoder_length=self.max_encoder_length,
            max_prediction_length=self.max_prediction_length,
            target_horizons=self.target_horizons,
            early_stopping_patience=self.early_stopping_patience,
            reduce_lr_patience=self.reduce_lr_patience,
            validation_split=self.validation_split,
            test_split=self.test_split,
            use_mixed_precision=self.use_mixed_precision,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

@dataclass
class ModelConfig:
    """
    Model-specific configuration class
    
    Contains parameters specific to model training and architecture.
    Can be derived from PipelineConfig or used independently.
    """
    
    # Model identification
    name: str = "default"
    model_type: str = "TFT"  # TFT, LSTM, Ensemble
    
    # Training parameters
    max_epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    
    # Architecture parameters
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    attention_head_size: int = 4
    
    # Time series parameters
    max_encoder_length: int = 30
    max_prediction_length: int = 5
    target_horizons: List[int] = field(default_factory=lambda: [5, 30, 90])
    
    # Regularization and early stopping
    early_stopping_patience: int = 15
    reduce_lr_patience: int = 8
    gradient_clip_val: float = 1.0
    
    # Data parameters
    validation_split: float = 0.2
    test_split: float = 0.1
    min_prediction_length: int = 1
    
    # Advanced parameters
    use_mixed_precision: bool = True
    accumulate_grad_batches: int = 1
    num_workers: int = 4
    pin_memory: bool = True
    
    def __post_init__(self):
        """Validate model configuration"""
        if self.model_type not in ["TFT", "LSTM", "Ensemble"]:
            raise ValueError(f"Invalid model type: {self.model_type}")
        if self.validation_split + self.test_split >= 1.0:
            raise ValueError("validation_split + test_split must be < 1.0")

# =============================================================================
# DECAY PARAMETERS CONFIGURATION
# =============================================================================

@dataclass
class DecayParameters:
    """
    Temporal decay parameters for specific horizon
    
    Mathematical framework:
    sentiment_weighted = Σ(sentiment_i * exp(-λ_h * age_i)) / Σ(exp(-λ_h * age_i))
    """
    
    horizon: int  # Forecast horizon in days
    lambda_decay: float  # Decay rate parameter
    lookback_days: int  # Maximum lookback window
    min_sentiment_count: int = 3  # Minimum articles for reliable sentiment
    confidence_threshold: float = 0.5  # Minimum confidence threshold
    
    def __post_init__(self):
        """Validate decay parameters"""
        if self.lambda_decay <= 0 or self.lambda_decay > 1:
            raise ValueError(f"Lambda decay must be in (0, 1], got {self.lambda_decay}")
        if self.lookback_days < self.horizon:
            logger.warning(f"Lookback ({self.lookback_days}) < horizon ({self.horizon})")

# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_default_config() -> PipelineConfig:
    """Get default pipeline configuration"""
    return PipelineConfig()

def get_quick_test_config() -> PipelineConfig:
    """Get configuration for quick testing"""
    config = PipelineConfig()
    config.symbols = ['AAPL', 'MSFT']  # Fewer symbols
    config.fnspid_sample_ratio = 0.05  # 5% sample
    config.max_epochs = 10  # Fewer epochs
    config.start_date = "2023-01-01"  # Shorter date range
    config.end_date = "2023-06-30"
    return config

def get_research_config() -> PipelineConfig:
    """Get configuration for comprehensive research"""
    config = PipelineConfig()
    config.fnspid_sample_ratio = 0.5  # 50% of data
    config.max_epochs = 200  # More thorough training
    config.enable_tensorboard = True
    config.save_intermediate_results = True
    config.validate_data_at_each_stage = True
    return config

def create_decay_parameters_from_config(config: PipelineConfig) -> Dict[int, DecayParameters]:
    """Convert pipeline config decay params to DecayParameters objects"""
    decay_params = {}
    for horizon, params in config.temporal_decay_params.items():
        decay_params[horizon] = DecayParameters(
            horizon=horizon,
            lambda_decay=params['lambda_decay'],
            lookback_days=params['lookback_days'],
            min_sentiment_count=params['min_sentiment_count'],
            confidence_threshold=params['confidence_threshold']
        )
    return decay_params

# =============================================================================
# PATH UTILITY FUNCTIONS
# =============================================================================

def ensure_directories_exist():
    """Ensure all required directories exist"""
    config = get_default_config()
    # Directories are created in __post_init__, so this just triggers it
    logger.info("✅ All directories ensured to exist")

def get_file_path(file_type: str, config: Optional[PipelineConfig] = None) -> Path:
    """Get standardized file path for given file type"""
    if config is None:
        config = get_default_config()
    
    path_mapping = {
        'fnspid_raw': config.fnspid_raw_path,
        'core_dataset': config.core_dataset_path,
        'enhanced_dataset': config.enhanced_dataset_path,
        'fnspid_daily_sentiment': config.fnspid_daily_sentiment_path,
        'temporal_decay_data': config.temporal_decay_data_path,
        'pipeline_report': config.pipeline_report_path,
        'symbol_mapping': config.symbol_mapping_path
    }
    
    if file_type not in path_mapping:
        raise ValueError(f"Unknown file type: {file_type}. Available: {list(path_mapping.keys())}")
    
    return path_mapping[file_type]

# =============================================================================
# CONFIGURATION VALIDATION
# =============================================================================

def validate_environment() -> Dict[str, bool]:
    """Validate environment and dependencies"""
    validation = {
        'directories_exist': False,
        'pytorch_available': False,
        'pytorch_forecasting_available': False,
        'transformers_available': False,
        'ta_available': False
    }
    
    try:
        # Check directories
        ensure_directories_exist()
        validation['directories_exist'] = True
        
        # Check dependencies
        try:
            import torch
            validation['pytorch_available'] = True
        except ImportError:
            pass
        
        try:
            import pytorch_forecasting
            validation['pytorch_forecasting_available'] = True
        except ImportError:
            pass
        
        try:
            import transformers
            validation['transformers_available'] = True
        except ImportError:
            pass
        
        try:
            import ta
            validation['ta_available'] = True
        except ImportError:
            pass
        
    except Exception as e:
        logger.error(f"Environment validation failed: {e}")
    
    return validation

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Configuration classes
    'PipelineConfig',
    'ModelConfig', 
    'DecayParameters',
    
    # Path constants
    'DATA_DIR', 'PROCESSED_DATA_DIR', 'RAW_DATA_DIR', 'RESULTS_DIR', 'MODELS_DIR',
    'MODELS_CHECKPOINTS_DIR', 'EVALUATION_RESULTS_DIR',
    
    # File name constants
    'FNSPID_RAW_FILE', 'CORE_DATASET_FILE', 'ENHANCED_DATASET_FILE',
    'FNSPID_DAILY_SENTIMENT_FILE', 'TEMPORAL_DECAY_DATA_FILE',
    
    # Convenience functions
    'get_default_config', 'get_quick_test_config', 'get_research_config',
    'create_decay_parameters_from_config', 'get_file_path',
    
    # Validation functions
    'validate_environment', 'ensure_directories_exist'
]

if __name__ == "__main__":
    # Test configuration
    print("🔧 Testing pipeline configuration...")
    
    # Test default config
    config = get_default_config()
    print(f"✅ Default config: {len(config.symbols)} symbols, {config.start_date} to {config.end_date}")
    
    # Test environment validation
    env_validation = validate_environment()
    print(f"🔍 Environment validation: {sum(env_validation.values())}/{len(env_validation)} checks passed")
    
    # Test file paths
    core_path = get_file_path('core_dataset')
    print(f"📁 Core dataset path: {core_path}")
    
    print("🎉 Configuration module working correctly!")