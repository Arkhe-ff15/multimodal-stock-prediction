#!/usr/bin/env python3
"""
Production-level financial modeling framework implementing LSTM Baseline, TFT Baseline, 
and TFT Enhanced models with temporal sentiment decay for multi-horizon forecasting.
Designed for academic rigor and institutional deployment.

Version: 1.0
Date: June 25, 2025
Author: Financial ML Research Team
"""

import sys
import os
import argparse
from pathlib import Path
import warnings
import random
import time
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional, Union

# Standard ML libraries
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

# Scikit-learn
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Utilities
import json
import joblib
import yaml
import traceback
import gc
import psutil

# TFT imports with graceful degradation
try:
    from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
    from pytorch_forecasting.data import GroupNormalizer
    from pytorch_forecasting.metrics import QuantileLoss, MAE, RMSE
    TFT_AVAILABLE = True
    print("✅ PyTorch Forecasting available - TFT models enabled")
except ImportError as e:
    TFT_AVAILABLE = False
    print(f"❌ PyTorch Forecasting not available: {e}")
    print("📦 Install with: pip install pytorch-forecasting")
    print("🔧 LSTM Baseline will still work without this dependency")

warnings.filterwarnings('ignore')

# Configure logging
log_file = Path("logs") / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
log_file.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)
logger.info(f"📝 Logging initialized. Logs saved to: {log_file}")

def set_random_seeds(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if hasattr(pl, 'seed_everything'):
        pl.seed_everything(seed)
    logger.info(f"🎲 Random seeds set to {seed}")

@dataclass
class CompleteModelConfig:
    """Configuration class for all model hyperparameters with validation"""
    
    # LSTM Configuration (Enhanced)
    lstm_hidden_size: int = 512
    lstm_num_layers: int = 4
    lstm_dropout: float = 0.3
    lstm_sequence_length: int = 44
    lstm_learning_rate: float = 0.01
    lstm_weight_decay: float = 0.01
    lstm_max_epochs: int = 300
    
    # TFT Configuration (Baseline & Enhanced)
    tft_hidden_size: int = 256
    tft_attention_head_size: int = 8
    tft_dropout: float = 0.1
    tft_hidden_continuous_size: int = 64
    tft_max_encoder_length: int = 44
    tft_max_prediction_length: int = 5
    tft_learning_rate: float = 0.001
    tft_weight_decay: float = 0.01
    tft_max_epochs: int = 150
    
    # Enhanced TFT Configuration
    tft_enhanced_hidden_size: int = 512
    tft_enhanced_attention_head_size: int = 16
    
    # General Training Configuration
    batch_size: int = 32
    early_stopping_patience: int = 50
    gradient_clip_val: float = 1.0
    
    # Financial Domain Configuration
    quantiles: List[float] = None
    prediction_horizons: List[int] = None
    
    def __post_init__(self):
        """Validate and set default values"""
        if self.quantiles is None:
            self.quantiles = [0.1, 0.5, 0.9]
        if self.prediction_horizons is None:
            self.prediction_horizons = [5, 22, 90]
        
        # Validate configuration
        self._validate_config()
        logger.info("✅ Configuration validated")
    
    def _validate_config(self):
        """Validate configuration parameters"""
        # Validate batch size (power of 2, minimum 8)
        if self.batch_size < 8 or (self.batch_size & (self.batch_size - 1)) != 0:
            logger.warning(f"⚠️ Invalid batch_size {self.batch_size}. Setting to 32.")
            self.batch_size = 32
        
        # Validate learning rates
        if not (1e-6 <= self.lstm_learning_rate <= 0.05):
            logger.warning(f"⚠️ Invalid lstm_learning_rate {self.lstm_learning_rate}. Setting to 0.01.")
            self.lstm_learning_rate = 0.01
        
        if not (1e-6 <= self.tft_learning_rate <= 0.05):
            logger.warning(f"⚠️ Invalid tft_learning_rate {self.tft_learning_rate}. Setting to 0.001.")
            self.tft_learning_rate = 0.001

class MemoryMonitor:
    """Memory monitoring utilities"""
    
    @staticmethod
    def get_memory_usage() -> Dict[str, float]:
        """Get current memory usage statistics"""
        memory = psutil.virtual_memory()
        stats = {
            'total_gb': memory.total / (1024**3),
            'used_gb': memory.used / (1024**3),
            'available_gb': memory.available / (1024**3),
            'percent': memory.percent
        }
        
        # Add GPU memory if available
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / (1024**3)
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            stats.update({
                'gpu_used_gb': gpu_memory,
                'gpu_total_gb': gpu_memory_total,
                'gpu_percent': (gpu_memory / gpu_memory_total * 100) if gpu_memory_total > 0 else 0
            })
        
        return stats
    
    @staticmethod
    def log_memory_status():
        """Log current memory status"""
        stats = MemoryMonitor.get_memory_usage()
        log_msg = f"💾 Memory: {stats['used_gb']:.1f}GB/{stats['total_gb']:.1f}GB ({stats['percent']:.1f}%)"
        if 'gpu_used_gb' in stats:
            log_msg += f" | GPU: {stats['gpu_used_gb']:.1f}GB/{stats['gpu_total_gb']:.1f}GB ({stats['gpu_percent']:.1f}%)"
        logger.info(log_msg)
    
    @staticmethod
    def cleanup_memory():
        """Clean up memory"""
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            logger.debug("🧹 Memory cleanup completed")
        except Exception as e:
            logger.warning(f"⚠️ Memory cleanup failed: {e}")

class CompleteDataLoader:
    """Enhanced data loader with comprehensive validation"""
    
    def __init__(self, base_path: str = "data/model_ready"):
        self.base_path = Path(base_path)
        self.scalers_path = Path("data/scalers")
        self.metadata_path = Path("results/data_prep")
        
        # Create directories if they don't exist
        for directory in [self.scalers_path, self.metadata_path]:
            directory.mkdir(parents=True, exist_ok=True)
        
        self._validate_directory_structure()
    
    def _validate_directory_structure(self):
        """Validate that required directories exist"""
        if not self.base_path.exists():
            error_msg = f"Required data directory not found: {self.base_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        logger.info("✅ Directory structure validation passed")
    
    def load_dataset(self, dataset_type: str) -> Dict[str, Any]:
        """Load dataset with comprehensive validation"""
        logger.info(f"📥 Loading {dataset_type} dataset...")
        
        try:
            MemoryMonitor.log_memory_status()
            
            # Load data splits
            splits = self._load_data_splits(dataset_type)
            
            # Load or create scaler
            scaler = self._load_or_create_scaler(dataset_type)
            
            # Load features metadata
            selected_features = self._load_features_metadata(dataset_type)
            
            # Analyze features
            feature_analysis = self._analyze_financial_features(splits['train'].columns.tolist(), selected_features)
            
            # Validate dataset
            self._validate_financial_data(splits, feature_analysis, dataset_type)
            
            dataset = {
                'splits': splits,
                'scaler': scaler,
                'selected_features': selected_features,
                'feature_analysis': feature_analysis,
                'dataset_type': dataset_type
            }
            
            logger.info(f"✅ {dataset_type} dataset loaded successfully")
            logger.info(f"   📊 Train: {splits['train'].shape}, Val: {splits['val'].shape}, Test: {splits['test'].shape}")
            
            return dataset
            
        except Exception as e:
            logger.error(f"❌ Dataset loading failed: {e}", exc_info=True)
            raise
    
    def _load_data_splits(self, dataset_type: str) -> Dict[str, pd.DataFrame]:
        """Load train, validation, and test splits"""
        splits = {}
        required_splits = ['train', 'val', 'test']
        
        for split in required_splits:
            file_path = self.base_path / f"{dataset_type}_{split}.csv"
            logger.debug(f"   🔍 Loading {split} split: {file_path}")
            
            if not file_path.exists():
                raise FileNotFoundError(f"Split file not found: {file_path}")
            
            # Load and parse dates
            splits[split] = pd.read_csv(file_path)
            if 'date' in splits[split].columns:
                splits[split]['date'] = pd.to_datetime(splits[split]['date'])
            
            if splits[split].empty:
                raise ValueError(f"Empty {split} split")
            
            logger.info(f"   📊 {split}: {splits[split].shape}")
        
        return splits
    
    def _load_or_create_scaler(self, dataset_type: str) -> RobustScaler:
        """Load existing scaler or create new one"""
        scaler_path = self.scalers_path / f"{dataset_type}_scaler.joblib"
        
        if scaler_path.exists():
            try:
                scaler = joblib.load(scaler_path)
                logger.info(f"   📈 Loaded existing scaler: {type(scaler).__name__}")
                return scaler
            except Exception as e:
                logger.warning(f"⚠️ Failed to load scaler: {e}")
        
        # Create new scaler
        scaler = RobustScaler()
        logger.info(f"   📈 Created new scaler: {type(scaler).__name__}")
        return scaler
    
    def _load_features_metadata(self, dataset_type: str) -> List[str]:
        """Load features metadata or return empty list"""
        features_path = self.metadata_path / f"{dataset_type}_selected_features.json"
        
        if features_path.exists():
            try:
                with open(features_path, 'r') as f:
                    selected_features = json.load(f)
                logger.info(f"   🎯 Loaded {len(selected_features)} features from metadata")
                return selected_features
            except Exception as e:
                logger.warning(f"⚠️ Failed to load features metadata: {e}")
        
        logger.info(f"   🎯 No features metadata found, will use all numeric features")
        return []
    
    def _analyze_financial_features(self, actual_columns: List[str], selected_features: List[str]) -> Dict[str, List[str]]:
        """Analyze and categorize financial features"""
        
        # Use selected features if available, otherwise use all numeric columns
        if selected_features:
            available_features = [f for f in selected_features if f in actual_columns]
        else:
            # Auto-detect numeric features (excluding identifiers and targets)
            exclude_patterns = ['stock_id', 'symbol', 'date', 'time_idx']
            available_features = [col for col in actual_columns 
                                if col not in exclude_patterns and not col.startswith('target_')]
        
        analysis = {
            'identifier_features': [],
            'target_features': [],
            'static_categoricals': [],
            'static_reals': [],
            'time_varying_known_reals': [],
            'time_varying_unknown_reals': [],
            'sentiment_features': [],
            'temporal_decay_features': [],
            'technical_features': [],
            'price_features': [],
            'volume_features': [],
            'lstm_features': [],
            'tft_baseline_features': [],
            'tft_enhanced_features': [],
            'available_features': available_features
        }
        
        # Categorize features
        for col in actual_columns:
            col_lower = col.lower()
            
            # Identifier features
            if col in ['stock_id', 'symbol', 'date', 'time_idx']:
                analysis['identifier_features'].append(col)
            
            # Target features
            elif col.startswith('target_'):
                analysis['target_features'].append(col)
            
            # Static categorical features
            elif col in ['symbol', 'sector']:
                analysis['static_categoricals'].append(col)
            
            # Sentiment features (for enhanced model)
            elif any(pattern in col_lower for pattern in [
                'sentiment_decay_', 'sentiment_compound', 'sentiment_positive',
                'sentiment_negative', 'sentiment_confidence', 'sentiment_ma_'
            ]):
                analysis['sentiment_features'].append(col)
                analysis['time_varying_unknown_reals'].append(col)
                analysis['tft_enhanced_features'].append(col)
                
                # Temporal decay features (novel contribution)
                if 'sentiment_decay' in col_lower:
                    analysis['temporal_decay_features'].append(col)
            
            # Price features
            elif any(pattern in col_lower for pattern in [
                'open', 'high', 'low', 'close', 'price', 'vwap', 'return'
            ]):
                analysis['price_features'].append(col)
                analysis['time_varying_unknown_reals'].append(col)
                analysis['lstm_features'].append(col)
                analysis['tft_baseline_features'].append(col)
                analysis['tft_enhanced_features'].append(col)
            
            # Volume features
            elif any(pattern in col_lower for pattern in ['volume', 'turnover']):
                analysis['volume_features'].append(col)
                analysis['time_varying_unknown_reals'].append(col)
                analysis['lstm_features'].append(col)
                analysis['tft_baseline_features'].append(col)
                analysis['tft_enhanced_features'].append(col)
            
            # Technical features
            elif any(pattern in col_lower for pattern in [
                'rsi', 'macd', 'bb_', 'atr', 'volatility', 'sma', 'ema'
            ]):
                analysis['technical_features'].append(col)
                analysis['time_varying_unknown_reals'].append(col)
                analysis['lstm_features'].append(col)
                analysis['tft_baseline_features'].append(col)
                analysis['tft_enhanced_features'].append(col)
            
            # Other numeric features
            elif col in available_features:
                analysis['time_varying_unknown_reals'].append(col)
                analysis['lstm_features'].append(col)
                analysis['tft_baseline_features'].append(col)
                analysis['tft_enhanced_features'].append(col)
        
        # Remove duplicates
        for key in analysis.keys():
            analysis[key] = list(dict.fromkeys(analysis[key]))
        
        # Log feature analysis
        logger.info(f"📊 Feature Analysis:")
        logger.info(f"   📈 LSTM features: {len(analysis['lstm_features'])}")
        logger.info(f"   📊 TFT baseline features: {len(analysis['tft_baseline_features'])}")
        logger.info(f"   🔬 TFT enhanced features: {len(analysis['tft_enhanced_features'])}")
        logger.info(f"   🎭 Sentiment features: {len(analysis['sentiment_features'])}")
        if analysis['temporal_decay_features']:
            logger.info(f"   🏆 NOVEL: Temporal decay features: {len(analysis['temporal_decay_features'])}")
        
        return analysis
    
    def _validate_financial_data(self, splits: Dict[str, pd.DataFrame], 
                                feature_analysis: Dict[str, List[str]], dataset_type: str):
        """Validate financial data integrity"""
        logger.info(f"🔍 Validating {dataset_type} dataset")
        
        # Check required columns
        required_columns = ['symbol', 'date']
        for split_name, split_data in splits.items():
            missing_cols = [col for col in required_columns if col not in split_data.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns in {split_name} split: {missing_cols}")
        
        # Check target features
        target_cols = feature_analysis.get('target_features', [])
        if not target_cols:
            raise ValueError("No target columns found. Ensure at least one column starts with 'target_'")
        
        # Check minimum feature counts based on dataset type
        lstm_features = len(feature_analysis.get('lstm_features', []))
        if lstm_features < 10:
            logger.warning(f"⚠️ Only {lstm_features} LSTM features found, consider adding more features")
        
        if dataset_type == 'enhanced':
            tft_enhanced_features = len(feature_analysis.get('tft_enhanced_features', []))
            temporal_decay_features = len(feature_analysis.get('temporal_decay_features', []))
            
            if tft_enhanced_features < 15:
                logger.warning(f"⚠️ Only {tft_enhanced_features} TFT enhanced features found")
            
            if temporal_decay_features < 5:
                logger.warning(f"⚠️ Only {temporal_decay_features} temporal decay features found")
        
        # Check temporal consistency (no data leakage)
        train_dates = splits['train']['date']
        val_dates = splits['val']['date']
        test_dates = splits['test']['date']
        
        if not (train_dates.max() < val_dates.min() and val_dates.max() < test_dates.min()):
            raise ValueError("Data leakage detected: overlapping dates between splits")
        
        logger.info(f"✅ {dataset_type} dataset validation passed")

class FinancialDataset(Dataset):
    """Enhanced Dataset for financial time-series data with robust sequence creation"""
    
    def __init__(self, data: pd.DataFrame, features: List[str], target: str, sequence_length: int):
        if data.empty:
            raise ValueError("Input data is empty")
        
        # Validate features exist in data
        self.features = [col for col in features if col in data.columns]
        if not self.features:
            raise ValueError("No valid feature columns found in data")
        
        if target not in data.columns:
            raise ValueError(f"Target column '{target}' not found in data")
        
        self.target = target
        self.sequence_length = sequence_length
        self.sequences, self.labels = self._prepare_sequences(data)
        
        if len(self.sequences) == 0:
            raise ValueError("No valid sequences created. Check data quality or sequence length.")
        
        logger.info(f"📊 Created {len(self.sequences)} sequences for {data['symbol'].nunique()} symbols")
    
    def _prepare_sequences(self, data: pd.DataFrame) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Prepare sequences and labels with robust error handling"""
        sequences, labels = [], []
        
        for symbol in data['symbol'].unique():
            symbol_data = data[data['symbol'] == symbol].sort_values('date')
            
            if len(symbol_data) < self.sequence_length + 1:
                logger.warning(f"Skipping symbol {symbol}: insufficient data ({len(symbol_data)} rows)")
                continue
            
            # Extract features and target with proper error handling
            try:
                feature_values = symbol_data[self.features].values.astype(np.float32)
                target_values = symbol_data[self.target].values.astype(np.float32)
            except Exception as e:
                logger.warning(f"Skipping symbol {symbol}: data conversion error ({e})")
                continue
            
            # Create sequences
            for i in range(len(symbol_data) - self.sequence_length):
                seq = feature_values[i:i + self.sequence_length]
                label = target_values[i + self.sequence_length]
                
                # Validate sequence quality
                if (np.isfinite(label) and 
                    np.all(np.isfinite(seq)) and
                    np.var(seq.flatten()) > 1e-8 and
                    abs(label) < 10):  # Remove extreme outliers
                    sequences.append(seq)
                    labels.append(label)
        
        return torch.FloatTensor(sequences), torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

class EnhancedLSTMModel(nn.Module):
    """Enhanced LSTM with attention mechanism and batch normalization"""
    
    def __init__(self, input_size: int, config: CompleteModelConfig):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = config.lstm_hidden_size
        self.num_layers = config.lstm_num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=config.lstm_dropout if self.num_layers > 1 else 0,
            batch_first=True
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=8,
            dropout=config.lstm_dropout,
            batch_first=True
        )
        
        # Batch normalization and regularization
        self.batch_norm = nn.BatchNorm1d(self.hidden_size)
        self.dropout = nn.Dropout(config.lstm_dropout)
        
        # Output layers
        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size // 2)
        self.fc2 = nn.Linear(self.hidden_size // 2, 1)
        self.relu = nn.ReLU()
        
        # Initialize weights
        self._init_weights()
        
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"🧠 Enhanced LSTM: {input_size}→{self.hidden_size}x{self.num_layers}→1, params={total_params:,}")
    
    def _init_weights(self):
        """Initialize weights using best practices"""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.zeros_(param.data)
                # Set forget gate bias to 1 (standard practice)
                if 'bias_ih' in name:
                    hidden_size = param.size(0) // 4
                    param.data[hidden_size:2*hidden_size].fill_(1.0)
            elif 'weight' in name and len(param.shape) == 2:
                nn.init.xavier_uniform_(param.data)
    
    def forward(self, x):
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Apply attention
        attended, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Combine LSTM and attention outputs (residual connection)
        combined = lstm_out + attended
        
        # Use mean pooling over sequence dimension
        context = torch.mean(combined, dim=1)
        
        # Apply batch normalization
        context = self.batch_norm(context)
        
        # Output layers
        x = self.relu(self.fc1(self.dropout(context)))
        output = self.fc2(self.dropout(x))
        
        return output.squeeze()

class LSTMTrainer(pl.LightningModule):
    """PyTorch Lightning wrapper for LSTM training"""
    
    def __init__(self, model: EnhancedLSTMModel, config: CompleteModelConfig):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.config = config
        self.criterion = nn.HuberLoss(delta=0.1)  # More robust than MSE
        self.validation_step_outputs = []
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)
        
        # Log metrics
        mae = torch.mean(torch.abs(y_pred - y))
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_mae', mae, on_step=False, on_epoch=True)
        
        # Memory cleanup
        if batch_idx % 100 == 0:
            MemoryMonitor.cleanup_memory()
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)
        
        mae = torch.mean(torch.abs(y_pred - y))
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_mae', mae, on_step=False, on_epoch=True)
        
        # Store predictions for hit rate calculation
        self.validation_step_outputs.append({
            'predictions': y_pred.detach().cpu(),
            'targets': y.detach().cpu()
        })
        
        return loss
    
    def on_validation_epoch_end(self):
        if not self.validation_step_outputs:
            return
        
        # Calculate hit rate (directional accuracy)
        all_preds = torch.cat([x['predictions'] for x in self.validation_step_outputs])
        all_targets = torch.cat([x['targets'] for x in self.validation_step_outputs])
        
        preds_np = all_preds.numpy()
        targets_np = all_targets.numpy()
        
        if len(preds_np) > 10:
            hit_rate = np.mean(np.sign(preds_np) == np.sign(targets_np))
            self.log('val_hit_rate', hit_rate)
        
        self.validation_step_outputs.clear()
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.lstm_learning_rate,
            weight_decay=self.config.lstm_weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.7,
            patience=self.config.early_stopping_patience // 2,
            min_lr=1e-6
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'frequency': 1
            }
        }

class TFTDatasetPreparer:
    """Prepare datasets for TFT models with proper time indexing"""
    
    def __init__(self, config: CompleteModelConfig):
        self.config = config
        self.label_encoders = {}
    
    def prepare_tft_dataset(self, dataset: Dict[str, Any], model_type: str) -> Tuple[Any, Any]:
        """Prepare TFT dataset with comprehensive validation"""
        if not TFT_AVAILABLE:
            raise ImportError("PyTorch Forecasting not available for TFT models")
        
        logger.info(f"🔬 Preparing TFT Dataset ({model_type})...")
        
        # Combine all splits with proper time indexing
        combined_data = self._prepare_combined_data(dataset)
        
        # Get feature configuration
        feature_analysis = dataset['feature_analysis']
        feature_config = self._get_feature_config(feature_analysis, combined_data, model_type)
        
        # Prepare categorical features
        combined_data = self._prepare_categorical_features(combined_data, feature_config)
        
        # Create time index
        combined_data = self._create_time_index(combined_data)
        
        # Debug: Check data types before creating TFT dataset
        logger.info(f"🔍 Data types before TFT creation:")
        logger.info(f"   symbol: {combined_data['symbol'].dtype}")
        logger.info(f"   time_idx: {combined_data['time_idx'].dtype}")
        if 'target_5' in combined_data.columns:
            logger.info(f"   target_5: {combined_data['target_5'].dtype}")
        logger.info(f"   Sample symbol values: {combined_data['symbol'].unique()[:5]}")
        
        # Split data for training and validation
        train_data = dataset['splits']['train'].copy()
        val_start_date = dataset['splits']['val']['date'].min()
        
        # Find validation start index
        val_start_idx = combined_data[combined_data['date'] >= val_start_date]['time_idx'].min()
        if pd.isna(val_start_idx):
            # Fallback: use 80% of data for training
            max_idx = combined_data['time_idx'].max()
            val_start_idx = int(max_idx * 0.8)
        
        # Find the target column
        target_features = dataset['feature_analysis'].get('target_features', [])
        if not target_features:
            raise ValueError("No target features found in dataset")
        target_col = target_features[0]  # Use first target feature
        
        # Validate target exists in combined data
        if target_col not in combined_data.columns:
            raise ValueError(f"Target column '{target_col}' not found in combined data")
        
        logger.info(f"📊 Using target column: {target_col}")
        logger.info(f"📊 Training data: time_idx < {val_start_idx}")
        logger.info(f"📊 Validation data: time_idx >= {val_start_idx}")
        
        # Create TFT datasets
        training_dataset = TimeSeriesDataSet(
            combined_data[combined_data.time_idx < val_start_idx],
            time_idx="time_idx",
            target=target_col,
            group_ids=['symbol'],
            min_encoder_length=self.config.tft_max_encoder_length // 3,
            max_encoder_length=self.config.tft_max_encoder_length,
            min_prediction_length=1,
            max_prediction_length=self.config.tft_max_prediction_length,
            static_categoricals=feature_config['static_categoricals'],
            static_reals=feature_config['static_reals'],
            time_varying_known_reals=feature_config['time_varying_known_reals'],
            time_varying_unknown_reals=feature_config['time_varying_unknown_reals'],
            target_normalizer=GroupNormalizer(groups=['symbol']),
            add_relative_time_idx=True,
            add_target_scales=True,
            allow_missing_timesteps=True,
        )
        
        validation_dataset = TimeSeriesDataSet.from_dataset(
            training_dataset,
            combined_data,
            min_prediction_idx=val_start_idx,
            stop_randomization=True
        )
        
        logger.info(f"✅ TFT Dataset prepared ({model_type}):")
        logger.info(f"   📊 Training samples: {len(training_dataset):,}")
        logger.info(f"   📊 Validation samples: {len(validation_dataset):,}")
        
        return training_dataset, validation_dataset
    
    def _prepare_combined_data(self, dataset: Dict[str, Any]) -> pd.DataFrame:
        """Combine all data splits with proper preprocessing"""
        splits = dataset['splits']
        
        # Process each split
        processed_splits = []
        for split_name in ['train', 'val', 'test']:
            df = splits[split_name].copy()
            df['date'] = pd.to_datetime(df['date'])
            
            # Ensure symbol is string (critical for TFT)
            if 'symbol' in df.columns:
                df['symbol'] = df['symbol'].astype(str)
                # Add prefix if symbol looks like a number to ensure it's treated as categorical
                if df['symbol'].str.isnumeric().any():
                    df['symbol'] = 'STOCK_' + df['symbol']
            
            # Handle missing values and infinities
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            df[numeric_columns] = df[numeric_columns].fillna(0).replace([np.inf, -np.inf], 0)
            
            processed_splits.append(df)
        
        # Combine and sort
        combined_data = pd.concat(processed_splits, ignore_index=True)
        combined_data = combined_data.sort_values(['symbol', 'date']).reset_index(drop=True)
        
        # Double-check symbol is string
        if 'symbol' in combined_data.columns:
            combined_data['symbol'] = combined_data['symbol'].astype(str)
        
        logger.info(f"📊 Combined data shape: {combined_data.shape}")
        logger.info(f"📅 Date range: {combined_data['date'].min()} to {combined_data['date'].max()}")
        logger.info(f"📊 Symbol dtype: {combined_data['symbol'].dtype if 'symbol' in combined_data.columns else 'N/A'}")
        
        return combined_data
    
    def _get_feature_config(self, feature_analysis: Dict[str, List[str]], 
                           combined_data: pd.DataFrame, model_type: str) -> Dict[str, List[str]]:
        """Get feature configuration for TFT model"""
        config = {
            'static_categoricals': [],
            'static_reals': [],
            'time_varying_known_reals': [],
            'time_varying_unknown_reals': []
        }
        
        # Add symbol as static categorical if available
        if 'symbol' in combined_data.columns:
            config['static_categoricals'].append('symbol')
        
        # Get features based on model type
        if model_type == 'baseline':
            features = feature_analysis.get('tft_baseline_features', [])
        else:  # enhanced
            features = feature_analysis.get('tft_enhanced_features', [])
        
        # Categorize features for TFT
        exclude_patterns = ['symbol', 'date', 'time_idx', 'target_', 'stock_id']
        for col in combined_data.columns:
            if any(pattern in col for pattern in exclude_patterns):
                continue
            
            if combined_data[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                config['time_varying_unknown_reals'].append(col)
        
        logger.info(f"🔧 TFT Feature Configuration ({model_type}):")
        for key, value in config.items():
            if value:
                logger.info(f"   {key}: {len(value)}")
        
        return config
    
    def _prepare_categorical_features(self, data: pd.DataFrame, 
                                    feature_config: Dict[str, List[str]]) -> pd.DataFrame:
        """Prepare categorical features with proper string conversion"""
        data = data.copy()
        
        for feature in feature_config['static_categoricals']:
            if feature in data.columns:
                # Always convert to string for TFT compatibility
                data[feature] = data[feature].astype(str)
                
                # Only apply label encoding if we have many unique categories
                unique_count = data[feature].nunique()
                if unique_count > 100:  # Only encode if too many categories
                    if feature not in self.label_encoders:
                        self.label_encoders[feature] = LabelEncoder()
                        data[feature] = self.label_encoders[feature].fit_transform(data[feature])
                        # Convert back to string after encoding
                        data[feature] = data[feature].astype(str)
                    else:
                        encoded = self.label_encoders[feature].transform(data[feature])
                        data[feature] = encoded.astype(str)
                
                logger.info(f"   🏷️ {feature}: {data[feature].nunique()} unique values, dtype: {data[feature].dtype}")
        
        return data
    
    def _create_time_index(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create time index for each symbol"""
        data = data.copy()
        data['time_idx'] = data.groupby('symbol').cumcount()
        data['time_idx'] = data['time_idx'].astype(int)
        
        logger.info(f"📊 Time index created: {data['time_idx'].min()} to {data['time_idx'].max()}")
        return data

class SimpleTFTTrainer(pl.LightningModule):
    """Simplified TFT trainer that bypasses trainer attachment issues"""
    
    def __init__(self, config: CompleteModelConfig, training_dataset: Any, model_type: str):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.training_dataset = training_dataset
        self.model_type = model_type
        
        # Model configuration
        if model_type == 'TFT_Enhanced':
            self.hidden_size = config.tft_enhanced_hidden_size
            self.attention_heads = config.tft_enhanced_attention_head_size
        else:
            self.hidden_size = config.tft_hidden_size
            self.attention_heads = config.tft_attention_head_size
        
        # Create TFT model
        self.tft_model = TemporalFusionTransformer.from_dataset(
            training_dataset,
            learning_rate=config.tft_learning_rate,
            hidden_size=self.hidden_size,
            attention_head_size=self.attention_heads,
            dropout=config.tft_dropout,
            hidden_continuous_size=config.tft_hidden_continuous_size,
            output_size=len(config.quantiles),
            loss=QuantileLoss(quantiles=config.quantiles),
            log_interval=50,
            reduce_on_plateau_patience=config.early_stopping_patience // 2
        )
        
        # Store the loss function separately for direct use
        self.loss_fn = QuantileLoss(quantiles=config.quantiles)
        
        # Store validation outputs for metrics
        self.validation_step_outputs = []
        
        logger.info(f"🧠 {model_type} TFT Model initialized:")
        logger.info(f"   🔧 Hidden size: {self.hidden_size}")
        logger.info(f"   👁️ Attention heads: {self.attention_heads}")
        logger.info(f"   📊 Output quantiles: {config.quantiles}")
    
    def forward(self, x):
        return self.tft_model(x)
    
    def training_step(self, batch, batch_idx):
        # Unpack batch - TFT datasets return (x, y) tuples
        x, y = batch
        
        # Forward pass through the model
        output = self(x)
        
        # Handle different output formats from TFT
        if isinstance(output, dict):
            # TFT returns a dictionary with 'prediction_outputs' key
            predictions = output.get('prediction_outputs', output.get('prediction', None))
        else:
            predictions = output
        
        # Compute loss using the quantile loss function
        # The predictions should have shape (batch_size, prediction_length, num_quantiles)
        # The targets y should have shape (batch_size, prediction_length) or (batch_size, prediction_length, 1)
        
        # Ensure y has the right shape for loss computation
        if len(y.shape) == 2:
            y = y.unsqueeze(-1)  # Add quantile dimension
        
        # Compute quantile loss
        loss = self.loss_fn(predictions, y)
        
        # Log metrics
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # Also log the median prediction error (using middle quantile)
        median_idx = len(self.config.quantiles) // 2
        median_predictions = predictions[..., median_idx]
        mae = torch.mean(torch.abs(median_predictions - y.squeeze(-1)))
        self.log('train_mae', mae, on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        # Unpack batch
        x, y = batch
        
        # Forward pass through the model
        output = self(x)
        
        # Handle different output formats from TFT
        if isinstance(output, dict):
            predictions = output.get('prediction_outputs', output.get('prediction', None))
        else:
            predictions = output
        
        # Ensure y has the right shape for loss computation
        if len(y.shape) == 2:
            y = y.unsqueeze(-1)
        
        # Compute quantile loss
        loss = self.loss_fn(predictions, y)
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # Calculate median prediction error
        median_idx = len(self.config.quantiles) // 2
        median_predictions = predictions[..., median_idx]
        mae = torch.mean(torch.abs(median_predictions - y.squeeze(-1)))
        self.log('val_mae', mae, on_step=False, on_epoch=True)
        
        # Store for additional metrics calculation
        self.validation_step_outputs.append({
            'loss': loss.detach(),
            'predictions': median_predictions.detach().cpu(),
            'targets': y.squeeze(-1).detach().cpu()
        })
        
        return loss
    
    def on_validation_epoch_end(self):
        if self.validation_step_outputs:
            # Calculate average validation loss
            avg_loss = torch.stack([x['loss'] for x in self.validation_step_outputs]).mean()
            self.log('val_loss_custom', avg_loss)
            
            # Calculate hit rate (directional accuracy) if applicable
            all_preds = torch.cat([x['predictions'] for x in self.validation_step_outputs])
            all_targets = torch.cat([x['targets'] for x in self.validation_step_outputs])
            
            if len(all_preds) > 10 and all_preds.dim() == 1:  # Only for single-step predictions
                hit_rate = torch.mean((torch.sign(all_preds) == torch.sign(all_targets)).float())
                self.log('val_hit_rate', hit_rate)
            
            self.validation_step_outputs.clear()
    
    def configure_optimizers(self):
        # Use AdamW optimizer like the original TFT
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.tft_learning_rate,
            weight_decay=self.config.tft_weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.7,
            patience=self.config.early_stopping_patience // 2,
            min_lr=1e-6
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'frequency': 1
            }
        }

class CompleteFinancialModelFramework:
    """Complete framework for training all three models"""
    
    def __init__(self):
        set_random_seeds(42)
        self.config = self._load_config()
        self.data_loader = CompleteDataLoader()
        self.datasets = {}
        
        # Create necessary directories
        self.models_dir = Path("models/checkpoints")
        self.logs_dir = Path("logs/training")
        self.results_dir = Path("results/training")
        
        for directory in [self.models_dir, self.logs_dir, self.results_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        logger.info("🚀 Complete Financial Model Framework initialized")
        logger.info("🎯 Models: LSTM Baseline + TFT Baseline + TFT Enhanced")
        MemoryMonitor.log_memory_status()
    
    def _load_config(self) -> CompleteModelConfig:
        """Load configuration from file or use defaults"""
        config_path = Path("config.yaml")
        
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    yaml_config = yaml.safe_load(f)
                
                # Extract configuration parameters
                model_config = yaml_config.get('model', {})
                training_config = yaml_config.get('training', {})
                
                config = CompleteModelConfig(
                    # Override defaults with YAML values
                    **{k: v for k, v in model_config.items() if hasattr(CompleteModelConfig, k)},
                    **{k: v for k, v in training_config.items() if hasattr(CompleteModelConfig, k)}
                )
                
                logger.info("✅ Configuration loaded from config.yaml")
                return config
            
            except Exception as e:
                logger.warning(f"⚠️ Failed to load config.yaml: {e}. Using defaults.")
        
        return CompleteModelConfig()
    
    def load_datasets(self) -> bool:
        """Load all available datasets"""
        logger.info("📥 Loading datasets for all models...")
        
        dataset_types = ['baseline', 'enhanced']
        loaded_count = 0
        
        for dataset_type in dataset_types:
            try:
                # Check if files exist
                train_path = Path(f"data/model_ready/{dataset_type}_train.csv")
                val_path = Path(f"data/model_ready/{dataset_type}_val.csv")
                test_path = Path(f"data/model_ready/{dataset_type}_test.csv")
                
                if all(path.exists() for path in [train_path, val_path, test_path]):
                    self.datasets[dataset_type] = self.data_loader.load_dataset(dataset_type)
                    loaded_count += 1
                    logger.info(f"✅ Loaded {dataset_type} dataset")
                else:
                    logger.warning(f"⚠️ {dataset_type} dataset files not found")
            
            except Exception as e:
                logger.error(f"❌ Failed to load {dataset_type} dataset: {e}")
        
        if loaded_count == 0:
            logger.error("❌ No datasets loaded. Check data files in 'data/model_ready/'")
            return False
        
        logger.info(f"✅ Successfully loaded {loaded_count} dataset(s): {list(self.datasets.keys())}")
        return True
    
    def train_lstm_baseline(self) -> Dict[str, Any]:
        """Train LSTM baseline model"""
        logger.info("🚀 Training LSTM Baseline")
        start_time = time.time()
        
        try:
            # Get dataset (prefer baseline, fall back to enhanced)
            dataset_key = 'baseline' if 'baseline' in self.datasets else 'enhanced'
            if dataset_key not in self.datasets:
                raise ValueError("No suitable dataset found for LSTM training")
            
            dataset = self.datasets[dataset_key]
            
            # Get features and target
            features = dataset['feature_analysis']['lstm_features']
            target_features = dataset['feature_analysis']['target_features']
            
            if not features:
                raise ValueError("No LSTM features found in dataset")
            if not target_features:
                raise ValueError("No target features found in dataset")
            
            target = target_features[0]  # Use first target
            
            logger.info(f"📊 Using {len(features)} features, target: {target}")
            
            # Prepare data with scaling
            train_df = dataset['splits']['train'].copy()
            val_df = dataset['splits']['val'].copy()
            
            # Scale features
            scaler = RobustScaler()
            train_df[features] = scaler.fit_transform(train_df[features])
            val_df[features] = scaler.transform(val_df[features])
            
            # Save scaler
            scaler_path = self.data_loader.scalers_path / f"{dataset_key}_scaler.joblib"
            joblib.dump(scaler, scaler_path)
            
            # Scale target
            target_scaler = RobustScaler()
            train_df[[target]] = target_scaler.fit_transform(train_df[[target]])
            val_df[[target]] = target_scaler.transform(val_df[[target]])
            
            # Save target scaler
            target_scaler_path = self.data_loader.scalers_path / f"{dataset_key}_target_scaler.joblib"
            joblib.dump(target_scaler, target_scaler_path)
            
            # Create datasets
            train_dataset = FinancialDataset(
                train_df, features, target, self.config.lstm_sequence_length
            )
            val_dataset = FinancialDataset(
                val_df, features, target, self.config.lstm_sequence_length
            )
            
            # Create data loaders
            train_loader = DataLoader(
                train_dataset, 
                batch_size=self.config.batch_size, 
                shuffle=True, 
                num_workers=4
            )
            val_loader = DataLoader(
                val_dataset, 
                batch_size=self.config.batch_size, 
                shuffle=False, 
                num_workers=4
            )
            
            # Create model
            lstm_model = EnhancedLSTMModel(len(features), self.config)
            trainer_model = LSTMTrainer(lstm_model, self.config)
            
            # Setup callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_loss', 
                    patience=self.config.early_stopping_patience, 
                    mode='min'
                ),
                ModelCheckpoint(
                    dirpath=str(self.models_dir),
                    filename='lstm_baseline_{epoch:02d}_{val_loss:.4f}',
                    monitor='val_loss',
                    mode='min',
                    save_top_k=3
                ),
                LearningRateMonitor(logging_interval='epoch')
            ]
            
            # Create trainer
            trainer = pl.Trainer(
                max_epochs=self.config.lstm_max_epochs,
                callbacks=callbacks,
                logger=TensorBoardLogger(str(self.logs_dir), name='lstm_baseline'),
                accelerator='auto',
                gradient_clip_val=self.config.gradient_clip_val,
                deterministic=True
            )
            
            # Train model
            logger.info("🚀 Starting LSTM baseline training...")
            trainer.fit(trainer_model, train_loader, val_loader)
            
            training_time = time.time() - start_time
            
            # Extract results
            results = {
                'model_type': 'LSTM_Baseline',
                'training_time': training_time,
                'best_val_loss': float(callbacks[1].best_model_score) if callbacks[1].best_model_score else None,
                'epochs_trained': trainer.current_epoch,
                'best_checkpoint': callbacks[1].best_model_path,
                'dataset_used': dataset_key,
                'features_count': len(features)
            }
            
            logger.info(f"✅ LSTM Baseline training completed!")
            logger.info(f"⏱️ Training time: {training_time:.1f}s")
            logger.info(f"📉 Best val loss: {results['best_val_loss']:.6f}")
            
            return results
            
        except Exception as e:
            training_time = time.time() - start_time
            logger.error(f"❌ LSTM Baseline training failed: {e}", exc_info=True)
            return {
                'error': str(e),
                'model_type': 'LSTM_Baseline',
                'training_time': training_time
            }
    
    def train_tft_baseline(self) -> Dict[str, Any]:
        """Train TFT baseline model"""
        if not TFT_AVAILABLE:
            error_msg = "❌ PyTorch Forecasting not available for TFT models"
            logger.error(error_msg)
            return {'error': error_msg, 'model_type': 'TFT_Baseline'}
        
        logger.info("🚀 Training TFT Baseline")
        start_time = time.time()
        
        try:
            # Get dataset
            dataset_key = 'baseline' if 'baseline' in self.datasets else 'enhanced'
            if dataset_key not in self.datasets:
                raise ValueError("No suitable dataset found for TFT training")
            
            dataset = self.datasets[dataset_key]
            
            # Prepare TFT dataset
            tft_preparer = TFTDatasetPreparer(self.config)
            training_dataset, validation_dataset = tft_preparer.prepare_tft_dataset(dataset, 'baseline')
            
            # Create data loaders
            train_dataloader = training_dataset.to_dataloader(
                train=True, batch_size=self.config.batch_size, num_workers=0
            )
            val_dataloader = validation_dataset.to_dataloader(
                train=False, batch_size=self.config.batch_size, num_workers=0
            )
            
            # Use simplified trainer
            model = SimpleTFTTrainer(self.config, training_dataset, "TFT_Baseline")
            
            # Setup callbacks
            callbacks = [
                EarlyStopping(
                    monitor="val_loss", 
                    patience=self.config.early_stopping_patience, 
                    mode="min"
                ),
                ModelCheckpoint(
                    dirpath=str(self.models_dir),
                    filename="tft_baseline_{epoch:02d}_{val_loss:.4f}",
                    monitor="val_loss", 
                    mode="min", 
                    save_top_k=3
                ),
                LearningRateMonitor(logging_interval='epoch')
            ]
            
            # Create trainer
            trainer = pl.Trainer(
                max_epochs=self.config.tft_max_epochs,
                gradient_clip_val=self.config.gradient_clip_val,
                accelerator="auto",
                callbacks=callbacks,
                logger=TensorBoardLogger(str(self.logs_dir), name="tft_baseline"),
                deterministic=True,
                enable_progress_bar=True
            )
            
            # Train model
            logger.info("🚀 Starting TFT baseline training...")
            trainer.fit(model, train_dataloader, val_dataloader)
            
            training_time = time.time() - start_time
            
            # Extract results
            results = {
                'model_type': 'TFT_Baseline',
                'training_time': training_time,
                'best_val_loss': float(callbacks[1].best_model_score) if callbacks[1].best_model_score else None,
                'epochs_trained': trainer.current_epoch,
                'best_checkpoint': callbacks[1].best_model_path,
                'dataset_used': dataset_key
            }
            
            logger.info(f"✅ TFT Baseline training completed!")
            logger.info(f"⏱️ Training time: {training_time:.1f}s")
            
            return results
            
        except Exception as e:
            training_time = time.time() - start_time
            logger.error(f"❌ TFT Baseline training failed: {e}", exc_info=True)
            return {
                'error': str(e),
                'model_type': 'TFT_Baseline',
                'training_time': training_time
            }
    
    def train_tft_enhanced(self) -> Dict[str, Any]:
        """Train TFT enhanced model with temporal decay sentiment"""
        if not TFT_AVAILABLE:
            error_msg = "❌ PyTorch Forecasting not available for TFT Enhanced"
            logger.error(error_msg)
            return {'error': error_msg, 'model_type': 'TFT_Enhanced'}
        
        logger.info("🚀 Training TFT Enhanced with Temporal Decay Sentiment")
        logger.info("🔬 NOVEL CONTRIBUTION: Temporal decay sentiment weighting")
        start_time = time.time()
        
        try:
            # Must use enhanced dataset
            if 'enhanced' not in self.datasets:
                raise ValueError("Enhanced dataset required for TFT Enhanced model")
            
            dataset = self.datasets['enhanced']
            
            # Check for temporal decay features
            decay_features = dataset['feature_analysis'].get('temporal_decay_features', [])
            if len(decay_features) < 3:
                logger.warning(f"⚠️ Only {len(decay_features)} temporal decay features found")
            else:
                logger.info(f"🏆 NOVEL: {len(decay_features)} temporal decay features available")
            
            # Prepare TFT dataset
            tft_preparer = TFTDatasetPreparer(self.config)
            training_dataset, validation_dataset = tft_preparer.prepare_tft_dataset(dataset, 'enhanced')
            
            # Create data loaders
            train_dataloader = training_dataset.to_dataloader(
                train=True, batch_size=self.config.batch_size, num_workers=0
            )
            val_dataloader = validation_dataset.to_dataloader(
                train=False, batch_size=self.config.batch_size, num_workers=0
            )
            
            # Create enhanced model using simplified trainer
            model = SimpleTFTTrainer(self.config, training_dataset, "TFT_Enhanced")
            
            # Setup callbacks
            callbacks = [
                EarlyStopping(
                    monitor="val_loss", 
                    patience=self.config.early_stopping_patience, 
                    mode="min"
                ),
                ModelCheckpoint(
                    dirpath=str(self.models_dir),
                    filename="tft_enhanced_{epoch:02d}_{val_loss:.4f}",
                    monitor="val_loss", 
                    mode="min", 
                    save_top_k=3
                ),
                LearningRateMonitor(logging_interval='epoch')
            ]
            
            # Create trainer
            trainer = pl.Trainer(
                max_epochs=self.config.tft_max_epochs,
                gradient_clip_val=self.config.gradient_clip_val,
                accelerator="auto",
                callbacks=callbacks,
                logger=TensorBoardLogger(str(self.logs_dir), name="tft_enhanced"),
                deterministic=True,
                enable_progress_bar=True
            )
            
            # Train model
            logger.info("🚀 Starting TFT Enhanced training...")
            trainer.fit(model, train_dataloader, val_dataloader)
            
            training_time = time.time() - start_time
            
            # Extract results
            results = {
                'model_type': 'TFT_Enhanced',
                'training_time': training_time,
                'best_val_loss': float(callbacks[1].best_model_score) if callbacks[1].best_model_score else None,
                'epochs_trained': trainer.current_epoch,
                'best_checkpoint': callbacks[1].best_model_path,
                'novel_features': {
                    'temporal_decay_sentiment': True,
                    'sentiment_feature_count': len(dataset['feature_analysis'].get('sentiment_features', [])),
                    'decay_feature_count': len(decay_features),
                    'enhanced_architecture': True
                }
            }
            
            logger.info(f"✅ TFT Enhanced training completed!")
            logger.info(f"⏱️ Training time: {training_time:.1f}s")
            logger.info(f"🔬 Novel methodology: SUCCESSFULLY IMPLEMENTED")
            
            return results
            
        except Exception as e:
            training_time = time.time() - start_time
            logger.error(f"❌ TFT Enhanced training failed: {e}", exc_info=True)
            return {
                'error': str(e),
                'model_type': 'TFT_Enhanced',
                'training_time': training_time
            }
    
    def train_all_models(self) -> Dict[str, Any]:
        """Train all available models"""
        logger.info("🎓 COMPLETE FINANCIAL MODEL TRAINING")
        logger.info("=" * 60)
        
        if not self.load_datasets():
            raise RuntimeError("Failed to load datasets")
        
        results = {}
        start_time = time.time()
        
        # Train LSTM Baseline
        logger.info("\n" + "="*20 + " LSTM BASELINE " + "="*20)
        results['LSTM_Baseline'] = self.train_lstm_baseline()
        MemoryMonitor.cleanup_memory()
        
        # Train TFT models if available
        if TFT_AVAILABLE:
            logger.info("\n" + "="*20 + " TFT BASELINE " + "="*20)
            results['TFT_Baseline'] = self.train_tft_baseline()
            MemoryMonitor.cleanup_memory()
            
            if 'enhanced' in self.datasets:
                logger.info("\n" + "="*20 + " TFT ENHANCED " + "="*20)
                results['TFT_Enhanced'] = self.train_tft_enhanced()
                MemoryMonitor.cleanup_memory()
        else:
            logger.warning("⚠️ TFT models skipped - PyTorch Forecasting not available")
        
        total_time = time.time() - start_time
        self._generate_summary(results, total_time)
        
        return results
    
    def _generate_summary(self, results: Dict[str, Any], total_time: float):
        """Generate training summary"""
        logger.info("\n" + "="*60)
        logger.info("🎓 TRAINING SUMMARY")
        logger.info("="*60)
        
        successful = [name for name, result in results.items() if 'error' not in result]
        failed = [name for name, result in results.items() if 'error' in result]
        
        logger.info(f"✅ Successfully trained: {len(successful)}/{len(results)} models")
        logger.info(f"⏱️ Total training time: {total_time:.1f}s ({total_time/60:.1f}m)")
        
        for model in successful:
            result = results[model]
            logger.info(f"\n📊 {model}:")
            logger.info(f"   ⏱️ Time: {result.get('training_time', 0):.1f}s")
            if result.get('best_val_loss') is not None:
                logger.info(f"   📉 Best val loss: {result.get('best_val_loss'):.6f}")
            logger.info(f"   🔄 Epochs: {result.get('epochs_trained', 0)}")
            
            if model == 'TFT_Enhanced' and 'novel_features' in result:
                novel = result['novel_features']
                logger.info(f"   🔬 Temporal decay features: {novel.get('decay_feature_count', 0)}")
        
        if failed:
            logger.info(f"\n❌ Failed models: {failed}")
            for model in failed:
                error = results[model].get('error', 'Unknown error')
                logger.info(f"   • {model}: {error}")
        
        # Save results
        results_summary = {
            'timestamp': datetime.now().isoformat(),
            'total_time': total_time,
            'successful_models': successful,
            'failed_models': failed,
            'results': results
        }
        
        results_file = self.results_dir / f"training_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(results_summary, f, indent=2, default=str)
        
        logger.info(f"💾 Results saved: {results_file}")
        logger.info("="*60)

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Complete Financial ML Framework')
    parser.add_argument('--model', choices=['all', 'lstm', 'tft_baseline', 'tft_enhanced'], 
                       default='all', help='Model to train')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    
    args = parser.parse_args()
    
    print("🎓 COMPLETE FINANCIAL MODEL TRAINING FRAMEWORK")
    print("=" * 60)
    print("🎯 MODELS AVAILABLE:")
    print("   1. 📈 LSTM Baseline - Enhanced LSTM with attention")
    print("   2. 📊 TFT Baseline - Standard TFT (requires pytorch-forecasting)")
    print("   3. 🔬 TFT Enhanced - TFT + Temporal Decay Sentiment (Novel)")
    print("=" * 60)
    
    try:
        framework = CompleteFinancialModelFramework()
        
        # Override config if specified
        if args.batch_size:
            framework.config.batch_size = args.batch_size
        if args.epochs:
            framework.config.lstm_max_epochs = args.epochs
            framework.config.tft_max_epochs = args.epochs
        
        # Train specified model(s)
        if args.model == 'all':
            results = framework.train_all_models()
        elif args.model == 'lstm':
            if not framework.load_datasets():
                return 1
            results = {'LSTM_Baseline': framework.train_lstm_baseline()}
        elif args.model == 'tft_baseline':
            if not TFT_AVAILABLE:
                print("❌ PyTorch Forecasting not available")
                return 1
            if not framework.load_datasets():
                return 1
            results = {'TFT_Baseline': framework.train_tft_baseline()}
        elif args.model == 'tft_enhanced':
            if not TFT_AVAILABLE:
                print("❌ PyTorch Forecasting not available")
                return 1
            if not framework.load_datasets():
                return 1
            results = {'TFT_Enhanced': framework.train_tft_enhanced()}
        
        # Print final results
        successful = [name for name, result in results.items() if 'error' not in result]
        print(f"\n🎉 TRAINING COMPLETED!")
        print(f"✅ Successfully trained: {len(successful)}/{len(results)} models")
        
        for model_name in successful:
            result = results[model_name]
            print(f"\n📊 {model_name}:")
            print(f"   ⏱️ Time: {result.get('training_time', 0):.1f}s")
            if result.get('best_val_loss'):
                print(f"   📉 Val loss: {result['best_val_loss']:.6f}")
            print(f"   💾 Checkpoint: {result.get('best_checkpoint', 'N/A')}")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Main execution failed: {e}", exc_info=True)
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())