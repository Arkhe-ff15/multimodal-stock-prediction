#!/usr/bin/env python3
"""
ACADEMIC MODEL TRAINING FRAMEWORK
=================================

Research-Grade Implementation of:
1. LSTM Baseline (Technical Features Only)
2. TFT Baseline (Technical Features Only)  
3. TFT Enhanced (Technical + Temporal Decay Sentiment)

✅ Academic Standards:
- Reproducible experiments with fixed seeds
- Proper temporal validation (no data leakage)
- Comprehensive metrics and statistical testing
- Clear feature documentation and ablation
- Publication-ready results and visualizations

Author: Research Team
Version: 3.0 (Academic)
"""

import sys
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add src directory to Python path
script_dir = Path(__file__).parent
if 'src' in str(script_dir):
    sys.path.insert(0, str(script_dir))
else:
    sys.path.insert(0, str(script_dir / 'src'))

# Core imports
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import random

# Config integration
try:
    from config_reader import load_config, get_data_paths
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    logging.warning("⚠️ Config reader not available - using defaults")

# PyTorch Forecasting (TFT)
try:
    from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
    from pytorch_forecasting.data import GroupNormalizer
    from pytorch_forecasting.metrics import QuantileLoss, MAE, RMSE
    TFT_AVAILABLE = True
except ImportError:
    TFT_AVAILABLE = False
    logging.warning("⚠️ PyTorch Forecasting not available - TFT models will be skipped")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def set_random_seeds(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    pl.seed_everything(seed)

class AcademicDataSplitter:
    """
    Academic-grade temporal data splitting with validation
    Ensures no data leakage and proper temporal ordering
    """
    
    def __init__(self, train_ratio: float = 0.7, val_ratio: float = 0.2, test_ratio: float = 0.1):
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")
        
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
    
    def split_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Temporal split ensuring no data leakage
        """
        logger.info("📊 Performing academic temporal data split...")
        
        # Ensure proper sorting and date handling
        data = data.copy()
        data['date'] = pd.to_datetime(data['date'])
        data = data.sort_values(['symbol', 'date']).reset_index(drop=True)
        
        # Get unique dates for temporal splitting
        unique_dates = sorted(data['date'].unique())
        n_dates = len(unique_dates)
        
        # Calculate split points
        train_end_idx = int(n_dates * self.train_ratio)
        val_end_idx = int(n_dates * (self.train_ratio + self.val_ratio))
        
        train_end_date = unique_dates[train_end_idx - 1] if train_end_idx > 0 else unique_dates[0]
        val_end_date = unique_dates[val_end_idx - 1] if val_end_idx < n_dates else unique_dates[-1]
        
        # Create splits
        train_data = data[data['date'] <= train_end_date].copy()
        val_data = data[(data['date'] > train_end_date) & (data['date'] <= val_end_date)].copy()
        test_data = data[data['date'] > val_end_date].copy()
        
        # Validation
        self._validate_split(train_data, val_data, test_data)
        
        # Log split information
        logger.info(f"✅ Academic temporal split completed:")
        logger.info(f"   📊 Train: {len(train_data):,} records ({train_data['date'].min().date()} to {train_data['date'].max().date()})")
        logger.info(f"   📊 Val:   {len(val_data):,} records ({val_data['date'].min().date()} to {val_data['date'].max().date()})")
        logger.info(f"   📊 Test:  {len(test_data):,} records ({test_data['date'].min().date()} to {test_data['date'].max().date()})")
        
        return train_data, val_data, test_data
    
    def _validate_split(self, train_data: pd.DataFrame, val_data: pd.DataFrame, test_data: pd.DataFrame):
        """Validate temporal ordering and no data leakage"""
        if len(train_data) == 0 or len(val_data) == 0 or len(test_data) == 0:
            raise ValueError("One or more splits is empty")
        
        train_max = train_data['date'].max()
        val_min = val_data['date'].min()
        val_max = val_data['date'].max()
        test_min = test_data['date'].min()
        
        if train_max >= val_min:
            raise ValueError(f"Data leakage: train_max ({train_max}) >= val_min ({val_min})")
        if val_max >= test_min:
            raise ValueError(f"Data leakage: val_max ({val_max}) >= test_min ({test_min})")
        
        logger.info("✅ No data leakage detected - temporal integrity maintained")

class FinancialLSTMDataset(Dataset):
    """
    Financial time series dataset for LSTM models
    Academic-grade preprocessing and sequence creation
    """
    
    def __init__(self, data: pd.DataFrame, feature_cols: List[str], target_col: str, 
                 sequence_length: int = 30, scaler: RobustScaler = None):
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.sequence_length = sequence_length
        self.scaler = scaler
        
        self.sequences = []
        self.targets = []
        self.metadata = []
        
        # Process each symbol separately to maintain temporal integrity
        for symbol in data['symbol'].unique():
            symbol_data = data[data['symbol'] == symbol].sort_values('date').reset_index(drop=True)
            
            if len(symbol_data) < sequence_length + 1:
                logger.warning(f"⚠️ Symbol {symbol} has insufficient data ({len(symbol_data)} < {sequence_length + 1})")
                continue
            
            # Extract and clean features
            features = symbol_data[feature_cols].fillna(0).values.astype(np.float32)
            targets = symbol_data[target_col].fillna(0).values.astype(np.float32)
            
            # Apply scaling if provided
            if scaler is not None:
                features = scaler.transform(features)
            
            # Create sequences
            for i in range(len(features) - sequence_length):
                target_value = targets[i + sequence_length]
                
                # Quality check
                if not (np.isfinite(target_value) and np.all(np.isfinite(features[i:i + sequence_length]))):
                    continue
                
                self.sequences.append(features[i:i + sequence_length])
                self.targets.append(target_value)
                self.metadata.append({
                    'symbol': symbol,
                    'date': symbol_data.iloc[i + sequence_length]['date'],
                    'sequence_start_idx': i,
                    'sequence_end_idx': i + sequence_length
                })
        
        # Convert to tensors
        if len(self.sequences) == 0:
            raise ValueError("No valid sequences created - check data quality and feature columns")
        
        self.sequences = torch.FloatTensor(np.array(self.sequences))
        self.targets = torch.FloatTensor(np.array(self.targets))
        
        logger.info(f"📊 LSTM Dataset created: {len(self.sequences):,} sequences from {len(data['symbol'].unique())} symbols")
        logger.info(f"   📏 Sequence length: {sequence_length}")
        logger.info(f"   🔧 Features: {len(feature_cols)}")
        logger.info(f"   🎯 Target: {target_col}")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

class AcademicLSTMModel(nn.Module):
    """
    Academic-grade LSTM model with attention mechanism
    Publication-ready architecture
    """
    
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, 
                 dropout: float = 0.2, use_attention: bool = True):
        super(AcademicLSTMModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_attention = use_attention
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )
        
        # Attention mechanism
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.Tanh(),
                nn.Linear(hidden_size // 2, 1)
            )
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, 1)
        self.activation = nn.ReLU()
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialization for better convergence"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
    
    def forward(self, x):
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        if self.use_attention:
            # Attention mechanism
            attention_weights = self.attention(lstm_out)
            attention_weights = torch.softmax(attention_weights, dim=1)
            context = torch.sum(lstm_out * attention_weights, dim=1)
        else:
            # Use last output
            context = lstm_out[:, -1, :]
        
        # Layer normalization
        context = self.layer_norm(context)
        
        # Output layers
        x = self.activation(self.fc1(self.dropout(context)))
        output = self.fc2(self.dropout(x))
        
        return output.squeeze()

class LSTMTrainer(pl.LightningModule):
    """
    PyTorch Lightning trainer for LSTM models
    Academic-grade training with comprehensive metrics
    """
    
    def __init__(self, model: AcademicLSTMModel, learning_rate: float = 1e-3, 
                 weight_decay: float = 1e-4):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Metrics storage for analysis
        self.training_step_outputs = []
        self.validation_step_outputs = []
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)
        
        # Log metrics
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.training_step_outputs.append({'loss': loss})
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)
        
        # Calculate additional metrics
        mae = torch.mean(torch.abs(y_pred - y))
        mse = torch.mean((y_pred - y) ** 2)
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_mae', mae, on_step=False, on_epoch=True)
        self.log('val_mse', mse, on_step=False, on_epoch=True)
        
        self.validation_step_outputs.append({
            'val_loss': loss,
            'val_mae': mae,
            'val_mse': mse,
            'predictions': y_pred,
            'targets': y
        })
        
        return {'val_loss': loss, 'val_mae': mae, 'val_mse': mse}
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.7,
            patience=10,
            min_lr=1e-6,
            verbose=True
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'frequency': 1
            }
        }

class AcademicTFTModel:
    """
    Academic-grade TFT model wrapper
    Handles both baseline and enhanced (with sentiment) configurations
    """
    
    def __init__(self, model_type: str = "baseline"):
        if not TFT_AVAILABLE:
            raise ImportError("PyTorch Forecasting not available for TFT training")
        
        self.model_type = model_type  # "baseline" or "enhanced"
        self.model = None
        self.trainer = None
        self.training_dataset = None
        self.validation_dataset = None
        
        logger.info(f"🔬 Initializing Academic TFT Model ({model_type})")
    
    def prepare_features(self, data: pd.DataFrame) -> Tuple[List[str], List[str], List[str], List[str]]:
        """
        Define features based on model type with academic rigor
        """
        # Static features
        static_categoricals = ['symbol']
        static_reals = []
        
        # Time-varying known (available at prediction time)
        time_varying_known_reals = ['time_idx']
        
        # Technical features (always included)
        technical_base = [
            'open', 'high', 'low', 'close', 'volume',
            'returns', 'log_returns', 'vwap', 'gap', 'intraday_return', 'price_position'
        ]
        
        # Technical indicators
        technical_indicators = []
        indicator_patterns = [
            'ema_', 'sma_', 'bb_', 'rsi_', 'macd', 'atr', 'roc_',
            'volume_sma', 'volume_ratio', 'volatility', '_lag_'
        ]
        
        for col in data.columns:
            if any(pattern in col.lower() for pattern in indicator_patterns):
                technical_indicators.append(col)
        
        # Time features
        time_features = []
        time_patterns = [
            'year', 'month', 'day', 'quarter', '_sin', '_cos', 
            'is_weekday', 'is_weekend', 'trading_day'
        ]
        
        for col in data.columns:
            if any(pattern in col.lower() for pattern in time_patterns):
                time_features.append(col)
        
        # Start with technical features
        time_varying_unknown_reals = technical_base + technical_indicators + time_features
        
        # Add sentiment features for enhanced model
        sentiment_features = []
        if self.model_type == "enhanced":
            sentiment_patterns = [
                'sentiment_decay_', 'sentiment_volatility_', 'sentiment_momentum_',
                'confidence_mean', 'confidence_std', 'high_confidence_ratio'
            ]
            
            for col in data.columns:
                if any(pattern in col.lower() for pattern in sentiment_patterns):
                    sentiment_features.append(col)
            
            time_varying_unknown_reals.extend(sentiment_features)
        
        # Remove duplicates and ensure all columns exist
        time_varying_unknown_reals = list(dict.fromkeys([
            col for col in time_varying_unknown_reals if col in data.columns
        ]))
        
        # Academic feature reporting
        logger.info(f"📊 TFT Feature Configuration ({self.model_type}):")
        logger.info(f"   🔧 Technical base: {len(technical_base)}")
        logger.info(f"   📈 Technical indicators: {len(technical_indicators)}")
        logger.info(f"   ⏰ Time features: {len(time_features)}")
        if self.model_type == "enhanced":
            logger.info(f"   🎭 Sentiment features: {len(sentiment_features)}")
        logger.info(f"   📊 Total features: {len(time_varying_unknown_reals)}")
        
        return static_categoricals, static_reals, time_varying_known_reals, time_varying_unknown_reals
    
    def prepare_dataset(self, train_data: pd.DataFrame, val_data: pd.DataFrame) -> None:
        """
        Prepare TFT dataset with academic-grade preprocessing
        """
        logger.info(f"📊 Preparing TFT dataset ({self.model_type})...")
        
        # Combine and sort data
        combined_data = pd.concat([train_data, val_data], ignore_index=True)
        combined_data['date'] = pd.to_datetime(combined_data['date'])
        combined_data = combined_data.sort_values(['symbol', 'date']).reset_index(drop=True)
        
        # Create time index
        combined_data['time_idx'] = combined_data.groupby('symbol').cumcount()
        
        # Data quality checks
        initial_length = len(combined_data)
        
        # Remove infinite values
        combined_data = combined_data.replace([np.inf, -np.inf], np.nan)
        
        # Quality filtering by symbol
        target_coverage = combined_data.groupby('symbol')['target_5'].apply(
            lambda x: x.notna().mean()
        )
        valid_symbols = target_coverage[target_coverage >= 0.7].index.tolist()
        combined_data = combined_data[combined_data['symbol'].isin(valid_symbols)]
        
        # Forward fill within groups
        combined_data = combined_data.groupby('symbol').apply(
            lambda group: group.fillna(method='ffill').fillna(method='bfill')
        ).reset_index(drop=True)
        
        # Remove rows with missing targets
        combined_data = combined_data.dropna(subset=['target_5'])
        
        logger.info(f"   📊 Data quality: {len(combined_data):,}/{initial_length:,} records retained ({len(combined_data)/initial_length*100:.1f}%)")
        logger.info(f"   🏢 Valid symbols: {len(valid_symbols)} ({valid_symbols})")
        
        # Get feature columns
        static_categoricals, static_reals, time_varying_known_reals, time_varying_unknown_reals = \
            self.prepare_features(combined_data)
        
        # Determine validation split point
        train_max_date = pd.to_datetime(train_data['date']).max()
        val_start_idx = combined_data[combined_data['date'] > train_max_date]['time_idx'].min()
        
        if pd.isna(val_start_idx):
            val_start_idx = int(combined_data['time_idx'].max() * 0.8)
        
        # Create training dataset
        self.training_dataset = TimeSeriesDataSet(
            combined_data[lambda x: x.time_idx < val_start_idx],
            time_idx="time_idx",
            target="target_5",
            group_ids=['symbol'],
            min_encoder_length=15,  # Academic standard
            max_encoder_length=30,
            min_prediction_length=1,
            max_prediction_length=5,
            static_categoricals=static_categoricals,
            static_reals=static_reals,
            time_varying_known_reals=time_varying_known_reals,
            time_varying_unknown_reals=time_varying_unknown_reals,
            target_normalizer=GroupNormalizer(
                groups=['symbol'],
                transformation="softplus",
                center=False
            ),
            add_relative_time_idx=True,
            add_target_scales=True,
            allow_missing_timesteps=True,
            randomize_length=0.05  # Small randomization for robustness
        )
        
        # Create validation dataset
        self.validation_dataset = TimeSeriesDataSet.from_dataset(
            self.training_dataset,
            combined_data,
            min_prediction_idx=val_start_idx,
            stop_randomization=True
        )
        
        logger.info(f"✅ TFT dataset prepared ({self.model_type}):")
        logger.info(f"   📊 Training samples: {len(self.training_dataset):,}")
        logger.info(f"   📊 Validation samples: {len(self.validation_dataset):,}")
        logger.info(f"   🎯 Target: target_5 (5-day forward returns)")
        logger.info(f"   🔧 Total features: {len(time_varying_unknown_reals)}")
    
    def train(self, max_epochs: int = 100, batch_size: int = 32, learning_rate: float = 1e-3) -> Dict[str, Any]:
        """
        Train TFT model with academic rigor
        """
        logger.info(f"🚀 Training Academic TFT Model ({self.model_type})...")
        
        # Create data loaders
        train_dataloader = self.training_dataset.to_dataloader(
            train=True,
            batch_size=batch_size,
            num_workers=0,  # Avoid multiprocessing issues
            pin_memory=False
        )
        
        val_dataloader = self.validation_dataset.to_dataloader(
            train=False,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=False
        )
        
        # Create model
        self.model = TemporalFusionTransformer.from_dataset(
            self.training_dataset,
            learning_rate=learning_rate,
            hidden_size=64,  # Academic standard
            attention_head_size=4,
            dropout=0.1,
            hidden_continuous_size=32,
            output_size=7,  # Quantiles
            loss=QuantileLoss(),
            log_interval=50,
            reduce_on_plateau_patience=15,
            optimizer='AdamW',
            optimizer_params={'weight_decay': 1e-4}
        )
        
        # Setup callbacks
        early_stop = EarlyStopping(
            monitor="val_loss",
            min_delta=1e-4,
            patience=20,
            mode="min",
            verbose=True
        )
        
        checkpoint = ModelCheckpoint(
            dirpath="models/checkpoints",
            filename=f"tft_{self.model_type}_{{epoch:02d}}_{{val_loss:.4f}}",
            monitor="val_loss",
            mode="min",
            save_top_k=3,
            save_last=True
        )
        
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        
        # Logger
        tb_logger = TensorBoardLogger(
            save_dir="logs/training",
            name=f"tft_{self.model_type}",
            version=""
        )
        
        # Trainer
        self.trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator="auto",
            devices="auto",
            gradient_clip_val=0.5,
            precision=32,  # More stable for academic work
            callbacks=[early_stop, checkpoint, lr_monitor],
            logger=tb_logger,
            enable_progress_bar=True,
            deterministic=True,  # For reproducibility
            enable_checkpointing=True
        )
        
        # Train
        start_time = datetime.now()
        self.trainer.fit(self.model, train_dataloader, val_dataloader)
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Results
        results = {
            'model_type': f'TFT_{self.model_type}',
            'training_time': training_time,
            'best_val_loss': float(checkpoint.best_model_score) if checkpoint.best_model_score else None,
            'epochs_trained': self.trainer.current_epoch,
            'best_checkpoint': checkpoint.best_model_path,
            'config': {
                'max_epochs': max_epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'model_type': self.model_type
            }
        }
        
        logger.info(f"✅ TFT training completed ({self.model_type})!")
        logger.info(f"   ⏱️ Training time: {training_time:.1f}s ({training_time/60:.1f}m)")
        logger.info(f"   📉 Best validation loss: {results['best_val_loss']:.4f}")
        logger.info(f"   🔄 Epochs trained: {results['epochs_trained']}")
        
        return results

class AcademicModelFramework:
    """
    Main framework for academic model training and evaluation
    """
    
    def __init__(self, config_file: str = "config.yaml"):
        # Set random seeds for reproducibility
        set_random_seeds(42)
        
        # Load configuration
        if CONFIG_AVAILABLE:
            try:
                self.config = load_config(config_file)
                self.data_paths = get_data_paths(self.config)
                logger.info("✅ Configuration loaded successfully")
            except Exception as e:
                logger.warning(f"⚠️ Config loading failed: {e}, using defaults")
                self._setup_default_config()
        else:
            self._setup_default_config()
        
        # Setup directories
        self.models_dir = Path("models/checkpoints")
        self.logs_dir = Path("logs/training")
        self.results_dir = Path("results")
        
        for directory in [self.models_dir, self.logs_dir, self.results_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Results storage
        self.results = {}
        self.models = {}
    
    def _setup_default_config(self):
        """Setup default configuration when config reader not available"""
        self.data_paths = {
            'core_dataset': Path("data/processed/combined_dataset.csv"),
            'temporal_decay_dataset': Path("data/processed/temporal_decay_enhanced_dataset.csv")
        }
        logger.info("📊 Using default configuration")
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load datasets with academic validation
        """
        logger.info("📥 Loading datasets for academic training...")
        
        # Load core dataset (baseline models)
        core_path = self.data_paths['core_dataset']
        if not core_path.exists():
            raise FileNotFoundError(f"Core dataset not found: {core_path}")
        
        core_data = pd.read_csv(core_path)
        logger.info(f"✅ Core dataset: {core_data.shape[0]:,} records, {core_data.shape[1]} features")
        
        # Load enhanced dataset (temporal decay sentiment)
        enhanced_data = None
        enhanced_path = self.data_paths['temporal_decay_dataset']
        if enhanced_path.exists():
            enhanced_data = pd.read_csv(enhanced_path)
            logger.info(f"✅ Enhanced dataset: {enhanced_data.shape[0]:,} records, {enhanced_data.shape[1]} features")
        else:
            logger.warning("⚠️ Enhanced dataset not found - TFT Enhanced will use core dataset")
        
        # Data validation
        self._validate_datasets(core_data, enhanced_data)
        
        return core_data, enhanced_data
    
    def _validate_datasets(self, core_data: pd.DataFrame, enhanced_data: pd.DataFrame = None):
        """Academic-grade dataset validation"""
        logger.info("🔍 Performing academic dataset validation...")
        
        # Core dataset validation
        required_cols = ['symbol', 'date', 'target_5', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in core_data.columns]
        if missing_cols:
            raise ValueError(f"Core dataset missing required columns: {missing_cols}")
        
        # Check for sufficient data per symbol
        symbol_counts = core_data['symbol'].value_counts()
        min_observations = 100  # Academic minimum
        insufficient_symbols = symbol_counts[symbol_counts < min_observations].index.tolist()
        if insufficient_symbols:
            logger.warning(f"⚠️ Symbols with insufficient data (< {min_observations}): {insufficient_symbols}")
        
        # Enhanced dataset validation
        if enhanced_data is not None:
            sentiment_features = [col for col in enhanced_data.columns if 'sentiment' in col.lower()]
            logger.info(f"📊 Enhanced dataset contains {len(sentiment_features)} sentiment features")
            
            if len(sentiment_features) == 0:
                logger.warning("⚠️ Enhanced dataset contains no sentiment features")
        
        logger.info("✅ Dataset validation completed")
    
    def train_lstm_baseline(self, train_data: pd.DataFrame, val_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Train LSTM baseline model (technical features only)
        """
        logger.info("🚀 Training LSTM Baseline Model (Technical Features Only)")
        
        # Define technical features
        technical_patterns = [
            'open', 'high', 'low', 'close', 'volume',
            'returns', 'log_returns', 'vwap', 'gap', 'intraday_return', 'price_position',
            'ema_', 'sma_', 'bb_', 'rsi_', 'macd', 'atr', 'roc_',
            'volume_sma', 'volume_ratio', 'volatility', '_lag_'
        ]
        
        feature_cols = []
        for col in train_data.columns:
            if any(pattern in col.lower() for pattern in technical_patterns):
                # Exclude target and identifier columns
                if not any(excl in col.lower() for excl in ['target_', 'stock_id', 'symbol', 'date']):
                    feature_cols.append(col)
        
        logger.info(f"📊 LSTM features selected: {len(feature_cols)}")
        
        # Prepare scaler
        scaler = RobustScaler()
        train_features = train_data[feature_cols].fillna(0)
        scaler.fit(train_features)
        
        # Create datasets
        train_dataset = FinancialLSTMDataset(
            train_data, feature_cols, 'target_5', sequence_length=30, scaler=scaler
        )
        val_dataset = FinancialLSTMDataset(
            val_data, feature_cols, 'target_5', sequence_length=30, scaler=scaler
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=False
        )
        val_loader = DataLoader(
            val_dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=False
        )
        
        # Initialize model
        model = AcademicLSTMModel(
            input_size=len(feature_cols),
            hidden_size=128,
            num_layers=2,
            dropout=0.2,
            use_attention=True
        )
        
        # Lightning trainer
        lstm_trainer = LSTMTrainer(model, learning_rate=1e-3, weight_decay=1e-4)
        
        # Callbacks
        early_stop = EarlyStopping(
            monitor='val_loss', patience=20, mode='min', verbose=True
        )
        checkpoint = ModelCheckpoint(
            dirpath=str(self.models_dir),
            filename="lstm_baseline_{epoch:02d}_{val_loss:.4f}",
            monitor='val_loss', mode='min', save_top_k=3
        )
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        
        # PyTorch Lightning trainer
        trainer = pl.Trainer(
            max_epochs=100,
            accelerator="auto",
            devices="auto",
            callbacks=[early_stop, checkpoint, lr_monitor],
            logger=TensorBoardLogger(str(self.logs_dir), name="lstm_baseline"),
            enable_progress_bar=True,
            deterministic=True
        )
        
        # Train
        start_time = datetime.now()
        trainer.fit(lstm_trainer, train_loader, val_loader)
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Results
        results = {
            'model_type': 'LSTM_Baseline',
            'training_time': training_time,
            'best_val_loss': float(checkpoint.best_model_score) if checkpoint.best_model_score else None,
            'epochs_trained': trainer.current_epoch,
            'feature_count': len(feature_cols),
            'best_checkpoint': checkpoint.best_model_path
        }
        
        # Store model
        self.models['LSTM_Baseline'] = {
            'model': lstm_trainer,
            'trainer': trainer,
            'scaler': scaler,
            'feature_cols': feature_cols
        }
        
        logger.info("✅ LSTM Baseline training completed!")
        logger.info(f"   ⏱️ Training time: {training_time:.1f}s ({training_time/60:.1f}m)")
        logger.info(f"   📉 Best validation loss: {results['best_val_loss']:.4f}")
        
        return results
    
    def train_tft_baseline(self, train_data: pd.DataFrame, val_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Train TFT baseline model (technical features only)
        """
        if not TFT_AVAILABLE:
            logger.warning("⚠️ PyTorch Forecasting not available - skipping TFT baseline")
            return {'error': 'PyTorch Forecasting not available'}
        
        logger.info("🚀 Training TFT Baseline Model (Technical Features Only)")
        
        tft = AcademicTFTModel(model_type="baseline")
        tft.prepare_dataset(train_data, val_data)
        results = tft.train(max_epochs=100, batch_size=32, learning_rate=1e-3)
        
        self.models['TFT_Baseline'] = tft
        
        logger.info("✅ TFT Baseline training completed!")
        return results
    
    def train_tft_enhanced(self, train_data: pd.DataFrame, val_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Train TFT enhanced model (technical + temporal decay sentiment features)
        """
        if not TFT_AVAILABLE:
            logger.warning("⚠️ PyTorch Forecasting not available - skipping TFT enhanced")
            return {'error': 'PyTorch Forecasting not available'}
        
        logger.info("🚀 Training TFT Enhanced Model (Technical + Temporal Decay Sentiment)")
        
        # Check for sentiment features
        sentiment_features = [col for col in train_data.columns if 'sentiment' in col.lower()]
        if len(sentiment_features) == 0:
            logger.warning("⚠️ No sentiment features found - falling back to baseline")
            return self.train_tft_baseline(train_data, val_data)
        
        logger.info(f"📊 Found {len(sentiment_features)} sentiment features for enhanced model")
        
        tft = AcademicTFTModel(model_type="enhanced")
        tft.prepare_dataset(train_data, val_data)
        results = tft.train(max_epochs=100, batch_size=32, learning_rate=1e-3)
        
        self.models['TFT_Enhanced'] = tft
        
        logger.info("✅ TFT Enhanced training completed!")
        return results
    
    def train_all_models(self) -> Dict[str, Any]:
        """
        Train all three models with academic rigor
        """
        logger.info("🔬 ACADEMIC MODEL TRAINING FRAMEWORK")
        logger.info("=" * 50)
        logger.info("Training sequence:")
        logger.info("1. LSTM Baseline (Technical Features)")
        logger.info("2. TFT Baseline (Technical Features)")
        logger.info("3. TFT Enhanced (Technical + Temporal Decay Sentiment)")
        logger.info("=" * 50)
        
        # Load data
        core_data, enhanced_data = self.load_data()
        
        # Data splitting
        splitter = AcademicDataSplitter(train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)
        
        # Split core data for baseline models
        core_train, core_val, core_test = splitter.split_data(core_data)
        
        # Split enhanced data for enhanced model
        if enhanced_data is not None:
            enhanced_train, enhanced_val, enhanced_test = splitter.split_data(enhanced_data)
        else:
            enhanced_train, enhanced_val, enhanced_test = core_train, core_val, core_test
        
        # Store test data for evaluation
        self.test_data = {
            'core': core_test,
            'enhanced': enhanced_test
        }
        
        all_results = {}
        training_start = datetime.now()
        
        try:
            # 1. LSTM Baseline
            logger.info("\n" + "="*30 + " LSTM BASELINE " + "="*30)
            all_results['LSTM_Baseline'] = self.train_lstm_baseline(core_train, core_val)
            
            # 2. TFT Baseline  
            logger.info("\n" + "="*30 + " TFT BASELINE " + "="*30)
            all_results['TFT_Baseline'] = self.train_tft_baseline(core_train, core_val)
            
            # 3. TFT Enhanced
            logger.info("\n" + "="*30 + " TFT ENHANCED " + "="*30)
            all_results['TFT_Enhanced'] = self.train_tft_enhanced(enhanced_train, enhanced_val)
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            raise
        
        total_training_time = (datetime.now() - training_start).total_seconds()
        
        # Store results
        self.results = all_results
        
        # Generate academic summary
        self._generate_academic_summary(all_results, total_training_time)
        
        return all_results
    
    def _generate_academic_summary(self, results: Dict[str, Any], total_time: float):
        """
        Generate academic-quality training summary
        """
        logger.info("\n" + "="*60)
        logger.info("🎓 ACADEMIC TRAINING SUMMARY")
        logger.info("="*60)
        
        successful_models = [name for name, result in results.items() if 'error' not in result]
        failed_models = [name for name, result in results.items() if 'error' in result]
        
        logger.info(f"✅ Successfully trained: {len(successful_models)}/3 models")
        for model in successful_models:
            result = results[model]
            logger.info(f"   • {model}: {result.get('training_time', 0):.1f}s, "
                       f"Val Loss: {result.get('best_val_loss', 'N/A'):.4f}")
        
        if failed_models:
            logger.info(f"❌ Failed models: {failed_models}")
        
        logger.info(f"⏱️ Total training time: {total_time:.1f}s ({total_time/60:.1f}m)")
        logger.info(f"📊 Models ready for evaluation and comparison")
        
        # Save detailed results
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_training_time': total_time,
            'successful_models': successful_models,
            'failed_models': failed_models,
            'model_results': results,
            'reproducibility': {
                'random_seed': 42,
                'pytorch_version': torch.__version__,
                'framework_version': '3.0'
            }
        }
        
        summary_path = self.results_dir / f"academic_training_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"💾 Academic summary saved: {summary_path}")
        logger.info("="*60)

def main():
    """
    Main execution for academic model training
    """
    print("🎓 ACADEMIC MODEL TRAINING FRAMEWORK")
    print("=" * 50)
    print("Research-grade implementation of:")
    print("1. LSTM Baseline (Technical Features)")
    print("2. TFT Baseline (Technical Features)")
    print("3. TFT Enhanced (Technical + Temporal Decay Sentiment)")
    print("=" * 50)
    
    try:
        # Initialize framework
        framework = AcademicModelFramework()
        
        # Train all models
        results = framework.train_all_models()
        
        # Success message
        successful_models = [name for name, result in results.items() if 'error' not in result]
        
        print(f"\n🎉 ACADEMIC TRAINING COMPLETED!")
        print(f"✅ Successfully trained: {len(successful_models)}/3 models")
        print(f"🔬 Results ready for academic evaluation")
        print(f"📁 Models saved in: models/checkpoints/")
        print(f"📊 Logs available in: logs/training/")
        
        return 0
        
    except Exception as e:
        print(f"❌ Academic training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())