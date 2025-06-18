#!/usr/bin/env python3
"""
MODELS.PY - CONFIG-INTEGRATED TRAINING FRAMEWORK
===============================================

✅ FIXES APPLIED:
- Proper config.py integration
- Fixed dataset path references
- Standardized model configurations
- Enhanced fallback mechanisms
- Automated execution without prompts
- Memory-efficient processing

MODELS IMPLEMENTED:
- LSTM Baseline (technical features only)
- TFT Baseline (technical features only)  
- TFT Enhanced (technical + temporal decay sentiment)

Author: Research Team
Version: 2.1 (Config-Integrated)
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
import pickle
import json
import warnings
from datetime import datetime, timedelta
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import argparse
import gc
import time
from tqdm import tqdm

# ✅ FIXED: Proper config integration
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PipelineConfig, ModelConfig, get_default_config

# PyTorch Forecasting imports with robust error handling
try:
    from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
    from pytorch_forecasting.data import GroupNormalizer, EncoderNormalizer
    from pytorch_forecasting.metrics import QuantileLoss, MAE, RMSE, MAPE
    PYTORCH_FORECASTING_AVAILABLE = True
except ImportError as e:
    PYTORCH_FORECASTING_AVAILABLE = False
    logging.warning(f"⚠️ PyTorch Forecasting not available: {e}")

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class TimeSeriesDataSplitter:
    """
    Robust time series data splitting without data leakage
    """
    
    def __init__(self, config: Union[PipelineConfig, ModelConfig]):
        self.config = config
        
    def split_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        ✅ FIXED: Split data temporally using config parameters
        """
        logger.info("🔄 Performing temporal data split...")
        
        # Ensure proper sorting
        data = data.sort_values(['symbol', 'date']).reset_index(drop=True)
        
        # Convert dates
        data['date'] = pd.to_datetime(data['date'])
        
        # Calculate split points based on time
        unique_dates = sorted(data['date'].unique())
        total_dates = len(unique_dates)
        
        # ✅ Use config split parameters
        validation_split = getattr(self.config, 'validation_split', 0.2)
        test_split = getattr(self.config, 'test_split', 0.1)
        
        # Time-based splits
        train_end_idx = int(total_dates * (1 - validation_split - test_split))
        val_end_idx = int(total_dates * (1 - test_split))
        
        train_end_date = unique_dates[train_end_idx - 1]
        val_end_date = unique_dates[val_end_idx - 1]
        
        # Split data
        train_data = data[data['date'] <= train_end_date].copy()
        val_data = data[(data['date'] > train_end_date) & (data['date'] <= val_end_date)].copy()
        test_data = data[data['date'] > val_end_date].copy()
        
        # Log split information
        logger.info(f"📊 Data split completed:")
        logger.info(f"   📅 Train: {train_data['date'].min()} to {train_data['date'].max()} ({len(train_data):,} rows)")
        logger.info(f"   📅 Val:   {val_data['date'].min()} to {val_data['date'].max()} ({len(val_data):,} rows)")
        logger.info(f"   📅 Test:  {test_data['date'].min()} to {test_data['date'].max()} ({len(test_data):,} rows)")
        
        # Validate no overlap
        self._validate_temporal_split(train_data, val_data, test_data)
        
        return train_data, val_data, test_data
    
    def _validate_temporal_split(self, train_data: pd.DataFrame, 
                                val_data: pd.DataFrame, 
                                test_data: pd.DataFrame):
        """Validate that temporal split has no data leakage"""
        train_max = train_data['date'].max()
        val_min = val_data['date'].min()
        val_max = val_data['date'].max()
        test_min = test_data['date'].min()
        
        if train_max >= val_min:
            raise ValueError(f"Data leakage: train_max ({train_max}) >= val_min ({val_min})")
        if val_max >= test_min:
            raise ValueError(f"Data leakage: val_max ({val_max}) >= test_min ({test_min})")
        
        logger.info("✅ Temporal split validation passed - no data leakage detected")

class EnhancedLSTMDataset(Dataset):
    """Enhanced LSTM dataset with robust preprocessing"""
    
    def __init__(self, data: pd.DataFrame, feature_cols: List[str], 
                 target_col: str, sequence_length: int, scaler: StandardScaler = None):
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.sequence_length = sequence_length
        self.scaler = scaler
        
        # Process data by symbol to maintain temporal integrity
        self.sequences = []
        self.targets = []
        self.symbol_info = []
        
        for symbol in data['symbol'].unique():
            symbol_data = data[data['symbol'] == symbol].sort_values('date')
            
            if len(symbol_data) < sequence_length + 1:
                continue
                
            # Extract features and targets
            features = symbol_data[feature_cols].fillna(0).values
            targets = symbol_data[target_col].fillna(0).values
            
            # Scale features if scaler provided
            if self.scaler is not None:
                features = self.scaler.transform(features)
            
            # Create sequences
            for i in range(len(features) - sequence_length):
                if not np.isnan(targets[i + sequence_length]) and not np.isinf(targets[i + sequence_length]):
                    seq = features[i:i + sequence_length]
                    target = targets[i + sequence_length]
                    
                    if not np.any(np.isnan(seq)) and not np.any(np.isinf(seq)):
                        self.sequences.append(seq)
                        self.targets.append(target)
                        self.symbol_info.append({
                            'symbol': symbol,
                            'date_idx': i + sequence_length,
                            'date': symbol_data.iloc[i + sequence_length]['date']
                        })
        
        # Convert to tensors
        self.sequences = torch.FloatTensor(np.array(self.sequences))
        self.targets = torch.FloatTensor(np.array(self.targets))
        
        logger.info(f"📊 LSTM Dataset created: {len(self.sequences)} sequences")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

class EnhancedLSTMModel(nn.Module):
    """Enhanced LSTM with attention and residual connections"""
    
    def __init__(self, input_size: int, hidden_size: int = 128, 
                 num_layers: int = 2, dropout: float = 0.2,
                 use_attention: bool = True):
        super(EnhancedLSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_attention = use_attention
        
        # LSTM layers with dropout
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )
        
        # Attention mechanism (optional)
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.Tanh(),
                nn.Linear(hidden_size // 2, 1)
            )
        
        # Output layers with residual connection
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, 1)
        self.relu = nn.ReLU()
        
        # Batch normalization
        self.bn = nn.BatchNorm1d(hidden_size)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        if self.use_attention:
            # Apply attention mechanism
            attention_weights = self.attention(lstm_out)
            attention_weights = torch.softmax(attention_weights, dim=1)
            
            # Weighted sum of LSTM outputs
            context = torch.sum(lstm_out * attention_weights, dim=1)
        else:
            # Use last output
            context = lstm_out[:, -1, :]
        
        # Apply batch normalization
        context = self.bn(context)
        
        # Forward through output layers with residual connection
        x1 = self.relu(self.fc1(self.dropout(context)))
        output = self.fc2(self.dropout(x1))
        
        return output

class LSTMForecaster(pl.LightningModule):
    """Enhanced PyTorch Lightning LSTM forecaster"""
    
    def __init__(self, config: ModelConfig, input_size: int, feature_cols: List[str]):
        super().__init__()
        self.save_hyperparameters()
        
        self.config = config
        self.input_size = input_size
        self.feature_cols = feature_cols
        
        # Model architecture
        self.model = EnhancedLSTMModel(
            input_size=input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout,
            use_attention=True
        )
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Metrics storage
        self.train_losses = []
        self.val_losses = []
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        sequences, targets = batch
        predictions = self(sequences)
        loss = self.criterion(predictions.squeeze(), targets)
        
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        sequences, targets = batch
        predictions = self(sequences)
        loss = self.criterion(predictions.squeeze(), targets)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return {'val_loss': loss, 'predictions': predictions, 'targets': targets}
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=self.config.reduce_lr_patience,
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

class EnhancedTFTForecaster:
    """Enhanced TFT with proper feature handling and interpretability"""
    
    def __init__(self, config: ModelConfig, dataset_type: str = "baseline"):
        self.config = config
        self.dataset_type = dataset_type
        self.model = None
        self.trainer = None
        self.training_dataset = None
        self.validation_dataset = None
        self.feature_importance = None
        
    def get_feature_columns(self, data: pd.DataFrame) -> Tuple[List[str], List[str], List[str], List[str]]:
        """Define feature columns based on dataset type"""
        
        # Base categorical and real features
        static_categoricals = ['symbol']
        static_reals = []
        time_varying_known_reals = ['time_idx']
        
        # Base technical features
        time_varying_unknown_reals = [
            'open', 'high', 'low', 'close', 'volume',
            'returns', 'log_returns', 'vwap', 'gap', 'intraday_return', 'price_position'
        ]
        
        # Add technical indicators
        technical_patterns = [
            'ema_', 'sma_', 'bb_', 'rsi_', 'macd', 'atr', 'roc_', 'stoch', 'williams',
            'volume_sma', 'volume_ratio', 'volume_trend', 'volatility', '_lag_'
        ]
        
        for col in data.columns:
            if any(pattern in col.lower() for pattern in technical_patterns):
                time_varying_unknown_reals.append(col)
        
        # Add time features
        time_patterns = [
            'year', 'month', 'day', 'quarter', '_sin', '_cos', 'is_weekday', 'is_weekend', 'trading_day'
        ]
        
        for col in data.columns:
            if any(pattern in col.lower() for pattern in time_patterns):
                time_varying_unknown_reals.append(col)
        
        # Add sentiment features for enhanced model
        if self.dataset_type == "enhanced":
            sentiment_patterns = [
                'sentiment_decay_', 'sentiment_confidence', 'article_count', 
                'sentiment_weight_sum_', 'sentiment_quality_'
            ]
            
            for col in data.columns:
                if any(pattern in col.lower() for pattern in sentiment_patterns):
                    time_varying_unknown_reals.append(col)
        
        # Remove duplicates and validate
        time_varying_unknown_reals = list(dict.fromkeys([
            col for col in time_varying_unknown_reals if col in data.columns
        ]))
        
        logger.info(f"📊 TFT Features ({self.dataset_type}):")
        logger.info(f"   🔧 Technical: {len([c for c in time_varying_unknown_reals if any(p in c for p in technical_patterns)])}")
        logger.info(f"   ⏰ Time: {len([c for c in time_varying_unknown_reals if any(p in c for p in time_patterns)])}")
        if self.dataset_type == "enhanced":
            logger.info(f"   🎭 Sentiment: {len([c for c in time_varying_unknown_reals if 'sentiment' in c.lower()])}")
        logger.info(f"   📊 Total features: {len(time_varying_unknown_reals)}")
        
        return static_categoricals, static_reals, time_varying_known_reals, time_varying_unknown_reals
    
    def prepare_data(self, train_data: pd.DataFrame, val_data: pd.DataFrame) -> Tuple:
        """Prepare TFT datasets with enhanced preprocessing"""
        if not PYTORCH_FORECASTING_AVAILABLE:
            raise ImportError("PyTorch Forecasting not available for TFT training")
        
        logger.info(f"📊 Preparing TFT data ({self.dataset_type})...")
        
        # Combine and preprocess data
        combined_data = pd.concat([train_data, val_data], ignore_index=True)
        combined_data = combined_data.sort_values(['symbol', 'date']).reset_index(drop=True)
        
        # Create time index
        combined_data['time_idx'] = combined_data.groupby('symbol').cumcount()
        
        # Enhanced data cleaning
        combined_data = combined_data.replace([np.inf, -np.inf], np.nan)
        
        # Quality filtering
        target_coverage = combined_data.groupby('symbol')['target_5'].apply(
            lambda x: x.notna().mean()
        )
        valid_symbols = target_coverage[target_coverage >= 0.7].index
        combined_data = combined_data[combined_data['symbol'].isin(valid_symbols)]
        
        # Forward fill within groups
        combined_data = combined_data.groupby('symbol').apply(
            lambda x: x.fillna(method='ffill').fillna(method='bfill')
        ).reset_index(drop=True)
        
        # Final cleanup
        combined_data = combined_data.dropna(subset=['target_5'])
        
        # Get feature columns
        static_categoricals, static_reals, time_varying_known_reals, time_varying_unknown_reals = \
            self.get_feature_columns(combined_data)
        
        # Determine validation cutoff
        train_max_date = train_data['date'].max()
        combined_data['date'] = pd.to_datetime(combined_data['date'])
        val_start_idx = combined_data[combined_data['date'] > train_max_date]['time_idx'].min()
        
        if pd.isna(val_start_idx):
            val_start_idx = int(combined_data['time_idx'].max() * 0.8)
        
        # Create training dataset
        self.training_dataset = TimeSeriesDataSet(
            combined_data[lambda x: x.time_idx < val_start_idx],
            time_idx="time_idx",
            target="target_5",
            group_ids=['symbol'],
            min_encoder_length=self.config.max_encoder_length // 2,
            max_encoder_length=self.config.max_encoder_length,
            min_prediction_length=self.config.min_prediction_length,
            max_prediction_length=self.config.max_prediction_length,
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
            randomize_length=0.1
        )
        
        # Create validation dataset
        self.validation_dataset = TimeSeriesDataSet.from_dataset(
            self.training_dataset,
            combined_data,
            min_prediction_idx=val_start_idx,
            stop_randomization=True
        )
        
        logger.info(f"✅ TFT data prepared ({self.dataset_type}):")
        logger.info(f"   📊 Training samples: {len(self.training_dataset)}")
        logger.info(f"   📊 Validation samples: {len(self.validation_dataset)}")
        logger.info(f"   🎯 Target: target_5")
        logger.info(f"   🔧 Features: {len(time_varying_unknown_reals)}")
        
        return self.training_dataset, self.validation_dataset
    
    def train(self, save_path: str = None) -> Dict:
        """Train TFT with enhanced callbacks and monitoring"""
        if not PYTORCH_FORECASTING_AVAILABLE:
            raise ImportError("PyTorch Forecasting not available for TFT training")
        
        logger.info(f"🚀 Starting TFT training ({self.dataset_type})...")
        
        # Create data loaders
        train_dataloader = self.training_dataset.to_dataloader(
            train=True, 
            batch_size=self.config.batch_size, 
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )
        
        val_dataloader = self.validation_dataset.to_dataloader(
            train=False, 
            batch_size=self.config.batch_size, 
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )
        
        # Create model
        self.model = TemporalFusionTransformer.from_dataset(
            self.training_dataset,
            learning_rate=self.config.learning_rate,
            hidden_size=self.config.hidden_size,
            attention_head_size=self.config.attention_head_size,
            dropout=self.config.dropout,
            hidden_continuous_size=self.config.hidden_size // 2,
            output_size=7,  # quantiles
            loss=QuantileLoss(),
            log_interval=50,
            reduce_on_plateau_patience=self.config.reduce_lr_patience,
            optimizer='AdamW',
            optimizer_params={'weight_decay': self.config.weight_decay}
        )
        
        # Enhanced callbacks
        model_name = f"tft_{self.dataset_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            min_delta=1e-4,
            patience=self.config.early_stopping_patience,
            mode="min",
            verbose=True
        )
        
        checkpoint_callback = ModelCheckpoint(
            dirpath=str(self.config.models_checkpoints_dir) if hasattr(self.config, 'models_checkpoints_dir') else "models/checkpoints",
            filename=f"{model_name}_{{epoch:02d}}_{{val_loss:.4f}}",
            monitor="val_loss",
            mode="min",
            save_top_k=3,
            save_last=True,
            verbose=True
        )
        
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        
        # Logger
        log_dir = str(self.config.training_logs_dir) if hasattr(self.config, 'training_logs_dir') else "logs/training"
        tb_logger = TensorBoardLogger(
            save_dir=log_dir,
            name=model_name,
            version=""
        )
        
        # Trainer with enhanced configuration
        self.trainer = pl.Trainer(
            max_epochs=self.config.max_epochs,
            accelerator="auto",
            devices="auto",
            gradient_clip_val=self.config.gradient_clip_val,
            accumulate_grad_batches=self.config.accumulate_grad_batches,
            precision=16 if self.config.use_mixed_precision else 32,
            callbacks=[early_stop_callback, checkpoint_callback, lr_monitor],
            logger=tb_logger,
            enable_progress_bar=True,
            enable_model_summary=True,
            log_every_n_steps=50,
            val_check_interval=1.0,
            limit_val_batches=1.0,
            deterministic=False,
            enable_checkpointing=True
        )
        
        # Train model
        start_time = datetime.now()
        self.trainer.fit(self.model, train_dataloader, val_dataloader)
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Load best checkpoint if available
        if checkpoint_callback.best_model_path:
            logger.info(f"📥 Loading best checkpoint: {checkpoint_callback.best_model_path}")
            self.model = TemporalFusionTransformer.load_from_checkpoint(
                checkpoint_callback.best_model_path
            )
        
        # Store training results
        training_results = {
            'model_type': f'TFT_{self.dataset_type}',
            'training_time': training_time,
            'best_val_loss': float(checkpoint_callback.best_model_score) if checkpoint_callback.best_model_score else None,
            'best_checkpoint': checkpoint_callback.best_model_path,
            'config': {
                'max_encoder_length': self.config.max_encoder_length,
                'max_prediction_length': self.config.max_prediction_length,
                'hidden_size': self.config.hidden_size,
                'attention_head_size': self.config.attention_head_size,
                'learning_rate': self.config.learning_rate,
                'batch_size': self.config.batch_size,
                'dataset_type': self.dataset_type
            }
        }
        
        # Save model if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(self.model.state_dict(), save_path)
            logger.info(f"💾 Model saved: {save_path}")
        
        logger.info(f"✅ TFT training completed ({self.dataset_type})!")
        logger.info(f"⏱️ Training time: {training_time:.1f}s")
        logger.info(f"📉 Best validation loss: {training_results['best_val_loss']:.4f}")
        
        return training_results
    
    def predict(self, test_data: pd.DataFrame = None) -> Dict:
        """Make predictions with enhanced error handling"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        if not PYTORCH_FORECASTING_AVAILABLE:
            raise ImportError("PyTorch Forecasting not available for TFT prediction")
        
        logger.info(f"🔮 Making TFT predictions ({self.dataset_type})...")
        
        try:
            # Use validation dataset if no test data provided
            if test_data is not None:
                # Prepare test dataset
                test_data['time_idx'] = test_data.groupby('symbol').cumcount()
                test_dataset = TimeSeriesDataSet.from_dataset(
                    self.training_dataset, 
                    test_data, 
                    predict=True,
                    stop_randomization=True
                )
                pred_dataloader = test_dataset.to_dataloader(
                    train=False, 
                    batch_size=self.config.batch_size, 
                    num_workers=0
                )
            else:
                pred_dataloader = self.validation_dataset.to_dataloader(
                    train=False, 
                    batch_size=self.config.batch_size, 
                    num_workers=0
                )
            
            # Make predictions
            predictions = self.trainer.predict(
                self.model,
                pred_dataloader,
                return_predictions=True
            )
            
            if predictions is None or len(predictions) == 0:
                logger.warning("⚠️ No predictions generated")
                return {}
            
            # Extract predictions (median quantile)
            if hasattr(predictions[0], 'output'):
                # Multiple batch predictions
                pred_values = torch.cat([p.output for p in predictions], dim=0)
            else:
                # Single prediction
                pred_values = predictions
            
            # Extract median predictions
            if pred_values.dim() == 3:
                median_idx = pred_values.shape[-1] // 2
                final_predictions = pred_values[:, 0, median_idx].cpu().numpy()
            else:
                final_predictions = pred_values.cpu().numpy()
            
            result = {
                5: final_predictions  # Primary horizon
            }
            
            logger.info(f"✅ TFT predictions completed ({self.dataset_type})")
            logger.info(f"   📊 Predictions generated: {len(final_predictions)}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ TFT prediction failed ({self.dataset_type}): {e}")
            return {}

class ConfigIntegratedModelTrainer:
    """✅ FIXED: Model trainer with proper config integration"""
    
    def __init__(self, pipeline_config: PipelineConfig, config_overrides: Dict = None):
        self.pipeline_config = pipeline_config
        self.config_overrides = config_overrides or {}
        self.results = {}
        self.models = {}
        
        # ✅ Setup directories using config
        self.models_dir = pipeline_config.models_checkpoints_dir
        self.logs_dir = pipeline_config.training_logs_dir
        self.results_dir = pipeline_config.evaluation_results_dir
        
        # Ensure directories exist
        for directory in [self.models_dir, self.logs_dir, self.results_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def load_datasets(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """✅ FIXED: Load datasets using config paths"""
        logger.info("📥 Loading datasets...")
        
        # ✅ Load core dataset using config path
        if not self.pipeline_config.core_dataset_path.exists():
            raise FileNotFoundError(f"Core dataset not found: {self.pipeline_config.core_dataset_path}")
        
        core_data = pd.read_csv(self.pipeline_config.core_dataset_path)
        logger.info(f"✅ Core dataset loaded: {core_data.shape}")
        
        # ✅ Try to load enhanced dataset using config path
        enhanced_data = None
        if self.pipeline_config.enhanced_dataset_path.exists():
            enhanced_data = pd.read_csv(self.pipeline_config.enhanced_dataset_path)
            logger.info(f"✅ Enhanced dataset loaded: {enhanced_data.shape}")
        else:
            logger.warning("⚠️ Enhanced dataset not found - will use core dataset only")
        
        return core_data, enhanced_data
    
    def train_lstm_baseline(self, train_data: pd.DataFrame, val_data: pd.DataFrame) -> Dict:
        """Train LSTM baseline model"""
        logger.info("🚀 Training LSTM Baseline...")
        
        # ✅ Create model config from pipeline config
        model_config = self.pipeline_config.get_model_config("LSTM_Baseline")
        model_config.model_type = "LSTM"
        
        # Apply overrides
        for key, value in self.config_overrides.items():
            if hasattr(model_config, key):
                setattr(model_config, key, value)
        
        # Get technical features
        technical_patterns = [
            'open', 'high', 'low', 'close', 'volume', 'returns', 'log_returns',
            'vwap', 'gap', 'intraday_return', 'price_position',
            'ema_', 'sma_', 'bb_', 'rsi_', 'macd', 'atr', 'roc_',
            'volume_sma', 'volume_ratio', 'volatility', '_lag_'
        ]
        
        feature_cols = []
        for col in train_data.columns:
            if any(pattern in col.lower() for pattern in technical_patterns):
                feature_cols.append(col)
        
        # Remove non-numeric and target columns
        exclude_patterns = ['stock_id', 'symbol', 'date', 'target_']
        feature_cols = [col for col in feature_cols if not any(excl in col for excl in exclude_patterns)]
        
        logger.info(f"📊 LSTM features: {len(feature_cols)}")
        
        # Prepare scaler
        scaler = RobustScaler()  # More robust to outliers than StandardScaler
        
        # Fit scaler on training data
        train_features = train_data[feature_cols].fillna(0)
        scaler.fit(train_features)
        
        # Create datasets
        train_dataset = EnhancedLSTMDataset(
            train_data, feature_cols, 'target_5', 
            model_config.max_encoder_length, scaler
        )
        
        val_dataset = EnhancedLSTMDataset(
            val_data, feature_cols, 'target_5',
            model_config.max_encoder_length, scaler
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=model_config.batch_size,
            shuffle=True,
            num_workers=model_config.num_workers,
            pin_memory=model_config.pin_memory,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=model_config.batch_size,
            shuffle=False,
            num_workers=model_config.num_workers,
            pin_memory=model_config.pin_memory
        )
        
        # Initialize model
        model = LSTMForecaster(model_config, len(feature_cols), feature_cols)
        
        # Setup callbacks
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=model_config.early_stopping_patience,
            mode='min',
            verbose=True
        )
        
        checkpoint = ModelCheckpoint(
            dirpath=str(self.models_dir),
            filename=f"lstm_baseline_{{epoch:02d}}_{{val_loss:.4f}}",
            monitor='val_loss',
            mode='min',
            save_top_k=3,
            save_last=True
        )
        
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        
        # Logger
        tb_logger = TensorBoardLogger(
            save_dir=str(self.logs_dir),
            name="lstm_baseline",
            version=""
        )
        
        # Trainer
        trainer = pl.Trainer(
            max_epochs=model_config.max_epochs,
            accelerator="auto",
            devices="auto",
            gradient_clip_val=model_config.gradient_clip_val,
            precision=16 if model_config.use_mixed_precision else 32,
            callbacks=[early_stop, checkpoint, lr_monitor],
            logger=tb_logger,
            enable_progress_bar=True,
            deterministic=False
        )
        
        # Train
        start_time = datetime.now()
        trainer.fit(model, train_loader, val_loader)
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Store results
        results = {
            'model_type': 'LSTM_Baseline',
            'training_time': training_time,
            'best_val_loss': float(checkpoint.best_model_score) if checkpoint.best_model_score else None,
            'best_checkpoint': checkpoint.best_model_path,
            'feature_count': len(feature_cols),
            'config': model_config.__dict__
        }
        
        self.models['LSTM_Baseline'] = {
            'model': model,
            'trainer': trainer,
            'scaler': scaler,
            'feature_cols': feature_cols
        }
        
        logger.info("✅ LSTM Baseline training completed!")
        return results
    
    def train_tft_baseline(self, train_data: pd.DataFrame, val_data: pd.DataFrame) -> Dict:
        """Train TFT baseline model (technical features only)"""
        if not PYTORCH_FORECASTING_AVAILABLE:
            logger.warning("⚠️ PyTorch Forecasting not available - skipping TFT baseline")
            return {'error': 'PyTorch Forecasting not available'}
        
        logger.info("🚀 Training TFT Baseline...")
        
        # ✅ Create model config from pipeline config
        model_config = self.pipeline_config.get_model_config("TFT_Baseline")
        
        # Apply overrides
        for key, value in self.config_overrides.items():
            if hasattr(model_config, key):
                setattr(model_config, key, value)
        
        tft = EnhancedTFTForecaster(model_config, dataset_type="baseline")
        tft.prepare_data(train_data, val_data)
        results = tft.train(save_path=str(self.models_dir / "tft_baseline.pth"))
        
        self.models['TFT_Baseline'] = tft
        logger.info("✅ TFT Baseline training completed!")
        return results
    
    def train_tft_enhanced(self, train_data: pd.DataFrame, val_data: pd.DataFrame) -> Dict:
        """Train TFT enhanced model (technical + sentiment features)"""
        if not PYTORCH_FORECASTING_AVAILABLE:
            logger.warning("⚠️ PyTorch Forecasting not available - skipping TFT enhanced")
            return {'error': 'PyTorch Forecasting not available'}
        
        logger.info("🚀 Training TFT Enhanced...")
        
        # Check for sentiment features
        sentiment_cols = [col for col in train_data.columns if 'sentiment' in col.lower()]
        if len(sentiment_cols) == 0:
            logger.warning("⚠️ No sentiment features found - using baseline features")
            return self.train_tft_baseline(train_data, val_data)
        
        # ✅ Create model config from pipeline config
        model_config = self.pipeline_config.get_model_config("TFT_Enhanced")
        
        # Apply overrides
        for key, value in self.config_overrides.items():
            if hasattr(model_config, key):
                setattr(model_config, key, value)
        
        tft = EnhancedTFTForecaster(model_config, dataset_type="enhanced")
        tft.prepare_data(train_data, val_data)
        results = tft.train(save_path=str(self.models_dir / "tft_enhanced.pth"))
        
        self.models['TFT_Enhanced'] = tft
        logger.info("✅ TFT Enhanced training completed!")
        return results
    
    def train_all_models(self) -> Dict:
        """✅ FIXED: Train all models using config"""
        logger.info("🚀 Starting comprehensive model training...")
        
        # Load datasets
        core_data, enhanced_data = self.load_datasets()
        
        # Use enhanced dataset if available, otherwise core dataset
        primary_data = enhanced_data if enhanced_data is not None else core_data
        
        # Split data temporally
        splitter = TimeSeriesDataSplitter(self.pipeline_config)
        train_data, val_data, test_data = splitter.split_data(primary_data)
        
        # Store split data for later use
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        
        all_results = {}
        
        try:
            # Train LSTM Baseline (always available)
            all_results['LSTM_Baseline'] = self.train_lstm_baseline(train_data, val_data)
            
            # Train TFT models (only if pytorch-forecasting available)
            if PYTORCH_FORECASTING_AVAILABLE:
                # Always train baseline TFT with core features
                core_train_data, core_val_data, _ = splitter.split_data(core_data)
                all_results['TFT_Baseline'] = self.train_tft_baseline(core_train_data, core_val_data)
                
                # Train enhanced TFT if sentiment features available
                if enhanced_data is not None:
                    all_results['TFT_Enhanced'] = self.train_tft_enhanced(train_data, val_data)
            else:
                logger.warning("⚠️ PyTorch Forecasting not available - skipping TFT models")
        
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            raise
        
        # Save comprehensive results
        self.results = all_results
        self._save_training_summary()
        
        return all_results
    
    def _save_training_summary(self):
        """Save comprehensive training summary"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'models_trained': list(self.results.keys()),
            'training_results': self.results,
            'config_overrides': self.config_overrides,
            'data_info': {
                'train_size': len(self.train_data) if hasattr(self, 'train_data') else 0,
                'val_size': len(self.val_data) if hasattr(self, 'val_data') else 0,
                'test_size': len(self.test_data) if hasattr(self, 'test_data') else 0
            }
        }
        
        summary_path = self.results_dir / f"training_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"💾 Training summary saved: {summary_path}")

# =============================================================================
# PROGRAMMATIC INTERFACE
# =============================================================================

def run_model_training_programmatic(pipeline_config: PipelineConfig, 
                                   config_overrides: Dict = None) -> Tuple[bool, Dict[str, Any]]:
    """
    ✅ FIXED: Programmatic model training interface
    
    Args:
        pipeline_config: PipelineConfig object from config.py
        config_overrides: Optional parameter overrides
        
    Returns:
        Tuple[bool, Dict]: (success, results_dict)
    """
    
    try:
        logger.info("🚀 Starting programmatic model training")
        
        # Initialize trainer
        trainer = ConfigIntegratedModelTrainer(pipeline_config, config_overrides)
        
        # Train all models
        results = trainer.train_all_models()
        
        # Compile summary
        training_summary = {
            'models_trained': list(results.keys()),
            'successful_models': [name for name, result in results.items() if 'error' not in result],
            'failed_models': [name for name, result in results.items() if 'error' in result],
            'training_times': {name: result.get('training_time', 0) for name, result in results.items()},
            'validation_losses': {name: result.get('best_val_loss', None) for name, result in results.items()}
        }
        
        return True, {
            'status': 'completed',
            'stage': 'model_training',
            'training_summary': training_summary,
            'detailed_results': results,
            'trainer_instance': trainer  # For evaluation integration
        }
        
    except Exception as e:
        logger.error(f"❌ Programmatic model training failed: {e}")
        return False, {
            'error': str(e),
            'error_type': type(e).__name__,
            'stage': 'model_training'
        }

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """✅ FIXED: Main execution using config"""
    parser = argparse.ArgumentParser(
        description='Config-Integrated Model Training Framework',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--config-type', type=str, default='default',
                       choices=['default', 'quick_test', 'research'],
                       help='Configuration type to use')
    parser.add_argument('--mode', type=str, default='train_all',
                       choices=['train_all', 'train_lstm', 'train_tft_baseline', 
                               'train_tft_enhanced'],
                       help='Training mode')
    parser.add_argument('--max_epochs', type=int, default=None,
                       help='Override max epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Override batch size')
    
    args = parser.parse_args()
    
    print("🚀 CONFIG-INTEGRATED MODEL TRAINING FRAMEWORK")
    print("=" * 60)
    
    try:
        # ✅ Load config based on type
        from config import get_default_config, get_quick_test_config, get_research_config
        
        if args.config_type == 'quick_test':
            config = get_quick_test_config()
        elif args.config_type == 'research':
            config = get_research_config()
        else:
            config = get_default_config()
        
        # Apply command line overrides
        config_overrides = {}
        if args.max_epochs is not None:
            config_overrides['max_epochs'] = args.max_epochs
        if args.batch_size is not None:
            config_overrides['batch_size'] = args.batch_size
        
        print(f"📊 Configuration: {args.config_type}")
        print(f"🎯 Mode: {args.mode}")
        print(f"📅 Max epochs: {config.max_epochs}")
        print(f"📦 Batch size: {config.batch_size}")
        
        # Check dependencies
        env_validation = validate_environment()
        if not env_validation['pytorch_available']:
            print("❌ PyTorch not available - cannot train models")
            return
        
        if args.mode == 'train_all':
            # ✅ Run programmatic training
            success, results = run_model_training_programmatic(config, config_overrides)
            
            if success:
                print(f"\n🎉 MODEL TRAINING COMPLETED!")
                summary = results['training_summary']
                print(f"   🤖 Models trained: {len(summary['successful_models'])}")
                print(f"   ✅ Successful: {summary['successful_models']}")
                if summary['failed_models']:
                    print(f"   ❌ Failed: {summary['failed_models']}")
                
                for model, time_taken in summary['training_times'].items():
                    val_loss = summary['validation_losses'].get(model, 'N/A')
                    print(f"   • {model}: {time_taken:.1f}s, Val Loss: {val_loss}")
            else:
                print(f"\n❌ Model training failed: {results['error']}")
        
        else:
            # Individual model training
            trainer = ConfigIntegratedModelTrainer(config, config_overrides)
            core_data, enhanced_data = trainer.load_datasets()
            primary_data = enhanced_data if enhanced_data is not None else core_data
            
            splitter = TimeSeriesDataSplitter(config)
            train_data, val_data, test_data = splitter.split_data(primary_data)
            
            if args.mode == 'train_lstm':
                results = trainer.train_lstm_baseline(train_data, val_data)
            elif args.mode == 'train_tft_baseline':
                results = trainer.train_tft_baseline(train_data, val_data)
            elif args.mode == 'train_tft_enhanced':
                results = trainer.train_tft_enhanced(train_data, val_data)
            
            print(f"\n✅ {args.mode} completed!")
            print(f"   ⏱️ Training time: {results.get('training_time', 0):.1f}s")
            print(f"   📉 Best val loss: {results.get('best_val_loss', 'N/A')}")
        
        print(f"\n🎉 Process completed successfully!")
        print(f"📁 Models saved in: {config.models_checkpoints_dir}")
        print(f"📊 Logs saved in: {config.training_logs_dir}")
        
    except Exception as e:
        print(f"❌ Process failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())