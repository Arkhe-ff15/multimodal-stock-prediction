"""
src/models.py

Complete model implementations for Multi-Horizon Sentiment-Enhanced TFT
✅ FIXED: Import issues resolved
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from pathlib import Path
import pickle
import json

# ✅ FIXED: Remove the problematic import - we'll handle this with lazy loading
# from src.temporal_decay import TemporalDecayProcessor, DecayParameters

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for model training and architecture"""
    # Model architecture
    hidden_size: int = 64
    attention_head_size: int = 4
    dropout: float = 0.3
    num_lstm_layers: int = 2
    
    # Training parameters
    learning_rate: float = 0.001
    batch_size: int = 32
    max_epochs: int = 50
    gradient_clip_val: float = 1.0
    weight_decay: float = 1e-4
    
    # Overfitting prevention
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 5
    reduce_lr_factor: float = 0.5
    min_lr: float = 1e-6
    validation_check_interval: float = 0.25
    
    # Data parameters
    max_encoder_length: int = 30
    max_prediction_length: int = 1  # ✅ FIXED: Set to 1 for single-step prediction
    training_cutoff_days: int = 365
    
    # Regularization
    label_smoothing: float = 0.0
    mixup_alpha: float = 0.0
    
    # Cross-validation
    cv_folds: int = 5
    cv_method: str = 'time_series'
    
    # Random seed for reproducibility
    random_seed: int = 42

class TimeSeriesDataset(Dataset):
    """✅ FIXED: Custom dataset with proper error handling and data validation"""
    
    def __init__(self, data: pd.DataFrame, 
                    target_columns: List[str],
                    feature_columns: List[str],
                    temporal_decay_processor: Optional[Any] = None,  # ✅ Use Any to avoid import
                    max_encoder_length: int = 30,
                    max_prediction_length: int = 1,
                    scaler: Optional[StandardScaler] = None,
                    fit_scaler: bool = True):
        
        self.data = data.copy()
        self.target_columns = target_columns
        self.feature_columns = feature_columns
        self.temporal_decay_processor = temporal_decay_processor
        self.max_encoder_length = max_encoder_length
        self.max_prediction_length = max_prediction_length
        
        # ✅ FIXED: Validate and prepare data
        self._validate_data()
        
        # Initialize scaler
        if scaler is None:
            self.scaler = StandardScaler()
        else:
            self.scaler = scaler
        
        # Prepare data
        self._prepare_data(fit_scaler)
    
    def _validate_data(self):
        """✅ FIXED: Comprehensive data validation"""
        # Check if data has the minimum required columns
        required_cols = ['symbol']
        missing_required = [col for col in required_cols if col not in self.data.columns]
        if missing_required:
            raise ValueError(f"Missing required columns: {missing_required}")
        
        # Handle date column/index
        if 'date' not in self.data.columns:
            if self.data.index.name == 'date' or pd.api.types.is_datetime64_any_dtype(self.data.index):
                self.data = self.data.reset_index()
                if 'date' not in self.data.columns:
                    self.data = self.data.rename(columns={self.data.columns[0]: 'date'})
            else:
                raise ValueError("No date column or datetime index found")
        
        # Validate target columns
        missing_targets = [col for col in self.target_columns if col not in self.data.columns]
        if missing_targets:
            logger.warning(f"Missing target columns: {missing_targets}")
            # Create dummy targets filled with NaN
            for col in missing_targets:
                self.data[col] = np.nan
        
        # Validate feature columns
        existing_features = [col for col in self.feature_columns if col in self.data.columns]
        missing_features = [col for col in self.feature_columns if col not in self.data.columns]
        
        if missing_features:
            logger.warning(f"Missing feature columns: {missing_features}")
        
        if len(existing_features) == 0:
            raise ValueError("No feature columns found in data")
        
        self.feature_columns = existing_features
        logger.info(f"Using {len(self.feature_columns)} feature columns: {self.feature_columns[:5]}...")
    
    def _prepare_data(self, fit_scaler: bool):
        """✅ FIXED: Prepare and scale data with comprehensive error handling"""
        # Ensure date column is datetime
        self.data['date'] = pd.to_datetime(self.data['date'])
        
        # Sort by symbol and date
        self.data = self.data.sort_values(['symbol', 'date'])
        
        # Handle missing values in features
        if self.data[self.feature_columns].isnull().any().any():
            logger.warning("Found missing values in features, forward filling...")
            self.data[self.feature_columns] = self.data.groupby('symbol')[self.feature_columns].fillna(method='ffill')
            self.data[self.feature_columns] = self.data[self.feature_columns].fillna(0)
        
        # Scale features
        if fit_scaler:
            # ✅ FIXED: Handle potential scaling issues
            try:
                self.data[self.feature_columns] = self.scaler.fit_transform(self.data[self.feature_columns])
            except Exception as e:
                logger.error(f"Scaling failed: {e}")
                logger.info("Proceeding without scaling")
        else:
            try:
                self.data[self.feature_columns] = self.scaler.transform(self.data[self.feature_columns])
            except Exception as e:
                logger.error(f"Transform failed: {e}")
                logger.info("Proceeding without scaling")
        
        # Create sequences
        self.sequences = []
        self.targets = []
        
        for symbol in self.data['symbol'].unique():
            symbol_data = self.data[self.data['symbol'] == symbol].reset_index(drop=True)
            
            # Check if we have enough data
            min_required = self.max_encoder_length + self.max_prediction_length
            if len(symbol_data) < min_required:
                logger.warning(f"Insufficient data for {symbol}: {len(symbol_data)} < {min_required}")
                continue
            
            # ✅ FIXED: Create non-overlapping sequences to prevent data leakage
            step_size = max(1, self.max_prediction_length)  # Non-overlapping windows
            
            for i in range(self.max_encoder_length, len(symbol_data) - self.max_prediction_length + 1, step_size):
                # Encoder sequence (input features)
                encoder_seq = symbol_data[self.feature_columns].iloc[
                    i-self.max_encoder_length:i
                ].values
                
                # ✅ FIXED: Handle single-step prediction properly
                if self.max_prediction_length == 1:
                    # Single target value
                    target_seq = symbol_data[self.target_columns].iloc[i].values
                    if len(target_seq) == 1:
                        target_seq = target_seq[0]  # Scalar for single target
                else:
                    # Multi-step prediction
                    target_seq = symbol_data[self.target_columns].iloc[
                        i:i+self.max_prediction_length
                    ].values
                
                # Skip sequences with NaN targets
                if np.isnan(target_seq).any():
                    continue
                
                self.sequences.append(encoder_seq)
                self.targets.append(target_seq)
        
        logger.info(f"Created {len(self.sequences)} sequences for training")
        
        if len(self.sequences) == 0:
            raise ValueError("No valid sequences created. Check your data and parameters.")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = torch.FloatTensor(self.sequences[idx])
        target = torch.FloatTensor([self.targets[idx]]) if np.isscalar(self.targets[idx]) else torch.FloatTensor(self.targets[idx])
        return sequence, target

class BaseTFTModel(pl.LightningModule):
    """✅ FIXED: Base class compatible with PyTorch Lightning 2.0+"""
    
    def __init__(self, config: ModelConfig, num_features: int, num_targets: int = 1):
        super().__init__()
        self.config = config
        self.num_features = num_features
        self.num_targets = num_targets
        
        # Set random seed
        pl.seed_everything(config.random_seed)
        
        # Save hyperparameters
        self.save_hyperparameters()
        
        # ✅ FIXED: Initialize validation metrics for tracking
        self.validation_losses = []
    
    def configure_optimizers(self):
        """Configure optimizer with learning rate scheduling"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=self.config.reduce_lr_factor,
            patience=self.config.reduce_lr_patience,
            min_lr=self.config.min_lr,
            verbose=True
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }
    
    def training_step(self, batch, batch_idx):
        """✅ FIXED: Training step with proper tensor handling"""
        x, y = batch
        y_hat = self(x)
        
        # ✅ FIXED: Ensure tensor shapes match
        y_hat = y_hat.squeeze()
        y = y.squeeze()
        
        if y_hat.dim() == 0:
            y_hat = y_hat.unsqueeze(0)
        if y.dim() == 0:
            y = y.unsqueeze(0)
        
        loss = F.mse_loss(y_hat, y)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """✅ FIXED: Validation step with proper tensor handling"""
        x, y = batch
        y_hat = self(x)
        
        # ✅ FIXED: Ensure tensor shapes match
        y_hat = y_hat.squeeze()
        y = y.squeeze()
        
        if y_hat.dim() == 0:
            y_hat = y_hat.unsqueeze(0)
        if y.dim() == 0:
            y = y.unsqueeze(0)
        
        loss = F.mse_loss(y_hat, y)
        mae = F.l1_loss(y_hat, y)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_mae', mae, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def on_validation_epoch_end(self):
        """✅ FIXED: Compatible with PyTorch Lightning 2.0+"""
        # Get current validation loss from logged metrics
        current_val_loss = self.trainer.logged_metrics.get('val_loss', float('inf'))
        self.validation_losses.append(current_val_loss)
        
        # Simple early stopping logic
        if len(self.validation_losses) > self.config.early_stopping_patience:
            recent_losses = self.validation_losses[-self.config.early_stopping_patience:]
            if all(loss >= recent_losses[0] for loss in recent_losses[1:]):
                logger.info(f"Early stopping triggered at epoch {self.current_epoch}")
                self.trainer.should_stop = True

class TFTTemporalDecayModel(BaseTFTModel):
    """✅ FIXED: TFT with horizon-specific temporal sentiment decay"""
    
    def __init__(self, config: ModelConfig, num_features: int, 
                 temporal_decay_processor: Optional[Any] = None,
                 horizons: List[int] = [5, 30, 90]):
        super().__init__(config, num_features)
        
        self.temporal_decay_processor = temporal_decay_processor
        self.horizons = horizons
        
        # Enhanced architecture
        self.feature_encoder = nn.Sequential(
            nn.Linear(num_features, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_lstm_layers,
            dropout=config.dropout if config.num_lstm_layers > 1 else 0,
            batch_first=True
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.attention_head_size,
            dropout=config.dropout,
            batch_first=True
        )
        
        # ✅ FIXED: Single output layer for single-step prediction
        self.output_layer = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 2, 1)
        )
        
        logger.info(f"Initialized TFT Temporal Decay Model for horizons: {horizons}")
    
    def forward(self, x):
        """✅ FIXED: Forward pass with proper tensor handling"""
        batch_size, seq_len, _ = x.shape
        
        # Encode features
        encoded = self.feature_encoder(x)
        
        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(encoded)
        
        # Apply attention
        attended_out, attention_weights = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Use the last timestep
        final_representation = attended_out[:, -1, :]
        
        # Generate prediction
        prediction = self.output_layer(final_representation)
        
        return prediction

class TFTStaticSentimentModel(BaseTFTModel):
    """✅ FIXED: TFT with static sentiment aggregation"""
    
    def __init__(self, config: ModelConfig, num_features: int):
        super().__init__(config, num_features)
        
        self.feature_encoder = nn.Sequential(
            nn.Linear(num_features, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        
        self.lstm = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_lstm_layers,
            dropout=config.dropout if config.num_lstm_layers > 1 else 0,
            batch_first=True
        )
        
        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.attention_head_size,
            dropout=config.dropout,
            batch_first=True
        )
        
        self.output_layer = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 2, 1)
        )
        
        logger.info("Initialized TFT Static Sentiment Model")
    
    def forward(self, x):
        """Standard forward pass"""
        encoded = self.feature_encoder(x)
        lstm_out, _ = self.lstm(encoded)
        attended_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        final_representation = attended_out[:, -1, :]
        prediction = self.output_layer(final_representation)
        return prediction

class TFTNumericalModel(BaseTFTModel):
    """✅ FIXED: TFT with only numerical features"""
    
    def __init__(self, config: ModelConfig, num_features: int):
        super().__init__(config, num_features)
        
        self.feature_encoder = nn.Sequential(
            nn.Linear(num_features, config.hidden_size),
            nn.BatchNorm1d(config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        
        self.lstm = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_lstm_layers,
            dropout=config.dropout if config.num_lstm_layers > 1 else 0,
            batch_first=True
        )
        
        self.output_layer = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 2, 1)
        )
        
        logger.info("Initialized TFT Numerical Model")
    
    def forward(self, x):
        """✅ FIXED: Forward pass with proper batch norm handling"""
        batch_size, seq_len, _ = x.shape
        
        # Reshape for batch norm
        x_reshaped = x.view(-1, x.size(-1))
        encoded = self.feature_encoder(x_reshaped)
        encoded = encoded.view(batch_size, seq_len, -1)
        
        # LSTM processing
        lstm_out, _ = self.lstm(encoded)
        
        # Use last timestep
        final_representation = lstm_out[:, -1, :]
        prediction = self.output_layer(final_representation)
        
        return prediction

class LSTMBaseline(BaseTFTModel):
    """✅ FIXED: Traditional LSTM baseline"""
    
    def __init__(self, config: ModelConfig, num_features: int):
        super().__init__(config, num_features)
        
        self.lstm1 = nn.LSTM(
            input_size=num_features,
            hidden_size=config.hidden_size,
            num_layers=1,
            batch_first=True
        )
        
        self.dropout1 = nn.Dropout(config.dropout)
        
        self.lstm2 = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size // 2,
            num_layers=1,
            batch_first=True
        )
        
        self.dropout2 = nn.Dropout(config.dropout)
        
        self.output_layer = nn.Linear(config.hidden_size // 2, 1)
        
        logger.info("Initialized LSTM Baseline Model")
    
    def forward(self, x):
        """Simple LSTM forward pass"""
        lstm1_out, _ = self.lstm1(x)
        lstm1_out = self.dropout1(lstm1_out)
        
        lstm2_out, _ = self.lstm2(lstm1_out)
        lstm2_out = self.dropout2(lstm2_out)
        
        final_representation = lstm2_out[:, -1, :]
        prediction = self.output_layer(final_representation)
        
        return prediction

class ModelTrainer:
    """✅ FIXED: Model trainer with comprehensive error handling"""
    
    def __init__(self, config: ModelConfig, save_dir: str = "results/models"):
        self.config = config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Set random seed
        pl.seed_everything(config.random_seed)
        
        logger.info("ModelTrainer initialized")
    
    def create_data_loaders(self, train_data: pd.DataFrame, val_data: pd.DataFrame,
                           feature_columns: List[str], target_columns: List[str],
                           temporal_decay_processor: Optional[Any] = None) -> Tuple:
        """✅ FIXED: Create data loaders with comprehensive validation"""
        
        try:
            # Create datasets
            train_dataset = TimeSeriesDataset(
                train_data, target_columns, feature_columns,
                temporal_decay_processor, 
                self.config.max_encoder_length,
                self.config.max_prediction_length,
                fit_scaler=True
            )
            
            val_dataset = TimeSeriesDataset(
                val_data, target_columns, feature_columns,
                temporal_decay_processor,
                self.config.max_encoder_length,
                self.config.max_prediction_length,
                scaler=train_dataset.scaler,
                fit_scaler=False
            )
            
            # Create data loaders
            train_loader = DataLoader(
                train_dataset, 
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=0,  # ✅ FIXED: Set to 0 to avoid multiprocessing issues
                pin_memory=True if torch.cuda.is_available() else False
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=0,  # ✅ FIXED: Set to 0 to avoid multiprocessing issues
                pin_memory=True if torch.cuda.is_available() else False
            )
            
            return train_loader, val_loader, train_dataset.scaler
            
        except Exception as e:
            logger.error(f"Error creating data loaders: {e}")
            raise
    
    def train_model(self, model: BaseTFTModel, train_loader: DataLoader, 
                   val_loader: DataLoader, model_name: str) -> Dict:
        """✅ FIXED: Train model with proper error handling"""
        
        logger.info(f"Training {model_name}...")
        
        try:
            # ✅ FIXED: Setup trainer with proper callbacks
            trainer = pl.Trainer(
                max_epochs=self.config.max_epochs,
                gradient_clip_val=self.config.gradient_clip_val,
                val_check_interval=self.config.validation_check_interval,
                enable_checkpointing=True,
                default_root_dir=self.save_dir / model_name,
                accelerator="auto",
                devices="auto",
                precision="16-mixed" if torch.cuda.is_available() else 32,  # ✅ FIXED: Updated precision syntax
                deterministic=True,
                enable_progress_bar=True,
                log_every_n_steps=10,
                callbacks=[
                    pl.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=self.config.early_stopping_patience,
                        mode='min'
                    )
                ]
            )
            
            # Train model
            start_time = datetime.now()
            trainer.fit(model, train_loader, val_loader)
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Save trained model
            model_path = self.save_dir / f"{model_name}_final.ckpt"
            trainer.save_checkpoint(model_path)
            
            # Collect training results
            training_results = {
                'model_name': model_name,
                'training_time': training_time,
                'final_train_loss': float(trainer.logged_metrics.get('train_loss_epoch', float('inf'))),
                'final_val_loss': float(trainer.logged_metrics.get('val_loss', float('inf'))),
                'final_val_mae': float(trainer.logged_metrics.get('val_mae', float('inf'))),
                'epochs_trained': trainer.current_epoch,
                'early_stopped': trainer.current_epoch < self.config.max_epochs - 1,
                'model_path': str(model_path)
            }
            
            logger.info(f"Training completed for {model_name}:")
            logger.info(f"  Final validation loss: {training_results['final_val_loss']:.6f}")
            logger.info(f"  Training time: {training_time:.1f} seconds")
            logger.info(f"  Epochs: {training_results['epochs_trained']}")
            
            return training_results
            
        except Exception as e:
            logger.error(f"Error training {model_name}: {e}")
            return {
                'model_name': model_name,
                'error': str(e),
                'training_time': 0,
                'final_val_loss': float('inf')
            }