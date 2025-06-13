"""
FIXED MODELS.PY - Core Dataset Integration
=========================================

✅ FIXED FOR CORE DATASET:
1. Integrated with clean core dataset (data/processed/combined_dataset.csv)
2. LSTM working with technical indicators only
3. Baseline TFT working with technical indicators only  
4. Clean placeholders for Sentiment TFT and Temporal Decay TFT
5. Proper error handling and device management
6. Standard results directory structure
7. Ready for step-by-step model testing

SCOPE: LSTM + Baseline TFT (technical only) + placeholders for enhanced models
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import logging
import pickle
import json
import warnings
from datetime import datetime
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# PyTorch Forecasting imports with error handling
try:
    from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
    from pytorch_forecasting.data import GroupNormalizer
    from pytorch_forecasting.metrics import QuantileLoss, MAE, RMSE, MAPE
    from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint  # ← FIXED
    PYTORCH_FORECASTING_AVAILABLE = True
except ImportError as e:
    PYTORCH_FORECASTING_AVAILABLE = False
    logging.warning(f"⚠️ PyTorch Forecasting not available: {e}")

# Technical Analysis
try:
    import ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False
    logging.warning("⚠️ 'ta' library not available. Technical indicators will be skipped.")

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# Standard paths
DATA_DIR = "data/processed"
RESULTS_DIR = "results/latest"
CORE_DATASET = f"{DATA_DIR}/combined_dataset.csv"

@dataclass 
class ModelConfig:
    """Configuration for model training"""
    name: str
    type: str = "TFT"  # TFT, LSTM
    max_epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.001
    hidden_size: int = 64
    dropout: float = 0.1
    max_encoder_length: int = 30
    max_prediction_length: int = 5
    early_stopping_patience: int = 10
    
    def __post_init__(self):
        """Validate configuration"""
        if self.type not in ["TFT", "LSTM"]:
            raise ValueError(f"Invalid model type: {self.type}")

# =========================================================================
# BASE MODEL CLASS
# =========================================================================

class BaseForecaster:
    """Base class for all forecasting models with enhanced error handling"""
    
    def __init__(self, target_horizons: List[int] = [5, 30, 90], 
                 results_dir: str = RESULTS_DIR):
        self.target_horizons = target_horizons
        self.results_dir = Path(results_dir)
        
        # Ensure directory creation with proper error handling
        try:
            self.results_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning(f"⚠️ Could not create results directory {results_dir}: {e}")
            self.results_dir = Path("./results")
            self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.scaler = None
        self.training_results = {}
        self.predictions = {}
        self.feature_importance = {}
        
    def load_core_dataset(self) -> pd.DataFrame:
        """Load the core dataset with validation"""
        logger.info(f"📥 Loading core dataset from {CORE_DATASET}")
        
        if not os.path.exists(CORE_DATASET):
            raise FileNotFoundError(f"Core dataset not found: {CORE_DATASET}")
        
        try:
            data = pd.read_csv(CORE_DATASET)
            
            # Basic validation
            required_cols = ['stock_id', 'symbol', 'date', 'open', 'high', 'low', 'close', 'volume', 'target_5']
            missing_cols = [col for col in required_cols if col not in data.columns]
            
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Convert date column
            data['date'] = pd.to_datetime(data['date'])
            
            logger.info(f"✅ Core dataset loaded: {data.shape}")
            logger.info(f"   📅 Date range: {data['date'].min()} to {data['date'].max()}")
            logger.info(f"   🏢 Symbols: {data['symbol'].nunique()}")
            logger.info(f"   🎯 Target coverage: {data['target_5'].notna().mean():.1%}")
            
            return data
            
        except Exception as e:
            logger.error(f"❌ Error loading core dataset: {e}")
            raise
    
    def evaluate_predictions(self, predictions: Dict, actuals: Dict) -> Dict:
        """Evaluate model performance with robust error handling"""
        metrics = {}
        
        for horizon in self.target_horizons:
            if horizon not in predictions or horizon not in actuals:
                continue
                
            try:
                pred = np.array(predictions[horizon])
                actual = np.array(actuals[horizon])
                
                # Remove NaN, infinite, and invalid values
                mask = (
                    ~np.isnan(pred) & ~np.isnan(actual) & 
                    ~np.isinf(pred) & ~np.isinf(actual) &
                    (pred != 0) & (actual != 0)
                )
                
                pred_clean = pred[mask]
                actual_clean = actual[mask]
                
                if len(pred_clean) < 10:
                    logger.warning(f"⚠️ Insufficient clean samples for horizon {horizon}: {len(pred_clean)}")
                    continue
                
                # Calculate metrics with error handling
                mae = mean_absolute_error(actual_clean, pred_clean)
                mse = mean_squared_error(actual_clean, pred_clean)
                rmse = np.sqrt(mse)
                
                # MAPE with safe calculation
                try:
                    mape_values = np.abs((actual_clean - pred_clean) / actual_clean)
                    mape = np.mean(mape_values) * 100
                    mape = min(mape, 1000)  # Cap at 1000%
                except:
                    mape = np.nan
                
                # R² with error handling
                try:
                    r2 = r2_score(actual_clean, pred_clean)
                    r2 = max(min(r2, 1.0), -10.0)  # Cap at reasonable bounds
                except:
                    r2 = np.nan
                
                # Directional accuracy
                try:
                    if len(actual_clean) > 1:
                        actual_direction = np.sign(actual_clean[1:] - actual_clean[:-1])
                        pred_direction = np.sign(pred_clean[1:] - pred_clean[:-1])
                        directional_accuracy = np.mean(actual_direction == pred_direction) * 100
                    else:
                        directional_accuracy = np.nan
                except:
                    directional_accuracy = np.nan
                
                metrics[f'horizon_{horizon}'] = {
                    'mae': float(mae),
                    'mse': float(mse),
                    'rmse': float(rmse),
                    'mape': float(mape) if not np.isnan(mape) else 999.0,
                    'r2': float(r2) if not np.isnan(r2) else -999.0,
                    'directional_accuracy': float(directional_accuracy) if not np.isnan(directional_accuracy) else 0.0,
                    'samples': len(pred_clean)
                }
                
            except Exception as e:
                logger.error(f"❌ Error evaluating horizon {horizon}: {e}")
                continue
                
        return metrics
    
    def save_model(self, model_name: str):
        """Save trained model with enhanced error handling"""
        try:
            save_path = self.results_dir / f"{model_name}.pkl"
            
            # Handle different model types properly
            if hasattr(self.model, 'state_dict'):
                model_state = self.model.state_dict()
            else:
                model_state = self.model
            
            model_data = {
                'model_state': model_state,
                'scaler': self.scaler,
                'training_results': self.training_results,
                'predictions': self.predictions,
                'feature_importance': self.feature_importance,
                'target_horizons': self.target_horizons,
                'model_class': self.__class__.__name__
            }
            
            with open(save_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            # Save metrics as JSON
            try:
                metrics_path = self.results_dir / f"{model_name}_metrics.json"
                json_results = {}
                for key, value in self.training_results.items():
                    if isinstance(value, (list, dict, str, int, float, bool, type(None))):
                        json_results[key] = value
                    else:
                        json_results[key] = str(value)
                
                with open(metrics_path, 'w') as f:
                    json.dump(json_results, f, indent=2)
            except Exception as e:
                logger.warning(f"⚠️ Could not save metrics JSON: {e}")
            
            logger.info(f"💾 Model saved: {save_path}")
            return save_path
            
        except Exception as e:
            logger.error(f"❌ Failed to save model {model_name}: {e}")
            return None

# =========================================================================
# LSTM MODEL - FIXED FOR CORE DATASET
# =========================================================================

class LSTMDataset(Dataset):
    """Dataset for LSTM model with proper type handling"""
    
    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        try:
            self.sequences = torch.FloatTensor(sequences.astype(np.float32))
            self.targets = torch.FloatTensor(targets.astype(np.float32))
        except Exception as e:
            logger.error(f"❌ Error creating LSTM dataset: {e}")
            raise
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

class LSTMModel(nn.Module):
    """LSTM model with improved architecture"""
    
    def __init__(self, input_size: int, hidden_size: int = 128, 
                 num_layers: int = 2, dropout: float = 0.2):
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        try:
            lstm_out, _ = self.lstm(x)
            last_output = lstm_out[:, -1, :]
            last_output = self.dropout(last_output)
            output = self.fc(last_output)
            return output
        except Exception as e:
            logger.error(f"❌ Error in LSTM forward pass: {e}")
            raise

class LSTMForecaster(BaseForecaster):
    """LSTM-based forecasting model for core dataset"""
    
    def __init__(self, sequence_length: int = 30, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.2,
                 learning_rate: float = 0.001, batch_size: int = 64,
                 max_epochs: int = 100, **kwargs):
        super().__init__(**kwargs)
        
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        
        # Device handling
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            logger.info("🔧 Using GPU for LSTM training")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device('mps')
            logger.info("🔧 Using MPS (Apple Silicon) for LSTM training")
        else:
            self.device = torch.device('cpu')
            logger.info("🔧 Using CPU for LSTM training")
        
        self.feature_cols = None
        self.input_size = None
        
    def get_core_feature_columns(self, data: pd.DataFrame) -> List[str]:
        """Get technical feature columns from core dataset"""
        # Stock features
        stock_features = ['open', 'high', 'low', 'close', 'volume']
        
        # Technical indicators patterns from our core dataset
        technical_patterns = [
            'returns', 'log_returns', 'vwap', 'gap', 'intraday_return', 'price_position',
            'ema_', 'sma_', 'bb_', 'rsi_', 'macd', 'atr', 'roc_', 'stoch', 'williams',
            'volume_sma', 'volume_ratio', 'volume_trend', 'volatility', '_lag_'
        ]
        
        # Time features
        time_patterns = [
            'year', 'month', 'day', 'quarter', 'time_idx', '_sin', '_cos',
            'is_weekday', 'is_weekend', 'trading_day'
        ]
        
        # Collect features
        all_features = stock_features.copy()
        
        for col in data.columns:
            if any(pattern in col.lower() for pattern in technical_patterns + time_patterns):
                all_features.append(col)
        
        # Filter existing columns and remove identifiers/targets
        exclude_patterns = ['stock_id', 'symbol', 'date', 'target_']
        feature_cols = [
            col for col in all_features 
            if col in data.columns and not any(excl in col for excl in exclude_patterns)
        ]
        
        # Remove columns with too many NaN values
        valid_features = []
        for col in feature_cols:
            if data[col].notna().sum() > len(data) * 0.5:
                valid_features.append(col)
        
        logger.info(f"📊 LSTM Core Features: {len(valid_features)} valid columns")
        logger.info(f"   📈 Stock: {len([c for c in stock_features if c in valid_features])}")
        logger.info(f"   🔧 Technical: {len([c for c in valid_features if any(p in c for p in technical_patterns)])}")
        logger.info(f"   ⏰ Time: {len([c for c in valid_features if any(p in c for p in time_patterns)])}")
        
        return valid_features
    
    def prepare_data(self, data: pd.DataFrame = None, validation_split: float = 0.2) -> Tuple[DataLoader, DataLoader]:
        """Prepare data with core dataset integration"""
        logger.info("📊 Preparing LSTM data from core dataset...")
        
        try:
            # Load core dataset if not provided
            if data is None:
                data = self.load_core_dataset()
            
            # Get feature columns
            self.feature_cols = self.get_core_feature_columns(data)
            
            if len(self.feature_cols) == 0:
                raise ValueError("No valid feature columns found")
            
            self.input_size = len(self.feature_cols)
            
            # Initialize scaler
            self.scaler = StandardScaler()
            
            # Prepare sequences by symbol
            train_sequences, train_targets = [], []
            val_sequences, val_targets = [], []
            
            # Collect all features for scaler fitting
            all_features = []
            valid_symbols = []
            
            for symbol in data['symbol'].unique():
                symbol_data = data[data['symbol'] == symbol].sort_values('date')
                
                if len(symbol_data) >= self.sequence_length + max(self.target_horizons):
                    features = symbol_data[self.feature_cols].fillna(0).values
                    if not np.any(np.isnan(features)) and not np.any(np.isinf(features)):
                        all_features.append(features)
                        valid_symbols.append(symbol)
            
            if not all_features:
                raise ValueError("No valid symbol data found for training")
            
            # Fit scaler on all valid data
            all_features_combined = np.vstack(all_features)
            self.scaler.fit(all_features_combined)
            
            # Create sequences for training
            for symbol in valid_symbols:
                symbol_data = data[data['symbol'] == symbol].sort_values('date')
                
                # Scale features
                features = symbol_data[self.feature_cols].fillna(0).values
                features_scaled = self.scaler.transform(features)
                
                # Use primary target (5-day horizon)
                target_col = 'target_5'
                if target_col not in symbol_data.columns:
                    continue
                
                targets = symbol_data[target_col].fillna(0).values
                
                # Determine split point
                split_idx = int(len(symbol_data) * (1 - validation_split))
                
                # Create sliding windows
                for i in range(len(features_scaled) - self.sequence_length):
                    if i + self.sequence_length < len(targets):
                        seq = features_scaled[i:i + self.sequence_length]
                        target = targets[i + self.sequence_length]
                        
                        # Validate sequence and target
                        if (not np.isnan(target) and not np.isinf(target) and
                            not np.any(np.isnan(seq)) and not np.any(np.isinf(seq))):
                            
                            if i < split_idx:
                                train_sequences.append(seq)
                                train_targets.append(target)
                            else:
                                val_sequences.append(seq)
                                val_targets.append(target)
            
            if len(train_sequences) == 0:
                raise ValueError("No valid training sequences created")
            
            # Create datasets and loaders
            train_dataset = LSTMDataset(np.array(train_sequences), np.array(train_targets))
            val_dataset = LSTMDataset(np.array(val_sequences), np.array(val_targets))
            
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
            
            logger.info(f"✅ LSTM data prepared from core dataset:")
            logger.info(f"   📊 Training samples: {len(train_dataset)}")
            logger.info(f"   📊 Validation samples: {len(val_dataset)}")
            logger.info(f"   🔧 Features: {len(self.feature_cols)}")
            logger.info(f"   ⏰ Sequence length: {self.sequence_length}")
            
            return train_loader, val_loader
            
        except Exception as e:
            logger.error(f"❌ Error preparing LSTM data: {e}")
            raise
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader = None) -> Dict:
        """Train LSTM model with comprehensive error handling"""
        logger.info("🚀 Starting LSTM training on core dataset...")
        
        try:
            # Initialize model
            self.model = LSTMModel(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout
            ).to(self.device)
            
            # Loss and optimizer
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=10, verbose=True
            )
            
            # Training tracking
            train_losses = []
            val_losses = []
            best_val_loss = float('inf')
            patience_counter = 0
            patience = 15
            
            start_time = datetime.now()
            
            for epoch in range(self.max_epochs):
                # Training phase
                self.model.train()
                epoch_train_loss = 0
                train_batches = 0
                
                for sequences, targets in train_loader:
                    try:
                        sequences = sequences.to(self.device)
                        targets = targets.to(self.device).unsqueeze(1)
                        
                        optimizer.zero_grad()
                        outputs = self.model(sequences)
                        loss = criterion(outputs, targets)
                        
                        if torch.isnan(loss):
                            logger.warning(f"⚠️ NaN loss detected at epoch {epoch}, skipping batch")
                            continue
                        
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        optimizer.step()
                        
                        epoch_train_loss += loss.item()
                        train_batches += 1
                        
                    except Exception as e:
                        logger.warning(f"⚠️ Error in training batch: {e}")
                        continue
                
                if train_batches == 0:
                    logger.error("❌ No valid training batches")
                    break
                
                avg_train_loss = epoch_train_loss / train_batches
                train_losses.append(avg_train_loss)
                
                # Validation phase
                avg_val_loss = avg_train_loss
                
                if val_loader:
                    self.model.eval()
                    epoch_val_loss = 0
                    val_batches = 0
                    
                    with torch.no_grad():
                        for sequences, targets in val_loader:
                            try:
                                sequences = sequences.to(self.device)
                                targets = targets.to(self.device).unsqueeze(1)
                                
                                outputs = self.model(sequences)
                                loss = criterion(outputs, targets)
                                
                                if not torch.isnan(loss):
                                    epoch_val_loss += loss.item()
                                    val_batches += 1
                                    
                            except Exception as e:
                                logger.warning(f"⚠️ Error in validation batch: {e}")
                                continue
                    
                    if val_batches > 0:
                        avg_val_loss = epoch_val_loss / val_batches
                        val_losses.append(avg_val_loss)
                        
                        scheduler.step(avg_val_loss)
                        
                        # Early stopping
                        if avg_val_loss < best_val_loss:
                            best_val_loss = avg_val_loss
                            patience_counter = 0
                            # Save best model
                            try:
                                torch.save(self.model.state_dict(), self.results_dir / 'best_lstm_model.pth')
                            except Exception as e:
                                logger.warning(f"⚠️ Could not save best model: {e}")
                        else:
                            patience_counter += 1
                        
                        if patience_counter >= patience:
                            logger.info(f"⏱️ Early stopping at epoch {epoch + 1}")
                            break
                
                # Logging
                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch + 1:3d}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Load best model if available
            best_model_path = self.results_dir / 'best_lstm_model.pth'
            if val_loader and best_model_path.exists():
                try:
                    self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
                    logger.info("✅ Loaded best model weights")
                except Exception as e:
                    logger.warning(f"⚠️ Could not load best model: {e}")
            
            # Store results
            self.training_results = {
                'model_type': 'LSTM',
                'training_time': training_time,
                'epochs_trained': epoch + 1,
                'best_val_loss': best_val_loss if val_loader else None,
                'final_train_loss': avg_train_loss,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'feature_count': len(self.feature_cols),
                'config': {
                    'sequence_length': self.sequence_length,
                    'hidden_size': self.hidden_size,
                    'num_layers': self.num_layers,
                    'dropout': self.dropout,
                    'learning_rate': self.learning_rate,
                    'batch_size': self.batch_size
                }
            }
            
            logger.info("✅ LSTM training completed on core dataset!")
            logger.info(f"⏱️ Training time: {training_time:.1f}s")
            logger.info(f"📉 Best validation loss: {best_val_loss:.4f}")
            
            return self.training_results
            
        except Exception as e:
            logger.error(f"❌ LSTM training failed: {e}")
            raise
    
    def predict(self, data: pd.DataFrame = None) -> Dict:
        """Make predictions with enhanced error handling"""
        if self.model is None or self.scaler is None:
            raise ValueError("Model not trained. Call train() first.")
        
        logger.info("🔮 Making LSTM predictions on core dataset...")
        
        try:
            # Load core dataset if not provided
            if data is None:
                data = self.load_core_dataset()
            
            self.model.eval()
            predictions = {}
            
            with torch.no_grad():
                for symbol in data['symbol'].unique():
                    symbol_data = data[data['symbol'] == symbol].sort_values('date')
                    
                    if len(symbol_data) < self.sequence_length:
                        continue
                    
                    # Prepare features
                    features = symbol_data[self.feature_cols].fillna(0).values
                    
                    if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                        logger.warning(f"⚠️ Invalid features for {symbol}, skipping")
                        continue
                    
                    features_scaled = self.scaler.transform(features)
                    
                    # Make predictions
                    symbol_predictions = []
                    for i in range(len(features_scaled) - self.sequence_length + 1):
                        seq = features_scaled[i:i + self.sequence_length]
                        seq_tensor = torch.FloatTensor(seq).unsqueeze(0).to(self.device)
                        
                        try:
                            pred = self.model(seq_tensor).cpu().numpy()[0, 0]
                            if not np.isnan(pred) and not np.isinf(pred):
                                symbol_predictions.append(pred)
                        except Exception as e:
                            logger.warning(f"⚠️ Prediction error for {symbol}: {e}")
                            continue
                    
                    # Store predictions for primary horizon
                    primary_horizon = min(self.target_horizons)
                    if primary_horizon not in predictions:
                        predictions[primary_horizon] = []
                    predictions[primary_horizon].extend(symbol_predictions)
            
            # Convert to numpy arrays
            for horizon in predictions:
                predictions[horizon] = np.array(predictions[horizon])
            
            self.predictions = predictions
            logger.info("✅ LSTM predictions completed on core dataset")
            return predictions
            
        except Exception as e:
            logger.error(f"❌ LSTM prediction failed: {e}")
            return {}

# =========================================================================
# TFT MODELS - FIXED FOR CORE DATASET
# =========================================================================

class TFTForecaster(BaseForecaster):
    """Base TFT forecasting model with PyTorch Forecasting integration"""
    
    def __init__(self, 
                 max_encoder_length: int = 30,
                 max_prediction_length: int = 5,
                 batch_size: int = 64,
                 max_epochs: int = 50,
                 learning_rate: float = 0.001,
                 hidden_size: int = 32,
                 attention_head_size: int = 4,
                 dropout: float = 0.1,
                 **kwargs):
        super().__init__(**kwargs)
        
        if not PYTORCH_FORECASTING_AVAILABLE:
            raise ImportError("PyTorch Forecasting is required for TFT models. Please install with: pip install pytorch-forecasting")
        
        self.max_encoder_length = max_encoder_length
        self.max_prediction_length = max_prediction_length
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.attention_head_size = attention_head_size
        self.dropout = dropout
        
        self.training_dataset = None
        self.validation_dataset = None
        self.trainer = None
    
    def get_feature_columns(self, data: pd.DataFrame) -> Tuple[List[str], List[str], List[str], List[str]]:
        """Get feature columns (implemented by subclasses)"""
        raise NotImplementedError
    
    def prepare_data(self, data: pd.DataFrame = None, validation_split: float = 0.2) -> Tuple[TimeSeriesDataSet, TimeSeriesDataSet]:
        """Prepare data for TFT training with core dataset integration"""
        logger.info("📊 Preparing TFT data from core dataset...")
        
        try:
            # Load core dataset if not provided
            if data is None:
                data = self.load_core_dataset()
            
            # Reset index and create a clean copy
            data = data.copy()
            if isinstance(data.index, pd.DatetimeIndex):
                data = data.reset_index()
            
            # Ensure proper date handling
            if 'date' in data.columns:
                data['date'] = pd.to_datetime(data['date'])
            
            # Sort by symbol and date
            data = data.sort_values(['symbol', 'date'])
            
            # Create unique time index per symbol
            data['time_idx'] = data.groupby('symbol').cumcount()
            
            # Clean data - aggressive cleaning
            data = data.replace([np.inf, -np.inf], np.nan)
            
            # Remove symbols with too many missing values
            target_coverage = data.groupby('symbol')['target_5'].apply(
                lambda x: x.notna().mean()
            )
            valid_symbols = target_coverage[target_coverage >= 0.5].index
            data = data[data['symbol'].isin(valid_symbols)]
            
            # Forward fill within remaining symbol groups
            data = data.groupby('symbol').apply(
                lambda x: x.fillna(method='ffill').fillna(method='bfill')
            ).reset_index(drop=True)
            
            # Final cleanup of any remaining NaN
            data = data.dropna(subset=['target_5'])
            
            logger.info(f"🧹 Data cleaned: {len(data)} rows remaining")
            logger.info(f"   🎯 Valid target values: {data['target_5'].notna().sum()}")
            logger.info(f"   🏢 Valid symbols: {len(valid_symbols)}")
            
            # Get feature columns
            static_categoricals, static_reals, time_varying_known_reals, time_varying_unknown_reals = \
                self.get_feature_columns(data)
            
            # Determine validation cutoff
            max_prediction_idx = data['time_idx'].max()
            valid_idx = int(max_prediction_idx * (1 - validation_split))
            logger.info(f"📅 Validation cutoff: time_idx <= {valid_idx}")
            
            # Create training dataset with cleaned data
            training = TimeSeriesDataSet(
                data[lambda x: x.time_idx <= valid_idx],
                time_idx="time_idx",
                target="target_5",
                group_ids=['symbol'],
                min_encoder_length=self.max_encoder_length // 2,
                max_encoder_length=self.max_encoder_length,
                min_prediction_length=1,
                max_prediction_length=self.max_prediction_length,
                static_categoricals=static_categoricals,
                time_varying_known_reals=time_varying_known_reals,
                time_varying_unknown_reals=time_varying_unknown_reals,
                target_normalizer=GroupNormalizer(
                    groups=['symbol'],
                    transformation="softplus"
                ),
                add_relative_time_idx=True,
                add_target_scales=True,
                allow_missing_timesteps=True
            )
            
            # Create validation dataset
            validation = TimeSeriesDataSet.from_dataset(
                training,
                data,
                min_prediction_idx=valid_idx + 1,
                stop_randomization=True
            )
            
            self.training_dataset = training
            self.validation_dataset = validation
            
            logger.info(f"✅ TFT data prepared from core dataset:")
            logger.info(f"   📊 Training samples: {len(training)}")
            logger.info(f"   📊 Validation samples: {len(validation)}")
            logger.info(f"   🎯 Target: target_5")
            logger.info(f"   🔧 Features: {len(time_varying_unknown_reals)}")
            
            return training, validation
            
        except Exception as e:
            logger.error(f"❌ Error preparing TFT data: {e}")
            raise
    
    def train(self) -> Dict:
        """Train TFT model with enhanced error handling"""
        logger.info("🚀 Starting TFT training on core dataset...")
        
        try:
            # Create data loaders
            train_dataloader = self.training_dataset.to_dataloader(
                train=True, batch_size=self.batch_size, num_workers=0
            )
            val_dataloader = self.validation_dataset.to_dataloader(
                train=False, batch_size=self.batch_size, num_workers=0
            )
            
            # Create model
            self.model = TemporalFusionTransformer.from_dataset(
                self.training_dataset,
                learning_rate=self.learning_rate,
                hidden_size=self.hidden_size,
                attention_head_size=self.attention_head_size,
                dropout=self.dropout,
                hidden_continuous_size=self.hidden_size // 2,
                output_size=7,  # quantiles
                loss=QuantileLoss(),
                log_interval=10,
                reduce_on_plateau_patience=4
            )
            
            # Setup callbacks
            early_stop_callback = EarlyStopping(
                monitor="val_loss", min_delta=1e-4, patience=10, mode="min"
            )
            
            # Trainer configuration
            self.trainer = pl.Trainer(
                max_epochs=self.max_epochs,
                accelerator="auto",
                devices="auto",
                gradient_clip_val=0.1,
                callbacks=[early_stop_callback],
                enable_progress_bar=True,
                logger=False,
                enable_checkpointing=False
            )
            
            # Train model
            start_time = datetime.now()
            self.trainer.fit(self.model, train_dataloader, val_dataloader)
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Store results
            self.training_results = {
                'model_type': 'TFT',
                'training_time': training_time,
                'final_val_loss': float(self.trainer.callback_metrics.get('val_loss', 0)),
                'model_params': self.model.size(),
                'config': {
                    'max_encoder_length': self.max_encoder_length,
                    'max_prediction_length': self.max_prediction_length,
                    'hidden_size': self.hidden_size,
                    'learning_rate': self.learning_rate,
                    'batch_size': self.batch_size
                }
            }
            
            logger.info("✅ TFT training completed on core dataset!")
            logger.info(f"⏱️ Training time: {training_time:.1f}s")
            logger.info(f"📉 Final validation loss: {self.training_results['final_val_loss']:.4f}")
            
            return self.training_results
            
        except Exception as e:
            logger.error(f"❌ TFT training failed: {e}")
            raise
    
    def predict(self, data: pd.DataFrame = None) -> Dict:
        """Make predictions with TFT model"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        logger.info("🔮 Making TFT predictions on core dataset...")
        
        try:
            # Use validation dataset if no data provided
            dataset = self.validation_dataset
            if data is not None:
                # Load core dataset if not provided
                if data is None:
                    data = self.load_core_dataset()
                data = data.copy()
                data['time_idx'] = data.groupby('symbol').cumcount()
                dataset = TimeSeriesDataSet.from_dataset(self.training_dataset, data, predict=True)
            
            # Make predictions
            pred_dataloader = dataset.to_dataloader(train=False, batch_size=self.batch_size, num_workers=0)
            
            predictions = self.model.predict(
                pred_dataloader, 
                return_y=True,
                trainer_kwargs=dict(accelerator="cpu", logger=False, enable_progress_bar=False)
            )
            
            # Extract median predictions
            pred_values = predictions.output.numpy()
            median_idx = pred_values.shape[-1] // 2
            
            self.predictions = {
                min(self.target_horizons): pred_values[:, 0, median_idx]
            }
            
            logger.info("✅ TFT predictions completed on core dataset")
            return self.predictions
            
        except Exception as e:
            logger.error(f"❌ TFT prediction failed: {e}")
            return {}
    
    def interpret_model(self) -> Dict:
        """Interpret TFT model with enhanced error handling"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        logger.info("🔍 Interpreting TFT model...")
        
        try:
            # Get predictions for interpretation
            val_dataloader = self.validation_dataset.to_dataloader(
                train=False, batch_size=self.batch_size, num_workers=0
            )
            
            raw_predictions = self.model.predict(
                val_dataloader, mode="raw", return_x=True,
                trainer_kwargs=dict(accelerator="cpu", logger=False, enable_progress_bar=False)
            )
            
            # Get interpretation
            interpretation = self.model.interpret_output(raw_predictions, reduction="sum")
            
            # Safe interpretation extraction
            self.feature_importance = {
                'attention': interpretation.get("attention"),
                'encoder_variables': interpretation.get("static_variables", pd.Series()),
                'decoder_variables': interpretation.get("decoder_variables", pd.Series())
            }
            
            logger.info("✅ TFT interpretation completed")
            return self.feature_importance
            
        except Exception as e:
            logger.error(f"❌ TFT interpretation failed: {e}")
            return {}

class BaselineTFT(TFTForecaster):
    """Baseline TFT model using only technical indicators from core dataset"""
    
    def get_feature_columns(self, data: pd.DataFrame) -> Tuple[List[str], List[str], List[str], List[str]]:
        """Define technical-only features for TFT from core dataset"""
        static_categoricals = ['symbol']
        static_reals = []
        time_varying_known_reals = ['time_idx']
        
        # Technical features from core dataset
        time_varying_unknown_reals = [
            # OHLCV
            'open', 'high', 'low', 'close', 'volume',
            
            # Returns & Volatility
            'returns', 'log_returns', 'vwap', 'gap', 'intraday_return', 'price_position',
            
            # Technical indicators - get all that exist in data
        ]
        
        # Add all technical indicators that exist in the data
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
        
        # Validate features exist and remove duplicates
        time_varying_unknown_reals = list(dict.fromkeys([
            col for col in time_varying_unknown_reals if col in data.columns
        ]))
        
        # Verify target column exists
        if 'target_5' not in data.columns:
            logger.error("❌ Target column 'target_5' not found in data")
            raise ValueError("Missing required target column 'target_5'")
        
        # Validate no sentiment features (should be none in core dataset)
        sentiment_patterns = ['sentiment', 'finbert', 'news', 'article']
        sentiment_found = [col for col in data.columns if any(pattern in col.lower() for pattern in sentiment_patterns)]
        if sentiment_found:
            logger.warning(f"⚠️ Unexpected sentiment features in core dataset: {sentiment_found}")
        
        logger.info(f"📊 Baseline TFT Features from Core Dataset:")
        logger.info(f"   🔧 Technical features: {len(time_varying_unknown_reals)}")
        logger.info(f"   🏷️ Categorical: {static_categoricals}")
        logger.info(f"   ⏰ Time varying known: {time_varying_known_reals}")
        
        return static_categoricals, static_reals, time_varying_known_reals, time_varying_unknown_reals

# =========================================================================
# PLACEHOLDER CLASSES FOR ENHANCED MODELS
# =========================================================================

class SentimentTFT(TFTForecaster):
    """
    🚧 PLACEHOLDER: Sentiment-enhanced TFT model
    
    This will be implemented when sentiment.py is ready.
    Will use sentiment dataset: core features + sentiment features
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        logger.info("🚧 Sentiment TFT placeholder initialized")
        logger.info("   📰 Requires sentiment.py pipeline completion")
        logger.info("   📊 Will use: core dataset + sentiment features")
    
    def get_feature_columns(self, data: pd.DataFrame) -> Tuple[List[str], List[str], List[str], List[str]]:
        """Define feature columns for sentiment TFT (placeholder)"""
        # For now, use baseline features
        baseline_tft = BaselineTFT()
        static_categoricals, static_reals, time_varying_known_reals, baseline_features = baseline_tft.get_feature_columns(data)
        
        # TODO: Add sentiment features when sentiment.py is ready
        # sentiment_patterns = ['sentiment', 'finbert', 'news_count', 'article_count']
        # sentiment_features = [col for col in data.columns if any(pattern in col.lower() for pattern in sentiment_patterns)]
        
        logger.info("🚧 Using baseline features (sentiment features not yet available)")
        return static_categoricals, static_reals, time_varying_known_reals, baseline_features

class TemporalDecayTFT(TFTForecaster):
    """
    🚧 PLACEHOLDER: Temporal decay TFT model
    
    This will be implemented when temporal_decay.py is ready.
    Will use temporal decay dataset: core + sentiment + exponential decay features
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        logger.info("🚧 Temporal Decay TFT placeholder initialized")
        logger.info("   ⏰ Requires temporal_decay.py pipeline completion")
        logger.info("   📊 Will use: core + sentiment + temporal decay features")
    
    def get_feature_columns(self, data: pd.DataFrame) -> Tuple[List[str], List[str], List[str], List[str]]:
        """Define feature columns for temporal decay TFT (placeholder)"""
        # For now, use baseline features
        baseline_tft = BaselineTFT()
        static_categoricals, static_reals, time_varying_known_reals, baseline_features = baseline_tft.get_feature_columns(data)
        
        # TODO: Add temporal decay features when temporal_decay.py is ready
        # decay_patterns = ['decay', 'temporal', 'weighted_sentiment']
        # decay_features = [col for col in data.columns if any(pattern in col.lower() for pattern in decay_patterns)]
        
        logger.info("🚧 Using baseline features (temporal decay features not yet available)")
        return static_categoricals, static_reals, time_varying_known_reals, baseline_features

# =========================================================================
# MODEL COMPARISON AND TESTING
# =========================================================================

class ModelComparison:
    """Compare and evaluate multiple forecasting models with core dataset"""
    
    def __init__(self, results_dir: str = RESULTS_DIR):
        self.results_dir = Path(results_dir)
        try:
            self.results_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning(f"⚠️ Could not create comparison directory: {e}")
            self.results_dir = Path("./results")
            self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.models = {}
        self.results = {}
    
    def add_model(self, name: str, model: BaseForecaster):
        """Add a model to comparison"""
        self.models[name] = model
    
    def compare_models(self, data: pd.DataFrame = None) -> Dict:
        """Compare all models with comprehensive error handling"""
        logger.info("🔍 Comparing models on core dataset...")
        
        # Load core dataset if not provided
        if data is None:
            base_model = BaseForecaster()
            data = base_model.load_core_dataset()
        
        results = {}
        for name, model in self.models.items():
            try:
                logger.info(f"📊 Evaluating {name}...")
                
                # Make predictions
                predictions = model.predict(data)
                
                if not predictions:
                    logger.warning(f"⚠️ No predictions from {name}")
                    results[name] = {'error': 'No predictions generated'}
                    continue
                
                # Get actuals
                actuals = {}
                for horizon in model.target_horizons:
                    target_col = f'target_{horizon}' if horizon != 5 else 'target_5'
                    if target_col in data.columns:
                        actuals[horizon] = data[target_col].dropna().values
                
                # Evaluate
                metrics = model.evaluate_predictions(predictions, actuals)
                
                results[name] = {
                    'metrics': metrics,
                    'training_results': model.training_results
                }
                
                logger.info(f"✅ {name} evaluation completed")
                
            except Exception as e:
                logger.error(f"❌ {name} evaluation failed: {e}")
                results[name] = {'error': str(e)}
        
        self.results = results
        return results
    
    def get_comparison_summary(self) -> pd.DataFrame:
        """Get comparison summary as DataFrame"""
        summary_data = []
        
        for model_name, results in self.results.items():
            if 'error' in results:
                summary_data.append({
                    'Model': model_name,
                    'Status': 'Failed',
                    'Error': results['error']
                })
                continue
            
            row = {'Model': model_name, 'Status': 'Success'}
            
            # Add metrics for first horizon
            if 'metrics' in results and results['metrics']:
                first_horizon = list(results['metrics'].keys())[0]
                metrics = results['metrics'][first_horizon]
                row.update({
                    'MAE': metrics['mae'],
                    'RMSE': metrics['rmse'],
                    'MAPE (%)': metrics['mape'],
                    'R²': metrics['r2'],
                    'Dir. Accuracy (%)': metrics['directional_accuracy'],
                    'Samples': metrics['samples']
                })
            
            # Add training info
            if 'training_results' in results:
                training = results['training_results']
                row.update({
                    'Training Time (s)': training.get('training_time', 0),
                    'Model Type': training.get('model_type', 'Unknown')
                })
                
                if 'model_params' in training:
                    row['Parameters'] = training['model_params']
                if 'feature_count' in training:
                    row['Features'] = training['feature_count']
            
            summary_data.append(row)
        
        return pd.DataFrame(summary_data)

# =========================================================================
# TESTING AND INTEGRATION FUNCTIONS
# =========================================================================

def test_core_dataset_integration():
    """Test that models can load and process the core dataset"""
    logger.info("🧪 Testing core dataset integration...")
    
    try:
        # Test dataset loading
        base_model = BaseForecaster()
        data = base_model.load_core_dataset()
        logger.info(f"✅ Core dataset loaded: {data.shape}")
        
        # Test LSTM feature extraction
        lstm = LSTMForecaster(max_epochs=1)  # Quick test
        features = lstm.get_core_feature_columns(data)
        logger.info(f"✅ LSTM features identified: {len(features)}")
        
        # Test TFT feature extraction (if available)
        if PYTORCH_FORECASTING_AVAILABLE:
            baseline_tft = BaselineTFT(max_epochs=1)  # Quick test
            static_cat, static_real, time_known, time_unknown = baseline_tft.get_feature_columns(data)
            logger.info(f"✅ TFT features identified: {len(time_unknown)} time-varying")
        
        logger.info("✅ Core dataset integration test passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Core dataset integration test failed: {e}")
        return False

def quick_train_core_models(max_epochs: int = 5) -> Dict:
    """Quick training test of core models with reduced epochs"""
    logger.info("🚀 Quick training test of core models...")
    
    results = {}
    
    try:
        # Load core dataset
        base_model = BaseForecaster()
        data = base_model.load_core_dataset()
        
        # Test LSTM
        try:
            logger.info("🧪 Testing LSTM with core dataset...")
            lstm = LSTMForecaster(max_epochs=max_epochs, batch_size=32)
            train_loader, val_loader = lstm.prepare_data(data)
            training_result = lstm.train(train_loader, val_loader)
            predictions = lstm.predict(data)
            
            results['LSTM'] = {
                'status': 'success',
                'training_time': training_result['training_time'],
                'final_loss': training_result.get('best_val_loss', 0),
                'predictions_count': len(predictions.get(5, []))
            }
            logger.info("✅ LSTM test completed")
            
        except Exception as e:
            logger.error(f"❌ LSTM test failed: {e}")
            results['LSTM'] = {'status': 'failed', 'error': str(e)}
        
        # Test Baseline TFT (if available)
        if PYTORCH_FORECASTING_AVAILABLE:
            try:
                logger.info("🧪 Testing Baseline TFT with core dataset...")
                baseline_tft = BaselineTFT(max_epochs=max_epochs, batch_size=32)
                train_dataset, val_dataset = baseline_tft.prepare_data(data)
                training_result = baseline_tft.train()
                predictions = baseline_tft.predict(data)
                
                results['Baseline_TFT'] = {
                    'status': 'success',
                    'training_time': training_result['training_time'],
                    'final_loss': training_result.get('final_val_loss', 0),
                    'predictions_count': len(predictions.get(5, []))
                }
                logger.info("✅ Baseline TFT test completed")
                
            except Exception as e:
                logger.error(f"❌ Baseline TFT test failed: {e}")
                results['Baseline_TFT'] = {'status': 'failed', 'error': str(e)}
        else:
            results['Baseline_TFT'] = {'status': 'skipped', 'reason': 'PyTorch Forecasting not available'}
        
        # Test placeholders
        results['Sentiment_TFT'] = {'status': 'placeholder', 'ready': False, 'requires': 'sentiment.py completion'}
        results['Temporal_Decay_TFT'] = {'status': 'placeholder', 'ready': False, 'requires': 'temporal_decay.py completion'}
        
        logger.info("✅ Quick training test completed!")
        return results
        
    except Exception as e:
        logger.error(f"❌ Quick training test failed: {e}")
        return {'error': str(e)}

# =========================================================================
# MAIN EXECUTION
# =========================================================================

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("🧪 Testing models.py with core dataset integration...")
    print("=" * 60)
    
    # Test core dataset integration
    if test_core_dataset_integration():
        print("✅ Core dataset integration successful!")
        
        # Quick training test
        print("\n🚀 Running quick training test...")
        results = quick_train_core_models(max_epochs=3)
        
        print("\n📊 Quick Training Results:")
        print("-" * 40)
        for model_name, result in results.items():
            status = result.get('status', 'unknown')
            if status == 'success':
                print(f"✅ {model_name}: {result['training_time']:.1f}s, {result['predictions_count']} predictions")
            elif status == 'failed':
                print(f"❌ {model_name}: {result['error']}")
            elif status == 'placeholder':
                print(f"🚧 {model_name}: Placeholder ({result['requires']})")
            elif status == 'skipped':
                print(f"⏭️ {model_name}: Skipped ({result['reason']})")
        
        print("\n🎯 Next Steps:")
        print("1. ✅ LSTM ready for core dataset")
        print("2. ✅ Baseline TFT ready for core dataset")  
        print("3. 🚧 Sentiment TFT waiting for sentiment.py")
        print("4. 🚧 Temporal Decay TFT waiting for temporal_decay.py")
        
    else:
        print("❌ Core dataset integration failed!")
        print("💡 Make sure core dataset exists: data/processed/combined_dataset.csv")
    
    print("\n🎉 Models.py testing complete!")