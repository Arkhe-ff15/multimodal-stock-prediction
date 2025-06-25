import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.preprocessing import RobustScaler
import joblib
import logging
from dataclasses import dataclass
import numpy as np
import torch.nn as nn
from sklearn.metrics import mean_absolute_error

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CompleteModelConfig:
    """Configuration class for LSTM model hyperparameters."""
    lstm_hidden_size: int = 256
    lstm_num_layers: int = 3
    lstm_dropout: float = 0.2
    lstm_sequence_length: int = 44
    lstm_learning_rate: float = 0.005
    lstm_weight_decay: float = 0.01
    lstm_max_epochs: int = 200
    early_stopping_patience: int = 50
    gradient_clip_val: float = 1.0
    batch_size: int = 32

class FinancialDataset(Dataset):
    """Custom Dataset for financial time-series data."""
    def __init__(self, data: pd.DataFrame, features: list, target: str, sequence_length: int):
        self.data = data
        self.features = features
        self.target = target
        self.sequence_length = sequence_length
        self.sequences, self.labels = self._prepare_sequences()
        logger.info(f"Created {len(self.sequences)} sequences for {self.data['symbol'].nunique()} symbols")

    def _prepare_sequences(self):
        """Prepare sequences and labels for training."""
        sequences, labels = [], []
        for symbol in self.data['symbol'].unique():
            symbol_data = self.data[self.data['symbol'] == symbol].sort_values('date')
            if len(symbol_data) < self.sequence_length + 1:
                logger.warning(f"Skipping symbol {symbol}: insufficient data ({len(symbol_data)} rows)")
                continue
            for i in range(len(symbol_data) - self.sequence_length):
                seq = symbol_data[self.features].iloc[i:i + self.sequence_length].values
                label = symbol_data[self.target].iloc[i + self.sequence_length]
                sequences.append(seq)
                labels.append(label)
        return torch.FloatTensor(sequences), torch.FloatTensor(labels)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

class LSTMModel(pl.LightningModule):
    """LSTM model definition using PyTorch Lightning."""
    def __init__(self, input_size, hidden_size, num_layers, dropout, learning_rate, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y.unsqueeze(-1))
        self.log('train_loss', loss, prog_bar=True)
        grad_norm = sum(p.grad.norm() for p in self.parameters() if p.grad is not None)
        self.log('grad_norm', grad_norm, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y.unsqueeze(-1))
        self.log('val_loss', loss, prog_bar=True)
        mae = mean_absolute_error(y.cpu().numpy(), y_hat.cpu().numpy().flatten())
        hit_rate_val = self._hit_rate(y_hat.cpu().numpy().flatten(), y.cpu().numpy())
        self.log('val_mae', mae, prog_bar=True)
        self.log('val_hit_rate', hit_rate_val, prog_bar=True)
        return loss

    def _hit_rate(self, y_pred, y_true):
        """Calculate hit rate for directional accuracy."""
        return np.mean(np.sign(y_pred) == np.sign(y_true))

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.config.lstm_weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }

class CompleteFinancialModelFramework:
    """Framework for managing financial model training."""
    def __init__(self, data_loader, datasets: dict, config=None):
        self.data_loader = data_loader
        self.datasets = datasets
        self.config = config or CompleteModelConfig()

    def train_lstm_baseline(self):
        """Train the LSTM baseline model."""
        dataset = self.datasets.get('baseline')
        if not dataset or 'splits' not in dataset:
            raise ValueError("Dataset 'baseline' with splits not found in datasets dictionary")

        train_df = dataset['splits']['train']
        val_df = dataset['splits']['val']
        features = dataset['feature_analysis']['lstm_features']
        target = dataset['feature_analysis']['target_features'][0]

        # Normalize features
        scaler = RobustScaler()
        train_df[features] = scaler.fit_transform(train_df[features])
        val_df[features] = scaler.transform(val_df[features])
        joblib.dump(scaler, 'data/scalers/baseline_scaler.joblib')

        # Normalize target
        target_scaler = RobustScaler()
        train_df[[target]] = target_scaler.fit_transform(train_df[[target]])
        val_df[[target]] = target_scaler.transform(val_df[[target]])
        joblib.dump(target_scaler, 'data/scalers/baseline_target_scaler.joblib')

        # Create datasets
        train_dataset = FinancialDataset(train_df, features, target, self.config.lstm_sequence_length)
        val_dataset = FinancialDataset(val_df, features, target, self.config.lstm_sequence_length)

        if len(train_dataset) == 0 or len(val_dataset) == 0:
            raise ValueError("No valid sequences created. Check data or sequence length.")

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=4)

        # Initialize model
        model = LSTMModel(
            input_size=len(features),
            hidden_size=self.config.lstm_hidden_size,
            num_layers=self.config.lstm_num_layers,
            dropout=self.config.lstm_dropout,
            learning_rate=self.config.lstm_learning_rate,
            config=self.config
        )

        # Set up trainer
        trainer = pl.Trainer(
            max_epochs=self.config.lstm_max_epochs,
            callbacks=[
                EarlyStopping(monitor='val_loss', patience=self.config.early_stopping_patience, mode='min'),
                ModelCheckpoint(dirpath='models/checkpoints', filename='lstm_baseline_{epoch}_{val_loss:.2f}'),
                LearningRateMonitor(logging_interval='epoch')
            ],
            logger=TensorBoardLogger('logs/tensorboard', name='lstm_baseline'),
            accelerator='auto',
            gradient_clip_val=self.config.gradient_clip_val
        )

        # Train model
        logger.info("Starting LSTM baseline training...")
        trainer.fit(model, train_loader, val_loader)
        logger.info("Training completed.")

if __name__ == "__main__":
    # Example usage with dummy data_loader and datasets
    train_df = pd.read_csv('data/model_ready/baseline_train.csv', parse_dates=['date'])
    val_df = pd.read_csv('data/model_ready/baseline_val.csv', parse_dates=['date'])
    features = [col for col in train_df.columns if col not in ['stock_id', 'symbol', 'date', 'target_5']]
    target = 'target_5'

    datasets = {
        'baseline': {
            'splits': {'train': train_df, 'val': val_df},
            'feature_analysis': {
                'lstm_features': features,
                'target_features': [target]
            }
        }
    }

    framework = CompleteFinancialModelFramework(data_loader=None, datasets=datasets)
    framework.train_lstm_baseline()