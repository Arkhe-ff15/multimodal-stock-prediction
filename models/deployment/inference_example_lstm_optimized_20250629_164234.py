"""
LSTM Model Inference Example
Generated: 20250629_164234
Model: LSTM_Optimized
"""

import torch
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

class LSTMInference:
    def __init__(self, model_path, feature_scaler_path, target_scaler_path, metadata_path):
        """Initialize LSTM inference"""
        # Load model
        self.model_data = torch.load(model_path, map_location='cpu')
        self.model_config = self.model_data['model_config']
        
        # Load scalers
        self.feature_scaler = joblib.load(feature_scaler_path)
        self.target_scaler = joblib.load(target_scaler_path)
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            import json
            self.metadata = json.load(f)
        
        # Rebuild model architecture
        self.model = self._rebuild_model()
        self.model.load_state_dict(self.model_data['model_state_dict'])
        self.model.eval()
        
        print(f"✅ LSTM model loaded successfully")
        print(f"📊 Features: {len(self.metadata['features'])}")
        print(f"🎯 Target: {self.metadata['target']}")
    
    def _rebuild_model(self):
        """Rebuild LSTM model from config"""
        # Import your OptimizedLSTMModel class here
        from models import OptimizedLSTMModel, OptimizedFinancialConfig
        
        # Create config from saved parameters
        config = OptimizedFinancialConfig()
        for key, value in self.model_config.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        # Rebuild model
        model = OptimizedLSTMModel(
            input_size=self.model_config['input_size'],
            config=config
        )
        
        return model
    
    def preprocess_data(self, data: pd.DataFrame) -> torch.Tensor:
        """Preprocess data for inference"""
        # Select features
        feature_data = data[self.metadata['features']].copy()
        
        # Handle missing values
        feature_data = feature_data.fillna(0)
        
        # Scale features
        scaled_features = self.feature_scaler.transform(feature_data)
        
        # Create sequences
        sequence_length = self.model_config['sequence_length']
        sequences = []
        
        for i in range(len(scaled_features) - sequence_length + 1):
            seq = scaled_features[i:i + sequence_length]
            sequences.append(seq)
        
        if not sequences:
            raise ValueError(f"Not enough data for sequences. Need at least {sequence_length} rows.")
        
        return torch.FloatTensor(sequences)
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        # Preprocess
        sequences = self.preprocess_data(data)
        
        # Predict
        with torch.no_grad():
            predictions = self.model(sequences)
            predictions = predictions.cpu().numpy()
        
        # Inverse transform
        predictions_reshaped = predictions.reshape(-1, 1)
        predictions_original = self.target_scaler.inverse_transform(predictions_reshaped)
        
        return predictions_original.flatten()

# Usage example
if __name__ == "__main__":
    # File paths
    model_path = "lstm_model_20250629_164234.pth"
    feature_scaler_path = "data/scalers/baseline_scaler.joblib"
    target_scaler_path = "data/scalers/baseline_target_scaler.joblib"
    metadata_path = "model_metadata_lstm_optimized_20250629_164234.json"
    
    # Initialize inference
    lstm_inference = LSTMInference(
        model_path, feature_scaler_path, 
        target_scaler_path, metadata_path
    )
    
    # Example data (replace with your actual data)
    # data = pd.read_csv("your_data.csv")
    # predictions = lstm_inference.predict(data)
    # print(f"Predictions: {predictions}")
    
    print("🚀 LSTM inference setup complete!")
    print("📋 Modify the data loading section with your actual data")
