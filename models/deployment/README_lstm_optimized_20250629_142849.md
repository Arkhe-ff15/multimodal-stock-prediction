# LSTM_Optimized Model Deployment Guide

## Model Information
- **Model Type**: LSTM_Optimized
- **Created**: 20250629_142849
- **Target**: target_5
- **Features**: 33 features
- **Dataset**: baseline
- **Architecture**: LSTM
- **Parameters**: 463,617

## Performance Metrics

## Files for Deployment
- **Model**: `lstm_model_20250629_142849.pth`
- **Metadata**: `model_metadata_lstm_optimized_20250629_142849.json`
- **Features**: `features_lstm_optimized_20250629_142849.json`
- **Feature Scaler**: `baseline_scaler.joblib`
- **Target Scaler**: `baseline_target_scaler.joblib`
- **Inference Example**: `inference_example.py`

## Deployment Requirements
```bash
pip install torch>=1.12.0 numpy pandas scikit-learn joblib
```

## Directory Structure
```
models/
├── deployment/
│   ├── lstm_model_20250629_142849.pth           # Main model file
│   ├── model_metadata_lstm_optimized_20250629_142849.json     # Model configuration
│   ├── features_lstm_optimized_20250629_142849.json       # Feature information
│   └── inference_example.py  # Example code
└── scalers/
    ├── baseline_scaler.joblib     # Feature scaler
    └── baseline_target_scaler.joblib      # Target scaler
```

## Quick Start
1. Load the inference example: `inference_example.py`
2. Modify input data format to match your needs
3. Run inference on new data

## Model Architecture Details
- **Type**: LSTM
- **Optimization**: performance_optimized
- **Preprocessing**: RobustScaler
- **Sequence Length**: 50

## Important Notes
- Ensure all preprocessing steps match the training pipeline exactly
- Use the same feature order as training
- Apply the same scaling transformations
- Handle missing values the same way (fill with 0)
- Create sequences with the exact same length as training
