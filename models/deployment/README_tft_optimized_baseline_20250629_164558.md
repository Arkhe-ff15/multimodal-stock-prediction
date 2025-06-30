# TFT_Optimized_Baseline Model Deployment Guide

## Model Information
- **Model Type**: TFT_Optimized_Baseline
- **Created**: 20250629_164558
- **Target**: target_5
- **Features**: 33 features
- **Dataset**: baseline
- **Architecture**: TemporalFusionTransformer
- **Parameters**: 628,824

## Performance Metrics

## Files for Deployment
- **Model**: `tft_model_tft_optimized_baseline_20250629_164558.pth`
- **Metadata**: `model_metadata_tft_optimized_baseline_20250629_164558.json`
- **Features**: `features_tft_optimized_baseline_20250629_164558.json`
- **Feature Scaler**: `baseline_scaler.joblib`
- **Target Scaler**: `baseline_target_scaler.joblib`
- **Inference Example**: `inference_example.py`

## Deployment Requirements
```bash
pip install torch>=1.12.0 numpy pandas scikit-learn joblib
pip install pytorch-forecasting
```

## Directory Structure
```
models/
├── deployment/
│   ├── tft_model_tft_optimized_baseline_20250629_164558.pth           # Main model file
│   ├── model_metadata_tft_optimized_baseline_20250629_164558.json     # Model configuration
│   ├── features_tft_optimized_baseline_20250629_164558.json       # Feature information
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
- **Type**: TemporalFusionTransformer
- **Optimization**: performance_optimized
- **Preprocessing**: RobustScaler

## Important Notes
- Ensure all preprocessing steps match the training pipeline exactly
- Use the same feature order as training
- Apply the same scaling transformations
- Handle missing values the same way (fill with 0)
- Format data for multi-horizon forecasting
