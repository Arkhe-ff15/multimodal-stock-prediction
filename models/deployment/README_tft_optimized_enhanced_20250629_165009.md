# TFT_Optimized_Enhanced Model Deployment Guide

## Model Information
- **Model Type**: TFT_Optimized_Enhanced
- **Created**: 20250629_165009
- **Target**: target_5
- **Features**: 53 features
- **Dataset**: enhanced
- **Architecture**: TemporalFusionTransformer
- **Parameters**: 867,564

## Performance Metrics

## Files for Deployment
- **Model**: `tft_model_tft_optimized_enhanced_20250629_165009.pth`
- **Metadata**: `model_metadata_tft_optimized_enhanced_20250629_165009.json`
- **Features**: `features_tft_optimized_enhanced_20250629_165009.json`
- **Feature Scaler**: `enhanced_scaler.joblib`
- **Target Scaler**: `enhanced_target_scaler.joblib`
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
│   ├── tft_model_tft_optimized_enhanced_20250629_165009.pth           # Main model file
│   ├── model_metadata_tft_optimized_enhanced_20250629_165009.json     # Model configuration
│   ├── features_tft_optimized_enhanced_20250629_165009.json       # Feature information
│   └── inference_example.py  # Example code
└── scalers/
    ├── enhanced_scaler.joblib     # Feature scaler
    └── enhanced_target_scaler.joblib      # Target scaler
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
