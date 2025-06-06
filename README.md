# Multi-Horizon Sentiment-Enhanced TFT

## Core Innovation ⭐
**Horizon-specific temporal sentiment decay for financial forecasting**

Question: Can exponential sentiment decay parameters tailored to different forecasting horizons (5, 30, 90 days) improve TFT performance?

## Quick Start
```bash
# Setup environment
pip install -r requirements.txt

# Run complete experiment
python run_experiment.py

# Open analysis notebooks
jupyter notebook notebooks/
```

## Project Structure
- `src/` - Core implementation
- `experiments/` - Training scripts  
- `notebooks/` - Analysis & visualization
- `configs/` - Configuration files
- `results/` - Model outputs & plots

## Key Features
- **Temporal Decay**: Horizon-specific sentiment weighting
- **Overfitting Prevention**: Early stopping, dropout, regularization
- **Robust Evaluation**: Time-series CV, statistical testing
- **Rich Visualization**: Interactive plots, model comparisons

## Models Compared
1. TFT-Temporal-Decay (Our contribution)
2. TFT-Static-Sentiment 
3. TFT-Numerical (Baseline)
4. LSTM (Traditional baseline)

## Expected Results
Strongest improvements for short-term forecasts, diminishing but significant for long-term predictions.
