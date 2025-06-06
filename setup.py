#!/usr/bin/env python3
"""
Multi-Horizon Sentiment-Enhanced TFT Project Setup
Creates the complete project structure from scratch
"""

import os
import yaml
from pathlib import Path

def create_project_structure():
    """Create the complete project directory structure"""
    
    # Base project directory
    project_name = "sentiment_tft"
    base_dir = Path(project_name)
    
    # Define directory structure
    directories = [
        "data/raw",
        "data/processed", 
        "data/sentiment",
        "src",
        "experiments",
        "notebooks",
        "configs",
        "results/models",
        "results/plots",
        "results/metrics",
        "tests"
    ]
    
    # Create directories
    for directory in directories:
        (base_dir / directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    return base_dir

def create_data_config(base_dir):
    """Create data configuration file"""
    config = {
        'data': {
            'stocks': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],
            'start_date': '2023-01-01',
            'end_date': '2024-12-01',
            'validation_split': 0.15,
            'test_split': 0.15,
            'features': {
                'price': ['open', 'high', 'low', 'close', 'volume'],
                'technical': ['rsi', 'macd', 'bb_upper', 'bb_lower'],
                'lags': [1, 2, 3, 5, 10]
            }
        },
        'sentiment': {
            'sources': ['yahoo_finance_news', 'newsapi_fallback'],
            'quality_threshold': 0.70,
            'relevance_threshold': 0.85,
            'max_articles_per_day': 20,
            'text_max_length': 512
        }
    }
    
    config_path = base_dir / "configs" / "data_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    print(f"✓ Created: {config_path}")

def create_model_config(base_dir):
    """Create model configuration with overfitting prevention"""
    config = {
        'temporal_decay': {
            # Core innovation: horizon-specific decay parameters
            'lambda_5': 0.3,    # Fast decay for 5-day forecasts
            'lambda_30': 0.1,   # Medium decay for 30-day forecasts  
            'lambda_90': 0.05,  # Slow decay for 90-day forecasts
            'lookback_days': {5: 10, 30: 30, 90: 60},
            'min_sentiment_count': 3  # Minimum articles for reliable sentiment
        },
        'model': {
            'hidden_size': 64,  # Smaller to prevent overfitting
            'lstm_layers': 2,
            'attention_head_size': 4,
            'dropout': 0.3,     # Higher dropout for regularization
            'learning_rate': 0.001,
            'batch_size': 32,
            'max_epochs': 50,
            'early_stopping_patience': 10,
            'reduce_lr_patience': 5
        },
        'training': {
            # Overfitting prevention strategies
            'weight_decay': 1e-4,
            'gradient_clip_val': 1.0,
            'validation_check_interval': 0.25,
            'monitor_metric': 'val_loss',
            'mode': 'min',
            # Cross-validation for robust evaluation
            'cv_folds': 5,
            'cv_method': 'time_series'  # Respects temporal order
        },
        'horizons': [5, 30, 90],
        'random_seed': 42
    }
    
    config_path = base_dir / "configs" / "model_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    print(f"✓ Created: {config_path}")

def create_requirements(base_dir):
    """Create requirements.txt with all dependencies"""
    requirements = [
        "torch>=1.12.0",
        "pytorch-forecasting>=0.10.0", 
        "pytorch-lightning>=1.8.0",
        "transformers>=4.20.0",
        "yfinance>=0.2.0",
        "newsapi-python>=0.2.6",
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.2.0",
        "matplotlib>=3.6.0",
        "seaborn>=0.12.0",
        "plotly>=5.11.0",
        "jupyter>=1.0.0",
        "PyYAML>=6.0",
        "tqdm>=4.64.0",
        "ta>=0.10.2",  # Technical analysis library
        "scipy>=1.9.0",
        "statsmodels>=0.13.0",
        "optuna>=3.0.0",  # For hyperparameter optimization
        "wandb>=0.13.0"   # For experiment tracking
    ]
    
    req_path = base_dir / "requirements.txt"
    with open(req_path, 'w') as f:
        f.write('\n'.join(requirements))
    print(f"✓ Created: {req_path}")

def create_readme(base_dir):
    """Create comprehensive README"""
    readme_content = """# Multi-Horizon Sentiment-Enhanced TFT

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
"""
    
    readme_path = base_dir / "README.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    print(f"✓ Created: {readme_path}")

def create_gitignore(base_dir):
    """Create .gitignore file"""
    gitignore_content = """# Data files
data/raw/
data/processed/
*.csv
*.parquet

# Model checkpoints
results/models/
*.ckpt
*.pth

# Jupyter notebooks checkpoints
.ipynb_checkpoints/

# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env/
.env
.venv/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
logs/
*.log
wandb/

# Temporary files
tmp/
temp/
"""
    
    gitignore_path = base_dir / ".gitignore"
    with open(gitignore_path, 'w') as f:
        f.write(gitignore_content)
    print(f"✓ Created: {gitignore_path}")

def create_init_files(base_dir):
    """Create __init__.py files for Python modules"""
    init_dirs = ['src', 'experiments', 'tests']
    
    for directory in init_dirs:
        init_path = base_dir / directory / "__init__.py"
        with open(init_path, 'w') as f:
            f.write(f'"""Multi-Horizon Sentiment-Enhanced TFT - {directory} module"""\n')
        print(f"✓ Created: {init_path}")

if __name__ == "__main__":
    print("🚀 Setting up Multi-Horizon Sentiment-Enhanced TFT Project...\n")
    
    # Create project structure
    base_dir = create_project_structure()
    print()
    
    # Create configuration files
    create_data_config(base_dir)
    create_model_config(base_dir)
    create_requirements(base_dir)
    create_readme(base_dir)
    create_gitignore(base_dir)
    create_init_files(base_dir)
    
    print(f"\n🎉 Project setup complete!")
    print(f"📁 Project created in: {base_dir.resolve()}")
    print(f"\nNext steps:")
    print(f"1. cd {base_dir}")
    print(f"2. pip install -r requirements.txt")
    print(f"3. Start with implementing temporal_decay.py")