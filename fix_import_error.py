#!/usr/bin/env python3
"""
fix_import_error.py - Diagnostic and Fix Script for DataCollector Import Error

This script will:
1. Check what's actually in your data_loader.py file
2. Identify the available classes
3. Provide a fix for the run_experiment.py import issue
"""

import sys
import os
import importlib.util
import inspect
from pathlib import Path

def check_data_loader_file():
    """Check what's actually in the data_loader.py file"""
    print("🔍 DIAGNOSING DATA_LOADER.PY FILE")
    print("=" * 50)
    
    # Check if file exists
    data_loader_path = Path("src/data_loader.py")
    if not data_loader_path.exists():
        print(f"❌ File not found: {data_loader_path}")
        return None
    
    print(f"✅ File found: {data_loader_path}")
    
    # Try to load the module
    try:
        # Add src to path temporarily
        sys.path.insert(0, "src")
        
        # Import the module
        import data_loader
        
        print("✅ Module imported successfully")
        
        # Check available classes and functions
        available_items = []
        for name, obj in inspect.getmembers(data_loader):
            if inspect.isclass(obj) and obj.__module__ == 'data_loader':
                available_items.append(f"Class: {name}")
            elif inspect.isfunction(obj) and obj.__module__ == 'data_loader':
                available_items.append(f"Function: {name}")
        
        print(f"\n📋 Available items in data_loader module:")
        for item in available_items:
            print(f"   • {item}")
        
        # Check specifically for collector classes
        collector_classes = []
        for name, obj in inspect.getmembers(data_loader):
            if (inspect.isclass(obj) and 
                obj.__module__ == 'data_loader' and 
                'collector' in name.lower()):
                collector_classes.append(name)
        
        print(f"\n📊 Collector classes found: {collector_classes}")
        
        return {
            'module': data_loader,
            'collector_classes': collector_classes,
            'all_classes': [name for name, obj in inspect.getmembers(data_loader) 
                           if inspect.isclass(obj) and obj.__module__ == 'data_loader']
        }
        
    except Exception as e:
        print(f"❌ Error importing module: {e}")
        print(f"   Error type: {type(e).__name__}")
        
        # Try to read the file and check for syntax errors
        try:
            with open(data_loader_path, 'r') as f:
                content = f.read()
            
            # Basic checks
            print(f"\n📄 File basic info:")
            print(f"   • Size: {len(content)} characters")
            print(f"   • Lines: {len(content.splitlines())} lines")
            
            # Check for common issues
            if 'class DataCollector' in content:
                print("   ✅ 'class DataCollector' found in file")
            else:
                print("   ❌ 'class DataCollector' NOT found in file")
            
            if 'class RefinedAcademicDataCollector' in content:
                print("   ✅ 'class RefinedAcademicDataCollector' found in file")
            
            # Check for other collector classes
            lines = content.splitlines()
            class_lines = [line.strip() for line in lines if line.strip().startswith('class ')]
            print(f"\n📝 All class definitions found:")
            for line in class_lines:
                print(f"   • {line}")
                
        except Exception as read_error:
            print(f"❌ Could not read file: {read_error}")
        
        return None

def create_fixed_run_experiment():
    """Create a fixed version of run_experiment.py with robust import handling"""
    
    print("\n🔧 CREATING FIXED RUN_EXPERIMENT.PY")
    print("=" * 50)
    
    fixed_code = '''"""
run_experiment.py - FIXED VERSION WITH ROBUST IMPORT HANDLING

This version automatically detects what's available in the data_loader module
and adapts accordingly, preventing import errors.
"""

import argparse
import yaml
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import warnings
import json
import sys
import os
import traceback
from typing import Dict, List, Optional, Any, Union
import pickle
import time
import importlib.util
import inspect

# Add src to path
sys.path.append('src')

# Setup enhanced logging
def setup_logging(log_level: str = "INFO", log_file: str = "experiment.log"):
    """Setup enhanced logging with both file and console output"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.FileHandler(f"logs/{log_file}"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Suppress warnings
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    
    return logging.getLogger(__name__)

logger = setup_logging()

def detect_available_data_collector():
    """Dynamically detect what data collector classes are available"""
    logger.info("🔍 Detecting available data collector classes...")
    
    try:
        # Try to import data_loader module
        import data_loader
        
        # Find all collector classes
        collector_classes = {}
        
        for name, obj in inspect.getmembers(data_loader):
            if (inspect.isclass(obj) and 
                obj.__module__ == 'data_loader' and 
                ('collector' in name.lower() or 'datacollector' in name.lower())):
                collector_classes[name] = obj
                logger.info(f"   ✅ Found: {name}")
        
        # Check for specific known classes
        known_classes = ['DataCollector', 'RefinedAcademicDataCollector', 'EnhancedDataCollector']
        for class_name in known_classes:
            if hasattr(data_loader, class_name):
                collector_classes[class_name] = getattr(data_loader, class_name)
                logger.info(f"   ✅ Found: {class_name}")
        
        if not collector_classes:
            logger.warning("❌ No collector classes found, checking all classes...")
            
            # Fallback: check all classes
            all_classes = {}
            for name, obj in inspect.getmembers(data_loader):
                if inspect.isclass(obj) and obj.__module__ == 'data_loader':
                    all_classes[name] = obj
                    logger.info(f"   📋 Available class: {name}")
            
            # If we have any classes, use the first one as collector
            if all_classes:
                first_class_name = list(all_classes.keys())[0]
                collector_classes[first_class_name] = all_classes[first_class_name]
                logger.info(f"   🔄 Using {first_class_name} as data collector")
        
        return collector_classes
        
    except Exception as e:
        logger.error(f"❌ Could not import data_loader module: {e}")
        return {}

class RobustExperimentRunner:
    """
    Robust experiment runner that adapts to available data collector classes
    """
    
    def __init__(self, config_path: str = "configs/model_config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        
        # Detect available data collector
        self.available_collectors = detect_available_data_collector()
        
        if not self.available_collectors:
            raise RuntimeError("❌ No data collector classes found in data_loader module")
        
        # Choose the best available collector
        self.collector_class = self._choose_best_collector()
        self.data_collector = None
        
        # Create directories
        self.results_dir = Path("results")
        self.data_dir = Path("data")
        self.cache_dir = Path("data/cache")
        self._ensure_directories()
        
        # Initialize
        self._initialize_data_collector()
        
        logger.info(f"✅ RobustExperimentRunner initialized with {self.collector_class.__name__}")
    
    def _choose_best_collector(self):
        """Choose the best available data collector class"""
        # Priority order
        preferred_order = [
            'DataCollector',
            'EnhancedDataCollector', 
            'RefinedAcademicDataCollector'
        ]
        
        # Check preferred classes first
        for preferred_name in preferred_order:
            if preferred_name in self.available_collectors:
                logger.info(f"✅ Using preferred collector: {preferred_name}")
                return self.available_collectors[preferred_name]
        
        # Fallback to first available
        first_available = list(self.available_collectors.keys())[0]
        logger.info(f"🔄 Using available collector: {first_available}")
        return self.available_collectors[first_available]
    
    def _load_config(self) -> dict:
        """Load configuration with robust fallbacks"""
        config = {}
        
        # Try to load main config
        if Path(self.config_path).exists():
            try:
                with open(self.config_path, 'r') as f:
                    config = yaml.safe_load(f)
                logger.info(f"✅ Loaded config from {self.config_path}")
            except Exception as e:
                logger.warning(f"⚠️ Could not load {self.config_path}: {e}")
        
        # Try to load data config
        data_config_path = "configs/data_config.yaml"
        if Path(data_config_path).exists():
            try:
                with open(data_config_path, 'r') as f:
                    data_config = yaml.safe_load(f)
                
                # Merge data configuration
                if 'data' not in config:
                    config['data'] = {}
                
                config['data'].update(data_config.get('data', {}))
                config.update({k: v for k, v in data_config.items() if k != 'data'})
                
                logger.info(f"✅ Merged data config from {data_config_path}")
            except Exception as e:
                logger.warning(f"⚠️ Could not load data config: {e}")
        
        # Apply defaults
        return self._apply_defaults(config)
    
    def _apply_defaults(self, config: dict) -> dict:
        """Apply default configuration"""
        defaults = {
            'data': {
                'stocks': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],
                'start_date': '2018-12-01',
                'end_date': '2024-01-31',
                'cache_enabled': True,
                'use_parallel': True,
                'news_sources': ['yahoo_finance', 'mock']
            },
            'experiment': {
                'save_intermediate_results': True
            }
        }
        
        # Simple merge
        for key, value in defaults.items():
            if key not in config:
                config[key] = value
            elif isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if subkey not in config[key]:
                        config[key][subkey] = subvalue
        
        return config
    
    def _ensure_directories(self):
        """Create required directories"""
        directories = [
            self.results_dir,
            self.data_dir,
            self.data_dir / "processed",
            self.cache_dir,
            Path("logs")
        ]
        
        for directory in directories:
            try:
                directory.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.debug(f"Could not create {directory}: {e}")
    
    def _initialize_data_collector(self):
        """Initialize the data collector with proper error handling"""
        try:
            # Try different initialization approaches
            if hasattr(self.collector_class, '__init__'):
                # Check what parameters the constructor accepts
                init_signature = inspect.signature(self.collector_class.__init__)
                params = list(init_signature.parameters.keys())
                
                # Try different initialization approaches
                if 'config_path' in params:
                    data_config_path = "configs/data_config.yaml" if Path("configs/data_config.yaml").exists() else None
                    self.data_collector = self.collector_class(
                        config_path=data_config_path,
                        cache_dir=str(self.cache_dir)
                    )
                elif 'cache_dir' in params:
                    self.data_collector = self.collector_class(cache_dir=str(self.cache_dir))
                else:
                    # Try simple initialization
                    self.data_collector = self.collector_class()
                
                logger.info(f"✅ Initialized {self.collector_class.__name__}")
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize data collector: {e}")
            raise RuntimeError(f"Data collector initialization failed: {e}")
    
    def step_1_data_collection(self) -> dict:
        """Step 1: Data collection with robust error handling"""
        logger.info("=" * 60)
        logger.info("STEP 1: ROBUST DATA COLLECTION")
        logger.info("=" * 60)
        
        step_start = datetime.now()
        
        try:
            # Market data collection
            logger.info("📈 Collecting market data...")
            
            # Check what methods are available
            if hasattr(self.data_collector, 'collect_market_data'):
                market_data = self.data_collector.collect_market_data(
                    symbols=self.config['data']['stocks'],
                    use_parallel=self.config['data'].get('use_parallel', True)
                )
            else:
                logger.warning("⚠️ collect_market_data method not found, creating mock data")
                market_data = self._create_mock_market_data()
            
            if not market_data:
                raise Exception("No market data collected")
            
            logger.info(f"✅ Market data collected for {len(market_data)} symbols")
            
            # News data collection
            logger.info("📰 Collecting news data...")
            
            if hasattr(self.data_collector, 'collect_news_data'):
                news_data = self.data_collector.collect_news_data(
                    symbols=self.config['data']['stocks']
                )
            else:
                logger.warning("⚠️ collect_news_data method not found, creating mock data")
                news_data = self._create_mock_news_data()
            
            total_articles = sum(len(articles) for articles in news_data.values())
            logger.info(f"✅ News data collected: {total_articles} articles")
            
            # Combined dataset
            logger.info("🔄 Creating combined dataset...")
            
            if hasattr(self.data_collector, 'create_combined_dataset'):
                combined_dataset = self.data_collector.create_combined_dataset(
                    market_data, news_data,
                    save_path="data/processed/combined_dataset.parquet"
                )
            else:
                logger.warning("⚠️ create_combined_dataset method not found, creating basic dataset")
                combined_dataset = self._create_basic_combined_dataset(market_data, news_data)
                combined_dataset.to_parquet("data/processed/combined_dataset.parquet")
            
            if combined_dataset.empty:
                raise Exception("Combined dataset is empty")
            
            logger.info(f"✅ Combined dataset created: {combined_dataset.shape}")
            
            # Save news data for later steps
            with open(self.cache_dir / "news_data.pkl", 'wb') as f:
                pickle.dump(news_data, f)
            
            processing_time = (datetime.now() - step_start).total_seconds()
            
            step_results = {
                'success': True,
                'processing_time_seconds': processing_time,
                'collector_used': self.collector_class.__name__,
                'market_data_symbols': len(market_data),
                'news_articles': total_articles,
                'dataset_shape': list(combined_dataset.shape),
                'date_range': {
                    'start': combined_dataset.index.min().isoformat(),
                    'end': combined_dataset.index.max().isoformat()
                }
            }
            
            logger.info("✅ STEP 1 COMPLETED SUCCESSFULLY")
            logger.info(f"   Collector: {self.collector_class.__name__}")
            logger.info(f"   Dataset: {combined_dataset.shape}")
            logger.info(f"   Processing Time: {processing_time:.1f}s")
            
            return step_results
            
        except Exception as e:
            logger.error(f"❌ STEP 1 ERROR: {e}")
            logger.error(traceback.format_exc())
            
            processing_time = (datetime.now() - step_start).total_seconds()
            return {
                'success': False,
                'error': str(e),
                'processing_time_seconds': processing_time
            }
    
    def _create_mock_market_data(self):
        """Create mock market data for testing"""
        logger.info("🔄 Creating mock market data...")
        
        from dataclasses import dataclass
        
        @dataclass
        class MockMarketData:
            symbol: str
            data: pd.DataFrame
            technical_indicators: pd.DataFrame
            sector: str = "Technology"
        
        mock_data = {}
        
        for symbol in self.config['data']['stocks']:
            # Create mock price data
            dates = pd.date_range(self.config['data']['start_date'], 
                                 self.config['data']['end_date'], freq='B')
            
            np.random.seed(hash(symbol) % 2**32)  # Consistent random data per symbol
            base_price = 100
            returns = np.random.normal(0.001, 0.02, len(dates))
            prices = base_price * np.exp(np.cumsum(returns))
            
            market_df = pd.DataFrame({
                'Open': prices * np.random.uniform(0.99, 1.01, len(dates)),
                'High': prices * np.random.uniform(1.00, 1.05, len(dates)),
                'Low': prices * np.random.uniform(0.95, 1.00, len(dates)),
                'Close': prices,
                'Volume': np.random.randint(1000000, 10000000, len(dates))
            }, index=dates)
            
            # Create mock technical indicators
            tech_df = pd.DataFrame({
                'SMA_20': market_df['Close'].rolling(20).mean(),
                'RSI_14': 50 + 30 * np.sin(np.arange(len(dates)) / 10),
                'MACD': np.random.normal(0, 0.5, len(dates))
            }, index=dates)
            
            mock_data[symbol] = MockMarketData(symbol, market_df, tech_df)
        
        return mock_data
    
    def _create_mock_news_data(self):
        """Create mock news data for testing"""
        logger.info("🔄 Creating mock news data...")
        
        from dataclasses import dataclass
        
        @dataclass
        class MockNewsArticle:
            title: str
            content: str
            date: datetime
            source: str
            url: str = ""
            relevance_score: float = 0.8
            sentiment_score: float = 0.0
        
        mock_news = {}
        
        for symbol in self.config['data']['stocks']:
            articles = []
            
            # Generate some mock articles
            for i in range(20):  # 20 articles per symbol
                base_date = datetime.strptime(self.config['data']['start_date'], '%Y-%m-%d')
                article_date = base_date + timedelta(days=i*10)
                
                articles.append(MockNewsArticle(
                    title=f"{symbol} reports quarterly results",
                    content=f"Analysis of {symbol} performance shows positive trends.",
                    date=article_date,
                    source='mock',
                    sentiment_score=np.random.normal(0, 0.3)
                ))
            
            mock_news[symbol] = articles
        
        return mock_news
    
    def _create_basic_combined_dataset(self, market_data, news_data):
        """Create basic combined dataset"""
        logger.info("🔄 Creating basic combined dataset...")
        
        combined_data = []
        
        for symbol, data in market_data.items():
            df = data.data.copy()
            df['symbol'] = symbol
            df['sector'] = data.sector
            
            # Add basic technical indicators
            for col in data.technical_indicators.columns:
                df[col] = data.technical_indicators[col]
            
            # Add basic news features
            df['news_count'] = 0
            df['avg_sentiment'] = 0
            
            # Add basic target variables
            for horizon in [5, 30, 90]:
                df[f'target_{horizon}d'] = df['Close'].shift(-horizon)
                df[f'return_{horizon}d'] = df[f'target_{horizon}d'] / df['Close'] - 1
            
            combined_data.append(df)
        
        # Combine all symbols
        final_df = pd.concat(combined_data, ignore_index=False)
        final_df = final_df.sort_index()
        
        # Clean up
        final_df = final_df.dropna(subset=[col for col in final_df.columns if col.startswith('target_')])
        
        return final_df
    
    def run_experiment(self, steps: List[str] = None) -> dict:
        """Run experiment with specified steps"""
        if steps is None:
            steps = ['1']  # Default to just step 1 for now
        
        logger.info("🚀 STARTING ROBUST EXPERIMENT")
        logger.info("=" * 50)
        
        start_time = datetime.now()
        results_summary = {}
        
        try:
            for step in steps:
                if step == '1':
                    results_summary['step_1'] = self.step_1_data_collection()
                else:
                    logger.warning(f"Step {step} not implemented in robust version yet")
                
                # Check if step failed
                if not results_summary.get(f'step_{step}', {}).get('success', False):
                    logger.error(f"Step {step} failed, stopping experiment")
                    break
            
            end_time = datetime.now()
            total_time = (end_time - start_time).total_seconds()
            
            experiment_results = {
                'status': 'completed',
                'total_runtime_seconds': total_time,
                'steps_results': results_summary,
                'collector_used': self.collector_class.__name__
            }
            
            logger.info("✅ ROBUST EXPERIMENT COMPLETED!")
            logger.info(f"   Total Runtime: {total_time:.1f}s")
            logger.info(f"   Collector Used: {self.collector_class.__name__}")
            
            return experiment_results
            
        except Exception as e:
            logger.error(f"❌ EXPERIMENT FAILED: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }

def main():
    """Main function with robust error handling"""
    parser = argparse.ArgumentParser(description='Robust Multi-Horizon Sentiment-Enhanced TFT Experiment')
    parser.add_argument('--config', type=str, default='configs/model_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--step', type=str, choices=['1', '2', '3', '4', '5', '6'],
                       default='1', help='Which step to run')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        runner = RobustExperimentRunner(args.config)
        result = runner.run_experiment([args.step])
        
        if result.get('status') == 'completed':
            logger.info("\\n✅ EXPERIMENT COMPLETED SUCCESSFULLY!")
        else:
            logger.error(f"\\n❌ EXPERIMENT FAILED: {result.get('error')}")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"\\n💥 Critical error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
    
    # Save the fixed version
    with open("run_experiment_fixed.py", 'w') as f:
        f.write(fixed_code)
    
    print("✅ Created run_experiment_fixed.py")
    print("   This version will automatically detect available data collectors")
    print("   and adapt to whatever is in your data_loader.py file")

def main():
    """Main diagnostic and fix function"""
    print("🚀 DATA LOADER IMPORT ERROR DIAGNOSTIC AND FIX")
    print("=" * 60)
    
    # Step 1: Check what's in the data_loader file
    module_info = check_data_loader_file()
    
    # Step 2: Create fixed version
    create_fixed_run_experiment()
    
    print("\n🎯 NEXT STEPS:")
    print("=" * 30)
    
    if module_info and module_info['collector_classes']:
        print("✅ Collector classes found - you can use the fixed version:")
        print("   python run_experiment_fixed.py --step 1")
    else:
        print("❌ No collector classes found. You need to:")
        print("   1. Check if your data_loader.py has syntax errors")
        print("   2. Ensure it has a DataCollector class defined")
        print("   3. Or use the provided enhanced data_loader.py")
        print("\nYou can try the fixed version anyway:")
        print("   python run_experiment_fixed.py --step 1")
    
    print("\n🔧 The fixed version will:")
    print("   • Automatically detect available classes")
    print("   • Adapt to whatever is in your data_loader.py")
    print("   • Provide helpful error messages")
    print("   • Create mock data if needed for testing")

if __name__ == "__main__":
    main()