#!/usr/bin/env python3
"""
Quick TFT Dataset Creator
========================
Creates smaller datasets optimized for 2-4 hour TFT training
while maintaining research quality.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime, timedelta
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QuickDatasetCreator:
    """Creates optimized datasets for faster TFT training"""
    
    def __init__(self):
        self.base_dataset_path = "data/processed/temporal_decay_enhanced_dataset.csv"
        self.output_dir = Path("data/processed/quick_datasets")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def create_dataset_variants(self):
        """Create multiple dataset variants for different training speeds"""
        
        logger.info("📊 CREATING QUICK TFT DATASETS")
        logger.info("=" * 50)
        
        # Load full dataset
        logger.info("📥 Loading full dataset...")
        try:
            full_data = pd.read_csv(self.base_dataset_path)
            full_data['date'] = pd.to_datetime(full_data['date'])
            logger.info(f"   ✅ Loaded: {full_data.shape[0]:,} records, {full_data.shape[1]} features")
        except Exception as e:
            logger.error(f"❌ Failed to load dataset: {e}")
            return
        
        # Define dataset variants
        variants = {
            "quick_5_symbols": {
                "name": "Quick 5 Symbols (2-3 hours)",
                "symbols": ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA'],
                "date_range": ("2018-01-01", "2024-01-30"),
                "target_records": 7500,
                "target_time": "2-3 hours"
            },
            "recent_4_years": {
                "name": "Recent 4 Years All Symbols (3-4 hours)", 
                "symbols": ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'NFLX'],
                "date_range": ("2020-01-01", "2024-01-30"),
                "target_records": 8000,
                "target_time": "3-4 hours"
            },
            "dev_dataset": {
                "name": "Development Dataset (1.5-2.5 hours)",
                "symbols": ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA'],
                "date_range": ("2020-01-01", "2024-01-30"),
                "target_records": 5000,
                "target_time": "1.5-2.5 hours"
            },
            "mini_dataset": {
                "name": "Mini Dataset (1-1.5 hours)",
                "symbols": ['AAPL', 'MSFT', 'GOOGL'],
                "date_range": ("2021-01-01", "2024-01-30"),
                "target_records": 3000,
                "target_time": "1-1.5 hours"
            }
        }
        
        results = {}
        
        # Create each variant
        for variant_name, config in variants.items():
            logger.info(f"\n🔨 Creating {config['name']}...")
            
            # Filter data
            filtered_data = self._filter_dataset(
                full_data, 
                symbols=config['symbols'],
                date_range=config['date_range']
            )
            
            if filtered_data.empty:
                logger.warning(f"   ⚠️ No data for {variant_name}")
                continue
            
            # Save dataset
            output_path = self.output_dir / f"{variant_name}.csv"
            filtered_data.to_csv(output_path, index=False)
            
            # Calculate metrics
            metrics = self._calculate_metrics(filtered_data, config)
            results[variant_name] = {
                "config": config,
                "metrics": metrics,
                "path": str(output_path)
            }
            
            logger.info(f"   ✅ Saved: {metrics['records']:,} records")
            logger.info(f"   📊 Symbols: {metrics['symbols']}")
            logger.info(f"   📅 Date range: {metrics['date_range']}")
            logger.info(f"   ⏱️ Est. training time: {config['target_time']}")
            logger.info(f"   💾 File: {output_path}")
        
        # Save summary
        self._save_summary(results)
        
        # Recommendations
        self._print_recommendations(results)
        
        return results
    
    def _filter_dataset(self, data: pd.DataFrame, symbols: list, date_range: tuple) -> pd.DataFrame:
        """Filter dataset by symbols and date range"""
        
        # Filter by symbols
        symbol_mask = data['symbol'].isin(symbols)
        filtered_data = data[symbol_mask].copy()
        
        # Filter by date range
        start_date, end_date = date_range
        date_mask = (
            (filtered_data['date'] >= pd.to_datetime(start_date)) &
            (filtered_data['date'] <= pd.to_datetime(end_date))
        )
        filtered_data = filtered_data[date_mask]
        
        # Sort for consistency
        filtered_data = filtered_data.sort_values(['symbol', 'date']).reset_index(drop=True)
        
        return filtered_data
    
    def _calculate_metrics(self, data: pd.DataFrame, config: dict) -> dict:
        """Calculate dataset metrics"""
        
        metrics = {
            "records": len(data),
            "features": data.shape[1],
            "symbols": data['symbol'].nunique(),
            "symbol_list": sorted(data['symbol'].unique().tolist()),
            "date_range": (
                str(data['date'].min().date()),
                str(data['date'].max().date())
            ),
            "target_coverage": {
                "target_5": float(data['target_5'].notna().mean()),
                "target_30d": float(data['target_30d'].notna().mean()) if 'target_30d' in data.columns else 0.0,
                "target_90d": float(data['target_90d'].notna().mean()) if 'target_90d' in data.columns else 0.0
            },
            "sentiment_coverage": float((data['sentiment_decay_5d_compound'] != 0).mean()) if 'sentiment_decay_5d_compound' in data.columns else 0.0,
            "estimated_training_time": config['target_time'],
            "estimated_memory_gb": f"{max(6, int(len(data) / 1000 * 0.8))}-{max(10, int(len(data) / 1000 * 1.2))}"
        }
        
        return metrics
    
    def _save_summary(self, results: dict):
        """Save dataset summary"""
        
        summary_path = self.output_dir / "dataset_summary.json"
        
        summary = {
            "created_at": datetime.now().isoformat(),
            "variants": results,
            "recommendations": {
                "development": "dev_dataset (fastest iteration)",
                "proof_of_concept": "quick_5_symbols (balanced)",
                "recent_focus": "recent_4_years (current patterns)",
                "debugging": "mini_dataset (ultra-fast)"
            }
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"\n💾 Summary saved: {summary_path}")
    
    def _print_recommendations(self, results: dict):
        """Print usage recommendations"""
        
        logger.info("\n💡 USAGE RECOMMENDATIONS")
        logger.info("=" * 35)
        
        recommendations = [
            ("🚀 **DEVELOPMENT**", "dev_dataset", "Best for model development & hyperparameter tuning"),
            ("⚡ **PROOF OF CONCEPT**", "quick_5_symbols", "Balanced speed vs quality for initial results"),
            ("📈 **RECENT PATTERNS**", "recent_4_years", "Focus on current market conditions"),
            ("🔧 **DEBUGGING**", "mini_dataset", "Ultra-fast for testing pipeline issues"),
        ]
        
        for purpose, variant, description in recommendations:
            if variant in results:
                metrics = results[variant]['metrics']
                logger.info(f"\n{purpose}:")
                logger.info(f"   📁 File: {variant}.csv")
                logger.info(f"   📊 Records: {metrics['records']:,}")
                logger.info(f"   ⏱️ Training: {metrics['estimated_training_time']}")
                logger.info(f"   💾 Memory: {metrics['estimated_memory_gb']}GB")
                logger.info(f"   🎯 Use case: {description}")
        
        logger.info(f"\n📂 All datasets saved in: {self.output_dir}")
        
        # Quick start commands
        logger.info(f"\n🚀 QUICK START COMMANDS:")
        logger.info(f"# Copy your preferred dataset:")
        logger.info(f"cp {self.output_dir}/dev_dataset.csv data/processed/combined_dataset.csv")
        logger.info(f"")
        logger.info(f"# Then run TFT training normally")
        logger.info(f"# Training will be 2-3x faster!")

def main():
    """Create quick datasets for faster TFT training"""
    
    creator = QuickDatasetCreator()
    results = creator.create_dataset_variants()
    
    if results:
        logger.info("\n✅ QUICK DATASETS CREATED SUCCESSFULLY!")
        logger.info(f"📊 {len(results)} variants ready for TFT training")
        logger.info("🚀 Choose based on your speed vs quality needs")
    else:
        logger.error("❌ Failed to create quick datasets")

if __name__ == "__main__":
    main()