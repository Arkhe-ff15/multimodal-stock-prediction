#!/usr/bin/env python3
"""
OPTIMIZED FNSPID PROCESSOR - Pipeline Integration v2.0
=====================================================

✅ ENHANCED FOR YOUR PIPELINE:
- Serves as efficient filter for 22GB FNSPID dataset  
- Filters for 2018-2024 time period matching data.py
- Targets specific symbols from data.py configuration
- Outputs standardized format for sentiment.py integration
- Memory-efficient chunked processing
- Timezone-safe date handling
- Quality control and data validation

PURPOSE: Smart filter to reduce 22GB → manageable dataset for sentiment analysis
OUTPUT: Clean sentiment data ready for temporal_decay.py processing
"""

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging
import sys
import os
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
import warnings
import json
import re
import gc
import time
from tqdm import tqdm
import pytz
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('fnspid_processing.log')
    ]
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# Paths aligned with your pipeline
DATA_DIR = "data/processed"
RAW_DIR = "data/raw"
FNSPID_NEWS_FILE = f"{RAW_DIR}/nasdaq_exteral_data.csv"
CORE_DATASET = f"{DATA_DIR}/combined_dataset.csv"

# Pipeline-aligned output files
FILTERED_FNSPID_OUTPUT = f"{DATA_DIR}/fnspid_filtered_articles.csv"
ARTICLE_SENTIMENT_OUTPUT = f"{DATA_DIR}/fnspid_article_sentiments.csv"
DAILY_SENTIMENT_OUTPUT = f"{DATA_DIR}/fnspid_daily_sentiment.csv"

@dataclass
class PipelineConfig:
    """Configuration aligned with your data.py pipeline"""
    # Core symbols from your data.py configuration
    target_symbols: List[str]
    
    # Date range matching your core dataset (2018-2024)
    start_date: str = "2018-01-01"
    end_date: str = "2024-01-31"
    
    # Processing parameters for 22GB dataset
    chunk_size: int = 50000
    sample_ratio: float = 0.15  # 15% sample for efficiency
    
    # Quality filters
    min_article_length: int = 100
    min_confidence_threshold: float = 0.6
    max_articles_per_symbol_per_day: int = 50
    
    # Memory management
    finbert_batch_size: int = 16
    enable_gc_frequency: int = 10  # Every 10 chunks
    
    def __post_init__(self):
        """Validate configuration"""
        if not self.target_symbols:
            raise ValueError("target_symbols cannot be empty")
        
        # Ensure dates are properly formatted
        self.start_date = pd.to_datetime(self.start_date).strftime('%Y-%m-%d')
        self.end_date = pd.to_datetime(self.end_date).strftime('%Y-%m-%d')

class OptimizedFNSPIDProcessor:
    """
    Optimized processor for filtering large FNSPID dataset and extracting sentiment
    
    Designed specifically for your pipeline:
    1. Efficiently filter 22GB dataset to 2018-2024, target symbols
    2. Apply high-quality sentiment analysis with FinBERT
    3. Output standardized format for temporal_decay.py integration
    4. Memory-efficient processing for large datasets
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        
        print("🚀 OPTIMIZED FNSPID PROCESSOR v2.0 - PIPELINE INTEGRATION")
        print("=" * 70)
        print(f"📊 Target symbols: {config.target_symbols}")
        print(f"📅 Date range: {config.start_date} to {config.end_date}")
        print(f"📦 Chunk size: {config.chunk_size:,}")
        print(f"🎯 Sample ratio: {config.sample_ratio:.1%}")
        print(f"📁 Source: {FNSPID_NEWS_FILE}")
        
        # Check file existence
        if not os.path.exists(FNSPID_NEWS_FILE):
            raise FileNotFoundError(f"❌ FNSPID dataset not found: {FNSPID_NEWS_FILE}")
        
        file_size_gb = os.path.getsize(FNSPID_NEWS_FILE) / (1024**3)
        print(f"📊 File size: {file_size_gb:.1f} GB")
        
        # Load and cache target symbols
        self.all_target_symbols = self._load_all_target_symbols()
        
        # Debug flag for first chunk inspection
        self._first_chunk_debug = False
        
        # Initialize processing statistics
        self.stats = {
            'total_rows_processed': 0,
            'date_filtered_rows': 0,
            'symbol_filtered_rows': 0,
            'quality_filtered_rows': 0,
            'final_articles': 0,
            'sentiment_processed': 0,
            'processing_start_time': time.time(),
            'chunks_processed': 0
        }
        
        # Initialize FinBERT for sentiment analysis
        self._initialize_finbert()
        
        print("✅ Initialization completed!")
    
    def _load_all_target_symbols(self) -> Set[str]:
        """Load and combine all target symbols from config and core dataset"""
        target_symbols = set(self.config.target_symbols)
        
        # Also include core dataset symbols for broader coverage
        try:
            if os.path.exists(CORE_DATASET):
                core_sample = pd.read_csv(CORE_DATASET, usecols=['symbol'], nrows=10000)
                core_symbols = set(core_sample['symbol'].unique())
                print(f"📊 Core dataset symbols: {list(core_symbols)[:10]}...")
                target_symbols.update(core_symbols)
            else:
                print(f"⚠️ Core dataset not found, using config symbols only")
        except Exception as e:
            print(f"⚠️ Could not load core symbols: {e}")
        
        print(f"🎯 Total target symbols: {len(target_symbols)}")
        return target_symbols
    
    def _initialize_finbert(self):
        """Initialize FinBERT with optimizations for large-scale processing"""
        print("\n🤖 Initializing FinBERT for sentiment analysis...")
        
        try:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"   📱 Device: {self.device}")
            
            model_name = "ProsusAI/finbert"
            print(f"   📥 Loading {model_name}...")
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            
            # Optimize for inference
            if self.device.type == 'cuda':
                self.model = self.model.half()  # Use FP16 for memory efficiency
            
            # Label mapping for FinBERT
            self.label_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}
            
            print("   ✅ FinBERT loaded and optimized!")
            
        except Exception as e:
            print(f"   ❌ FinBERT initialization failed: {e}")
            raise
    
    def _apply_smart_filters(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """Apply intelligent filters to reduce dataset size while maintaining quality"""
        
        original_size = len(chunk)
        
        # 1. Date filtering (critical for pipeline alignment)
        if 'Date' in chunk.columns:
            try:
                # Handle various date formats in FNSPID
                chunk['Date'] = pd.to_datetime(chunk['Date'], errors='coerce', utc=True)
                
                # Filter to pipeline date range
                start_date = pd.to_datetime(self.config.start_date, utc=True)
                end_date = pd.to_datetime(self.config.end_date, utc=True)
                
                chunk = chunk[(chunk['Date'] >= start_date) & (chunk['Date'] <= end_date)]
                self.stats['date_filtered_rows'] += len(chunk)
                
                if len(chunk) > 0:
                    print(f"      📅 Date filter: {original_size:,} → {len(chunk):,} articles")
                else:
                    return pd.DataFrame()  # Return empty if no data in range
                    
            except Exception as e:
                print(f"      ⚠️ Date filtering failed: {e}")
                return pd.DataFrame()
        
        # 2. Symbol filtering (align with your target symbols)
        if 'Stock_symbol' in chunk.columns and len(chunk) > 0:
            # Use cached target symbols
            chunk = chunk[chunk['Stock_symbol'].isin(self.all_target_symbols)]
            self.stats['symbol_filtered_rows'] += len(chunk)
            
            if len(chunk) == 0:
                return pd.DataFrame()
            
            print(f"      🏢 Symbol filter: {original_size:,} → {len(chunk):,} articles")
            
            # Show which symbols were found
            found_symbols = set(chunk['Stock_symbol'].unique())
            if found_symbols:
                print(f"         Found symbols: {list(found_symbols)[:5]}{'...' if len(found_symbols) > 5 else ''}")
        
        # 3. Content quality filtering
        if 'Article' in chunk.columns:
            # First, remove articles with missing content
            chunk = chunk.dropna(subset=['Article', 'Article_title'])
            
            if len(chunk) == 0:
                return pd.DataFrame()
            
            # Ensure Article column is string type
            chunk['Article'] = chunk['Article'].astype(str)
            chunk['Article_title'] = chunk['Article_title'].astype(str)
            
            # Remove articles that are too short
            chunk = chunk[chunk['Article'].str.len() >= self.config.min_article_length]
            
            # Remove duplicate articles (common in financial news)
            chunk = chunk.drop_duplicates(subset=['Article'], keep='first')
            
            # Additional quality checks
            # Remove articles that are mostly whitespace
            chunk = chunk[chunk['Article'].str.strip().str.len() >= self.config.min_article_length]
            
            # Remove articles with generic/placeholder content
            placeholder_patterns = ['no content', 'content not available', 'please try again', 'error loading']
            for pattern in placeholder_patterns:
                chunk = chunk[~chunk['Article'].str.lower().str.contains(pattern, na=False)]
            
            self.stats['quality_filtered_rows'] += len(chunk)
            
            if len(chunk) == 0:
                return pd.DataFrame()
            
            print(f"      🔧 Quality filter: {original_size:,} → {len(chunk):,} articles")
        
        # 4. Smart sampling to manage volume while preserving diversity
        if len(chunk) > 0:
            # Group by symbol and date to ensure balanced sampling
            if 'Stock_symbol' in chunk.columns and 'Date' in chunk.columns:
                chunk['date_only'] = chunk['Date'].dt.date
                
                # Limit articles per symbol per day to prevent bias
                chunk = chunk.groupby(['Stock_symbol', 'date_only']).apply(
                    lambda x: x.sample(n=min(len(x), self.config.max_articles_per_symbol_per_day), 
                                     random_state=42)
                ).reset_index(drop=True)
                
                # Apply global sampling ratio
                if len(chunk) > 0:
                    sample_size = max(1, int(len(chunk) * self.config.sample_ratio))
                    chunk = chunk.sample(n=min(sample_size, len(chunk)), random_state=42)
        
        return chunk
    
    def process_fnspid_dataset(self) -> pd.DataFrame:
        """
        Main processing function: filter 22GB FNSPID dataset efficiently
        """
        print(f"\n📊 PROCESSING FNSPID DATASET - SMART FILTERING")
        print("=" * 60)
        
        try:
            filtered_articles = []
            chunk_iterator = pd.read_csv(FNSPID_NEWS_FILE, chunksize=self.config.chunk_size)
            
            print(f"📥 Processing dataset in chunks of {self.config.chunk_size:,}...")
            
            for chunk_idx, chunk in enumerate(tqdm(chunk_iterator, desc="Processing chunks")):
                self.stats['chunks_processed'] = chunk_idx + 1
                self.stats['total_rows_processed'] += len(chunk)
                
                # Apply smart filters
                filtered_chunk = self._apply_smart_filters(chunk)
                
                if not filtered_chunk.empty:
                    filtered_articles.append(filtered_chunk)
                    self.stats['final_articles'] += len(filtered_chunk)
                
                # Memory management
                if chunk_idx % self.config.enable_gc_frequency == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                # Progress reporting
                if chunk_idx % 20 == 0 and chunk_idx > 0:
                    current_total = sum(len(df) for df in filtered_articles)
                    print(f"   📊 Processed {chunk_idx} chunks, {current_total:,} articles filtered")
                
                # Early stopping for very large datasets (optional)
                if len(filtered_articles) > 0:
                    current_total = sum(len(df) for df in filtered_articles)
                    if current_total >= 100000:  # Reasonable limit for processing
                        print(f"   🎯 Reached processing limit: {current_total:,} articles")
                        break
            
            # Combine all filtered articles
            if filtered_articles:
                final_dataset = pd.concat(filtered_articles, ignore_index=True)
                
                # Final cleanup and sorting
                final_dataset = final_dataset.sort_values(['Stock_symbol', 'Date'])
                final_dataset = final_dataset.reset_index(drop=True)
                
                print(f"\n✅ FILTERING COMPLETED:")
                print(f"   📊 Total rows processed: {self.stats['total_rows_processed']:,}")
                print(f"   📅 Date-filtered: {self.stats['date_filtered_rows']:,}")
                print(f"   🏢 Symbol-filtered: {self.stats['symbol_filtered_rows']:,}")
                print(f"   🔧 Quality-filtered: {self.stats['quality_filtered_rows']:,}")
                print(f"   ✅ Final articles: {len(final_dataset):,}")
                
                # Show symbol distribution
                if 'Stock_symbol' in final_dataset.columns:
                    symbol_dist = final_dataset['Stock_symbol'].value_counts()
                    print(f"\n🏢 Symbol distribution:")
                    for symbol, count in symbol_dist.head(10).items():
                        print(f"      {symbol}: {count:,} articles")
                
                return final_dataset
            else:
                print("❌ No articles passed filtering criteria")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"❌ FNSPID processing failed: {e}")
            traceback.print_exc()
            return pd.DataFrame()
    
    def analyze_sentiment_batch(self, articles: List[str]) -> List[Dict]:
        """
        Efficient batch sentiment analysis with FinBERT
        """
        results = []
        batch_size = self.config.finbert_batch_size
        
        for i in range(0, len(articles), batch_size):
            batch_articles = articles[i:i+batch_size]
            batch_results = []
            
            for article in batch_articles:
                try:
                    # Combine title and content for better sentiment analysis
                    text = article[:512]  # Truncate for FinBERT input limits
                    
                    # Tokenize
                    inputs = self.tokenizer(
                        text,
                        return_tensors="pt",
                        truncation=True,
                        padding=True,
                        max_length=512
                    ).to(self.device)
                    
                    # Get predictions
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
                        probabilities = probabilities.cpu().numpy()[0]
                    
                    # Create result
                    result = {
                        'negative': float(probabilities[0]),
                        'neutral': float(probabilities[1]),
                        'positive': float(probabilities[2]),
                        'compound': float(probabilities[2] - probabilities[0]),  # Simple compound score
                        'label': self.label_mapping[np.argmax(probabilities)],
                        'confidence': float(np.max(probabilities))
                    }
                    
                    batch_results.append(result)
                    
                except Exception as e:
                    print(f"⚠️ Sentiment analysis failed for one article: {e}")
                    # Default neutral sentiment for failed cases
                    batch_results.append({
                        'negative': 0.33, 'neutral': 0.34, 'positive': 0.33,
                        'compound': 0.0, 'label': 'neutral', 'confidence': 0.34
                    })
            
            results.extend(batch_results)
            
            # Memory cleanup for large batches
            if i % (batch_size * 5) == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        return results
    
    def create_article_level_sentiment(self, filtered_articles: pd.DataFrame) -> pd.DataFrame:
        """
        Create article-level sentiment analysis results
        """
        print(f"\n🔍 ANALYZING SENTIMENT - ARTICLE LEVEL")
        print("=" * 50)
        
        if filtered_articles.empty:
            print("❌ No articles to analyze")
            return pd.DataFrame()
        
        try:
            # Prepare articles for sentiment analysis
            articles_for_analysis = []
            for _, row in filtered_articles.iterrows():
                title = str(row.get('Article_title', ''))
                content = str(row.get('Article', ''))
                combined = f"{title} {content}"
                articles_for_analysis.append(combined)
            
            print(f"📝 Analyzing {len(articles_for_analysis):,} articles...")
            
            # Batch sentiment analysis
            sentiment_results = self.analyze_sentiment_batch(articles_for_analysis)
            
            # Combine with original data
            article_sentiment_df = filtered_articles.copy()
            
            # Add sentiment columns
            for i, result in enumerate(sentiment_results):
                for key, value in result.items():
                    article_sentiment_df.loc[i, f'sentiment_{key}'] = value
            
            self.stats['sentiment_processed'] = len(sentiment_results)
            
            print(f"✅ Article-level sentiment analysis completed!")
            print(f"   📊 Articles processed: {len(sentiment_results):,}")
            
            # Show sentiment distribution
            if sentiment_results:
                sentiment_dist = pd.Series([r['label'] for r in sentiment_results]).value_counts()
                print(f"   🎭 Sentiment distribution:")
                for sentiment, count in sentiment_dist.items():
                    percentage = (count / len(sentiment_results)) * 100
                    print(f"      {sentiment}: {count:,} ({percentage:.1f}%)")
            
            return article_sentiment_df
            
        except Exception as e:
            print(f"❌ Article sentiment analysis failed: {e}")
            traceback.print_exc()
            return pd.DataFrame()
    
    def create_daily_aggregated_sentiment(self, article_sentiment_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create daily aggregated sentiment features for temporal_decay.py
        """
        print(f"\n📊 CREATING DAILY AGGREGATED SENTIMENT")
        print("=" * 50)
        
        if article_sentiment_df.empty:
            print("❌ No sentiment data to aggregate")
            return pd.DataFrame()
        
        try:
            # Prepare data for aggregation
            df = article_sentiment_df.copy()
            
            # Convert date to date-only for daily aggregation
            df['date'] = pd.to_datetime(df['Date']).dt.date
            
            # Aggregate sentiment by symbol and date
            daily_sentiment = df.groupby(['Stock_symbol', 'date']).agg({
                'sentiment_compound': ['mean', 'std', 'count', 'min', 'max'],
                'sentiment_positive': ['mean', 'std'],
                'sentiment_negative': ['mean', 'std'],
                'sentiment_neutral': 'mean',
                'sentiment_confidence': ['mean', 'min']
            }).round(6)
            
            # Flatten column names
            daily_sentiment.columns = ['_'.join(col).strip() for col in daily_sentiment.columns]
            daily_sentiment = daily_sentiment.reset_index()
            
            # Rename columns for pipeline compatibility
            column_mapping = {
                'Stock_symbol': 'symbol',
                'sentiment_compound_mean': 'sentiment_compound',
                'sentiment_compound_std': 'sentiment_volatility',
                'sentiment_compound_count': 'article_count',
                'sentiment_compound_min': 'sentiment_min',
                'sentiment_compound_max': 'sentiment_max',
                'sentiment_positive_mean': 'sentiment_positive',
                'sentiment_negative_mean': 'sentiment_negative',
                'sentiment_neutral_mean': 'sentiment_neutral',
                'sentiment_confidence_mean': 'sentiment_confidence'
            }
            
            daily_sentiment = daily_sentiment.rename(columns=column_mapping)
            
            # Add derived features for temporal decay processing
            daily_sentiment['sentiment_intensity'] = np.abs(daily_sentiment['sentiment_compound'])
            daily_sentiment['sentiment_score'] = daily_sentiment['sentiment_compound']  # For temporal_decay.py
            daily_sentiment['confidence'] = daily_sentiment['sentiment_confidence']  # For temporal_decay.py
            
            # Ensure date is string for compatibility
            daily_sentiment['date'] = daily_sentiment['date'].astype(str)
            
            print(f"✅ Daily aggregation completed!")
            print(f"   📊 Daily records: {len(daily_sentiment):,}")
            print(f"   🏢 Symbols: {daily_sentiment['symbol'].nunique()}")
            print(f"   📅 Date range: {daily_sentiment['date'].min()} to {daily_sentiment['date'].max()}")
            
            return daily_sentiment
            
        except Exception as e:
            print(f"❌ Daily aggregation failed: {e}")
            traceback.print_exc()
            return pd.DataFrame()
    
    def save_pipeline_outputs(self, 
                            filtered_articles: pd.DataFrame,
                            article_sentiment_df: pd.DataFrame, 
                            daily_sentiment_df: pd.DataFrame):
        """
        Save outputs in formats ready for your pipeline
        """
        print(f"\n💾 SAVING PIPELINE OUTPUTS")
        print("=" * 40)
        
        try:
            # Ensure output directory exists
            os.makedirs(DATA_DIR, exist_ok=True)
            
            # 1. Save filtered articles (for reference)
            if not filtered_articles.empty:
                filtered_articles.to_csv(FILTERED_FNSPID_OUTPUT, index=False)
                print(f"✅ Filtered articles: {FILTERED_FNSPID_OUTPUT}")
            
            # 2. Save article-level sentiment (for detailed analysis)
            if not article_sentiment_df.empty:
                article_sentiment_df.to_csv(ARTICLE_SENTIMENT_OUTPUT, index=False)
                print(f"✅ Article sentiments: {ARTICLE_SENTIMENT_OUTPUT}")
            
            # 3. Save daily sentiment (for temporal_decay.py)
            if not daily_sentiment_df.empty:
                daily_sentiment_df.to_csv(DAILY_SENTIMENT_OUTPUT, index=False)
                print(f"✅ Daily sentiment: {DAILY_SENTIMENT_OUTPUT}")
                
                # This is the key file for your pipeline
                print(f"   🎯 Ready for temporal_decay.py integration!")
            
            # 4. Save processing statistics
            self.stats['processing_time'] = time.time() - self.stats['processing_start_time']
            stats_file = f"{DATA_DIR}/fnspid_processing_stats.json"
            with open(stats_file, 'w') as f:
                json.dump(self.stats, f, indent=2, default=str)
            print(f"✅ Processing stats: {stats_file}")
            
        except Exception as e:
            print(f"❌ Save failed: {e}")
            traceback.print_exc()
    
    def run_complete_pipeline(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Run the complete FNSPID processing pipeline
        """
        print(f"\n🚀 STARTING COMPLETE FNSPID PIPELINE")
        print("=" * 80)
        
        try:
            # Step 1: Filter FNSPID dataset
            filtered_articles = self.process_fnspid_dataset()
            
            if filtered_articles.empty:
                print("❌ No articles after filtering - check configuration")
                return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
            
            # Step 2: Analyze sentiment at article level
            article_sentiment_df = self.create_article_level_sentiment(filtered_articles)
            
            # Step 3: Create daily aggregated sentiment
            daily_sentiment_df = self.create_daily_aggregated_sentiment(article_sentiment_df)
            
            # Step 4: Save outputs for pipeline integration
            self.save_pipeline_outputs(filtered_articles, article_sentiment_df, daily_sentiment_df)
            
            # Step 5: Generate final report
            self._generate_final_report()
            
            print(f"\n🎉 FNSPID PIPELINE COMPLETED SUCCESSFULLY!")
            return filtered_articles, article_sentiment_df, daily_sentiment_df
            
        except Exception as e:
            print(f"\n❌ FNSPID PIPELINE FAILED: {e}")
            traceback.print_exc()
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    def _generate_final_report(self):
        """Generate comprehensive processing report"""
        print(f"\n📋 FINAL PROCESSING REPORT")
        print("=" * 60)
        
        processing_time = self.stats['processing_time']
        
        print(f"⏱️  Processing Time: {processing_time:.1f} seconds ({processing_time/60:.1f} minutes)")
        print(f"📊 Processing Statistics:")
        print(f"   📄 Total rows processed: {self.stats['total_rows_processed']:,}")
        print(f"   📅 Date-filtered rows: {self.stats['date_filtered_rows']:,}")
        print(f"   🏢 Symbol-filtered rows: {self.stats['symbol_filtered_rows']:,}")
        print(f"   🔧 Quality-filtered rows: {self.stats['quality_filtered_rows']:,}")
        print(f"   ✅ Final articles: {self.stats['final_articles']:,}")
        print(f"   🔬 Sentiment analyzed: {self.stats['sentiment_processed']:,}")
        print(f"   📦 Chunks processed: {self.stats['chunks_processed']:,}")
        
        # Calculate efficiency metrics
        if self.stats['total_rows_processed'] > 0:
            filter_efficiency = (self.stats['final_articles'] / self.stats['total_rows_processed']) * 100
            print(f"   📈 Filter efficiency: {filter_efficiency:.2f}%")
        
        print(f"\n📁 Output Files:")
        print(f"   📄 Filtered articles: {FILTERED_FNSPID_OUTPUT}")
        print(f"   📊 Article sentiments: {ARTICLE_SENTIMENT_OUTPUT}")
        print(f"   📈 Daily sentiment: {DAILY_SENTIMENT_OUTPUT}")
        
        print(f"\n🎯 Pipeline Integration:")
        print("1. ✅ FNSPID dataset filtered and processed")
        print("2. ✅ Sentiment analysis completed with FinBERT")
        print("3. ✅ Daily aggregation ready for temporal_decay.py")
        print("4. 🔄 Next: Run temporal_decay.py for decay processing")
        print("5. 🤖 Then: Train TFT models with enhanced features")


def create_pipeline_config_from_core() -> PipelineConfig:
    """
    Create pipeline configuration by reading from your core dataset
    """
    try:
        if os.path.exists(CORE_DATASET):
            # Read sample of core dataset to get symbols
            core_sample = pd.read_csv(CORE_DATASET, nrows=1000)
            target_symbols = core_sample['symbol'].unique().tolist()
            
            # Get date range
            date_sample = pd.read_csv(CORE_DATASET, usecols=['date'], nrows=10000)
            date_sample['date'] = pd.to_datetime(date_sample['date'])
            start_date = date_sample['date'].min().strftime('%Y-%m-%d')
            end_date = date_sample['date'].max().strftime('%Y-%m-%d')
            
            print(f"📊 Loaded configuration from core dataset:")
            print(f"   🏢 Symbols: {len(target_symbols)} ({', '.join(target_symbols[:5])}...)")
            print(f"   📅 Date range: {start_date} to {end_date}")
            
            return PipelineConfig(
                target_symbols=target_symbols,
                start_date=start_date,
                end_date=end_date
            )
        else:
            print(f"⚠️ Core dataset not found, using default configuration")
            return PipelineConfig(
                target_symbols=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'JPM'],
                start_date="2018-01-01",
                end_date="2024-01-31"
            )
    except Exception as e:
        print(f"⚠️ Could not load core dataset config: {e}")
        return PipelineConfig(
            target_symbols=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'JPM'],
            start_date="2018-01-01", 
            end_date="2024-01-31"
        )


def main():
    """
    Main execution with pipeline integration
    """
    print("🚀 OPTIMIZED FNSPID PROCESSOR - PIPELINE INTEGRATION")
    print("=" * 70)
    print("✅ Designed for your 22GB FNSPID dataset")
    print("📊 Optimized for 2018-2024 time period")
    print("🎯 Targets symbols from your core dataset")
    print("=" * 70)
    
    # Check dependencies
    try:
        import pytz
    except ImportError:
        print("❌ pytz library required for timezone handling")
        print("💡 Install with: pip install pytz")
        return
    
    # Configuration options
    print("\n⚙️ Processing Configuration:")
    print("1. Auto-detect from core dataset (recommended)")
    print("2. Quick processing (5% sample, 3 symbols)")
    print("3. Moderate processing (15% sample, 7 symbols)")
    print("4. Comprehensive processing (30% sample, all symbols)")
    print("5. Custom configuration")
    
    choice = input("\nSelect option (1-5): ").strip()
    
    if choice == "1" or choice == "":
        print("🔍 Auto-detecting configuration from core dataset...")
        config = create_pipeline_config_from_core()
        
    elif choice == "2":
        config = PipelineConfig(
            target_symbols=['AAPL', 'MSFT', 'GOOGL'],
            sample_ratio=0.05,
            chunk_size=25000,
            max_articles_per_symbol_per_day=20
        )
        print("⚡ Quick processing configuration selected")
        
    elif choice == "3":
        config = PipelineConfig(
            target_symbols=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'JPM'],
            sample_ratio=0.15,
            chunk_size=50000,
            max_articles_per_symbol_per_day=30
        )
        print("🚀 Moderate processing configuration selected")
        
    elif choice == "4":
        config = PipelineConfig(
            target_symbols=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'JPM', 'JNJ', 'PG', 'KO'],
            sample_ratio=0.30,
            chunk_size=75000,
            max_articles_per_symbol_per_day=50
        )
        print("🔬 Comprehensive processing configuration selected")
        
    elif choice == "5":
        # Custom configuration
        try:
            symbols_input = input("Target symbols (comma-separated): ") or "AAPL,MSFT,GOOGL"
            symbols = [s.strip().upper() for s in symbols_input.split(',')]
            
            sample_ratio = float(input("Sample ratio (0.05-0.5, default 0.15): ") or "0.15")
            sample_ratio = max(0.05, min(0.5, sample_ratio))
            
            start_date = input("Start date (YYYY-MM-DD, default 2018-01-01): ") or "2018-01-01"
            end_date = input("End date (YYYY-MM-DD, default 2024-01-31): ") or "2024-01-31"
            
            config = PipelineConfig(
                target_symbols=symbols,
                sample_ratio=sample_ratio,
                start_date=start_date,
                end_date=end_date
            )
            print("🛠️ Custom configuration created")
            
        except Exception as e:
            print(f"❌ Configuration error: {e}")
            return
    else:
        print("❌ Invalid choice, using auto-detection")
        config = create_pipeline_config_from_core()
    
    # Show processing plan
    print(f"\n📋 Processing Plan:")
    print(f"   🏢 Target symbols: {len(config.target_symbols)} symbols")
    print(f"   📅 Date range: {config.start_date} to {config.end_date}")
    print(f"   📦 Sample ratio: {config.sample_ratio:.1%}")
    print(f"   🔧 Chunk size: {config.chunk_size:,}")
    print(f"   📊 Max articles/symbol/day: {config.max_articles_per_symbol_per_day}")
    
    # Estimate processing time
    if os.path.exists(FNSPID_NEWS_FILE):
        file_size_gb = os.path.getsize(FNSPID_NEWS_FILE) / (1024**3)
        estimated_time = (file_size_gb * config.sample_ratio * 2)  # Rough estimate in minutes
        print(f"   ⏱️ Estimated time: {estimated_time:.0f}-{estimated_time*2:.0f} minutes")
    
    # Confirm execution
    print(f"\n🚀 Ready to process FNSPID dataset for your pipeline?")
    print("This will create sentiment data ready for temporal_decay.py")
    confirm = input("Proceed? (Y/n): ").strip().lower()
    
    if confirm in ['y', 'yes', '']:
        try:
            # Initialize processor
            processor = OptimizedFNSPIDProcessor(config)
            
            # Run complete pipeline
            filtered_articles, article_sentiment, daily_sentiment = processor.run_complete_pipeline()
            
            if not daily_sentiment.empty:
                print(f"\n🎉 SUCCESS! FNSPID processing completed for your pipeline")
                print(f"📊 Generated {len(daily_sentiment):,} daily sentiment records")
                print(f"🏢 Covering {daily_sentiment['symbol'].nunique()} symbols")
                print(f"📅 Date range: {daily_sentiment['date'].min()} to {daily_sentiment['date'].max()}")
                
                # Show sample results
                print(f"\n📋 Sample Daily Sentiment Results:")
                print(daily_sentiment.head())
                
                print(f"\n🎯 NEXT STEPS:")
                print("1. ✅ FNSPID sentiment data ready")
                print("2. ⏰ Run temporal_decay.py to calculate decay weights")
                print("3. 🔗 Integrate with core dataset using sentiment.py")
                print("4. 🤖 Train enhanced TFT models with temporal features")
                
                # Validate output for temporal_decay.py
                required_cols = ['symbol', 'date', 'sentiment_score', 'confidence', 'article_count']
                missing_cols = [col for col in required_cols if col not in daily_sentiment.columns]
                if missing_cols:
                    print(f"⚠️ Missing columns for temporal_decay.py: {missing_cols}")
                else:
                    print("✅ Output format validated for temporal_decay.py")
                
            else:
                print(f"\n⚠️ Processing completed but no sentiment data generated")
                print("💡 Try adjusting configuration or checking input data")
        
        except Exception as e:
            print(f"\n❌ Processing failed: {e}")
            traceback.print_exc()
    else:
        print("❌ Processing cancelled")


if __name__ == "__main__":
    main()