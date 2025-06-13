#!/usr/bin/env python3
"""
TIMEZONE-FIXED FNSPID PROCESSOR
===============================

✅ TIMEZONE FIX APPLIED:
- Handles UTC datetime columns properly
- Timezone-aware date filtering  
- Robust date parsing and comparison
- All other functionality preserved
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
from typing import Dict, List, Tuple, Optional, Any
import warnings
import json
import re
import gc
import time
from tqdm import tqdm
import pytz

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

# Paths
DATA_DIR = "data/processed"
FNSPID_NEWS_FILE = "data/raw/nasdaq_exteral_data.csv"
SENTIMENT_OUTPUT_FILE = f"{DATA_DIR}/fnspid_sentiment_dataset.csv"
ARTICLE_SENTIMENT_FILE = f"{DATA_DIR}/fnspid_article_sentiments.csv"

class TimezoneSafeFNSPIDProcessor:
    """FNSPID processor with timezone-safe date handling"""
    
    def __init__(self, sample_ratio=0.05, target_symbols=None, max_articles=1000):
        """Initialize with timezone-safe processing"""
        
        print("🚀 INITIALIZING TIMEZONE-SAFE FNSPID PROCESSOR")
        print("=" * 60)
        
        self.sample_ratio = sample_ratio
        self.target_symbols = target_symbols or ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
        self.max_articles = max_articles
        
        # Processing statistics
        self.stats = {
            'total_rows': 0,
            'sampled_rows': 0,
            'matched_articles': 0,
            'processed_articles': 0,
            'start_time': time.time()
        }
        
        print(f"⚙️ Configuration:")
        print(f"   📊 Sample ratio: {sample_ratio:.1%}")
        print(f"   🏢 Target symbols: {self.target_symbols}")
        print(f"   📰 Max articles per symbol: {max_articles}")
        print(f"   📁 Dataset file: {FNSPID_NEWS_FILE}")
        
        # Check file existence
        if not os.path.exists(FNSPID_NEWS_FILE):
            raise FileNotFoundError(f"❌ Dataset file not found: {FNSPID_NEWS_FILE}")
        
        file_size_gb = os.path.getsize(FNSPID_NEWS_FILE) / (1024**3)
        print(f"   📊 File size: {file_size_gb:.1f} GB")
        
        # Initialize FinBERT
        self._initialize_finbert()
        
        print("✅ Initialization completed!")
    
    def _initialize_finbert(self):
        """Initialize FinBERT with error handling"""
        print("\n🤖 Initializing FinBERT...")
        
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
                self.model = self.model.half()
            
            self.label_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}
            print("   ✅ FinBERT loaded successfully!")
            
        except Exception as e:
            print(f"   ❌ FinBERT initialization failed: {e}")
            raise
    
    def _safe_date_parse_and_filter(self, chunk):
        """Safely parse dates and apply timezone-aware filtering"""
        
        try:
            # Parse dates with UTC awareness
            chunk['Date'] = pd.to_datetime(chunk['Date'], errors='coerce', utc=True)
            
            # Remove rows with invalid dates
            chunk = chunk.dropna(subset=['Date'])
            
            # Create timezone-aware date boundaries
            utc = pytz.UTC
            start_date = utc.localize(datetime(2020, 1, 1))
            end_date = utc.localize(datetime(2024, 1, 31))
            
            # Apply date filter with timezone-aware comparison
            before_filter = len(chunk)
            chunk = chunk[(chunk['Date'] >= start_date) & (chunk['Date'] <= end_date)]
            after_filter = len(chunk)
            
            if before_filter > 0:
                filtered_pct = (before_filter - after_filter) / before_filter * 100
                if filtered_pct > 0:
                    print(f"      📅 Date filter: {before_filter:,} → {after_filter:,} ({filtered_pct:.1f}% filtered)")
            
            return chunk
            
        except Exception as e:
            print(f"      ⚠️ Date filtering failed: {e}")
            # Return chunk without date filtering if it fails
            return chunk
    
    def analyze_dataset_structure(self):
        """Analyze the dataset structure with timezone-safe handling"""
        print("\n🔍 ANALYZING DATASET STRUCTURE (TIMEZONE-SAFE)")
        print("=" * 50)
        
        try:
            # Read header and small sample
            print("📥 Reading dataset header...")
            header_df = pd.read_csv(FNSPID_NEWS_FILE, nrows=0)
            
            print("📥 Reading sample data...")
            sample_df = pd.read_csv(FNSPID_NEWS_FILE, nrows=100)
            
            print(f"✅ Sample loaded: {sample_df.shape}")
            print(f"📋 Columns: {list(sample_df.columns)}")
            
            # Analyze date column with timezone safety
            if 'Date' in sample_df.columns:
                print(f"\n📅 Date Column Analysis:")
                
                # Show raw date samples
                print(f"   📊 Raw date samples:")
                for i, date_val in enumerate(sample_df['Date'].head(5)):
                    print(f"      {i+1}. {date_val}")
                
                # Parse dates safely
                try:
                    sample_df['Date'] = pd.to_datetime(sample_df['Date'], errors='coerce', utc=True)
                    valid_dates = sample_df['Date'].dropna()
                    
                    if len(valid_dates) > 0:
                        print(f"   ✅ Parsed successfully: {len(valid_dates)}/{len(sample_df)} valid dates")
                        print(f"   📅 Date range: {valid_dates.min()} to {valid_dates.max()}")
                        print(f"   🌍 Timezone: {valid_dates.iloc[0].tz if hasattr(valid_dates.iloc[0], 'tz') else 'None'}")
                    else:
                        print(f"   ❌ No valid dates could be parsed")
                        
                except Exception as e:
                    print(f"   ❌ Date parsing failed: {e}")
            
            # Analyze other columns
            print(f"\n📊 Column Analysis:")
            for col in sample_df.columns:
                if col != 'Date':  # Skip date column (already analyzed)
                    non_null = sample_df[col].notna().sum()
                    unique_vals = sample_df[col].nunique()
                    
                    if sample_df[col].dtype == 'object' and non_null > 0:
                        avg_length = sample_df[col].astype(str).str.len().mean()
                        print(f"   📝 {col}: {non_null}/100 non-null, {unique_vals} unique, avg length: {avg_length:.1f}")
                    else:
                        print(f"   📊 {col}: {non_null}/100 non-null, {unique_vals} unique")
            
            # Show sample data (excluding complex date column for clarity)
            print(f"\n📋 Sample Data:")
            display_cols = ['Stock_symbol', 'Article_title']
            if all(col in sample_df.columns for col in display_cols):
                print(sample_df[display_cols].head(3))
            
            # Analyze stock symbols
            if 'Stock_symbol' in sample_df.columns:
                symbol_counts = sample_df['Stock_symbol'].value_counts()
                print(f"\n🏢 Stock Symbols in Sample:")
                print(symbol_counts.head(10))
            
            return sample_df
            
        except Exception as e:
            print(f"❌ Dataset structure analysis failed: {e}")
            traceback.print_exc()
            raise
    
    def create_smart_sample(self):
        """Create a smart sample with timezone-safe date filtering"""
        print(f"\n📊 CREATING SMART SAMPLE (TIMEZONE-SAFE)")
        print("=" * 50)
        
        try:
            chunks_processed = 0
            sampled_data = []
            
            print(f"📥 Processing dataset in chunks with timezone-safe filtering...")
            
            chunk_size = 10000
            max_chunks = int(1.0 / self.sample_ratio)  # Limit based on sample ratio
            
            chunk_iterator = pd.read_csv(FNSPID_NEWS_FILE, chunksize=chunk_size)
            
            for chunk in tqdm(chunk_iterator, desc="Processing chunks", total=max_chunks):
                chunks_processed += 1
                self.stats['total_rows'] += len(chunk)
                
                # Apply timezone-safe date filtering
                if 'Date' in chunk.columns:
                    chunk = self._safe_date_parse_and_filter(chunk)
                
                # Filter by target symbols
                if 'Stock_symbol' in chunk.columns and len(chunk) > 0:
                    chunk = chunk[chunk['Stock_symbol'].isin(self.target_symbols)]
                
                # Filter by content quality
                if len(chunk) > 0:
                    if 'Article' in chunk.columns:
                        chunk = chunk[chunk['Article'].notna()]
                        chunk = chunk[chunk['Article'].str.len() > 100]
                    
                    if 'Article_title' in chunk.columns:
                        chunk = chunk[chunk['Article_title'].notna()]
                        chunk = chunk[chunk['Article_title'].str.len() > 10]
                
                # Sample from filtered chunk
                if len(chunk) > 0:
                    sample_size = max(1, int(len(chunk) * self.sample_ratio))
                    chunk_sample = chunk.sample(n=min(sample_size, len(chunk)), random_state=42)
                    sampled_data.append(chunk_sample)
                    self.stats['sampled_rows'] += len(chunk_sample)
                
                # Progress updates
                if chunks_processed % 50 == 0:
                    gc.collect()
                    current_sample_size = sum(len(df) for df in sampled_data)
                    print(f"   📊 Processed {chunks_processed} chunks, sampled {current_sample_size:,} articles")
                
                # Stop conditions
                current_sample_size = sum(len(df) for df in sampled_data)
                if current_sample_size >= 50000 or chunks_processed >= max_chunks:
                    print(f"   🎯 Stopping: {current_sample_size:,} articles sampled from {chunks_processed} chunks")
                    break
            
            # Combine samples
            if sampled_data:
                final_sample = pd.concat(sampled_data, ignore_index=True)
                
                # Apply per-symbol limits
                limited_sample = []
                for symbol in self.target_symbols:
                    if 'Stock_symbol' in final_sample.columns:
                        symbol_data = final_sample[final_sample['Stock_symbol'] == symbol]
                        if len(symbol_data) > self.max_articles:
                            symbol_data = symbol_data.sample(n=self.max_articles, random_state=42)
                        if len(symbol_data) > 0:
                            limited_sample.append(symbol_data)
                
                final_sample = pd.concat(limited_sample, ignore_index=True) if limited_sample else pd.DataFrame()
                
                print(f"✅ Smart sample created:")
                print(f"   📊 Total sampled: {len(final_sample):,} articles")
                
                if 'Stock_symbol' in final_sample.columns and len(final_sample) > 0:
                    print(f"   🏢 Symbol distribution:")
                    symbol_dist = final_sample['Stock_symbol'].value_counts()
                    for symbol, count in symbol_dist.items():
                        print(f"      {symbol}: {count:,} articles")
                
                return final_sample
            else:
                raise ValueError("No data could be sampled - check your target symbols and date range")
                
        except Exception as e:
            print(f"❌ Sample creation failed: {e}")
            traceback.print_exc()
            raise
    
    def analyze_sentiment(self, sample_df):
        """Analyze sentiment with FinBERT"""
        print(f"\n🔍 ANALYZING SENTIMENT WITH FINBERT")
        print("=" * 40)
        
        try:
            if len(sample_df) == 0:
                print("❌ No data to analyze sentiment for")
                return sample_df
            
            results = []
            batch_size = 8  # Small batches for memory efficiency
            
            # Prepare texts
            texts = []
            for _, row in sample_df.iterrows():
                title = str(row.get('Article_title', ''))
                content = str(row.get('Article', ''))
                combined_text = f"{title} {content}"[:512]  # Truncate for FinBERT
                texts.append(combined_text)
            
            print(f"📝 Analyzing {len(texts):,} texts in batches of {batch_size}...")
            
            # Process in batches
            for i in tqdm(range(0, len(texts), batch_size), desc="FinBERT Analysis"):
                batch_texts = texts[i:i+batch_size]
                batch_results = []
                
                for text in batch_texts:
                    try:
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
                            'compound': float(probabilities[2] - probabilities[0]),
                            'label': self.label_mapping[np.argmax(probabilities)],
                            'confidence': float(np.max(probabilities))
                        }
                        
                        batch_results.append(result)
                        
                    except Exception as e:
                        print(f"⚠️ Sentiment analysis failed for one text: {e}")
                        batch_results.append({
                            'negative': 0.33, 'neutral': 0.34, 'positive': 0.33,
                            'compound': 0.0, 'label': 'neutral', 'confidence': 0.34
                        })
                
                results.extend(batch_results)
                
                # Memory cleanup
                if i % (batch_size * 10) == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            # Add results to dataframe
            for i, result in enumerate(results):
                for key, value in result.items():
                    sample_df.loc[i, key] = value
            
            self.stats['processed_articles'] = len(results)
            
            print(f"✅ Sentiment analysis completed:")
            print(f"   📊 Articles processed: {len(results):,}")
            
            # Show sentiment distribution
            if len(results) > 0:
                sentiment_dist = pd.Series([r['label'] for r in results]).value_counts()
                print(f"   🎭 Sentiment distribution:")
                for sentiment, count in sentiment_dist.items():
                    percentage = (count / len(results)) * 100
                    print(f"      {sentiment}: {count:,} ({percentage:.1f}%)")
            
            return sample_df
            
        except Exception as e:
            print(f"❌ Sentiment analysis failed: {e}")
            traceback.print_exc()
            raise
    
    def create_aggregated_features(self, sentiment_df):
        """Create daily aggregated sentiment features with timezone handling"""
        print(f"\n📊 CREATING AGGREGATED FEATURES")
        print("=" * 40)
        
        try:
            if len(sentiment_df) == 0:
                print("❌ No sentiment data to aggregate")
                return pd.DataFrame()
            
            # Convert date column to date only (remove timezone and time)
            if 'Date' in sentiment_df.columns:
                sentiment_df['date'] = pd.to_datetime(sentiment_df['Date']).dt.date
            else:
                print("❌ No Date column found for aggregation")
                return pd.DataFrame()
            
            # Aggregate by symbol and date
            aggregated = sentiment_df.groupby(['Stock_symbol', 'date']).agg({
                'compound': ['mean', 'std', 'min', 'max', 'count'],
                'positive': ['mean', 'std'],
                'negative': ['mean', 'std'],
                'neutral': 'mean',
                'confidence': ['mean', 'min']
            }).round(6)
            
            # Flatten columns
            aggregated.columns = ['_'.join(col).strip() for col in aggregated.columns]
            aggregated = aggregated.reset_index()
            
            # Rename for clarity
            aggregated = aggregated.rename(columns={
                'Stock_symbol': 'symbol',
                'compound_mean': 'sentiment_compound',
                'compound_std': 'sentiment_volatility',
                'compound_min': 'sentiment_min',
                'compound_max': 'sentiment_max',
                'compound_count': 'sentiment_count',
                'positive_mean': 'sentiment_positive',
                'negative_mean': 'sentiment_negative',
                'neutral_mean': 'sentiment_neutral',
                'confidence_mean': 'sentiment_confidence'
            })
            
            print(f"✅ Aggregated features created:")
            print(f"   📊 Daily records: {len(aggregated):,}")
            print(f"   🏢 Symbols: {aggregated['symbol'].nunique()}")
            print(f"   📅 Date range: {aggregated['date'].min()} to {aggregated['date'].max()}")
            
            return aggregated
            
        except Exception as e:
            print(f"❌ Feature aggregation failed: {e}")
            traceback.print_exc()
            raise
    
    def save_results(self, article_sentiments, aggregated_features):
        """Save results to files"""
        print(f"\n💾 SAVING RESULTS")
        print("=" * 40)
        
        try:
            # Ensure output directory exists
            os.makedirs(Path(DATA_DIR), exist_ok=True)
            
            # Save article-level sentiments
            if not article_sentiments.empty:
                article_sentiments.to_csv(ARTICLE_SENTIMENT_FILE, index=False)
                print(f"✅ Article sentiments saved: {ARTICLE_SENTIMENT_FILE}")
            
            # Save aggregated features
            if not aggregated_features.empty:
                aggregated_features.to_csv(SENTIMENT_OUTPUT_FILE, index=False)
                print(f"✅ Aggregated features saved: {SENTIMENT_OUTPUT_FILE}")
            
            # Save processing statistics
            self.stats['processing_time'] = time.time() - self.stats['start_time']
            stats_file = f"{DATA_DIR}/fnspid_processing_stats.json"
            with open(stats_file, 'w') as f:
                json.dump(self.stats, f, indent=2, default=str)
            print(f"✅ Statistics saved: {stats_file}")
            
        except Exception as e:
            print(f"❌ Save failed: {e}")
            traceback.print_exc()
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("\n🚀 STARTING TIMEZONE-SAFE FNSPID ANALYSIS")
        print("=" * 80)
        
        try:
            # Step 1: Analyze dataset structure
            sample_structure = self.analyze_dataset_structure()
            
            # Step 2: Create smart sample
            sample_df = self.create_smart_sample()
            
            if sample_df.empty:
                print("❌ No data sampled - check your configuration")
                return pd.DataFrame(), pd.DataFrame()
            
            # Step 3: Analyze sentiment
            sentiment_df = self.analyze_sentiment(sample_df)
            
            # Step 4: Create aggregated features
            aggregated_df = self.create_aggregated_features(sentiment_df)
            
            # Step 5: Save results
            if not aggregated_df.empty:
                self.save_results(sentiment_df, aggregated_df)
            
            # Step 6: Generate final report
            self.generate_final_report()
            
            print(f"\n🎉 ANALYSIS COMPLETED SUCCESSFULLY!")
            return sentiment_df, aggregated_df
            
        except Exception as e:
            print(f"\n❌ ANALYSIS FAILED: {e}")
            traceback.print_exc()
            return pd.DataFrame(), pd.DataFrame()
    
    def generate_final_report(self):
        """Generate final processing report"""
        print(f"\n📋 FINAL PROCESSING REPORT")
        print("=" * 60)
        
        processing_time = self.stats['processing_time']
        
        print(f"⏱️ Processing Time: {processing_time:.1f} seconds ({processing_time/60:.1f} minutes)")
        print(f"📊 Dataset Statistics:")
        print(f"   📄 Total rows processed: {self.stats['total_rows']:,}")
        print(f"   📊 Sampled rows: {self.stats['sampled_rows']:,}")
        print(f"   🔬 Sentiment analyzed: {self.stats['processed_articles']:,}")
        print(f"   📈 Sample efficiency: {(self.stats['sampled_rows']/max(self.stats['total_rows'], 1))*100:.2f}%")
        
        print(f"\n📁 Output Files:")
        print(f"   📄 Article sentiments: {ARTICLE_SENTIMENT_FILE}")
        print(f"   📊 Aggregated features: {SENTIMENT_OUTPUT_FILE}")
        
        print(f"\n🎯 Next Steps:")
        print("1. Review sentiment analysis results")
        print("2. Integrate with core dataset")
        print("3. Apply temporal decay processing")
        print("4. Train TFT models with sentiment features")

def main():
    """Main execution with user-friendly interface"""
    print("🚀 TIMEZONE-SAFE FNSPID PROCESSOR")
    print("=" * 50)
    print("✅ Fixed for UTC timezone handling")
    print("📊 Optimized for your 21.6 GB dataset")
    print("=" * 50)
    
    # Install pytz if needed
    try:
        import pytz
    except ImportError:
        print("❌ pytz library required for timezone handling")
        print("💡 Install with: pip install pytz")
        return
    
    # Configuration options
    print("\n⚙️ Processing Configuration:")
    print("1. Quick Test (1% sample, ~200MB processed)")
    print("2. Small Analysis (5% sample, ~1GB processed)")  
    print("3. Medium Analysis (10% sample, ~2GB processed)")
    print("4. Custom Configuration")
    
    choice = input("\nSelect option (1-4): ").strip()
    
    if choice == "1":
        processor = TimezoneSafeFNSPIDProcessor(
            sample_ratio=0.01,
            target_symbols=['AAPL', 'MSFT', 'GOOGL'],
            max_articles=200
        )
        print("⚡ Quick test configuration selected")
        
    elif choice == "2":
        processor = TimezoneSafeFNSPIDProcessor(
            sample_ratio=0.05,
            target_symbols=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA'],
            max_articles=500
        )
        print("🚀 Small analysis configuration selected")
        
    elif choice == "3":
        processor = TimezoneSafeFNSPIDProcessor(
            sample_ratio=0.10,
            target_symbols=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA'],
            max_articles=1000
        )
        print("🔬 Medium analysis configuration selected")
        
    elif choice == "4":
        # Custom configuration
        try:
            sample_ratio = float(input("Sample ratio (0.01-0.2, default 0.05): ") or "0.05")
            sample_ratio = max(0.001, min(0.2, sample_ratio))
            
            symbols_input = input("Symbols (comma-separated, default AAPL,MSFT,GOOGL): ") or "AAPL,MSFT,GOOGL"
            symbols = [s.strip().upper() for s in symbols_input.split(',')]
            
            max_articles = int(input("Max articles per symbol (default 1000): ") or "1000")
            
            processor = TimezoneSafeFNSPIDProcessor(
                sample_ratio=sample_ratio,
                target_symbols=symbols,
                max_articles=max_articles
            )
            print("🛠️ Custom configuration created")
            
        except Exception as e:
            print(f"❌ Configuration error: {e}")
            return
    else:
        print("❌ Invalid choice, using default")
        processor = TimezoneSafeFNSPIDProcessor()
    
    # Confirm execution
    print(f"\n🚀 Ready to process FNSPID dataset with timezone-safe handling?")
    confirm = input("Proceed? (y/N): ").strip().lower()
    
    if confirm in ['y', 'yes']:
        try:
            article_sentiments, aggregated_features = processor.run_complete_analysis()
            
            if not aggregated_features.empty:
                print(f"\n🎉 SUCCESS! FNSPID processing completed")
                print(f"📊 Generated {len(aggregated_features):,} daily sentiment records")
                print(f"🏢 Covering {aggregated_features['symbol'].nunique()} symbols")
                
                # Show sample results
                print(f"\n📋 Sample Results:")
                print(aggregated_features.head())
                
            else:
                print(f"\n⚠️ Processing completed but no results generated")
                print("💡 Try increasing sample_ratio or checking target symbols")
        
        except Exception as e:
            print(f"\n❌ Processing failed: {e}")
            traceback.print_exc()
    else:
        print("❌ Processing cancelled")

if __name__ == "__main__":
    main()