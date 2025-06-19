#!/usr/bin/env python3
import sys
import os
from pathlib import Path

# Add src directory to Python path so we can import config_reader
script_dir = Path(__file__).parent
if 'src' in str(script_dir):
    # Running from src directory
    sys.path.insert(0, str(script_dir))
else:
    # Running from project root
    sys.path.insert(0, str(script_dir / 'src'))


"""
FNSPID Processor - Fixed Simple Version
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from config_reader import load_config, get_data_paths

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FinBERT setup
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    FINBERT_AVAILABLE = True
except ImportError:
    FINBERT_AVAILABLE = False
    logger.warning("⚠️ FinBERT not available. Install with: pip install transformers torch")

class FNSPIDProcessor:
    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        self.data_paths = get_data_paths(self.config)
        self.symbols = self.config['data']['core']['symbols']
        self.start_date = self.config['data']['core']['start_date']
        self.end_date = self.config['data']['core']['end_date']
        self.setup_finbert()
        
    def setup_finbert(self):
        if not FINBERT_AVAILABLE:
            self.finbert_available = False
            return
        try:
            model_name = "ProsusAI/finbert"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
            self.model.eval()
            self.finbert_available = True
            logger.info(f"✅ FinBERT loaded on {self.device}")
        except Exception as e:
            logger.error(f"❌ FinBERT setup failed: {e}")
            self.finbert_available = False
    
    def load_and_filter_fnspid(self) -> pd.DataFrame:
        logger.info("📥 Loading FNSPID data...")
        fnspid_path = self.data_paths['raw_fnspid']
        if not fnspid_path.exists():
            raise FileNotFoundError(f"FNSPID file not found: {fnspid_path}")
        
        # Read sample to detect columns
        sample = pd.read_csv(fnspid_path, nrows=10)
        logger.info(f"📋 FNSPID columns: {list(sample.columns)}")
        
        column_mapping = {'Date': 'date', 'Article_title': 'headline', 'Stock_symbol': 'symbol'}
        missing_cols = [col for col in column_mapping.keys() if col not in sample.columns]
        if missing_cols:
            raise ValueError(f"Missing expected FNSPID columns: {missing_cols}")
        
        chunk_size = self.config['data']['fnspid']['production']['chunk_size']
        sample_ratio = self.config['data']['fnspid']['production']['sample_ratio']
        
        filtered_chunks = []
        total_processed = 0
        
        for chunk in pd.read_csv(fnspid_path, chunksize=chunk_size):
            chunk = chunk.rename(columns=column_mapping)
            chunk_filtered = chunk[
                (chunk['symbol'].isin(self.symbols)) &
                (pd.to_datetime(chunk['date']) >= self.start_date) &
                (pd.to_datetime(chunk['date']) <= self.end_date)
            ].copy()
            
            if sample_ratio < 1.0 and len(chunk_filtered) > 0:
                sample_size = max(1, int(len(chunk_filtered) * sample_ratio))
                chunk_filtered = chunk_filtered.sample(n=sample_size, random_state=42)
            
            if len(chunk_filtered) > 0:
                filtered_chunks.append(chunk_filtered)
            
            total_processed += len(chunk)
            if total_processed % 100000 == 0:
                logger.info(f"   📊 Processed {total_processed:,} rows")
        
        if not filtered_chunks:
            raise ValueError("No articles found matching criteria")
        
        articles_df = pd.concat(filtered_chunks, ignore_index=True)
        articles_df = articles_df.drop_duplicates(subset=['date', 'symbol', 'headline'])
        articles_df = articles_df.sort_values(['symbol', 'date']).reset_index(drop=True)
        
        logger.info(f"✅ FNSPID filtering completed: {len(articles_df):,} articles")
        return articles_df
    
    def analyze_sentiment(self, articles_df: pd.DataFrame) -> pd.DataFrame:
        if not self.finbert_available:
            logger.error("❌ FinBERT not available for sentiment analysis")
            raise RuntimeError("FinBERT not available")
        
        logger.info(f"🧠 Analyzing sentiment for {len(articles_df):,} articles...")
        
        sentiment_results = []
        batch_size = 16
        
        for i in range(0, len(articles_df), batch_size):
            batch = articles_df.iloc[i:i+batch_size]
            headlines = batch['headline'].tolist()
            
            inputs = self.tokenizer(headlines, return_tensors="pt", truncation=True, padding=True, max_length=512).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()
            
            for j, pred in enumerate(predictions):
                negative, neutral, positive = pred
                result = batch.iloc[j].copy()
                result['sentiment_compound'] = float(positive - negative)
                result['sentiment_positive'] = float(positive)
                result['sentiment_neutral'] = float(neutral)
                result['sentiment_negative'] = float(negative)
                result['confidence'] = float(np.max(pred))
                sentiment_results.append(result)
            
            if (i + batch_size) % 1000 == 0:
                progress = (i + batch_size) / len(articles_df) * 100
                logger.info(f"   🧠 Progress: {progress:.1f}%")
        
        sentiment_df = pd.DataFrame(sentiment_results)
        logger.info(f"✅ Sentiment analysis completed")
        return sentiment_df
    
    def aggregate_daily_sentiment(self, sentiment_df: pd.DataFrame) -> pd.DataFrame:
        logger.info("📊 Aggregating daily sentiment...")
        
        daily_sentiment = sentiment_df.groupby(['symbol', 'date']).agg({
            'sentiment_compound': lambda x: np.average(x, weights=sentiment_df.loc[x.index, 'confidence']),
            'sentiment_positive': 'mean',
            'sentiment_neutral': 'mean', 
            'sentiment_negative': 'mean',
            'confidence': 'mean',
            'headline': 'count'
        }).rename(columns={'headline': 'article_count'}).reset_index()
        
        logger.info(f"✅ Daily aggregation completed: {len(daily_sentiment):,} daily records")
        return daily_sentiment
    
    def run_complete_pipeline(self) -> pd.DataFrame:
        logger.info("🚀 Starting FNSPID processing pipeline")
        articles_df = self.load_and_filter_fnspid()
        sentiment_df = self.analyze_sentiment(articles_df)
        daily_sentiment = self.aggregate_daily_sentiment(sentiment_df)
        
        output_path = self.data_paths['fnspid_daily_sentiment']
        output_path.parent.mkdir(parents=True, exist_ok=True)
        daily_sentiment.to_csv(output_path, index=False)
        
        logger.info(f"💾 Daily sentiment saved: {output_path}")
        logger.info("✅ FNSPID processing completed!")
        return daily_sentiment

def main():
    try:
        processor = FNSPIDProcessor()
        daily_sentiment = processor.run_complete_pipeline()
        print(f"\n🎉 FNSPID Processing Completed Successfully!")
        print(f"📊 Daily sentiment records: {len(daily_sentiment):,}")
    except Exception as e:
        logger.error(f"❌ FNSPID processing failed: {e}")
        raise

if __name__ == "__main__":
    main()
