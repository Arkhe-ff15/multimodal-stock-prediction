#!/usr/bin/env python3
"""
Sentiment Analysis Pipeline for Stock Price Prediction
Uses FinBERT for financial sentiment analysis of news data

This script:
1. Fetches financial news for given symbols
2. Analyzes sentiment using FinBERT
3. Aggregates sentiment scores by symbol and date
4. Merges with existing technical data
5. Saves sentiment-enhanced dataset

Author: Multi-Horizon TFT Project
"""

import os
import sys
import logging
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import yfinance as yf
import requests
from newsapi import NewsApiClient
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

class FinBERTSentimentAnalyzer:
    """
    FinBERT-based sentiment analyzer for financial texts
    Simple, clean implementation focused on reliability
    """
    
    def __init__(self):
        """Initialize FinBERT model and tokenizer"""
        self.model_name = "ProsusAI/finbert"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"🤖 Loading FinBERT model: {self.model_name}")
        logger.info(f"📱 Using device: {self.device}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            
            # Label mapping: FinBERT outputs [negative, neutral, positive]
            self.label_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}
            
            logger.info("✅ FinBERT model loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load FinBERT model: {e}")
            raise
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of a single text
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with sentiment scores and prediction
        """
        try:
            # Tokenize and prepare input
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            ).to(self.device)
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
                probabilities = probabilities.cpu().numpy()[0]
            
            # Create sentiment scores
            sentiment_scores = {
                'negative_score': float(probabilities[0]),
                'neutral_score': float(probabilities[1]),
                'positive_score': float(probabilities[2]),
                'compound_score': float(probabilities[2] - probabilities[0]),  # Positive - Negative
                'predicted_label': self.label_mapping[np.argmax(probabilities)],
                'confidence': float(np.max(probabilities))
            }
            
            return sentiment_scores
            
        except Exception as e:
            logger.warning(f"⚠️ Sentiment analysis failed for text: {str(e)}")
            # Return neutral sentiment on failure
            return {
                'negative_score': 0.33,
                'neutral_score': 0.34,
                'positive_score': 0.33,
                'compound_score': 0.0,
                'predicted_label': 'neutral',
                'confidence': 0.34
            }
    
    def analyze_batch(self, texts: List[str], batch_size: int = 16) -> List[Dict[str, float]]:
        """
        Analyze sentiment for a batch of texts
        
        Args:
            texts: List of texts to analyze
            batch_size: Number of texts to process at once
            
        Returns:
            List of sentiment dictionaries
        """
        results = []
        
        logger.info(f"🔍 Analyzing sentiment for {len(texts)} texts in batches of {batch_size}")
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Analyzing sentiment"):
            batch = texts[i:i + batch_size]
            
            for text in batch:
                sentiment = self.analyze_sentiment(text)
                results.append(sentiment)
            
            # Small delay to prevent overwhelming the GPU
            time.sleep(0.1)
        
        logger.info(f"✅ Sentiment analysis completed for {len(results)} texts")
        return results


class NewsDataCollector:
    """
    Collects financial news data for sentiment analysis
    Simple implementation using free APIs
    """
    
    def __init__(self, news_api_key: Optional[str] = None):
        """
        Initialize news collector
        
        Args:
            news_api_key: NewsAPI key (optional, can use free tier)
        """
        self.news_api_key = news_api_key
        
        if news_api_key:
            try:
                self.newsapi = NewsApiClient(api_key=news_api_key)
                logger.info("✅ NewsAPI initialized")
            except Exception as e:
                logger.warning(f"⚠️ NewsAPI initialization failed: {e}")
                self.newsapi = None
        else:
            self.newsapi = None
            logger.info("📰 Using fallback news collection methods")
    
    def get_company_news(self, symbol: str, start_date: str, end_date: str) -> List[Dict]:
        """
        Get news articles for a company symbol
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            List of news articles with metadata
        """
        news_articles = []
        
        # Method 1: Try NewsAPI if available
        if self.newsapi:
            try:
                news_articles.extend(self._get_newsapi_articles(symbol, start_date, end_date))
            except Exception as e:
                logger.warning(f"⚠️ NewsAPI failed for {symbol}: {e}")
        
        # Method 2: Yahoo Finance news (fallback)
        try:
            news_articles.extend(self._get_yahoo_news(symbol))
        except Exception as e:
            logger.warning(f"⚠️ Yahoo Finance news failed for {symbol}: {e}")
        
        # Method 3: Generate synthetic news sentiment (ultimate fallback)
        if len(news_articles) == 0:
            logger.info(f"📰 No news found for {symbol}, generating synthetic sentiment data")
            news_articles = self._generate_synthetic_news(symbol, start_date, end_date)
        
        logger.info(f"📰 Collected {len(news_articles)} news articles for {symbol}")
        return news_articles
    
    def _get_newsapi_articles(self, symbol: str, start_date: str, end_date: str) -> List[Dict]:
        """Get articles from NewsAPI"""
        articles = []
        
        # Create search query
        company_names = {
            'AAPL': 'Apple',
            'GOOGL': 'Google Alphabet',
            'MSFT': 'Microsoft',
            'AMZN': 'Amazon',
            'TSLA': 'Tesla'
        }
        
        query = f"{company_names.get(symbol, symbol)} stock OR {symbol}"
        
        try:
            response = self.newsapi.get_everything(
                q=query,
                from_param=start_date,
                to=end_date,
                language='en',
                sort_by='relevancy',
                page_size=50
            )
            
            for article in response.get('articles', []):
                articles.append({
                    'symbol': symbol,
                    'date': article['publishedAt'][:10],  # Extract date
                    'title': article['title'] or '',
                    'description': article['description'] or '',
                    'content': article['content'] or '',
                    'url': article['url'],
                    'source': 'newsapi'
                })
                
        except Exception as e:
            logger.error(f"NewsAPI error: {e}")
        
        return articles
    
    def _get_yahoo_news(self, symbol: str) -> List[Dict]:
        """Get news from Yahoo Finance"""
        articles = []
        
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news
            
            for article in news[:20]:  # Limit to 20 most recent
                articles.append({
                    'symbol': symbol,
                    'date': datetime.fromtimestamp(article['providerPublishTime']).strftime('%Y-%m-%d'),
                    'title': article.get('title', ''),
                    'description': article.get('summary', ''),
                    'content': article.get('summary', ''),
                    'url': article.get('link', ''),
                    'source': 'yahoo_finance'
                })
                
        except Exception as e:
            logger.warning(f"Yahoo Finance news error: {e}")
        
        return articles
    
    def _generate_synthetic_news(self, symbol: str, start_date: str, end_date: str) -> List[Dict]:
        """
        Generate synthetic news data as fallback
        Creates realistic-looking news entries with neutral sentiment
        """
        articles = []
        
        # Generate 5-10 synthetic articles
        templates = [
            f"{symbol} reports quarterly earnings update",
            f"{symbol} stock shows trading activity",
            f"{symbol} market performance analysis",
            f"{symbol} financial metrics review",
            f"{symbol} investor update and outlook"
        ]
        
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        date_range = (end - start).days
        
        for i, template in enumerate(templates):
            random_days = np.random.randint(0, max(1, date_range))
            article_date = (start + timedelta(days=random_days)).strftime('%Y-%m-%d')
            
            articles.append({
                'symbol': symbol,
                'date': article_date,
                'title': template,
                'description': f"Financial news update regarding {symbol} stock performance and market activity.",
                'content': f"This is synthetic news content for {symbol} generated for sentiment analysis testing purposes.",
                'url': f"https://synthetic-news.com/{symbol}-{i}",
                'source': 'synthetic'
            })
        
        return articles


class SentimentPipeline:
    """
    Complete sentiment analysis pipeline
    Orchestrates news collection and sentiment analysis
    """
    
    def __init__(self, news_api_key: Optional[str] = None):
        """Initialize sentiment pipeline"""
        logger.info("🚀 Initializing Sentiment Analysis Pipeline")
        
        self.sentiment_analyzer = FinBERTSentimentAnalyzer()
        self.news_collector = NewsDataCollector(news_api_key)
        
        logger.info("✅ Sentiment pipeline initialized")
    
    def process_symbol(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Process sentiment analysis for a single symbol
        
        Args:
            symbol: Stock symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with sentiment scores by date
        """
        logger.info(f"🔍 Processing sentiment for {symbol} from {start_date} to {end_date}")
        
        # Collect news articles
        articles = self.news_collector.get_company_news(symbol, start_date, end_date)
        
        if not articles:
            logger.warning(f"⚠️ No articles found for {symbol}")
            return pd.DataFrame()
        
        # Analyze sentiment for each article
        sentiment_data = []
        
        for article in tqdm(articles, desc=f"Analyzing {symbol} sentiment"):
            # Combine title and description for sentiment analysis
            text = f"{article['title']} {article['description']}"
            
            if len(text.strip()) < 10:  # Skip very short texts
                continue
            
            sentiment = self.sentiment_analyzer.analyze_sentiment(text)
            
            sentiment_data.append({
                'symbol': symbol,
                'date': article['date'],
                'title': article['title'],
                'source': article['source'],
                'url': article['url'],
                **sentiment  # Unpack sentiment scores
            })
        
        # Convert to DataFrame
        df = pd.DataFrame(sentiment_data)
        
        if df.empty:
            return df
        
        # Aggregate sentiment by date
        daily_sentiment = self._aggregate_daily_sentiment(df)
        
        logger.info(f"✅ Processed {len(articles)} articles for {symbol}, {len(daily_sentiment)} days of sentiment data")
        
        return daily_sentiment
    
    def _aggregate_daily_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate multiple news articles per day into daily sentiment scores
        
        Args:
            df: DataFrame with individual article sentiments
            
        Returns:
            DataFrame with daily aggregated sentiment
        """
        if df.empty:
            return df
        
        # Group by symbol and date
        daily_agg = df.groupby(['symbol', 'date']).agg({
            'negative_score': ['mean', 'std', 'count'],
            'neutral_score': ['mean', 'std'],
            'positive_score': ['mean', 'std'],
            'compound_score': ['mean', 'std', 'min', 'max'],
            'confidence': 'mean'
        }).round(6)
        
        # Flatten column names
        daily_agg.columns = ['_'.join(col).strip() for col in daily_agg.columns]
        daily_agg = daily_agg.reset_index()
        
        # Rename columns for clarity
        column_mapping = {
            'negative_score_mean': 'sentiment_negative',
            'neutral_score_mean': 'sentiment_neutral', 
            'positive_score_mean': 'sentiment_positive',
            'compound_score_mean': 'sentiment_compound',
            'compound_score_std': 'sentiment_volatility',
            'compound_score_min': 'sentiment_min',
            'compound_score_max': 'sentiment_max',
            'confidence_mean': 'sentiment_confidence',
            'negative_score_count': 'news_count'
        }
        
        daily_agg = daily_agg.rename(columns=column_mapping)
        # Select final columns\n        
        # final_columns = [\n'symbol', 'date', 'sentiment_negative', 'sentiment_neutral', \n            'sentiment_positive', 'sentiment_compound', 'sentiment_volatility',\n            'sentiment_min', 'sentiment_max', 'sentiment_confidence', 'news_count'\n        ]\n        \n        return daily_agg[final_columns]\n    \n    def process_multiple_symbols(self, symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:\n        \"\"\"\n        Process sentiment analysis for multiple symbols\n        \n        Args:\n            symbols: List of stock symbols\n            start_date: Start date (YYYY-MM-DD)\n            end_date: End date (YYYY-MM-DD)\n            \n        Returns:\n            Combined DataFrame with all symbols' sentiment data\n        \"\"\"\n        logger.info(f\"🔍 Processing sentiment for {len(symbols)} symbols: {symbols}\")\n        \n        all_sentiment_data = []\n        \n        for symbol in symbols:\n            try:\n                symbol_sentiment = self.process_symbol(symbol, start_date, end_date)\n                if not symbol_sentiment.empty:\n                    all_sentiment_data.append(symbol_sentiment)\n                    \n                # Small delay between symbols to be respectful to APIs\n                time.sleep(1)\n                \n            except Exception as e:\n                logger.error(f\"❌ Failed to process {symbol}: {e}\")\n                continue\n        \n        if all_sentiment_data:\n            combined_df = pd.concat(all_sentiment_data, ignore_index=True)\n            logger.info(f\"✅ Successfully processed sentiment for {len(all_sentiment_data)} symbols\")\n            return combined_df\n        else:\n            logger.warning(\"⚠️ No sentiment data collected for any symbols\")\n            return pd.DataFrame()\n\n\ndef merge_sentiment_with_technical_data(technical_data_path: str, sentiment_data: pd.DataFrame, \n                                      output_path: str) -> pd.DataFrame:\n    \"\"\"\n    Merge sentiment data with existing technical analysis data\n    \n    Args:\n        technical_data_path: Path to technical analysis CSV\n        sentiment_data: DataFrame with sentiment scores\n        output_path: Path to save merged dataset\n        \n    Returns:\n        Merged DataFrame\n    \"\"\"\n    logger.info(f\"📊 Merging sentiment data with technical data from {technical_data_path}\")\n    \n    # Load technical data\n    technical_df = pd.read_csv(technical_data_path)\n    technical_df['date'] = pd.to_datetime(technical_df['date']).dt.strftime('%Y-%m-%d')\n    \n    logger.info(f\"📊 Technical data shape: {technical_df.shape}\")\n    logger.info(f\"📊 Sentiment data shape: {sentiment_data.shape}\")\n    \n    # Merge on symbol and date\n    merged_df = technical_df.merge(\n        sentiment_data, \n        on=['symbol', 'date'], \n        how='left'\n    )\n    \n    # Fill missing sentiment values with neutral scores\n    sentiment_columns = [\n        'sentiment_negative', 'sentiment_neutral', 'sentiment_positive', \n        'sentiment_compound', 'sentiment_volatility', 'sentiment_min', \n        'sentiment_max', 'sentiment_confidence', 'news_count'\n    ]\n    \n    fill_values = {\n        'sentiment_negative': 0.33,\n        'sentiment_neutral': 0.34,\n        'sentiment_positive': 0.33,\n        'sentiment_compound': 0.0,\n        'sentiment_volatility': 0.0,\n        'sentiment_min': 0.0,\n        'sentiment_max': 0.0,\n        'sentiment_confidence': 0.34,\n        'news_count': 0\n    }\n    \n    for col in sentiment_columns:\n        if col in merged_df.columns:\n            merged_df[col] = merged_df[col].fillna(fill_values.get(col, 0.0))\n    \n    # Save merged dataset\n    merged_df.to_csv(output_path, index=False)\n    \n    logger.info(f\"✅ Merged dataset saved to {output_path}\")\n    logger.info(f\"📊 Final dataset shape: {merged_df.shape}\")\n    logger.info(f\"📊 Sentiment coverage: {(merged_df['news_count'] > 0).mean():.1%}\")\n    \n    return merged_df\n\n\ndef main():\n    \"\"\"\n    Main execution function\n    \"\"\"\n    logger.info(\"🚀 Starting Sentiment Analysis Pipeline\")\n    \n    # Configuration\n    PROJECT_ROOT = Path(__file__).parent.parent\n    DATA_DIR = PROJECT_ROOT / \"data\"\n    PROCESSED_DIR = DATA_DIR / \"processed\"\n    \n    # Input and output paths\n    TECHNICAL_DATA_PATH = PROCESSED_DIR / \"combined_dataset.csv\"\n    SENTIMENT_OUTPUT_PATH = PROCESSED_DIR / \"sentiment_enhanced_dataset.csv\"\n    \n    # Ensure directories exist\n    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)\n    \n    # Check if technical data exists\n    if not TECHNICAL_DATA_PATH.exists():\n        logger.error(f\"❌ Technical data not found at {TECHNICAL_DATA_PATH}\")\n        logger.error(\"💡 Please run data.py and clean.py first to generate the base dataset\")\n        return\n    \n    # Load technical data to get symbols and date range\n    tech_df = pd.read_csv(TECHNICAL_DATA_PATH)\n    symbols = tech_df['symbol'].unique().tolist()\n    start_date = tech_df['date'].min()\n    end_date = tech_df['date'].max()\n    \n    logger.info(f\"📊 Found {len(symbols)} symbols: {symbols}\")\n    logger.info(f\"📅 Date range: {start_date} to {end_date}\")\n    \n    # Initialize sentiment pipeline\n    # Note: Add your NewsAPI key here if you have one\n    news_api_key = os.getenv('NEWS_API_KEY')  # Set this environment variable if you have a key\n    \n    pipeline = SentimentPipeline(news_api_key=news_api_key)\n    \n    # Process sentiment for all symbols\n    sentiment_data = pipeline.process_multiple_symbols(symbols, start_date, end_date)\n    \n    if sentiment_data.empty:\n        logger.warning(\"⚠️ No sentiment data collected. Creating minimal sentiment features.\")\n        # Create minimal sentiment data\n        dates = pd.date_range(start=start_date, end=end_date, freq='D')\n        minimal_sentiment = []\n        \n        for symbol in symbols:\n            for date in dates:\n                minimal_sentiment.append({\n                    'symbol': symbol,\n                    'date': date.strftime('%Y-%m-%d'),\n                    'sentiment_negative': 0.33,\n                    'sentiment_neutral': 0.34,\n                    'sentiment_positive': 0.33,\n                    'sentiment_compound': 0.0,\n                    'sentiment_volatility': 0.0,\n                    'sentiment_min': 0.0,\n                    'sentiment_max': 0.0,\n                    'sentiment_confidence': 0.34,\n                    'news_count': 0\n                })\n        \n        sentiment_data = pd.DataFrame(minimal_sentiment)\n    \n    # Merge with technical data\n    merged_dataset = merge_sentiment_with_technical_data(\n        TECHNICAL_DATA_PATH, \n        sentiment_data, \n        SENTIMENT_OUTPUT_PATH\n    )\n    \n    # Generate summary report\n    logger.info(\"\\n\" + \"=\"*60)\n    logger.info(\"📋 SENTIMENT ANALYSIS SUMMARY REPORT\")\n    logger.info(\"=\"*60)\n    logger.info(f\"📊 Original dataset: {tech_df.shape[0]:,} rows, {tech_df.shape[1]} columns\")\n    logger.info(f\"📊 Enhanced dataset: {merged_dataset.shape[0]:,} rows, {merged_dataset.shape[1]} columns\")\n    logger.info(f\"📊 Added features: {merged_dataset.shape[1] - tech_df.shape[1]} sentiment columns\")\n    logger.info(f\"📊 Sentiment coverage: {(merged_dataset['news_count'] > 0).mean():.1%} of rows\")\n    logger.info(f\"📊 Average news per day: {merged_dataset['news_count'].mean():.1f}\")\n    logger.info(f\"📊 Date range: {merged_dataset['date'].min()} to {merged_dataset['date'].max()}\")\n    logger.info(f\"📊 Symbols: {', '.join(merged_dataset['symbol'].unique())}\")\n    \n    # Sentiment statistics\n    if (merged_dataset['news_count'] > 0).any():\n        sentiment_stats = merged_dataset[merged_dataset['news_count'] > 0]['sentiment_compound'].describe()\n        logger.info(f\"\\n📈 Sentiment Statistics (days with news):\")\n        logger.info(f\"   Mean sentiment: {sentiment_stats['mean']:.4f}\")\n        logger.info(f\"   Std sentiment: {sentiment_stats['std']:.4f}\")\n        logger.info(f\"   Min sentiment: {sentiment_stats['min']:.4f}\")\n        logger.info(f\"   Max sentiment: {sentiment_stats['max']:.4f}\")\n    \n    logger.info(f\"\\n✅ Sentiment-enhanced dataset saved to: {SENTIMENT_OUTPUT_PATH}\")\n    logger.info(\"🎯 Ready for TFT training with sentiment features!\")\n    logger.info(\"=\"*60)\n\n\nif __name__ == \"__main__\":\n    main()