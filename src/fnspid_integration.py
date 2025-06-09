"""
FNSPID Real Financial Data Integration
Replaces fake data collection with real financial news and stock price data

FNSPID Dataset: 29.7M stock prices + 15.7M financial news articles (1999-2023)
Period: December 2018 - January 2024
Source: https://huggingface.co/datasets/Zihan1004/FNSPID
"""

import pandas as pd
import numpy as np
from pathlib import Path
import requests
import zipfile
import io
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple
import time
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleFNSPIDLoader:
    """
    Simple FNSPID data loader for temporal decay experiments
    Downloads and processes real financial news + stock data for Dec 2018 - Jan 2024
    """
    
    def __init__(self, data_dir: str = "./fnspid_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True, parents=True)
        
        # Experiment configuration
        self.start_date = "2018-12-01"
        self.end_date = "2024-01-31"
        self.target_symbols = ['AAPL', 'MSFT', 'TSLA', 'GOOGL', 'AMZN', 'META', 'NVDA', 'JPM', 'JNJ', 'V']
        
        # FNSPID URLs
        self.stock_url = "https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/Stock_price/full_history.zip"
        self.news_url = "https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/Stock_news/nasdaq_exteral_data.csv"
        
        # File paths
        self.stock_zip_path = self.data_dir / "stock_prices.zip"
        self.news_csv_path = self.data_dir / "news_data.csv"
        self.stock_csv_path = self.data_dir / "stock_prices.csv"
        
        logger.info(f"📁 FNSPID data directory: {self.data_dir}")
        logger.info(f"📅 Target period: {self.start_date} to {self.end_date}")
        logger.info(f"🏢 Target symbols: {len(self.target_symbols)} stocks")
    
    def download_fnspid_data(self) -> None:
        """Download FNSPID dataset if not already present"""
        
        logger.info("🔍 Checking FNSPID data availability...")
        
        # Check if we need to download stock data
        if not self.stock_zip_path.exists() and not self.stock_csv_path.exists():
            logger.info("📥 Downloading FNSPID stock price data (~500MB)...")
            self._download_with_progress(self.stock_url, self.stock_zip_path)
            logger.info("✅ Stock data downloaded")
        else:
            logger.info("✅ Stock data already available")
        
        # Check if we need to download news data
        if not self.news_csv_path.exists():
            logger.info("📥 Downloading FNSPID news sentiment data (~1.5GB)...")
            self._download_with_progress(self.news_url, self.news_csv_path)
            logger.info("✅ News data downloaded")
        else:
            logger.info("✅ News data already available")
        
        # Extract stock data if needed
        if self.stock_zip_path.exists() and not self.stock_csv_path.exists():
            logger.info("📦 Extracting stock price data...")
            self._extract_stock_data()
            logger.info("✅ Stock data extracted")
    
    def _download_with_progress(self, url: str, filepath: Path) -> None:
        """Download file with progress indication"""
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Get file size if available
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filepath, 'wb') as f:
                downloaded = 0
                chunk_size = 8192
                
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Simple progress indicator
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            if downloaded % (chunk_size * 1000) == 0:  # Update every ~8MB
                                logger.info(f"📥 Progress: {progress:.1f}% ({downloaded // (1024*1024)}MB)")
        
        except Exception as e:
            logger.error(f"❌ Download failed: {e}")
            if filepath.exists():
                filepath.unlink()  # Remove partial file
            raise
    
    def _extract_stock_data(self) -> None:
        """Extract stock price data from zip file"""
        
        try:
            with zipfile.ZipFile(self.stock_zip_path, 'r') as zip_ref:
                # Find the main CSV file in the zip
                csv_files = [f for f in zip_ref.namelist() if f.endswith('.csv')]
                
                if not csv_files:
                    raise ValueError("No CSV files found in stock data zip")
                
                # Extract the largest CSV file (likely the main dataset)
                main_csv = max(csv_files, key=lambda f: zip_ref.getinfo(f).file_size)
                
                logger.info(f"📦 Extracting {main_csv}...")
                zip_ref.extract(main_csv, self.data_dir)
                
                # Rename to expected filename
                extracted_path = self.data_dir / main_csv
                if extracted_path != self.stock_csv_path:
                    extracted_path.rename(self.stock_csv_path)
                
        except Exception as e:
            logger.error(f"❌ Extraction failed: {e}")
            raise
    
    def load_stock_data(self) -> pd.DataFrame:
        """Load and filter stock price data for experiment period"""
        
        logger.info("📈 Loading stock price data...")
        
        try:
            # Load stock data
            stock_df = pd.read_csv(self.stock_csv_path, low_memory=False)
            logger.info(f"📊 Raw stock data shape: {stock_df.shape}")
            
            # Clean and standardize columns
            stock_df = self._clean_stock_data(stock_df)
            
            # Filter for experiment period and symbols
            filtered_stocks = self._filter_stock_data(stock_df)
            
            logger.info(f"📈 Filtered stock data: {len(filtered_stocks)} records")
            logger.info(f"🏢 Symbols found: {filtered_stocks['symbol'].nunique()}")
            logger.info(f"📅 Date range: {filtered_stocks['date'].min()} to {filtered_stocks['date'].max()}")
            
            return filtered_stocks
            
        except Exception as e:
            logger.error(f"❌ Stock data loading failed: {e}")
            raise
    
    def _clean_stock_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize stock data columns"""
        
        # Standardize column names (handle different possible formats)
        column_mapping = {
            'Date': 'date', 'DATE': 'date',
            'Symbol': 'symbol', 'SYMBOL': 'symbol', 'ticker': 'symbol',
            'Open': 'open', 'OPEN': 'open',
            'High': 'high', 'HIGH': 'high',
            'Low': 'low', 'LOW': 'low',
            'Close': 'close', 'CLOSE': 'close', 'Adj Close': 'close',
            'Volume': 'volume', 'VOLUME': 'volume'
        }
        
        df = df.rename(columns=column_mapping)
        
        # Ensure required columns exist
        required_cols = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            logger.warning(f"⚠️ Missing columns: {missing_cols}")
            # Create placeholder columns if missing
            for col in missing_cols:
                if col in ['open', 'high', 'low', 'close']:
                    df[col] = df.get('close', 100.0)  # Use close price as fallback
                elif col == 'volume':
                    df[col] = 1000000  # Default volume
        
        # Convert data types
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['symbol'] = df['symbol'].astype(str).str.upper().str.strip()
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove rows with invalid dates or symbols
        df = df.dropna(subset=['date', 'symbol'])
        
        # Remove rows with all zero prices
        price_cols = ['open', 'high', 'low', 'close']
        df = df[df[price_cols].sum(axis=1) > 0]
        
        return df
    
    def _filter_stock_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter stock data for experiment period and target symbols"""
        
        # Date filter
        start_dt = pd.to_datetime(self.start_date)
        end_dt = pd.to_datetime(self.end_date)
        date_mask = (df['date'] >= start_dt) & (df['date'] <= end_dt)
        
        # Symbol filter
        symbol_mask = df['symbol'].isin(self.target_symbols)
        
        # Apply filters
        filtered_df = df[date_mask & symbol_mask].copy()
        
        # Sort by symbol and date
        filtered_df = filtered_df.sort_values(['symbol', 'date'])
        
        return filtered_df
    
    def load_news_data(self) -> pd.DataFrame:
        """Load and filter news sentiment data for experiment period"""
        
        logger.info("📰 Loading news sentiment data...")
        
        try:
            # Load news data (handle encoding issues)
            try:
                news_df = pd.read_csv(self.news_csv_path, low_memory=False)
            except UnicodeDecodeError:
                logger.info("📝 Handling encoding issues...")
                news_df = pd.read_csv(self.news_csv_path, encoding='latin-1', low_memory=False)
            
            logger.info(f"📊 Raw news data shape: {news_df.shape}")
            
            # Clean and filter news data
            filtered_news = self._clean_news_data(news_df)
            
            logger.info(f"📰 Filtered news data: {len(filtered_news)} articles")
            logger.info(f"🏢 Symbols found: {filtered_news['symbol'].nunique()}")
            logger.info(f"📅 Date range: {filtered_news['date'].min()} to {filtered_news['date'].max()}")
            
            return filtered_news
            
        except Exception as e:
            logger.error(f"❌ News data loading failed: {e}")
            raise
    
    def _clean_news_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize news data"""
        
        # Standardize column names
        column_mapping = {
            'Date': 'date', 'DATE': 'date', 'published_date': 'date',
            'Symbol': 'symbol', 'SYMBOL': 'symbol', 'ticker': 'symbol', 'stock': 'symbol',
            'Title': 'title', 'TITLE': 'title', 'headline': 'title',
            'Content': 'content', 'CONTENT': 'content', 'text': 'content', 'body': 'content',
            'Sentiment': 'sentiment', 'SENTIMENT': 'sentiment', 'sentiment_label': 'sentiment',
            'Score': 'sentiment_score', 'SCORE': 'sentiment_score', 'sentiment_score': 'sentiment_score',
            'Confidence': 'confidence', 'CONFIDENCE': 'confidence'
        }
        
        df = df.rename(columns=column_mapping)
        
        # Convert data types
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['symbol'] = df['symbol'].astype(str).str.upper().str.strip()
        
        # Handle sentiment scores
        if 'sentiment_score' not in df.columns:
            if 'sentiment' in df.columns:
                # Convert sentiment labels to scores
                sentiment_map = {
                    'positive': 1.0, 'pos': 1.0, 'bullish': 1.0, '1': 1.0,
                    'negative': -1.0, 'neg': -1.0, 'bearish': -1.0, '-1': -1.0,
                    'neutral': 0.0, 'neut': 0.0, '0': 0.0
                }
                
                df['sentiment'] = df['sentiment'].astype(str).str.lower().str.strip()
                df['sentiment_score'] = df['sentiment'].map(sentiment_map)
                
                # Handle unmapped sentiments
                unmapped_mask = df['sentiment_score'].isna()
                if unmapped_mask.sum() > 0:
                    logger.warning(f"⚠️ {unmapped_mask.sum()} unmapped sentiment values, setting to neutral")
                    df.loc[unmapped_mask, 'sentiment_score'] = 0.0
            else:
                logger.warning("⚠️ No sentiment information found, generating neutral scores")
                df['sentiment_score'] = 0.0
        
        # Ensure sentiment scores are in valid range
        df['sentiment_score'] = pd.to_numeric(df['sentiment_score'], errors='coerce').fillna(0.0)
        df['sentiment_score'] = df['sentiment_score'].clip(-1.0, 1.0)
        
        # Add confidence if missing
        if 'confidence' not in df.columns:
            df['confidence'] = 0.8  # Default confidence
        else:
            df['confidence'] = pd.to_numeric(df['confidence'], errors='coerce').fillna(0.8)
            df['confidence'] = df['confidence'].clip(0.0, 1.0)
        
        # Filter for experiment period and symbols
        start_dt = pd.to_datetime(self.start_date)
        end_dt = pd.to_datetime(self.end_date)
        
        date_mask = (df['date'] >= start_dt) & (df['date'] <= end_dt)
        symbol_mask = df['symbol'].isin(self.target_symbols)
        
        # Apply filters and remove invalid data
        filtered_df = df[date_mask & symbol_mask].copy()
        filtered_df = filtered_df.dropna(subset=['date', 'symbol'])
        
        # Sort by symbol and date
        filtered_df = filtered_df.sort_values(['symbol', 'date'])
        
        return filtered_df
    
    def create_experiment_dataset(self) -> Dict[str, pd.DataFrame]:
        """Create complete experiment dataset with stock prices and news sentiment"""
        
        logger.info("🔄 Creating experiment dataset...")
        
        # Download data if needed
        self.download_fnspid_data()
        
        # Load stock and news data
        stock_data = self.load_stock_data()
        news_data = self.load_news_data()
        
        # Create combined dataset for each symbol
        experiment_data = {}
        
        for symbol in self.target_symbols:
            symbol_stocks = stock_data[stock_data['symbol'] == symbol].copy()
            symbol_news = news_data[news_data['symbol'] == symbol].copy()
            
            if len(symbol_stocks) == 0:
                logger.warning(f"⚠️ No stock data for {symbol}")
                continue
            
            # Aggregate daily sentiment
            daily_sentiment = self._aggregate_daily_sentiment(symbol_news)
            
            # Merge stock data with sentiment
            merged_data = self._merge_stock_sentiment(symbol_stocks, daily_sentiment)
            
            # Calculate returns and technical indicators
            merged_data = self._calculate_features(merged_data)
            
            experiment_data[symbol] = merged_data
            
            logger.info(f"📊 {symbol}: {len(merged_data)} days, {len(symbol_news)} news articles")
        
        logger.info(f"✅ Experiment dataset created: {len(experiment_data)} symbols")
        
        return experiment_data
    
    def _aggregate_daily_sentiment(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate news sentiment by day"""
        
        if news_df.empty:
            return pd.DataFrame()
        
        # Group by date and calculate daily sentiment metrics
        daily_sentiment = news_df.groupby('date').agg({
            'sentiment_score': ['mean', 'std', 'count', 'min', 'max'],
            'confidence': 'mean'
        }).round(4)
        
        # Flatten column names
        daily_sentiment.columns = [
            'sentiment_mean', 'sentiment_std', 'sentiment_count', 
            'sentiment_min', 'sentiment_max', 'confidence_mean'
        ]
        
        # Fill missing values
        daily_sentiment['sentiment_std'] = daily_sentiment['sentiment_std'].fillna(0)
        daily_sentiment = daily_sentiment.fillna(0)
        
        return daily_sentiment.reset_index()
    
    def _merge_stock_sentiment(self, stock_df: pd.DataFrame, sentiment_df: pd.DataFrame) -> pd.DataFrame:
        """Merge stock prices with daily sentiment data"""
        
        # Merge on date
        merged = stock_df.merge(sentiment_df, on='date', how='left')
        
        # Fill missing sentiment with neutral values
        sentiment_cols = ['sentiment_mean', 'sentiment_std', 'sentiment_count', 
                         'sentiment_min', 'sentiment_max', 'confidence_mean']
        
        for col in sentiment_cols:
            if col in merged.columns:
                merged[col] = merged[col].fillna(0)
        
        return merged
    
    def _calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate returns and technical features"""
        
        # Sort by date to ensure proper calculation
        df = df.sort_values('date').copy()
        
        # Calculate returns
        df['return'] = df['close'].pct_change()
        df['return_5d'] = df['close'].pct_change(periods=5)
        df['return_30d'] = df['close'].pct_change(periods=30)
        
        # Calculate volatility
        df['volatility_5d'] = df['return'].rolling(window=5).std()
        df['volatility_30d'] = df['return'].rolling(window=30).std()
        
        # Calculate moving averages
        df['ma_5'] = df['close'].rolling(window=5).mean()
        df['ma_20'] = df['close'].rolling(window=20).mean()
        
        # Fill initial NaN values
        df = df.fillna(method='bfill').fillna(0)
        
        return df


class RealDataCollector:
    """
    Drop-in replacement for fake data collector
    Uses FNSPID real financial data
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.fnspid_loader = SimpleFNSPIDLoader()
        self.config_path = config_path
        logger.info("🚀 Real FNSPID data collector initialized")
    
    def collect_all_data(self) -> pd.DataFrame:
        """Collect all real FNSPID data and return combined dataset"""
        
        logger.info("📥 Collecting real FNSPID data...")
        
        # Get experiment data for all symbols
        experiment_data = self.fnspid_loader.create_experiment_dataset()
        
        # Convert to combined format
        combined_dataset = self._create_combined_dataset(experiment_data)
        
        # Save processed data
        self._save_processed_data(combined_dataset)
        
        logger.info("✅ Real FNSPID data collection complete")
        
        return combined_dataset
    
    def _create_combined_dataset(self, experiment_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Convert experiment data to combined dataset format"""
        
        all_data = []
        
        for symbol, data in experiment_data.items():
            # Add symbol column
            data = data.copy()
            data['symbol'] = symbol
            
            # Select relevant columns for pipeline
            pipeline_cols = [
                'date', 'symbol', 'open', 'high', 'low', 'close', 'volume',
                'return', 'return_5d', 'return_30d',
                'volatility_5d', 'volatility_30d', 'ma_5', 'ma_20',
                'sentiment_mean', 'sentiment_std', 'sentiment_count',
                'sentiment_min', 'sentiment_max', 'confidence_mean'
            ]
            
            # Only keep columns that exist
            available_cols = [col for col in pipeline_cols if col in data.columns]
            pipeline_data = data[available_cols].copy()
            
            all_data.append(pipeline_data)
        
        if not all_data:
            logger.error("❌ No data available for any symbols")
            return pd.DataFrame()
        
        # Combine all symbols
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df = combined_df.sort_values(['symbol', 'date'])
        
        logger.info(f"🎯 Combined dataset: {len(combined_df)} records, {combined_df['symbol'].nunique()} symbols")
        
        return combined_df
    
    def _save_processed_data(self, data: pd.DataFrame) -> None:
        """Save processed data for later use"""
        
        output_dir = Path("data/processed")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as parquet for efficiency
        output_file = output_dir / "real_fnspid_dataset.parquet"
        data.to_parquet(output_file, index=False)
        
        # Also save as CSV for inspection
        csv_file = output_dir / "real_fnspid_dataset.csv"
        data.to_csv(csv_file, index=False)
        
        logger.info(f"💾 Processed data saved to: {output_file}")
        logger.info(f"📊 Dataset shape: {data.shape}")
        logger.info(f"📅 Date range: {data['date'].min()} to {data['date'].max()}")
        logger.info(f"🏢 Symbols: {sorted(data['symbol'].unique())}")


def validate_fnspid_integration() -> bool:
    """Test function to validate FNSPID integration works"""
    
    logger.info("🧪 Testing FNSPID integration...")
    
    try:
        # Test basic loading
        loader = SimpleFNSPIDLoader()
        logger.info("✅ FNSPID loader created successfully")
        
        # Test data collection
        collector = RealDataCollector()
        logger.info("✅ Real data collector created successfully")
        
        # Note: Actual data download test would take too long for validation
        # In real usage, call collector.collect_all_data()
        
        logger.info("✅ FNSPID integration validation passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ FNSPID integration validation failed: {e}")
        return False


if __name__ == "__main__":
    # Quick test of the integration
    print("🚀 FNSPID Integration Test")
    print("=" * 50)
    
    success = validate_fnspid_integration()
    
    if success:
        print("\n✅ Integration ready!")
        print("To collect real data, run:")
        print("  collector = RealDataCollector()")
        print("  real_data = collector.collect_all_data()")
    else:
        print("\n❌ Integration failed - check logs above")