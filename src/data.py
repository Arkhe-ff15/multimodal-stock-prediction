"""
REFACTORED: Enhanced data collection with standard path integration
================================================================

✅ REFACTORING COMPLETE:
- Eliminated experiment directory dependencies
- Updated save_dataset() to use data/processed/combined_dataset.csv
- Added automatic backup creation before overwriting
- Simplified path management throughout
- Integrated with standard MLOps directory structure

All data operations now use predictable, standard locations.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple, Any
import logging
from datetime import datetime, timedelta
import warnings
import ta
import sys
import traceback
import shutil
import os

# Configure logging
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

# Standard path constants
DATA_DIR = "data/processed"
BACKUP_DIR = "data/backups"
MAIN_DATASET = f"{DATA_DIR}/combined_dataset.csv"

def create_backup(file_path: str) -> Optional[str]:
    """Create timestamped backup before overwriting - MOVED TO UTILS"""
    file_path = Path(file_path)
    
    if file_path.exists():
        # Ensure backup directory exists
        backup_dir = Path(BACKUP_DIR)
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamped backup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{file_path.stem}_backup_{timestamp}{file_path.suffix}"
        backup_path = backup_dir / backup_name
        
        try:
            shutil.copy2(file_path, backup_path)
            logger.info(f"💾 Data backup created: {backup_path}")
            return str(backup_path)
        except Exception as e:
            logger.warning(f"⚠️ Data backup failed for {file_path}: {e}")
            return None
    
    return None

def save_dataset(data: pd.DataFrame, file_path: str = None, create_backup_copy: bool = True) -> str:
    """
    REFACTORED: Save dataset to standard location with backup
    
    Args:
        data: DataFrame to save
        file_path: Optional custom path (defaults to MAIN_DATASET)
        create_backup_copy: Whether to create backup before overwriting
    
    Returns:
        Path where dataset was saved
    """
    if file_path is None:
        file_path = MAIN_DATASET
    
    # Ensure directory exists
    os.makedirs(Path(file_path).parent, exist_ok=True)
    
    # Create backup if requested and file exists
    backup_path = None
    if create_backup_copy:
        backup_path = create_backup(file_path)
    
    # Save dataset
    try:
        data.to_csv(file_path, index=True)
        logger.info(f"💾 Dataset saved to standard location: {file_path}")
        if backup_path:
            logger.info(f"💾 Previous version backed up to: {backup_path}")
        return file_path
    except Exception as e:
        logger.error(f"❌ Failed to save dataset to {file_path}: {e}")
        raise

def load_dataset(file_path: str = None) -> pd.DataFrame:
    """
    REFACTORED: Load dataset from standard location
    
    Args:
        file_path: Optional custom path (defaults to MAIN_DATASET)
    
    Returns:
        Loaded DataFrame
    """
    if file_path is None:
        file_path = MAIN_DATASET
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ Dataset not found at {file_path}")
    
    try:
        data = pd.read_csv(file_path, index_col=0, parse_dates=True)
        logger.info(f"📊 Dataset loaded from: {file_path} ({data.shape})")
        return data
    except Exception as e:
        logger.error(f"❌ Failed to load dataset from {file_path}: {e}")
        raise

class NumericalDataCollector:
    """Enhanced numerical data collector with standard path integration"""
    
    def __init__(self, symbols: List[str], start_date: str, end_date: str):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        logger.info(f"📊 NumericalDataCollector initialized for {len(symbols)} symbols")
        logger.info(f"📅 Date range: {start_date} to {end_date}")
        logger.info(f"💾 Will save to standard location: {MAIN_DATASET}")
    
    def collect_stock_data(self) -> pd.DataFrame:
        """Collect stock data for all symbols with technical indicators"""
        logger.info("📈 Collecting numerical stock data...")
        
        all_data = []
        successful_symbols = []
        
        for symbol in self.symbols:
            try:
                logger.info(f"📥 Processing {symbol}...")
                data = self._fetch_symbol_data(symbol)
                
                if not data.empty:
                    # Add technical indicators
                    data = self._add_technical_indicators(data)
                    
                    # Add symbol column
                    data['symbol'] = symbol
                    
                    all_data.append(data)
                    successful_symbols.append(symbol)
                    logger.info(f"✅ {symbol}: {data.shape}")
                else:
                    logger.warning(f"⚠️ No data for {symbol}")
                    
            except Exception as e:
                logger.error(f"❌ Error processing {symbol}: {e}")
                continue
        
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=False)
            logger.info(f"✅ Numerical data collection complete: {combined_data.shape}")
            logger.info(f"📊 Successful symbols: {successful_symbols}")
            return combined_data
        else:
            logger.error("❌ No data collected for any symbol")
            return pd.DataFrame()
    
    def _fetch_symbol_data(self, symbol: str) -> pd.DataFrame:
        """Fetch data for a single symbol"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(
                start=self.start_date,
                end=self.end_date,
                interval='1d'
            )
            
            if data.empty:
                return pd.DataFrame()
            
            # Reset index to make date a column
            data = data.reset_index()
            data.columns = data.columns.str.lower()
            
            return data
            
        except Exception as e:
            logger.error(f"❌ Failed to fetch data for {symbol}: {e}")
            return pd.DataFrame()
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive technical indicators"""
        try:
            # Basic price features
            data['returns'] = data['close'].pct_change()
            data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
            data['volatility'] = data['returns'].rolling(window=20).std()
            
            # Moving averages
            for period in [5, 10, 20, 50]:
                data[f'sma_{period}'] = ta.trend.sma_indicator(data['close'], window=period)
                data[f'ema_{period}'] = ta.trend.ema_indicator(data['close'], window=period)
            
            # Technical indicators
            data['rsi'] = ta.momentum.rsi(data['close'], window=14)
            data['macd'] = ta.trend.macd_diff(data['close'])
            data['bb_upper'] = ta.volatility.bollinger_hband(data['close'])
            data['bb_lower'] = ta.volatility.bollinger_lband(data['close'])
            data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['close']
            
            # Volume indicators
            data['volume_sma'] = data['volume'].rolling(window=20).mean()
            data['volume_ratio'] = data['volume'] / data['volume_sma']
            
            # Price position
            data['price_position'] = (data['close'] - data['low']) / (data['high'] - data['low'])
            
            # Additional features
            data['gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
            data['intraday_return'] = (data['close'] - data['open']) / data['open']
            
            return data
            
        except Exception as e:
            logger.error(f"❌ Error adding technical indicators: {e}")
            return data
    
    def add_target_variables(self, data: pd.DataFrame, horizons: List[int]) -> pd.DataFrame:
        """Add target variables for multiple prediction horizons"""
        logger.info("🎯 Adding target variables...")
        
        try:
            for horizon in horizons:
                # Forward returns for each horizon
                data[f'target_{horizon}d'] = (
                    data.groupby('symbol')['close']
                    .shift(-horizon) / data['close'] - 1
                )
            
            logger.info(f"✅ Added targets for horizons: {horizons}")
            return data
            
        except Exception as e:
            logger.error(f"❌ Error adding target variables: {e}")
            return data

class SentimentDataCollector:
    """Enhanced sentiment data collector with standard path integration"""
    
    def __init__(self, symbols: List[str], start_date: str, end_date: str, fnspid_data_dir: str = None):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        # Use standard directory if not specified
        self.fnspid_data_dir = Path(fnspid_data_dir) if fnspid_data_dir else Path(DATA_DIR)
        logger.info(f"📰 SentimentDataCollector initialized for {len(symbols)} symbols")
        logger.info(f"📁 FNSPID directory: {self.fnspid_data_dir}")
        logger.info(f"💾 Will integrate with dataset at: {MAIN_DATASET}")
    
    def collect_sentiment_data(self) -> pd.DataFrame:
        """Collect sentiment data from FNSPID dataset"""
        logger.info("📰 Collecting sentiment data...")
        
        try:
            # Find FNSPID file in standard location
            file_path = self._find_fnspid_file()
            
            if file_path is None:
                logger.warning("⚠️ No FNSPID file found, creating empty sentiment data")
                return self._create_empty_sentiment_data()
            
            # Load FNSPID data based on file size
            fnspid_data = self._load_fnspid_data(file_path)
            
            if fnspid_data.empty:
                logger.warning("⚠️ FNSPID file is empty or couldn't be loaded")
                return self._create_empty_sentiment_data()
            
            # Process sentiment data
            processed_data = self._process_sentiment_data(fnspid_data)
            
            logger.info(f"✅ Sentiment data collection complete: {processed_data.shape}")
            return processed_data
            
        except Exception as e:
            logger.error(f"❌ Sentiment data collection failed: {e}")
            return self._create_empty_sentiment_data()
    
    def _find_fnspid_file(self) -> Optional[Path]:
        """Find FNSPID data file in standard location"""
        possible_files = [
            # Look for our pre-filtered file FIRST in standard location
            self.fnspid_data_dir / "nasdaq_2018_2024.csv",
            # Fallback to original file locations
            self.fnspid_data_dir / "nasdaq_exteral_data.csv",
            self.fnspid_data_dir / "news_data.csv",
            self.fnspid_data_dir / "fnspid_data.csv",
            self.fnspid_data_dir / "financial_news.csv"
        ]
        
        for file_path in possible_files:
            if file_path.exists():
                file_size_mb = file_path.stat().st_size / (1024 * 1024)
                logger.info(f"📁 Found FNSPID file: {file_path} ({file_size_mb:.1f} MB)")
                return file_path
        
        logger.warning(f"⚠️ No FNSPID file found in {self.fnspid_data_dir}")
        logger.info(f"📂 Searched for: {[f.name for f in possible_files]}")
        return None
    
    def _load_fnspid_data(self, file_path: Path) -> pd.DataFrame:
        """Smart loading strategy based on file size"""
        logger.info(f"📥 Loading FNSPID data from {file_path}")
        
        try:
            # Check file size
            file_size_gb = file_path.stat().st_size / (1024 * 1024 * 1024)
            logger.info(f"📊 File size: {file_size_gb:.1f} GB")
            
            # Strategy based on file size
            if file_size_gb > 10:  # 10GB+ files
                logger.info("🚀 Using streaming date filter for massive file...")
                return self._load_fnspid_with_date_filter(file_path)
            elif file_size_gb > 1:  # 1-10GB files  
                logger.info("📊 Using chunked loading...")
                return self._load_fnspid_chunked(file_path)
            else:  # <1GB files
                logger.info("📁 Using direct loading...")
                return self._load_fnspid_direct(file_path)
                
        except Exception as e:
            logger.error(f"❌ Failed to load FNSPID data: {e}")
            return pd.DataFrame()
    
    def _load_fnspid_direct(self, file_path: Path) -> pd.DataFrame:
        """Load FNSPID file directly with standard path awareness"""
        logger.info(f"📁 Loading FNSPID file directly: {file_path}")
        
        try:
            # Load the filtered file with robust CSV parsing
            data = pd.read_csv(
                file_path, 
                low_memory=False,
                quoting=1,  # QUOTE_ALL - handles commas in content
                escapechar='\\',  # Handle escape characters
                on_bad_lines='skip',  # Skip malformed lines
                encoding='utf-8'
            )
            logger.info(f"📊 Loaded {len(data)} rows, {len(data.columns)} columns")
            
            # Normalize column names first
            data = self._normalize_fnspid_columns(data)
            
            # Log actual columns found
            logger.info(f"📋 Available columns: {list(data.columns)}")
            
            # Handle date column with UTC timezone
            if 'Date' in data.columns:
                logger.info("📅 Processing Date column...")
                data['Date'] = pd.to_datetime(data['Date'], errors='coerce', utc=True)
                # Convert to naive datetime (remove timezone for merging)
                data['Date'] = data['Date'].dt.tz_localize(None)
                logger.info(f"📅 Date range: {data['Date'].min()} to {data['Date'].max()}")
            
            # Smart symbol filtering - check both Symbol column and content
            if 'Symbol' in data.columns:
                # First, try filtering by Symbol column
                symbol_mask = data['Symbol'].isin(self.symbols)
                symbol_filtered = data[symbol_mask]
                logger.info(f"🎯 Articles with clean symbol tags: {len(symbol_filtered)}")
                
                # If we don't have enough data from symbol column, 
                # use the pre-filtered data (since we already filtered by content)
                if len(symbol_filtered) < 1000:  # Threshold for sufficient data
                    logger.info("📰 Using pre-filtered data (symbols found in content)")
                    # For pre-filtered data, assign symbols based on content
                    data = self._assign_symbols_from_content(data)
                else:
                    data = symbol_filtered
                    logger.info(f"📰 Using symbol column filtered data: {len(data)} articles")
            else:
                # No Symbol column, assign from content
                logger.info("📰 No Symbol column found, assigning from content")
                data = self._assign_symbols_from_content(data)
            
            # Filter for date range
            if 'Date' in data.columns:
                before_filter = len(data)
                start_date = pd.to_datetime(self.start_date)
                end_date = pd.to_datetime(self.end_date)
                
                date_mask = (data['Date'] >= start_date) & (data['Date'] <= end_date)
                data = data[date_mask]
                logger.info(f"📅 Date range filter: {before_filter} → {len(data)} articles")
            
            # Keep essential columns for sentiment analysis
            essential_columns = ['Symbol', 'Date', 'Title', 'Content']
            available_essential = [col for col in essential_columns if col in data.columns]
            
            if available_essential:
                data = data[available_essential].copy()
                logger.info(f"📋 Kept essential columns: {available_essential}")
            
            # Remove rows with missing essential data
            if 'Content' in data.columns:
                before_dropna = len(data)
                data = data.dropna(subset=['Content'])
                logger.info(f"🧹 Removed rows with missing content: {before_dropna} → {len(data)}")
            
            # Final symbol distribution
            if 'Symbol' in data.columns:
                symbol_counts = data['Symbol'].value_counts()
                logger.info(f"📊 Final articles per symbol: {dict(symbol_counts.head(10))}")
            
            logger.info(f"✅ FNSPID data loaded successfully: {len(data)} articles ready for sentiment analysis")
            return data
            
        except Exception as e:
            logger.error(f"❌ Failed to load FNSPID data: {e}")
            return pd.DataFrame()
    
    def _load_fnspid_with_date_filter(self, file_path: Path) -> pd.DataFrame:
        """Load FNSPID with aggressive date filtering for memory efficiency"""
        logger.info(f"📊 Loading massive file with 2018-2024 date filter...")
        logger.info(f"🎯 Target date range: {self.start_date} to {self.end_date}")
        
        # Target date range for filtering
        start_year = pd.to_datetime(self.start_date).year
        end_year = pd.to_datetime(self.end_date).year
        target_symbols = set(self.symbols)
        
        logger.info(f"📅 Filtering for years: {start_year}-{end_year}")
        logger.info(f"🏢 Filtering for symbols: {target_symbols}")
        
        relevant_data = []
        total_processed = 0
        total_kept = 0
        chunk_size = 2000  # Larger chunks since we're filtering aggressively
        max_relevant_articles = 50000  # Stop after finding enough relevant articles
        
        try:
            # Stream through file with aggressive filtering
            for chunk_num, chunk in enumerate(pd.read_csv(
                file_path, 
                chunksize=chunk_size,
                low_memory=False,
                encoding='utf-8',
                on_bad_lines='skip'
            )):
                total_processed += len(chunk)
                
                try:
                    # Normalize columns
                    chunk = self._normalize_fnspid_columns(chunk)
                    
                    # Filter by symbol FIRST (most selective)
                    if 'Symbol' in chunk.columns:
                        symbol_mask = chunk['Symbol'].isin(target_symbols)
                        chunk = chunk[symbol_mask]
                        
                        if chunk.empty:
                            continue
                    
                    # Filter by date SECOND
                    if 'Date' in chunk.columns:
                        # Convert dates efficiently
                        chunk['Date'] = pd.to_datetime(chunk['Date'], errors='coerce')
                        
                        # Filter by year range (fastest filtering)
                        year_mask = (
                            (chunk['Date'].dt.year >= start_year) & 
                            (chunk['Date'].dt.year <= end_year)
                        )
                        chunk = chunk[year_mask]
                        
                        if chunk.empty:
                            continue
                        
                        # More precise date filtering
                        start_date_ts = pd.to_datetime(self.start_date)
                        end_date_ts = pd.to_datetime(self.end_date)
                        
                        precise_mask = (
                            (chunk['Date'] >= start_date_ts) & 
                            (chunk['Date'] <= end_date_ts)
                        )
                        chunk = chunk[precise_mask]
                    
                    if not chunk.empty:
                        # Keep only essential columns to save memory
                        essential_cols = ['Symbol', 'Date', 'Title', 'Content', 'URL']
                        available_cols = [col for col in essential_cols if col in chunk.columns]
                        chunk = chunk[available_cols].copy()
                        
                        relevant_data.append(chunk)
                        total_kept += len(chunk)
                        
                        logger.info(f"📰 Chunk {chunk_num}: Kept {len(chunk)} articles (Total: {total_kept})")
                        
                        # Stop if we have enough data
                        if total_kept >= max_relevant_articles:
                            logger.info(f"🎯 Reached target of {max_relevant_articles} articles, stopping")
                            break
                
                except Exception as e:
                    logger.warning(f"⚠️ Error in chunk {chunk_num}: {e}")
                    continue
                
                # Progress and memory management
                if chunk_num % 50 == 0:
                    logger.info(f"📊 Processed {total_processed:,} rows, kept {total_kept:,} relevant articles")
                    # Garbage collection every 50 chunks
                    import gc
                    gc.collect()
                
                # Early termination if not finding relevant data
                if chunk_num > 100 and total_kept == 0:
                    logger.warning("⚠️ No relevant data found in first 100 chunks, stopping")
                    break
            
            # Combine results
            if relevant_data:
                logger.info(f"📰 Combining {len(relevant_data)} filtered chunks...")
                combined_data = pd.concat(relevant_data, ignore_index=True)
                
                # Final stats
                logger.info(f"✅ Successfully filtered massive file:")
                logger.info(f"   📊 Processed: {total_processed:,} total rows")
                logger.info(f"   📰 Kept: {len(combined_data):,} relevant articles")
                logger.info(f"   🏢 Symbols: {combined_data['Symbol'].nunique() if 'Symbol' in combined_data.columns else 'N/A'}")
                logger.info(f"   📅 Date range: {combined_data['Date'].min()} to {combined_data['Date'].max()}")
                
                return combined_data
            else:
                logger.warning("📰 No relevant articles found")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"❌ Streaming load failed: {e}")
            return pd.DataFrame()
    
    def _load_fnspid_chunked(self, file_path: Path) -> pd.DataFrame:
        """Load FNSPID in chunks for medium-sized files"""
        logger.info("📊 Loading file in chunks...")
        
        relevant_data = []
        chunk_size = 5000
        max_chunks = 30
        chunks_processed = 0
        total_articles = 0
        
        # Target symbols for filtering
        target_symbols = set(self.symbols)
        
        try:
            chunk_iterator = pd.read_csv(
                file_path, 
                chunksize=chunk_size,
                low_memory=False,
                encoding='utf-8',
                on_bad_lines='skip'
            )
            
            for i, chunk in enumerate(chunk_iterator):
                if chunks_processed >= max_chunks:
                    logger.info(f"📊 Reached chunk limit ({max_chunks})")
                    break
                
                try:
                    # Normalize column names first
                    chunk = self._normalize_fnspid_columns(chunk)
                    
                    # Filter for target symbols IMMEDIATELY
                    if 'Symbol' in chunk.columns:
                        symbol_mask = chunk['Symbol'].isin(target_symbols)
                        filtered_chunk = chunk[symbol_mask].copy()
                        
                        if not filtered_chunk.empty:
                            # Basic date filtering
                            if 'Date' in filtered_chunk.columns:
                                filtered_chunk['Date'] = pd.to_datetime(filtered_chunk['Date'], errors='coerce')
                                start_date = pd.to_datetime(self.start_date) - pd.Timedelta(days=30)
                                end_date = pd.to_datetime(self.end_date) + pd.Timedelta(days=1)
                                date_mask = (
                                    (filtered_chunk['Date'] >= start_date) & 
                                    (filtered_chunk['Date'] <= end_date)
                                )
                                filtered_chunk = filtered_chunk[date_mask].copy()
                            
                            if not filtered_chunk.empty:
                                relevant_data.append(filtered_chunk)
                                total_articles += len(filtered_chunk)
                                chunks_processed += 1
                                
                                logger.info(f"📰 Chunk {i}: Found {len(filtered_chunk)} relevant articles (Total: {total_articles})")
                
                except Exception as e:
                    logger.warning(f"⚠️ Error processing chunk {i}: {e}")
                    continue
                
                # Memory cleanup
                del chunk
                if i % 5 == 0:
                    import gc
                    gc.collect()
                
                # Progress logging
                if i % 10 == 0:
                    logger.info(f"📊 Processed chunk {i}, found {total_articles} relevant articles so far")
            
            if relevant_data:
                logger.info(f"📰 Combining {len(relevant_data)} filtered chunks...")
                combined_data = pd.concat(relevant_data, ignore_index=True)
                
                # Final cleanup
                del relevant_data
                import gc
                gc.collect()
                
                logger.info(f"📰 Successfully loaded {len(combined_data)} relevant articles")
                return combined_data
            else:
                logger.warning("📰 No relevant articles found in processed chunks")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"❌ Chunked load failed: {e}")
            return pd.DataFrame()
    
    def _normalize_fnspid_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize FNSPID column names to standard format"""
        # Create mapping for actual FNSPID file structure
        column_mapping = {
            # Actual FNSPID columns -> Standard names
            'Date': 'Date',
            'Stock_symbol': 'Symbol', 
            'Article_title': 'Title',
            'Article': 'Content',
            'Url': 'URL',
            
            # Alternative names (keep for compatibility)
            'date': 'Date',
            'symbol': 'Symbol',
            'title': 'Title',
            'content': 'Content',
            'url': 'URL',
            
            # Other possible variations
            'stock_symbol': 'Symbol',
            'article_title': 'Title',
            'article': 'Content'
        }
        
        # Apply column mapping
        data = data.rename(columns=column_mapping)
        
        # Log what columns we found and mapped
        original_cols = set(data.columns)
        mapped_cols = {old: new for old, new in column_mapping.items() if old in original_cols}
        if mapped_cols:
            logger.info(f"📊 Column mapping applied: {mapped_cols}")
        
        return data
    
    def _assign_symbols_from_content(self, data: pd.DataFrame) -> pd.DataFrame:
        """Assign symbols based on content analysis for pre-filtered data"""
        logger.info("🔍 Assigning symbols from content analysis...")
        
        # Create a list to store processed rows
        processed_rows = []
        
        for _, row in data.iterrows():
            # Check content for our target symbols
            content = str(row.get('Content', '')) + ' ' + str(row.get('Title', ''))
            content_upper = content.upper()
            
            # Find which target symbols appear in the content
            found_symbols = [symbol for symbol in self.symbols if symbol in content_upper]
            
            if found_symbols:
                # For simplicity, assign the first found symbol
                # In practice, you might want more sophisticated logic
                row = row.copy()
                row['Symbol'] = found_symbols[0]
                processed_rows.append(row)
        
        if processed_rows:
            result = pd.DataFrame(processed_rows)
            logger.info(f"📰 Assigned symbols to {len(result)} articles")
            return result
        else:
            logger.warning("⚠️ No articles found with target symbols in content")
            return pd.DataFrame()
    
    def _process_sentiment_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process sentiment data into time series format for standard structure"""
        logger.info("📊 Processing sentiment data into time series...")
        
        try:
            # Ensure we have required columns
            if 'Date' not in data.columns:
                logger.error("❌ No Date column found")
                return self._create_empty_sentiment_data()
            
            # Create date range for alignment with stock data
            start_date = pd.to_datetime(self.start_date)
            end_date = pd.to_datetime(self.end_date)
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            
            # Initialize result with all symbols and dates
            processed_data = []
            
            for symbol in self.symbols:
                # Filter data for this symbol
                symbol_data = data[data.get('Symbol', '') == symbol] if 'Symbol' in data.columns else data
                
                # Create time series for this symbol
                symbol_series = pd.DataFrame({
                    'date': date_range,
                    'symbol': symbol
                })
                
                # Aggregate sentiment by date
                if not symbol_data.empty and 'Date' in symbol_data.columns:
                    daily_sentiment = symbol_data.groupby(symbol_data['Date'].dt.date).agg({
                        'Title': 'count',  # News count
                        'Content': lambda x: len(' '.join(x.astype(str)))  # Total content length
                    }).reset_index()
                    
                    daily_sentiment.columns = ['date', 'news_count', 'content_length']
                    daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date'])
                    
                    # Merge with symbol series
                    symbol_series = symbol_series.merge(daily_sentiment, on='date', how='left')
                
                # Fill missing values with proper handling
                if 'news_count' in symbol_series.columns:
                    symbol_series['news_count'] = symbol_series['news_count'].fillna(0)
                else:
                    symbol_series['news_count'] = 0
                    
                if 'content_length' in symbol_series.columns:
                    symbol_series['content_length'] = symbol_series['content_length'].fillna(0)
                else:
                    symbol_series['content_length'] = 0
                
                # Add basic sentiment features
                symbol_series['sentiment_momentum'] = symbol_series['news_count'].rolling(window=7).mean()
                symbol_series['content_momentum'] = symbol_series['content_length'].rolling(window=7).mean()
                
                processed_data.append(symbol_series)
            
            if processed_data:
                final_data = pd.concat(processed_data, ignore_index=True)
                logger.info(f"📊 Processed sentiment data: {final_data.shape}")
                return final_data
            else:
                return self._create_empty_sentiment_data()
                
        except Exception as e:
            logger.error(f"❌ Error processing sentiment data: {e}")
            return self._create_empty_sentiment_data()
    
    def _create_empty_sentiment_data(self) -> pd.DataFrame:
        """Create empty sentiment data structure for standard format"""
        logger.info("📰 Creating empty sentiment data structure...")
        
        try:
            # Create date range
            start_date = pd.to_datetime(self.start_date)
            end_date = pd.to_datetime(self.end_date)
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            
            # Create empty data for all symbols
            empty_data = []
            for symbol in self.symbols:
                symbol_data = pd.DataFrame({
                    'date': date_range,
                    'symbol': symbol,
                    'news_count': 0,
                    'content_length': 0,
                    'sentiment_momentum': 0
                })
                empty_data.append(symbol_data)
            
            result = pd.concat(empty_data, ignore_index=True)
            logger.info(f"📰 Empty sentiment data created: {result.shape}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error creating empty sentiment data: {e}")
            return pd.DataFrame()

class DataMerger:
    """Enhanced data merger with standard path awareness"""
    
    def __init__(self):
        logger.info("🔗 DataMerger initialized for standard structure")
        logger.info(f"💾 Will use standard dataset location: {MAIN_DATASET}")
    
    def merge_datasets(self, numerical_data: pd.DataFrame, sentiment_data: pd.DataFrame) -> pd.DataFrame:
        """Merge numerical and sentiment datasets with standard path preparation"""
        logger.info("🔗 Merging numerical and sentiment datasets...")
        
        try:
            # Prepare datasets for merging
            numerical_prepared = self._prepare_numerical_for_merge(numerical_data)
            sentiment_prepared = self._prepare_sentiment_for_merge(sentiment_data)
            
            if numerical_prepared.empty:
                logger.error("❌ Numerical data is empty")
                return pd.DataFrame()
            
            if sentiment_prepared.empty:
                logger.warning("⚠️ Sentiment data is empty, adding empty sentiment columns")
                return self._add_empty_sentiment_columns(numerical_prepared)
            
            # Perform merge
            merged_data = self._perform_merge(numerical_prepared, sentiment_prepared)
            
            if merged_data.empty:
                logger.error("❌ Merge resulted in empty dataset")
                return numerical_prepared
            
            logger.info(f"✅ Datasets merged successfully: {merged_data.shape}")
            logger.info(f"💾 Ready for saving to: {MAIN_DATASET}")
            return merged_data
            
        except Exception as e:
            logger.error(f"❌ Merge operation failed: {e}")
            logger.info("🔄 Falling back to numerical data only")
            return self._add_empty_sentiment_columns(numerical_data)
    
    def _prepare_numerical_for_merge(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare numerical data for merging"""
        prepared = data.copy()
        
        # Ensure date index
        if not isinstance(prepared.index, pd.DatetimeIndex):
            if 'date' in prepared.columns:
                prepared['date'] = pd.to_datetime(prepared['date'])
                prepared = prepared.set_index('date')
            else:
                logger.warning("⚠️ No date column found in numerical data")
        
        # Reset index to make date a column for merging
        prepared = prepared.reset_index()
        if 'index' in prepared.columns:
            prepared = prepared.rename(columns={'index': 'date'})
        
        # Convert to datetime FIRST, then handle timezone
        prepared['date'] = pd.to_datetime(prepared['date'])
        if prepared['date'].dt.tz is not None:
            prepared['date'] = prepared['date'].dt.tz_localize(None)
            logger.info("🕐 Removed timezone from numerical data for merging")
            
        return prepared
    
    def _prepare_sentiment_for_merge(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare sentiment data for merging"""
        prepared = data.copy()
        
        # Ensure date index
        if not isinstance(prepared.index, pd.DatetimeIndex):
            if 'date' in prepared.columns:
                prepared['date'] = pd.to_datetime(prepared['date'])
                prepared = prepared.set_index('date')
            else:
                logger.warning("⚠️ No date column found in sentiment data")
        
        # Reset index to make date a column for merging
        prepared = prepared.reset_index()
        if 'index' in prepared.columns:
            prepared = prepared.rename(columns={'index': 'date'})
        
        # Ensure consistent datetime format (no timezone)
        prepared['date'] = pd.to_datetime(prepared['date'])
        if prepared['date'].dt.tz is not None:
            prepared['date'] = prepared['date'].dt.tz_localize(None)
            logger.info("🕐 Removed timezone from sentiment data for merging")
        
        return prepared
    
    def _perform_merge(self, numerical: pd.DataFrame, sentiment: pd.DataFrame) -> pd.DataFrame:
        """Perform the actual merge operation"""
        try:
            # Merge on date and symbol
            merged = numerical.merge(
                sentiment,
                on=['date', 'symbol'],
                how='left'
            )
            
            # Fill missing sentiment values
            sentiment_columns = [col for col in sentiment.columns if col not in ['date', 'symbol']]
            for col in sentiment_columns:
                if col in merged.columns:
                    merged[col] = merged[col].fillna(0)
            
            return merged
            
        except Exception as e:
            logger.error(f"❌ Merge operation failed: {e}")
            raise
    
    def _add_empty_sentiment_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add empty sentiment columns to numerical data"""
        logger.info("📰 Added empty sentiment columns to numerical data")
        
        data = data.copy()
        sentiment_columns = ['news_count', 'content_length', 'sentiment_momentum']
        
        for col in sentiment_columns:
            data[col] = 0
        
        return data

class CompleteDataCollector:
    """Complete data collection orchestrator with standard structure"""
    
    def __init__(self, config: dict):
        self.config = config
        self.symbols = config['data']['symbols']
        self.start_date = config['data']['start_date']
        self.end_date = config['data']['end_date']
        self.target_horizons = config['data'].get('target_horizons', [5, 30, 90])
        self.fnspid_data_dir = config['data'].get('fnspid_data_dir', DATA_DIR)
        
        logger.info("🚀 Starting complete dataset collection with standard structure...")
        logger.info(f"💾 Target dataset location: {MAIN_DATASET}")
    
    def collect_complete_dataset(self) -> pd.DataFrame:
        """Collect and merge complete dataset, save to standard location"""
        try:
            # Step 1: Collect numerical data
            logger.info("📊 Step 1: Collecting numerical data...")
            numerical_collector = NumericalDataCollector(
                symbols=self.symbols,
                start_date=self.start_date,
                end_date=self.end_date
            )
            
            numerical_data = numerical_collector.collect_stock_data()
            if numerical_data.empty:
                raise ValueError("Failed to collect numerical data")
            
            # Add target variables
            numerical_data = numerical_collector.add_target_variables(
                numerical_data, 
                self.target_horizons
            )
            
            # Step 2: Collect sentiment data
            logger.info("📰 Step 2: Collecting sentiment data...")
            sentiment_collector = SentimentDataCollector(
                symbols=self.symbols,
                start_date=self.start_date,
                end_date=self.end_date,
                fnspid_data_dir=self.fnspid_data_dir
            )
            
            sentiment_data = sentiment_collector.collect_sentiment_data()
            
            # Step 3: Merge datasets
            logger.info("🔗 Step 3: Merging datasets...")
            merger = DataMerger()
            complete_dataset = merger.merge_datasets(numerical_data, sentiment_data)
            
            if complete_dataset.empty:
                raise ValueError("Dataset merging failed")
            
            # Step 4: Save to standard location with backup
            logger.info(f"💾 Step 4: Saving complete dataset to standard location...")
            saved_path = save_dataset(complete_dataset, MAIN_DATASET, create_backup_copy=True)
            
            logger.info(f"🎉 Complete dataset collection finished: {complete_dataset.shape}")
            logger.info(f"💾 Dataset saved to: {saved_path}")
            return complete_dataset
            
        except Exception as e:
            logger.error(f"❌ Dataset collection failed: {e}")
            logger.error(f"📊 Error details: {traceback.format_exc()}")
            raise

# STANDALONE FUNCTIONS FOR EXPERIMENT RUNNER (REFACTORED)
def collect_complete_dataset(config: dict) -> pd.DataFrame:
    """
    REFACTORED: Standalone function for collecting complete dataset
    Now saves directly to standard location with backup
    """
    logger.info("📊 Collecting complete dataset (standard structure)...")
    
    try:
        # Create collector instance and run collection
        collector = CompleteDataCollector(config)
        dataset = collector.collect_complete_dataset()
        
        logger.info(f"✅ Complete dataset collection successful: {dataset.shape}")
        logger.info(f"💾 Dataset available at: {MAIN_DATASET}")
        return dataset
        
    except Exception as e:
        logger.error(f"❌ Complete dataset collection failed: {e}")
        raise

def get_data_summary(data: pd.DataFrame) -> Dict[str, Union[int, float, str, List]]:
    """
    Generate comprehensive data summary for experiment reporting
    ENHANCED for standard structure awareness
    """
    logger.info("📊 Generating data summary...")
    
    try:
        # Validate input data
        if data is None:
            logger.error("❌ Data is None")
            return {'error': 'Data is None', 'total_rows': 0, 'total_columns': 0}
        
        if not hasattr(data, 'columns'):
            logger.error("❌ Data has no columns attribute")
            return {'error': 'Data has no columns attribute', 'total_rows': 0, 'total_columns': 0}
        
        # Basic statistics with error handling
        try:
            total_rows = int(len(data))
            total_columns = int(len(data.columns))
            memory_usage = round(float(data.memory_usage(deep=True).sum()) / (1024 * 1024), 2)
        except Exception as e:
            logger.error(f"❌ Error in basic statistics: {e}")
            total_rows = 0
            total_columns = 0
            memory_usage = 0.0
        
        summary = {
            'total_rows': total_rows,
            'total_columns': total_columns,
            'memory_usage_mb': memory_usage,
            'standard_structure': {
                'data_location': MAIN_DATASET,
                'backup_location': BACKUP_DIR,
                'data_directory': DATA_DIR
            }
        }
        
        # Date range analysis
        try:
            if 'date' in data.columns:
                date_col = data['date']
                if len(date_col.dropna()) > 0:
                    summary['date_range'] = {
                        'start': str(date_col.min().date()) if pd.notna(date_col.min()) else 'N/A',
                        'end': str(date_col.max().date()) if pd.notna(date_col.max()) else 'N/A',
                        'total_days': int((date_col.max() - date_col.min()).days) if pd.notna(date_col.min()) else 0
                    }
                else:
                    summary['date_range'] = {'start': 'N/A', 'end': 'N/A', 'total_days': 0}
        except Exception as e:
            logger.warning(f"⚠️ Error in date analysis: {e}")
            summary['date_range'] = {'start': 'N/A', 'end': 'N/A', 'total_days': 0}
        
        # Symbol analysis
        try:
            if 'symbol' in data.columns:
                symbol_counts = data['symbol'].value_counts()
                summary['symbols'] = {
                    'count': int(len(symbol_counts)),
                    'list': list(symbol_counts.index),
                    'distribution': {str(k): int(v) for k, v in symbol_counts.to_dict().items()}
                }
        except Exception as e:
            logger.warning(f"⚠️ Error in symbol analysis: {e}")
            summary['symbols'] = {'count': 0, 'list': [], 'distribution': {}}
        
        # Column categorization - with extensive safety
        try:
            all_columns = list(data.columns)
            numeric_cols = list(data.select_dtypes(include=[np.number]).columns)
            target_cols = [str(col) for col in all_columns if str(col).startswith('target_')]
            sentiment_cols = [str(col) for col in all_columns if any(word in str(col).lower() for word in ['sentiment', 'news', 'content'])]
            technical_cols = [str(col) for col in numeric_cols if any(word in str(col).lower() for word in ['sma', 'ema', 'rsi', 'macd', 'bb_', 'volume'])]
            
            summary['columns'] = {
                'numeric_columns': len(numeric_cols),
                'target_columns': len(target_cols),
                'sentiment_columns': len(sentiment_cols),
                'technical_columns': len(technical_cols),
                'targets': target_cols,
                'sentiment': sentiment_cols,
                'technical': technical_cols
            }
        except Exception as e:
            logger.warning(f"⚠️ Error in column analysis: {e}")
            summary['columns'] = {
                'numeric_columns': 0,
                'target_columns': 0,
                'sentiment_columns': 0,
                'technical_columns': 0,
                'targets': [],
                'sentiment': [],
                'technical': []
            }
        
        # Data quality - with error handling
        try:
            missing_data = data.isnull().sum()
            total_cells = len(data) * len(data.columns)
            summary['data_quality'] = {
                'missing_values': int(missing_data.sum()),
                'missing_percentage': round(float((missing_data.sum() / total_cells) * 100), 2) if total_cells > 0 else 0.0,
                'columns_with_missing': {str(k): int(v) for k, v in missing_data[missing_data > 0].to_dict().items()}
            }
        except Exception as e:
            logger.warning(f"⚠️ Error in data quality analysis: {e}")
            summary['data_quality'] = {'missing_values': 0, 'missing_percentage': 0.0, 'columns_with_missing': {}}
        
        # Target variable statistics - with safe access
        try:
            target_cols = [str(col) for col in data.columns if str(col).startswith('target_')]
            if target_cols:
                target_stats = {}
                for col in target_cols:
                    try:
                        if col in data.columns:
                            target_data = data[col].dropna()
                            if len(target_data) > 0:
                                target_stats[str(col)] = {
                                    'mean': round(float(target_data.mean()), 4),
                                    'std': round(float(target_data.std()), 4),
                                    'min': round(float(target_data.min()), 4),
                                    'max': round(float(target_data.max()), 4),
                                    'valid_samples': int(len(target_data))
                                }
                    except Exception as e:
                        logger.warning(f"⚠️ Error processing target column {col}: {e}")
                        continue
                summary['target_statistics'] = target_stats
        except Exception as e:
            logger.warning(f"⚠️ Error in target statistics: {e}")
            summary['target_statistics'] = {}
        
        # Sentiment data statistics - with safe access
        try:
            sentiment_cols = [str(col) for col in data.columns if any(word in str(col).lower() for word in ['sentiment', 'news', 'content'])]
            if sentiment_cols:
                sentiment_stats = {}
                for col in sentiment_cols:
                    try:
                        if col in data.columns:
                            sentiment_data = data[col].dropna()
                            if len(sentiment_data) > 0:
                                sentiment_stats[str(col)] = {
                                    'mean': round(float(sentiment_data.mean()), 4),
                                    'total': int(sentiment_data.sum()),
                                    'non_zero_count': int((sentiment_data != 0).sum()),
                                    'coverage_percentage': round(float(((sentiment_data != 0).sum() / len(sentiment_data)) * 100), 2)
                                }
                    except Exception as e:
                        logger.warning(f"⚠️ Error processing sentiment column {col}: {e}")
                        continue
                summary['sentiment_statistics'] = sentiment_stats
        except Exception as e:
            logger.warning(f"⚠️ Error in sentiment statistics: {e}")
            summary['sentiment_statistics'] = {}
        
        logger.info(f"📊 Data summary generated successfully for standard structure")
        return summary
        
    except Exception as e:
        logger.error(f"❌ Critical error generating data summary: {e}")
        logger.error(f"❌ Error type: {type(e).__name__}")
        import traceback
        logger.error(f"❌ Traceback: {traceback.format_exc()}")
        return {
            'total_rows': 0,
            'total_columns': 0,
            'error': str(e),
            'error_type': type(e).__name__,
            'standard_structure': {
                'data_location': MAIN_DATASET,
                'backup_location': BACKUP_DIR,
                'data_directory': DATA_DIR
            }
        }

# Additional utility functions for standard structure
def check_dataset_exists(file_path: str = None) -> bool:
    """Check if dataset exists at standard location"""
    if file_path is None:
        file_path = MAIN_DATASET
    return os.path.exists(file_path)

def get_dataset_info(file_path: str = None) -> Dict[str, Any]:
    """Get basic info about dataset at standard location"""
    if file_path is None:
        file_path = MAIN_DATASET
    
    if not os.path.exists(file_path):
        return {'exists': False, 'path': file_path}
    
    try:
        stat = os.stat(file_path)
        return {
            'exists': True,
            'path': file_path,
            'size_mb': round(stat.st_size / (1024 * 1024), 2),
            'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
            'is_readable': os.access(file_path, os.R_OK),
            'is_writable': os.access(file_path, os.W_OK)
        }
    except Exception as e:
        return {'exists': True, 'path': file_path, 'error': str(e)}

def list_backups() -> List[Dict[str, Any]]:
    """List all available backups"""
    backup_dir = Path(BACKUP_DIR)
    if not backup_dir.exists():
        return []
    
    backups = []
    for backup_file in backup_dir.glob('*_backup_*.csv'):
        try:
            stat = backup_file.stat()
            backups.append({
                'name': backup_file.name,
                'path': str(backup_file),
                'size_mb': round(stat.st_size / (1024 * 1024), 2),
                'created': datetime.fromtimestamp(stat.st_mtime).isoformat()
            })
        except Exception as e:
            logger.warning(f"⚠️ Error reading backup {backup_file}: {e}")
    
    # Sort by creation time (newest first)
    backups.sort(key=lambda x: x['created'], reverse=True)
    return backups

if __name__ == "__main__":
    # Test the refactored data collection with standard structure
    print("🧪 Testing Refactored Data Collection (Standard Structure)")
    print("=" * 70)
    
    try:
        # Test configuration
        test_config = {
            'data': {
                'symbols': ['AAPL', 'MSFT'],
                'start_date': '2023-01-01',
                'end_date': '2023-03-31',
                'target_horizons': [5, 30],
                'fnspid_data_dir': DATA_DIR
            }
        }
        
        print(f"📁 Standard data location: {MAIN_DATASET}")
        print(f"💾 Backup location: {BACKUP_DIR}")
        
        # Check current dataset status
        dataset_info = get_dataset_info()
        print(f"📊 Current dataset: {dataset_info}")
        
        # Test dataset collection
        print("\n📊 Testing complete dataset collection...")
        dataset = collect_complete_dataset(test_config)
        
        print(f"✅ Dataset collection successful: {dataset.shape}")
        
        # Test data summary
        summary = get_data_summary(dataset)
        print(f"📋 Data summary generated: {len(summary)} keys")
        
        # List backups
        backups = list_backups()
        print(f"💾 Available backups: {len(backups)}")
        
        print("\n✅ Refactored data collection test completed!")
        print("Key benefits:")
        print("  • Predictable data location")
        print("  • Automatic backup creation")
        print("  • Standard MLOps structure")
        print("  • Simplified path management")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()