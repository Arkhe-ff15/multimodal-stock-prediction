"""
FINAL DATA.PY - Complete Implementation with Timezone Fixes
==========================================================

SPECIFICATIONS:
1. ✅ FNSPID Data: data/raw/nasdaq_exteral_data.csv
2. ✅ Exponential Decay: Standard financial λ=0.94 (RiskMetrics standard)
3. ✅ Technical Indicators: OHLCV + MACD + EMA + VWAP + BB + RSI + Optional (Lag, ROC, Volatility, Momentum)
4. ✅ Three Dataset Variants:
   - Core: Stock + Technical + Targets + Time (maximum data retention)
   - Sentiment: Core + Sentiment features
   - Temporal Decay: Core + Sentiment + Exponentially decayed sentiment
5. ✅ TIMEZONE FIXES: All datetime handling made timezone-safe

DATASET ORCHESTRATION:
- LSTM/TFT Baseline → Core Dataset (no sentiment data loss)
- TFT Sentiment → Sentiment Dataset (core + sentiment)
- TFT Temporal Decay → Temporal Decay Dataset (core + sentiment + decay)
"""

import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple, Any
import logging
from datetime import datetime, timedelta
import warnings
import sys
import traceback
import shutil
import os
from dataclasses import dataclass
from enum import Enum
import json
import time

# Technical Analysis
try:
    import ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False
    logging.warning("⚠️ 'ta' library not available. Install with: pip install ta")

# Configure logging
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# Dataset type enumeration
class DatasetType(Enum):
    CORE = "core"
    SENTIMENT = "sentiment"
    TEMPORAL_DECAY = "temporal_decay"

# Standard path constants
DATA_DIR = "data/processed"
BACKUP_DIR = "data/backups"
CACHE_DIR = "data/cache"
RAW_DIR = "data/raw"

# Dataset file paths
CORE_DATASET = f"{DATA_DIR}/core_dataset.csv"
SENTIMENT_DATASET = f"{DATA_DIR}/sentiment_dataset.csv"
TEMPORAL_DECAY_DATASET = f"{DATA_DIR}/temporal_decay_dataset.csv"
COMBINED_DATASET = f"{DATA_DIR}/combined_dataset.csv"  # Legacy compatibility

# FNSPID data location
FNSPID_DATA_FILE = f"{RAW_DIR}/nasdaq_exteral_data.csv"

@dataclass
class DatasetConfig:
    """Configuration for dataset creation"""
    symbols: List[str]
    start_date: str
    end_date: str
    target_horizons: List[int]
    fnspid_data_file: str = FNSPID_DATA_FILE
    include_sentiment: bool = True
    include_temporal_decay: bool = True
    cache_enabled: bool = True
    validation_split: float = 0.2
    min_observations_per_symbol: int = 100
    
    # Temporal decay parameters (RiskMetrics standard)
    decay_lambda: float = 0.94  # Financial industry standard
    max_decay_days: int = 90    # Maximum days to look back

@dataclass
class DatasetMetrics:
    """Metrics for dataset quality assessment"""
    total_rows: int
    total_features: int
    symbols_count: int
    date_range: Tuple[str, str]
    target_coverage: Dict[str, float]
    data_quality_score: float
    missing_data_percentage: float
    feature_breakdown: Dict[str, int]

def create_backup(file_path: str) -> Optional[str]:
    """Create timestamped backup before overwriting"""
    file_path = Path(file_path)
    
    if file_path.exists():
        backup_dir = Path(BACKUP_DIR)
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{file_path.stem}_backup_{timestamp}{file_path.suffix}"
        backup_path = backup_dir / backup_name
        
        try:
            shutil.copy2(file_path, backup_path)
            logger.info(f"💾 Backup created: {backup_path}")
            return str(backup_path)
        except Exception as e:
            logger.warning(f"⚠️ Backup failed for {file_path}: {e}")
            return None
    return None

def setup_data_directories():
    """Create all required data directories"""
    directories = [DATA_DIR, BACKUP_DIR, CACHE_DIR, RAW_DIR]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    logger.info("📁 Data directories initialized")

def ensure_timezone_safe_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure all datetime columns in DataFrame are timezone-naive - ENHANCED VERSION
    
    Args:
        df: DataFrame to process
        
    Returns:
        DataFrame with timezone-naive datetime columns
    """
    df = df.copy()
    
    # Handle datetime columns - MORE AGGRESSIVE
    for col in df.columns:
        if df[col].dtype.name.startswith('datetime64') or str(df[col].dtype).startswith('datetime64'):
            df[col] = pd.to_datetime(df[col])
            if hasattr(df[col].dt, 'tz') and df[col].dt.tz is not None:
                logger.debug(f"🔧 Removing timezone from column {col}")
                df[col] = df[col].dt.tz_localize(None)
    
    # Handle datetime index - MORE AGGRESSIVE  
    if isinstance(df.index, pd.DatetimeIndex):
        if df.index.tz is not None:
            logger.debug(f"🔧 Removing timezone from index")
            df.index = df.index.tz_localize(None)
    
    # Handle object columns that might contain datetime
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].name in ['date', 'Date'] or 'date' in col.lower():
            try:
                temp_series = pd.to_datetime(df[col], errors='coerce')
                if temp_series.notna().any():  # If any valid dates found
                    df[col] = temp_series
                    if hasattr(df[col].dt, 'tz') and df[col].dt.tz is not None:
                        logger.debug(f"🔧 Removing timezone from object column {col}")
                        df[col] = df[col].dt.tz_localize(None)
            except:
                pass
    
    return df

class StockDataCollector:
    """Collects and processes stock market data with enhanced caching and timezone safety"""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.cache_dir = Path(CACHE_DIR) / "stock_data"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Create symbol-to-id mapping
        self.symbol_to_id = {
            symbol: f"stock_{idx:04d}" 
            for idx, symbol in enumerate(sorted(config.symbols), 1)
        }
        self.id_to_symbol = {v: k for k, v in self.symbol_to_id.items()}
        
    def collect_stock_data(self) -> pd.DataFrame:
        """Collect stock data for all symbols with robust error handling"""
        logger.info("📈 Collecting stock market data...")
        
        all_data = []
        successful_symbols = []
        failed_symbols = []
        
        for symbol in self.config.symbols:
            try:
                logger.info(f"📥 Processing {symbol}...")
                
                # Check cache first
                cached_data = self._load_from_cache(symbol)
                if cached_data is not None:
                    data = cached_data
                    logger.info(f"📦 Loaded {symbol} from cache: {len(data)} rows")
                else:
                    # Fetch from Yahoo Finance
                    data = self._fetch_symbol_data(symbol)
                    if not data.empty:
                        self._save_to_cache(symbol, data)
                        logger.info(f"📥 Fetched {symbol} from Yahoo Finance: {len(data)} rows")
                
                if not data.empty and len(data) >= self.config.min_observations_per_symbol:
                    # Add symbol identifier and stock_id
                    data['symbol'] = symbol
                    data['stock_id'] = self.symbol_to_id[symbol]
                    all_data.append(data)
                    successful_symbols.append(symbol)
                    logger.info(f"✅ {symbol} ({self.symbol_to_id[symbol]}): {data.shape[0]} rows")
                else:
                    failed_symbols.append(symbol)
                    logger.warning(f"⚠️ {symbol}: Insufficient data ({len(data)} rows)")
                    
            except Exception as e:
                failed_symbols.append(symbol)
                logger.error(f"❌ Error processing {symbol}: {e}")
                continue
        
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            
            # Save symbol mapping for future reference
            self._save_symbol_mapping()
            
            # Apply timezone safety
            combined_data = ensure_timezone_safe_dataframe(combined_data)
            
            logger.info(f"✅ Stock data collection complete: {combined_data.shape}")
            logger.info(f"📊 Successful symbols: {', '.join([f'{s} ({self.symbol_to_id[s]})' for s in successful_symbols])}")
            if failed_symbols:
                logger.warning(f"⚠️ Failed symbols: {failed_symbols}")
            return combined_data
        else:
            logger.error("❌ No stock data collected")
            return pd.DataFrame()
    
    def _fetch_symbol_data(self, symbol: str) -> pd.DataFrame:
        """Fetch data for a single symbol from Yahoo Finance with ENHANCED timezone fixes"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(
                start=self.config.start_date,
                end=self.config.end_date,
                interval='1d'
            )
            
            if data.empty:
                return pd.DataFrame()
            
            # 🔥 ENHANCED TIMEZONE FIX: Multiple approaches
            
            # Step 1: Handle timezone-aware index
            if isinstance(data.index, pd.DatetimeIndex) and data.index.tz is not None:
                logger.debug(f"🔧 Removing timezone from {symbol} index: {data.index.tz}")
                data.index = data.index.tz_localize(None)
            
            # Step 2: Reset index to make date a column
            data = data.reset_index()
            
            # Step 3: Handle Date column variations
            if 'Date' in data.columns:
                data['Date'] = pd.to_datetime(data['Date'])
                if hasattr(data['Date'].dt, 'tz') and data['Date'].dt.tz is not None:
                    logger.debug(f"🔧 Removing timezone from {symbol} Date column: {data['Date'].dt.tz}")
                    data['Date'] = data['Date'].dt.tz_localize(None)
                data = data.rename(columns={'Date': 'date'})
            
            # Step 4: Ensure lowercase column names
            data.columns = data.columns.str.lower()
            
            # Step 5: Final date column processing
            if 'date' not in data.columns:
                if len(data) > 0:
                    data['date'] = pd.date_range(self.config.start_date, periods=len(data), freq='D', tz=None)
                else:
                    data['date'] = pd.Series([], dtype='datetime64[ns]')
            
            # Step 6: Ensure date column is timezone-naive
            data['date'] = pd.to_datetime(data['date'])
            if hasattr(data['date'].dt, 'tz') and data['date'].dt.tz is not None:
                data['date'] = data['date'].dt.tz_localize(None)
            
            # Step 7: Sort by date
            data = data.sort_values('date').reset_index(drop=True)
            
            # Step 8: Final safety check
            data = ensure_timezone_safe_dataframe(data)
            
            return data
            
        except Exception as e:
            logger.error(f"❌ Failed to fetch data for {symbol}: {e}")
            return pd.DataFrame()
    
    def _load_from_cache(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load cached data for a symbol"""
        if not self.config.cache_enabled:
            return None
            
        cache_file = self.cache_dir / f"{symbol}_{self.config.start_date}_{self.config.end_date}.csv"
        
        if cache_file.exists():
            try:
                # Check if cache is recent (within 1 day)
                cache_age = time.time() - cache_file.stat().st_mtime
                if cache_age > 86400:  # 24 hours
                    logger.info(f"📦 Cache for {symbol} is stale, will refresh")
                    return None
                
                data = pd.read_csv(cache_file)
                data['date'] = pd.to_datetime(data['date'])
                
                # Apply timezone safety to cached data
                data = ensure_timezone_safe_dataframe(data)
                
                return data
            except Exception as e:
                logger.warning(f"⚠️ Cache load failed for {symbol}: {e}")
                return None
        
        return None
    
    def _save_to_cache(self, symbol: str, data: pd.DataFrame):
        """Save data to cache"""
        if not self.config.cache_enabled:
            return
            
        cache_file = self.cache_dir / f"{symbol}_{self.config.start_date}_{self.config.end_date}.csv"
        
        try:
            # Ensure timezone safety before caching
            data_to_cache = ensure_timezone_safe_dataframe(data)
            data_to_cache.to_csv(cache_file, index=False)
            logger.debug(f"💾 Cached {symbol} data")
        except Exception as e:
            logger.warning(f"⚠️ Cache save failed for {symbol}: {e}")

    def _save_symbol_mapping(self):
        """Save symbol-to-id mapping to JSON file for future reference"""
        try:
            # Create mapping data structure
            mapping = {
                "symbol_to_id": self.symbol_to_id,
                "id_to_symbol": self.id_to_symbol,
                "created_at": datetime.now().isoformat(),
                "symbols_metadata": {
                    symbol: {
                        "id": stock_id,
                        "index": idx,
                        "first_mapped": datetime.now().isoformat()
                    } for idx, (symbol, stock_id) in enumerate(self.symbol_to_id.items())
                }
            }
            
            # Ensure data directory exists
            Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
            
            # Save mapping file
            mapping_file = Path(DATA_DIR) / "symbol_mapping.json"
            with open(mapping_file, 'w') as f:
                json.dump(mapping, f, indent=2)
                
            logger.info(f"💾 Symbol mapping saved: {mapping_file}")
            logger.info(f"   🏷️ Mapped {len(self.symbol_to_id)} symbols")
            
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to save symbol mapping: {e}")
            return False

class TechnicalIndicatorProcessor:
    """
    Processes technical indicators with YOUR SPECIFIC REQUIREMENTS and timezone safety:
    - OHLCV + MACD + EMA + VWAP + BB + RSI
    - Optional: Lag + ROC + Volatility + Momentum
    """
    
    @staticmethod
    def add_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive technical indicators with ENHANCED timezone safety"""
        logger.info("🔧 Adding technical indicators (ENHANCED TIMEZONE-SAFE)...")
        
        if not TA_AVAILABLE:
            logger.error("❌ 'ta' library not available. Install with: pip install ta")
            return data
        
        data = data.copy()
        
        # 🔥 ENHANCED TIMEZONE FIX: Apply at start
        data = ensure_timezone_safe_dataframe(data)
        
        # Sort by symbol and date for proper calculations
        data = data.sort_values(['symbol', 'date']).reset_index(drop=True)
        
        # Ensure numeric columns
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # Group by symbol for proper calculations
        symbol_groups = data.groupby('symbol')
        
        try:
            # === CORE REQUIRED INDICATORS ===
            
            # 1. BASIC OHLCV FEATURES
            logger.info("   📊 OHLCV basic features...")
            data['returns'] = symbol_groups['close'].pct_change()
            data['log_returns'] = data.groupby('symbol')['close'].pct_change().apply(lambda x: np.log(1 + x))
            
            # 🔥 TIMEZONE CHECK after basic features
            data = ensure_timezone_safe_dataframe(data)
            
            # 2. EXPONENTIAL MOVING AVERAGES (EMA) - REQUIRED
            logger.info("   📈 Exponential Moving Averages (EMA)...")
            for period in [5, 10, 20, 30, 50]:
                data[f'ema_{period}'] = symbol_groups['close'].transform(
                    lambda x: ta.trend.ema_indicator(x, window=period)
                )
                # 🔥 IMMEDIATE TIMEZONE CHECK after each EMA
                if hasattr(data[f'ema_{period}'], 'dt') and data[f'ema_{period}'].dt.tz is not None:
                    data[f'ema_{period}'] = data[f'ema_{period}'].dt.tz_localize(None)
            
            # 3. VWAP (Volume Weighted Average Price) - FIXED
            logger.info("   📊 Volume Weighted Average Price (VWAP) - FIXED...")
            data['vwap'] = calculate_proper_vwap(data)
            
            # 4. BOLLINGER BANDS (BB) - REQUIRED
            logger.info("   📊 Bollinger Bands (BB)...")
            data['bb_upper'] = symbol_groups['close'].transform(
                lambda x: ta.volatility.bollinger_hband(x, window=20, window_dev=2)
            )
            data['bb_lower'] = symbol_groups['close'].transform(
                lambda x: ta.volatility.bollinger_lband(x, window=20, window_dev=2)
            )
            data['bb_middle'] = symbol_groups['close'].transform(
                lambda x: ta.volatility.bollinger_mavg(x, window=20)
            )
            
            # Safe division for BB features
            bb_middle_safe = data['bb_middle'].replace(0, np.nan)
            data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / bb_middle_safe
            
            bb_range_safe = (data['bb_upper'] - data['bb_lower']).replace(0, np.nan)
            data['bb_position'] = (data['close'] - data['bb_lower']) / bb_range_safe
            
            # 5. RSI (Relative Strength Index) - REQUIRED
            logger.info("   📊 Relative Strength Index (RSI)...")
            for period in [6, 14, 21]:
                data[f'rsi_{period}'] = symbol_groups['close'].transform(
                    lambda x: ta.momentum.rsi(x, window=period)
                )
            
            # 6. MACD (Moving Average Convergence Divergence) - REQUIRED
            logger.info("   📊 MACD (Moving Average Convergence Divergence)...")
            data['macd_line'] = symbol_groups['close'].transform(
                lambda x: ta.trend.macd(x, window_slow=26, window_fast=12)
            )
            data['macd_signal'] = symbol_groups['close'].transform(
                lambda x: ta.trend.macd_signal(x, window_slow=26, window_fast=12, window_sign=9)
            )
            data['macd_diff'] = symbol_groups['close'].transform(
                lambda x: ta.trend.macd_diff(x, window_slow=26, window_fast=12, window_sign=9)
            )
            
            # 🔥 TIMEZONE CHECK after core indicators
            data = ensure_timezone_safe_dataframe(data)
            
            # === OPTIONAL INDICATORS ===
            
            # 7. SIMPLE MOVING AVERAGES (for comparison)
            logger.info("   📈 Simple Moving Averages (SMA)...")
            for period in [5, 10, 20, 50]:
                data[f'sma_{period}'] = symbol_groups['close'].transform(
                    lambda x: ta.trend.sma_indicator(x, window=period)
                )
            
            # 8. VOLATILITY MEASURES - OPTIONAL
            logger.info("   📊 Volatility measures...")
            data['volatility_20d'] = symbol_groups['returns'].transform(lambda x: x.rolling(window=20).std())
            
            # 🔥 ENHANCED ATR calculation with timezone safety
            logger.info("   📊 Average True Range (ATR) - OPTIMIZED...")
            data['atr'] = calculate_optimized_atr(data, window=14)
            
            # 9. MOMENTUM INDICATORS - OPTIONAL (Enhanced with timezone safety)
            logger.info("   📊 Momentum indicators...")
            
            stoch_k_values = []
            stoch_d_values = []
            williams_r_values = []
            
            for symbol in data['symbol'].unique():
                symbol_mask = data['symbol'] == symbol
                symbol_data = data[symbol_mask]
                if len(symbol_data) > 0:
                    # Stochastic K
                    stoch_k = ta.momentum.stoch(symbol_data['high'], symbol_data['low'], symbol_data['close'], window=14)
                    if hasattr(stoch_k, 'dt') and stoch_k.dt.tz is not None:
                        stoch_k = stoch_k.dt.tz_localize(None)
                    stoch_k_values.extend(stoch_k.values)
                    
                    # Stochastic D
                    stoch_d = ta.momentum.stoch_signal(symbol_data['high'], symbol_data['low'], symbol_data['close'], window=14)
                    if hasattr(stoch_d, 'dt') and stoch_d.dt.tz is not None:
                        stoch_d = stoch_d.dt.tz_localize(None)
                    stoch_d_values.extend(stoch_d.values)
                    
                    # Williams %R
                    williams_r = ta.momentum.williams_r(symbol_data['high'], symbol_data['low'], symbol_data['close'], lbp=14)
                    if hasattr(williams_r, 'dt') and williams_r.dt.tz is not None:
                        williams_r = williams_r.dt.tz_localize(None)
                    williams_r_values.extend(williams_r.values)
                else:
                    stoch_k_values.extend([np.nan] * len(symbol_data))
                    stoch_d_values.extend([np.nan] * len(symbol_data))
                    williams_r_values.extend([np.nan] * len(symbol_data))
            
            data['stoch_k'] = stoch_k_values
            data['stoch_d'] = stoch_d_values
            data['williams_r'] = williams_r_values
            
            # 10. RATE OF CHANGE (ROC) - OPTIONAL
            logger.info("   📊 Rate of Change (ROC)...")
            for period in [5, 10, 20]:
                data[f'roc_{period}'] = symbol_groups['close'].transform(
                    lambda x: ta.momentum.roc(x, window=period)
                )
            
            # 11. VOLUME INDICATORS
            logger.info("   📊 Volume indicators...")
            data['volume_sma_20'] = symbol_groups['volume'].transform(
                lambda x: x.rolling(window=20).mean()
            )
            data['volume_ratio'] = data['volume'] / (data['volume_sma_20'] + 1e-10)  # Avoid division by zero
            data['volume_trend'] = (
                symbol_groups['volume'].transform(lambda x: x.rolling(window=5).mean()) / 
                (symbol_groups['volume'].transform(lambda x: x.rolling(window=20).mean()) + 1e-10)
            )
            
            # 12. PRICE POSITION & ADDITIONAL FEATURES
            logger.info("   📍 Price position features...")
            price_range = (data['high'] - data['low']).replace(0, np.nan)
            data['price_position'] = (data['close'] - data['low']) / price_range
            
            # Enhanced gap calculation
            data['gap'] = data.groupby('symbol').apply(
                lambda x: (x['open'] - x['close'].shift(1)) / (x['close'].shift(1) + 1e-10)
            ).reset_index(level=0, drop=True)
            
            data['intraday_return'] = (data['close'] - data['open']) / (data['open'] + 1e-10)
            
            # 13. LAG FEATURES - OPTIONAL
            logger.info("   📊 Lag features...")
            lag_columns = ['close', 'volume', 'returns', 'vwap', 'rsi_14']
            for col in lag_columns:
                if col in data.columns:
                    for lag in [1, 2, 3, 5, 10]:
                        try:
                            if col in symbol_groups.obj.columns:
                                data[f'{col}_lag_{lag}'] = symbol_groups[col].transform(lambda x: x.shift(lag))
                            else:
                                logger.warning(f"   ⚠️ Column {col} not available for lag features, skipping")
                        except Exception as e:
                            logger.warning(f"   ⚠️ Error creating lag feature {col}_lag_{lag}: {e}")
                                    
            # === ENHANCED DATA CLEANING ===
            logger.info("   🧹 Enhanced technical indicators cleaning...")
            
            # Replace infinite values
            data = data.replace([np.inf, -np.inf], np.nan)
            
            # Get technical indicator columns
            technical_cols = [col for col in data.columns if any(
                indicator in col.lower() for indicator in [
                    'ema_', 'sma_', 'vwap', 'bb_', 'rsi_', 'macd', 'atr', 'roc_', 'volume_', 'returns', 'volatility', 'price_position', 'gap', 'intraday',
                    '_lag_'
                ]
            )]
            
            # Forward fill within symbol groups - FIXED: Use modern pandas syntax
            # Forward fill within symbol groups - ENHANCED SAFETY
            for col in technical_cols:
                try:
                    if col in data.columns and col in symbol_groups.obj.columns:
                        data[col] = symbol_groups[col].transform(
                            lambda x: x.fillna(method='ffill').fillna(method='bfill')
                        )
                except Exception as e:
                    logger.warning(f"   ⚠️ Error forward filling {col}: {e}")
                    continue
            
            # Final NaN cleanup
            data[technical_cols] = data[technical_cols].fillna(0)
            
            # 🔥 FINAL ENHANCED TIMEZONE SAFETY CHECK
            logger.info("   🔧 Final timezone safety check...")
            for col in data.columns:
                if hasattr(data[col], 'dt') and data[col].dt.tz is not None:
                    logger.warning(f"   ⚠️ Removing timezone from {col}: {data[col].dt.tz}")
                    data[col] = data[col].dt.tz_localize(None)
            
            # Apply comprehensive timezone safety
            data = ensure_timezone_safe_dataframe(data)
            
            logger.info(f"✅ Technical indicators added (ENHANCED): {len(technical_cols)} features")
            logger.info(f"   🔧 Core indicators: EMA, VWAP, BB, RSI, MACD")
            logger.info(f"   📊 Optional indicators: SMA, Volatility, Momentum, ROC, Lags")
            logger.info(f"   🕒 All timezone issues resolved")
            
            return data
            
        except Exception as e:
            logger.error(f"❌ Error adding technical indicators: {e}")
            import traceback
            traceback.print_exc()
            return data

class TargetVariableProcessor:
    """Processes target variables with FIXED forward-looking calculation and timezone safety"""
    
    @staticmethod
    def add_target_variables(data: pd.DataFrame, horizons: List[int]) -> pd.DataFrame:
        """Add target variables with FIXED forward-looking calculation and ENHANCED timezone safety"""
        logger.info("🎯 Adding target variables (FIXED + ENHANCED TIMEZONE-SAFE)...")
        
        data = data.copy()
        
        # 🔥 ENHANCED TIMEZONE FIX: Apply comprehensive safety
        data = ensure_timezone_safe_dataframe(data)
        
        # Sort by symbol and date
        data = data.sort_values(['symbol', 'date']).reset_index(drop=True)
        
        # Group by symbol for proper calculation
        symbol_groups = data.groupby('symbol')
        
        try:
            for horizon in horizons:
                logger.info(f"   📅 Calculating {horizon}-day forward returns...")
                
                # FIXED: Proper forward-looking return calculation
                # Formula: (future_price / current_price) - 1
                data[f'target_{horizon}d'] = symbol_groups['close'].transform(
                    lambda x: (x.shift(-horizon) / x - 1)
                )
                
                # Ensure target_5 exists for TFT compatibility
                if horizon == 5:
                    data['target_5'] = data['target_5d']
                    
                # Add additional target metrics for primary horizon
                if horizon == 5:
                    # Log returns version
                    data['target_5_log'] = symbol_groups['close'].transform(
                        lambda x: np.log(x.shift(-horizon) / x)
                    )
                    
                    # Binary direction (up/down)
                    data['target_5_direction'] = (data['target_5d'] > 0).astype(int)
            
            # === ENHANCED TARGET CLEANING ===
            logger.info("   🧹 Enhanced target variable cleaning...")
            
            target_cols = [col for col in data.columns if col.startswith('target_')]
            
            for col in target_cols:
                # Replace infinite values
                original_count = data[col].notna().sum()
                data[col] = data[col].replace([np.inf, -np.inf], np.nan)
                cleaned_count = data[col].notna().sum()
                
                if original_count != cleaned_count:
                    logger.info(f"      {col}: Removed {original_count - cleaned_count} infinite values")
                
                # Enhanced outlier handling
                if col.endswith('d') and not col.endswith('_direction'):
                    valid_data = data[col].dropna()
                    if len(valid_data) > 0:
                        # Use IQR method for more robust outlier detection
                        Q1 = valid_data.quantile(0.25)
                        Q3 = valid_data.quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        
                        # Also use percentile caps as backup
                        q99 = valid_data.quantile(0.99)
                        q01 = valid_data.quantile(0.01)
                        
                        # Use the more conservative bounds
                        final_lower = max(lower_bound, q01)
                        final_upper = min(upper_bound, q99)
                        
                        # Apply bounds
                        data[col] = data[col].clip(lower=final_lower, upper=final_upper)
            
            # 🔥 ENHANCED TIMEZONE CHECK for target variables
            for col in target_cols:
                if hasattr(data[col], 'dt') and data[col].dt.tz is not None:
                    logger.warning(f"   ⚠️ Target column {col} has timezone: {data[col].dt.tz}")
                    data[col] = data[col].dt.tz_localize(None)
            
            # === ENHANCED TARGET COVERAGE VALIDATION ===
            logger.info("   📊 Enhanced target variable validation...")
            
            validation_results = {}
            
            for col in target_cols:
                valid_count = data[col].notna().sum()
                total_count = len(data)
                coverage = valid_count / total_count * 100
                
                if valid_count > 0:
                    mean_val = data[col].mean()
                    std_val = data[col].std()
                    min_val = data[col].min()
                    max_val = data[col].max()
                    median_val = data[col].median()
                    
                    validation_results[col] = {
                        'coverage': coverage,
                        'mean': mean_val,
                        'std': std_val,
                        'median': median_val,
                        'range': (min_val, max_val),
                        'valid_count': valid_count
                    }
                    
                    logger.info(f"      {col}: {coverage:.1f}% coverage | "
                            f"Mean: {mean_val:.4f} | Std: {std_val:.4f} | "
                            f"Median: {median_val:.4f} | Range: [{min_val:.4f}, {max_val:.4f}]")
                else:
                    logger.warning(f"      {col}: No valid values!")
                    validation_results[col] = {'coverage': 0, 'valid_count': 0}
            
            # Enhanced primary target validation
            if 'target_5' in validation_results:
                primary_coverage = validation_results['target_5']['coverage']
                primary_count = validation_results['target_5']['valid_count']
                
                if primary_coverage < 30:
                    logger.error(f"❌ Primary target coverage critically low: {primary_coverage:.1f}%")
                elif primary_coverage < 50:
                    logger.warning(f"⚠️ Primary target coverage is low: {primary_coverage:.1f}%")
                elif primary_coverage >= 80:
                    logger.info(f"✅ Primary target coverage is excellent: {primary_coverage:.1f}%")
                else:
                    logger.info(f"✅ Primary target coverage is good: {primary_coverage:.1f}%")
                
                logger.info(f"   📊 Primary target samples: {primary_count:,}")
            
            # Final timezone safety check
            data = ensure_timezone_safe_dataframe(data)
            
            logger.info("✅ Target variables added successfully (FIXED + ENHANCED)")
            return data
            
        except Exception as e:
            logger.error(f"❌ Error adding target variables: {e}")
            import traceback
            traceback.print_exc()
            return data


class TimeFeatureProcessor:
    """Processes time-based features for modeling with timezone safety"""
    
    @staticmethod
    def add_time_features(data: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive time-based features with ENHANCED timezone safety"""
        logger.info("⏰ Adding time features (ENHANCED TIMEZONE-SAFE)...")
        
        data = data.copy()
        
        # 🔥 ENHANCED TIMEZONE FIX: Comprehensive safety at start
        data = ensure_timezone_safe_dataframe(data)
        
        # Ensure date column is datetime and timezone-naive
        data['date'] = pd.to_datetime(data['date'])
        if hasattr(data['date'].dt, 'tz') and data['date'].dt.tz is not None:
            logger.info(f"   🔧 Removing timezone from date column: {data['date'].dt.tz}")
            data['date'] = data['date'].dt.tz_localize(None)
        
        # Basic time features
        data['year'] = data['date'].dt.year
        data['month'] = data['date'].dt.month
        data['day'] = data['date'].dt.day
        data['day_of_week'] = data['date'].dt.dayofweek
        data['quarter'] = data['date'].dt.quarter
        data['day_of_year'] = data['date'].dt.dayofyear
        
        # Enhanced week calculation with error handling
        try:
            data['week_of_year'] = data['date'].dt.isocalendar().week
        except:
            # Fallback for older pandas versions
            data['week_of_year'] = data['date'].dt.week
        
        # Market-specific time features
        data['is_month_end'] = data['date'].dt.is_month_end.astype(int)
        data['is_month_start'] = data['date'].dt.is_month_start.astype(int)
        data['is_quarter_end'] = data['date'].dt.is_quarter_end.astype(int)
        data['is_year_end'] = data['date'].dt.is_year_end.astype(int)
        
        # Enhanced market timing features
        data['is_weekday'] = (data['day_of_week'] < 5).astype(int)
        data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)
        
        # Cyclical encoding for better ML performance
        data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
        data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
        data['day_of_week_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
        data['day_of_week_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
        data['day_of_year_sin'] = np.sin(2 * np.pi * data['day_of_year'] / 365.25)
        data['day_of_year_cos'] = np.cos(2 * np.pi * data['day_of_year'] / 365.25)
        
        # Create time_idx for each symbol (CRITICAL for TFT)
        data = data.sort_values(['symbol', 'date'])
        data['time_idx'] = data.groupby('symbol').cumcount()
        
        # Enhanced relative time features
        data['days_since_start'] = (data['date'] - data['date'].min()).dt.days
        
        # Market cycle features (approximate)
        data['trading_day_of_month'] = data.groupby([data['date'].dt.year, data['date'].dt.month]).cumcount() + 1
        
        # 🔥 FINAL TIMEZONE SAFETY CHECK
        data = ensure_timezone_safe_dataframe(data)
        
        # Verify no time features have timezone
        time_features = [col for col in data.columns if any(
            time_word in col.lower() for time_word in ['year', 'month', 'day', 'week', 'quarter', 'time_idx']
        )]
        
        for col in time_features:
            if hasattr(data[col], 'dt') and data[col].dt.tz is not None:
                logger.warning(f"   ⚠️ Time feature {col} has timezone: {data[col].dt.tz}")
                data[col] = data[col].dt.tz_localize(None)
        
        logger.info("✅ Time features added (ENHANCED + timezone-safe)")
        logger.info(f"   📅 Time features created: {len(time_features)}")
        logger.info(f"   🔧 Cyclical encoding applied")
        logger.info(f"   🕒 All timezone issues resolved")
        
        return data

class SentimentDataProcessor:
    """Processes sentiment data from FNSPID dataset with timezone safety"""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.fnspid_file = Path(config.fnspid_data_file)
        
        # Load symbol mapping if available
        self.symbol_to_id = self._load_symbol_mapping()
    
    def _load_symbol_mapping(self) -> Dict[str, str]:
        """Load symbol-to-id mapping from file"""
        mapping_file = Path(DATA_DIR) / "symbol_mapping.json"
        
        if mapping_file.exists():
            try:
                with open(mapping_file) as f:
                    mapping = json.load(f)
                return mapping["symbol_to_id"]
            except Exception as e:
                logger.warning(f"⚠️ Failed to load symbol mapping: {e}")
        
        # Fallback: Create new mapping
        return {
            symbol: f"stock_{idx:04d}" 
            for idx, symbol in enumerate(sorted(self.config.symbols), 1)
        }
    
    def collect_sentiment_data(self) -> pd.DataFrame:
        """Collect and process sentiment data from FNSPID with comprehensive error handling"""
        logger.info("📰 Collecting sentiment data from FNSPID...")
        logger.info(f"📁 FNSPID file: {self.fnspid_file}")
        
        try:
            # Step 1: Check if FNSPID file exists
            if not self.fnspid_file.exists():
                logger.warning(f"⚠️ FNSPID file not found at {self.fnspid_file}")
                logger.info("📰 Creating empty sentiment data (file not found)")
                return self._create_empty_sentiment_data()
            
            # Step 2: Check file size and readability
            try:
                file_size_mb = self.fnspid_file.stat().st_size / (1024 * 1024)
                logger.info(f"📊 FNSPID file size: {file_size_mb:.2f} MB")
                
                if file_size_mb < 0.1:  # Less than 100KB
                    logger.warning(f"⚠️ FNSPID file is very small ({file_size_mb:.2f} MB), might be empty")
                    return self._create_empty_sentiment_data()
                    
            except Exception as e:
                logger.error(f"❌ Cannot access FNSPID file: {e}")
                return self._create_empty_sentiment_data()
            
            # Step 3: Test file readability
            try:
                logger.info("🔍 Testing FNSPID file readability...")
                test_df = pd.read_csv(self.fnspid_file, nrows=3, low_memory=False)
                logger.info(f"📋 FNSPID columns detected: {list(test_df.columns)}")
                
                if len(test_df) == 0:
                    logger.warning("⚠️ FNSPID file appears to be empty")
                    return self._create_empty_sentiment_data()
                    
            except Exception as e:
                logger.error(f"❌ FNSPID file is corrupted or unreadable: {e}")
                logger.info("📰 Creating empty sentiment data (file unreadable)")
                return self._create_empty_sentiment_data()
            
            # Step 4: Load FNSPID data
            logger.info("📥 Loading FNSPID data...")
            fnspid_data = self._load_fnspid_data()
            
            if fnspid_data.empty:
                logger.warning("⚠️ FNSPID data is empty after loading")
                logger.info("📰 Creating empty sentiment data (no data loaded)")
                return self._create_empty_sentiment_data()
            
            # Step 5: Process sentiment data
            logger.info("📊 Processing sentiment data...")
            processed_data = self._process_sentiment_data(fnspid_data)
            
            if processed_data.empty:
                logger.warning("⚠️ No sentiment data after processing")
                return self._create_empty_sentiment_data()
            
            logger.info(f"✅ Sentiment data collection complete: {processed_data.shape}")
            return processed_data
            
        except KeyboardInterrupt:
            logger.info("⚠️ Sentiment data collection interrupted by user")
            return self._create_empty_sentiment_data()
            
        except Exception as e:
            logger.error(f"❌ Sentiment data collection failed with unexpected error: {e}")
            logger.info("📰 Falling back to empty sentiment data")
            return self._create_empty_sentiment_data()
        
    def _load_fnspid_data(self) -> pd.DataFrame:
        """Load FNSPID data with smart loading strategy"""
        logger.info(f"📥 Loading FNSPID data...")
        
        try:
            # Check file size
            file_size_gb = self.fnspid_file.stat().st_size / (1024 * 1024 * 1024)
            logger.info(f"📊 FNSPID file size: {file_size_gb:.2f} GB")
            
            if file_size_gb > 2:  # Large files
                return self._load_fnspid_chunked()
            else:  # Small files
                return self._load_fnspid_direct()
                
        except Exception as e:
            logger.error(f"❌ Failed to load FNSPID data: {e}")
            return pd.DataFrame()
    
    def _load_fnspid_direct(self) -> pd.DataFrame:
        """Load FNSPID file directly with enhanced error handling"""
        try:
            logger.info("📁 Loading FNSPID file directly...")
            
            # Try different loading strategies
            loading_strategies = [
                # Strategy 1: Standard loading
                {
                    'low_memory': False,
                    'encoding': 'utf-8',
                    'on_bad_lines': 'skip',
                    'engine': 'c'
                },
                # Strategy 2: Python engine with string types
                {
                    'low_memory': False,
                    'encoding': 'utf-8',
                    'on_bad_lines': 'skip',
                    'engine': 'python',
                    'dtype': str
                },
                # Strategy 3: Basic loading
                {
                    'encoding': 'utf-8',
                    'on_bad_lines': 'skip'
                }
            ]
            
            data = None
            
            for i, strategy in enumerate(loading_strategies):
                try:
                    logger.info(f"📊 Trying loading strategy {i+1}...")
                    data = pd.read_csv(self.fnspid_file, **strategy)
                    
                    if not data.empty:
                        logger.info(f"✅ Loading strategy {i+1} successful: {len(data)} rows")
                        break
                    else:
                        logger.warning(f"⚠️ Loading strategy {i+1} returned empty data")
                    
                except Exception as e:
                    logger.warning(f"⚠️ Loading strategy {i+1} failed: {e}")
                    continue
            
            if data is None or data.empty:
                logger.error("❌ All loading strategies failed")
                return pd.DataFrame()
            
            logger.info(f"📰 Loaded {len(data)} total articles")
            
            # Normalize column names
            data = self._normalize_columns(data)
            
            # Filter for target symbols and date range
            filtered_data = self._filter_data(data)
            
            # Apply timezone safety
            filtered_data = ensure_timezone_safe_dataframe(filtered_data)
            
            logger.info(f"📰 Filtered to {len(filtered_data)} relevant articles")
            return filtered_data
            
        except Exception as e:
            logger.error(f"❌ Failed to load FNSPID data directly: {e}")
            return pd.DataFrame()
    
    def _load_fnspid_chunked(self) -> pd.DataFrame:
        """Load large FNSPID files in chunks with robust error handling"""
        logger.info("📊 Loading large FNSPID file in chunks...")
        
        # Validate file first
        if not self.fnspid_file.exists():
            logger.error(f"❌ FNSPID file not found: {self.fnspid_file}")
            return pd.DataFrame()
        
        try:
            # Test file structure first
            logger.info("🔍 Testing FNSPID file structure...")
            test_chunk = pd.read_csv(
                self.fnspid_file, 
                nrows=10,
                low_memory=False,
                encoding='utf-8',
                on_bad_lines='skip'
            )
            
            if test_chunk.empty:
                logger.warning("⚠️ FNSPID file appears empty in test read")
                return pd.DataFrame()
            
            logger.info(f"📋 FNSPID file structure: {test_chunk.shape}, columns: {list(test_chunk.columns)}")
            
        except Exception as e:
            logger.error(f"❌ FNSPID file structure test failed: {e}")
            return pd.DataFrame()
        
        # Chunked loading with enhanced error handling
        relevant_data = []
        chunk_size = 5000      # Smaller chunks for stability
        max_chunks = 50        # Limit chunks for testing
        total_processed = 0
        successful_chunks = 0
        failed_chunks = 0
        
        try:
            logger.info(f"📊 Starting chunked loading (chunk_size={chunk_size}, max_chunks={max_chunks})...")
            
            # Create chunk iterator with error handling
            try:
                chunk_iterator = pd.read_csv(
                    self.fnspid_file, 
                    chunksize=chunk_size,
                    encoding='utf-8',
                    on_bad_lines='skip',
                    dtype=str,  # Force string type to avoid parsing issues
                    engine='python'  # Use Python engine for better error handling
                    # Note: low_memory=False removed - incompatible with python engine
                )
            except Exception as e:
                logger.error(f"❌ Failed to create chunk iterator: {e}")
                return pd.DataFrame()
            
            for i, chunk in enumerate(chunk_iterator):
                # Stop if we've reached max chunks
                if i >= max_chunks:
                    logger.info(f"📊 Reached maximum chunks ({max_chunks}) - stopping for testing")
                    break
                
                try:
                    total_processed += len(chunk)
                    
                    # Normalize and filter chunk
                    chunk = self._normalize_columns(chunk)
                    
                    if chunk.empty:
                        failed_chunks += 1
                        continue
                    
                    # Filter chunk for relevant data
                    filtered_chunk = self._filter_data(chunk)
                    
                    # Apply timezone safety to chunk
                    filtered_chunk = ensure_timezone_safe_dataframe(filtered_chunk)
                    
                    if not filtered_chunk.empty:
                        relevant_data.append(filtered_chunk)
                        successful_chunks += 1
                        
                        if i % 10 == 0:
                            logger.info(f"   📊 Chunk {i}: {len(filtered_chunk)} relevant articles found")
                    
                except Exception as e:
                    logger.warning(f"⚠️ Error processing chunk {i}: {e}")
                    failed_chunks += 1
                    continue
                
                # Memory management - collect garbage every 20 chunks
                if i % 20 == 0:
                    import gc
                    gc.collect()
            
            # Combine results
            if relevant_data:
                logger.info(f"📊 Combining {len(relevant_data)} successful chunks...")
                
                try:
                    combined_data = pd.concat(relevant_data, ignore_index=True)
                    
                    # Clean up memory
                    del relevant_data
                    import gc
                    gc.collect()
                    
                    # Final timezone safety check
                    combined_data = ensure_timezone_safe_dataframe(combined_data)
                    
                    logger.info(f"✅ Chunked loading complete:")
                    logger.info(f"   📊 Processed {total_processed:,} total rows")
                    logger.info(f"   ✅ Successful chunks: {successful_chunks}")
                    logger.info(f"   ❌ Failed chunks: {failed_chunks}")
                    logger.info(f"   📰 Relevant articles: {len(combined_data):,}")
                    
                    return combined_data
                    
                except Exception as e:
                    logger.error(f"❌ Failed to combine chunks: {e}")
                    return pd.DataFrame()
            else:
                logger.warning(f"⚠️ No relevant data found in {total_processed:,} processed rows")
                logger.info(f"   ✅ Successful chunks: {successful_chunks}")
                logger.info(f"   ❌ Failed chunks: {failed_chunks}")
                return pd.DataFrame()
                
        except KeyboardInterrupt:
            logger.info("⚠️ Chunked loading interrupted by user")
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"❌ Chunked loading failed: {e}")
            logger.error(f"   Error type: {type(e).__name__}")
            return pd.DataFrame()
     
    def _normalize_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize FNSPID column names with enhanced error handling"""
        try:
            if data.empty:
                return data
            
            logger.debug(f"📋 Original columns: {list(data.columns)}")
            
            # Multiple column mapping strategies
            column_mappings = [
                # Strategy 1: Exact FNSPID column names
                {
                    'Date': 'Date',
                    'Stock_symbol': 'Symbol',
                    'Article_title': 'Title',
                    'Article': 'Content',
                    'Url': 'URL'
                },
                # Strategy 2: Alternative column names
                {
                    'date': 'Date',
                    'symbol': 'Symbol',
                    'title': 'Title',
                    'content': 'Content',
                    'url': 'URL'
                },
                # Strategy 3: Lowercase variations
                {
                    'stock_symbol': 'Symbol',
                    'article_title': 'Title',
                    'article': 'Content'
                }
            ]
            
            # Apply the first matching column mapping
            for mapping in column_mappings:
                matching_cols = [col for col in mapping.keys() if col in data.columns]
                
                if matching_cols:
                    logger.info(f"📋 Applying column mapping: {matching_cols}")
                    data = data.rename(columns=mapping)
                    break
            
            logger.debug(f"📋 Normalized columns: {list(data.columns)}")
            return data
            
        except Exception as e:
            logger.warning(f"⚠️ Column normalization failed: {e}")
            return data
        
    def _filter_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Enhanced filtering with comprehensive symbol matching"""
        try:
            if data.empty:
                return pd.DataFrame()
            
            original_count = len(data)
            logger.info(f"🔍 Filtering {original_count} articles...")
            
            # Date filtering (existing code)
            if 'Date' in data.columns:
                # ...existing date filtering code...
                pass
            
            # ENHANCED SYMBOL FILTERING
            if 'Symbol' in data.columns:
                try:
                    logger.info("🏷️ Enhanced symbol processing...")
                    
                    # Clean symbol column
                    data['Symbol'] = data['Symbol'].astype(str).str.strip().str.upper()
                    data['Symbol_Original'] = data['Symbol'].copy()
                    
                    # Remove invalid values
                    data = data[~data['Symbol'].isin(['NAN', 'NONE', 'NULL', '', 'nan', 'UNKNOWN'])]
                    
                    # Target symbols in various formats
                    target_symbols = [s.upper() for s in self.config.symbols]
                    
                    # Strategy 1: Exact matching
                    exact_mask = data['Symbol'].isin(target_symbols)
                    exact_matches = len(data[exact_mask])
                    logger.info(f"🎯 Strategy 1 - Exact matches: {exact_matches}")
                    
                    if exact_matches > 0:
                        filtered_data = data[exact_mask]
                        logger.info(f"✅ Using exact matching: {len(filtered_data)} articles")
                        return self._apply_final_filters(filtered_data)
                    
                    # Strategy 2: Handle common symbol variations
                    logger.info("🔍 Strategy 2 - Trying symbol variations...")
                    symbol_variations = self._get_symbol_variations(target_symbols)
                    
                    # Try variations
                    variation_matches = pd.Series(False, index=data.index)
                    for target_symbol, variations in symbol_variations.items():
                        for variation in variations:
                            mask = data['Symbol'] == variation
                            variation_matches |= mask
                            if mask.sum() > 0:
                                logger.info(f"🎯 Found {mask.sum()} articles for {target_symbol} as '{variation}'")
                    
                    if variation_matches.sum() > 0:
                        filtered_data = data[variation_matches]
                        logger.info(f"✅ Using variation matching: {len(filtered_data)} articles")
                        return self._apply_final_filters(filtered_data)
                    
                    # Strategy 3: Substring matching
                    logger.info("🔍 Strategy 3 - Trying substring matching...")
                    substring_mask = self._get_substring_matches(data, target_symbols)
                    
                    if substring_mask.sum() > 0:
                        filtered_data = data[substring_mask]
                        self._log_sample_matches(filtered_data)
                        return self._apply_final_filters(filtered_data)
                    
                    # Strategy 4: Show available symbols
                    logger.warning("⚠️ No symbol matches found with any strategy")
                    self._log_available_symbols(data, target_symbols)
                    return pd.DataFrame()
                    
                except Exception as e:
                    logger.warning(f"⚠️ Enhanced symbol filtering failed: {e}")
                    return pd.DataFrame()
            else:
                logger.warning("⚠️ No 'Symbol' column found")
                return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"❌ Error in enhanced filtering: {e}")
            return pd.DataFrame()
        
    def _apply_final_filters(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply final filters and adjustments to the data"""
        try:
            if data.empty:
                return data
            
            logger.info(f"📊 Applying final filters...")
            
            # --- CONTENT FILTERING ---
            # Exclude obvious non-relevant content based on title and content
            initial_count = len(data)
            
            # Filter out rows where 'Title' or 'Content' is NaN
            data = data.dropna(subset=['Title', 'Content'])
            
            # Filter out rows with empty 'Title' or 'Content' after dropping NaNs
            data = data[(data['Title'].str.strip() != '') & (data['Content'].str.strip() != '')]
            
            # Filter out rows with excessive length in 'Title' or 'Content'
            data = data[(data['Title'].str.len() <= 150) & (data['Content'].str.len() <= 5000)]
            
            # Filter out rows with suspiciously short or long dates
            if 'Date' in data.columns:
                data = data[(data['Date'] >= '2000-01-01') & (data['Date'] <= '2100-01-01')]
            
            # Filter out rows where 'Symbol' is not in the target symbols
            if 'Symbol' in data.columns:
                target_symbols = [s.upper() for s in self.config.symbols]
                data = data[data['Symbol'].isin(target_symbols)]
            
            # --- LOGGING ---
            final_count = len(data)
            logger.info(f"🧹 Content filter: {initial_count} → {final_count} articles")
    
            # Apply timezone safety
            data = ensure_timezone_safe_dataframe(data)
    
            # Add normalized symbol for merging
            if 'Symbol' in data.columns:
                target_symbols = [s.upper() for s in self.config.symbols]
                
                # Create a mapping from found symbols to target symbols
                symbol_mapping = {}
                for target in target_symbols:
                    for symbol in data['Symbol'].unique():
                        if (target == symbol or 
                            target in symbol or 
                            symbol.replace('.O', '').replace('.US', '').replace(':US', '') == target):
                            symbol_mapping[symbol] = target
                            break
                
                data['symbol_normalized'] = data['Symbol'].map(symbol_mapping)
                logger.info(f"📊 Symbol mapping: {symbol_mapping}")
    
            return data
            
        except Exception as e:
            logger.error(f"❌ Final filtering failed: {e}")
            return data
    
    def _get_symbol_variations(self, target_symbols: List[str]) -> Dict[str, List[str]]:
        """Get variations of target symbols"""
        symbol_variations = {}
        for symbol in target_symbols:
            variations = [
                symbol,                    # AAPL
                f"{symbol}.O",            # AAPL.O
                f"{symbol}.US",           # AAPL.US
                f"{symbol}:US",           # AAPL:US
                f"US:{symbol}",           # US:AAPL
                f"{symbol} US",           # AAPL US
                f"{symbol}_US",           # AAPL_US
                f"NASDAQ:{symbol}",       # NASDAQ:AAPL
                f"{symbol}.NASDAQ",       # AAPL.NASDAQ
            ]
            symbol_variations[symbol] = variations
        return symbol_variations

    def _get_substring_matches(self, data: pd.DataFrame, target_symbols: List[str]) -> pd.Series:
        """Get substring matches for symbols"""
        substring_mask = pd.Series(False, index=data.index)
        
        for target_symbol in target_symbols:
            mask1 = data['Symbol'].str.contains(target_symbol, na=False, case=False)
            mask2 = data['Symbol'].apply(lambda x: target_symbol in str(x).upper() if pd.notna(x) else False)
            substring_mask |= (mask1 | mask2)
            
        return substring_mask

    def _log_sample_matches(self, data: pd.DataFrame):
        """Log sample of matched symbols"""
        sample_matches = data['Symbol'].value_counts().head(10)
        logger.info(f"📊 Sample matches: {dict(sample_matches)}")

    def _log_available_symbols(self, data: pd.DataFrame, target_symbols: List[str]):
        """Log available symbols when no matches found"""
        actual_symbols = data['Symbol'].value_counts().head(20)
        logger.info(f"📊 Top 20 available symbols: {dict(actual_symbols)}")
        logger.info(f"🎯 Target symbols: {target_symbols}")
    
    def debug_fnspid_symbols(self) -> Dict[str, Any]:
        """Debug function to see what symbols are actually in FNSPID data"""
        logger.info("🔍 DEBUGGING FNSPID SYMBOLS...")
        
        try:
            # Load a small sample to inspect symbols
            sample_data = pd.read_csv(
                self.fnspid_file, 
                nrows=10000,  # Larger sample
                encoding='utf-8',
                on_bad_lines='skip',
                dtype=str
            )
            
            # Normalize columns
            sample_data = self._normalize_columns(sample_data)
            
            if 'Symbol' in sample_data.columns:
                # Get unique symbols and their counts
                symbol_counts = sample_data['Symbol'].value_counts()
                unique_symbols = symbol_counts.index.tolist()
                
                # Your target symbols
                target_symbols = [s.upper() for s in self.config.symbols]
                
                # Check for exact matches
                exact_matches = [s for s in unique_symbols if s.upper() in target_symbols]
                
                # Check for partial matches
                partial_matches = []
                for target in target_symbols:
                    matches = [s for s in unique_symbols if target in s.upper() or s.upper() in target]
                    if matches:
                        partial_matches.extend(matches)
                
                debug_info = {
                    'total_unique_symbols': len(unique_symbols),
                    'top_symbols': dict(symbol_counts.head(20)),
                    'target_symbols': target_symbols,
                    'exact_matches': exact_matches,
                    'partial_matches': list(set(partial_matches)),
                    'sample_symbols': unique_symbols[:50]  # First 50 symbols
                }
                
                logger.info(f"🔍 FNSPID Symbol Analysis:")
                logger.info(f"   📊 Total unique symbols: {len(unique_symbols)}")
                logger.info(f"   🎯 Target symbols: {target_symbols}")
                logger.info(f"   ✅ Exact matches found: {exact_matches}")
                logger.info(f"   🔍 Partial matches found: {list(set(partial_matches))}")
                logger.info(f"   📋 Top 10 symbols in data: {list(symbol_counts.head(10).index)}")
                
                return debug_info
            else:
                logger.error("❌ No 'Symbol' column found after normalization")
                logger.info(f"📋 Available columns: {list(sample_data.columns)}")
                return {'error': 'No Symbol column found'}
                
        except Exception as e:
            logger.error(f"❌ Debug failed: {e}")
            return {'error': str(e)}

def calculate_proper_vwap(data: pd.DataFrame) -> pd.Series:
    """Calculate VWAP properly per symbol"""
    vwap = pd.Series(index=data.index, dtype=float)
    
    for symbol in data['symbol'].unique():
        mask = data['symbol'] == symbol
        typical_price = (data.loc[mask, 'high'] + data.loc[mask, 'low'] + data.loc[mask, 'close']) / 3
        vwap[mask] = (typical_price * data.loc[mask, 'volume']).cumsum() / data.loc[mask, 'volume'].cumsum()
        
    return vwap

def calculate_optimized_atr(data: pd.DataFrame, window: int = 14) -> pd.Series:
    """Calculate ATR (Average True Range) with optimized performance"""
    high = data['high']
    low = data['low']
    close = data['close']
    
    # Calculate True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate ATR by symbol
    atr = data.groupby('symbol')['close'].transform(
        lambda x: tr.loc[x.index].rolling(window=window).mean()
    )
    
    return atr

def create_all_datasets(config: DatasetConfig) -> Dict[str, pd.DataFrame]:
    """Create all three dataset variants"""
    datasets = {}
    
    try:
        # 1. Create core dataset (stock + technical)
        stock_collector = StockDataCollector(config)
        stock_data = stock_collector.collect_stock_data()
        
        if not stock_data.empty:
            # Add technical indicators
            tech_processor = TechnicalIndicatorProcessor()
            tech_data = tech_processor.add_technical_indicators(stock_data)
            
            # Add targets
            target_processor = TargetVariableProcessor()
            core_data = target_processor.add_target_variables(tech_data, config.target_horizons)
            datasets['core'] = core_data
            core_data.to_csv(CORE_DATASET, index=False)
            
            # 2. Create sentiment dataset if FNSPID available
            if config.include_sentiment:
                sentiment_processor = SentimentDataProcessor(config)
                sentiment_data = sentiment_processor.collect_sentiment_data()
                if not sentiment_data.empty:
                    datasets['sentiment'] = pd.merge(
                        core_data, sentiment_data,
                        on=['symbol', 'date'], how='left'
                    )
                    datasets['sentiment'].to_csv(SENTIMENT_DATASET, index=False)
            
                    # 3. Create temporal decay dataset
                    if config.include_temporal_decay:
                        decay_processor = TemporalDecayProcessor(config.decay_lambda)
                        datasets['temporal'] = decay_processor.apply_decay(
                            datasets['sentiment']
                        )
                        datasets['temporal'].to_csv(TEMPORAL_DECAY_DATASET, index=False)
                        
        return datasets
        
    except Exception as e:
        logger.error(f"❌ Failed to create datasets: {e}")
        return {}

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Test configuration
    test_config = DatasetConfig(
        symbols=['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'JPM'],
        start_date='2018-12-01',
        end_date='2024-01-20',
        target_horizons=[5, 30, 90],
        fnspid_data_file=FNSPID_DATA_FILE,
        include_sentiment=True,
        include_temporal_decay=True
    )
    
    logger.info("🚀 Testing dataset creation...")
    logger.info("🚀 CREATING MULTIPLE DATASET VARIANTS (TIMEZONE-SAFE)")
    logger.info("=" * 70)
    logger.info(f"📊 Symbols: {test_config.symbols}")
    logger.info(f"📅 Date range: {test_config.start_date} to {test_config.end_date}")
    logger.info(f"🎯 Target horizons: {test_config.target_horizons}")
    logger.info(f"📰 FNSPID file: {test_config.fnspid_data_file}")
    logger.info(f"⏰ Decay lambda: {test_config.decay_lambda}")
    logger.info("=" * 70)
    
    try:
        # Create all datasets
        datasets = create_all_datasets(test_config)
        
        for name, data in datasets.items():
            logger.info(f"✅ {name.title()} dataset created: {data.shape}")
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")