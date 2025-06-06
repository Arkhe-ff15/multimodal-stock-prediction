"""
src/data_loader.py - FIXED VERSION

Enhanced data collection system with proper error handling and compatibility
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import requests
import json
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
import warnings
from pathlib import Path
import re
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import yaml

# Suppress warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class NewsArticle:
    """Enhanced news article structure"""
    title: str
    content: str
    date: datetime
    source: str
    url: str = ""
    relevance_score: float = 0.5
    sentiment_score: Optional[float] = None
    word_count: int = 0
    
    def __post_init__(self):
        if isinstance(self.date, str):
            try:
                self.date = pd.to_datetime(self.date).to_pydatetime()
            except:
                self.date = datetime.now()
        
        if isinstance(self.content, str):
            self.word_count = len(self.content.split())
        
        # Basic sentiment scoring based on keywords
        self.sentiment_score = self._calculate_basic_sentiment()
    
    def _calculate_basic_sentiment(self) -> float:
        """Calculate basic sentiment score from keywords"""
        if not isinstance(self.title, str) or not isinstance(self.content, str):
            return 0.0
            
        positive_words = ['growth', 'profit', 'gain', 'rise', 'strong', 'beat', 'exceed', 'positive', 'bull', 'up']
        negative_words = ['loss', 'decline', 'fall', 'weak', 'miss', 'negative', 'bear', 'down', 'concern', 'risk']
        
        text = (self.title + " " + self.content).lower()
        
        pos_count = sum(1 for word in positive_words if word in text)
        neg_count = sum(1 for word in negative_words if word in text)
        
        if pos_count + neg_count == 0:
            return 0.0
        
        return (pos_count - neg_count) / (pos_count + neg_count)

@dataclass  
class MarketData:
    """Enhanced market data container"""
    symbol: str
    data: pd.DataFrame
    technical_indicators: pd.DataFrame
    sector: str = "Unknown"
    market_cap: str = "Unknown"

class DataCollector:
    """
    Enhanced data collection system with proper error handling and compatibility
    """
    
    def __init__(self, config_path: str = None, cache_dir: str = "data/cache"):
        """Initialize enhanced collector"""
        # Create cache directory
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Rate limiting with locks for thread safety
        self.last_request = {}
        self.request_lock = threading.Lock()
        
        # Session for requests
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Company info for better news collection
        self.company_info = self._get_company_info()
        
        logger.info("Enhanced DataCollector initialized")
        logger.info(f"Symbols: {len(self.config['symbols'])} stocks across sectors")
        logger.info(f"Date range: {self.config['start_date']} to {self.config['end_date']}")
        logger.info(f"News sources: {len(self.config['news_sources'])}")
    
    def _load_config(self, config_path: str = None) -> Dict:
        """Load configuration with proper fallbacks"""
        # Default configuration
        default_config = {
            'symbols': [
                # Large Cap Tech
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',
                # Financial Services
                'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C',
                # Healthcare & Pharma
                'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'TMO',
                # Consumer & Retail
                'WMT', 'HD', 'PG', 'KO', 'PEP', 'NKE',
                # Industrial & Energy
                'CAT', 'BA', 'GE', 'XOM', 'CVX', 'COP',
                # Communication & Media
                'VZ', 'T', 'CMCSA', 'DIS', 'NFLX',
                # Emerging Growth
                'CRM', 'ADBE', 'SHOP', 'ROKU', 'SQ', 'PYPL'
            ],
            'start_date': '2018-12-01',
            'end_date': '2024-01-31',
            'news_sources': ['yahoo_finance', 'mock'],  # Start with safe sources
            'max_articles_per_day': 15,
            'parallel_workers': 4,
            'enhanced_features': True,
            'api_keys': {
                'newsapi': None,  # Will be loaded from config if available
                'alphavantage': None
            },
            'sector_mapping': {
                # Tech
                'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology', 
                'AMZN': 'Technology', 'META': 'Technology', 'NVDA': 'Technology',
                'TSLA': 'Technology', 'CRM': 'Technology', 'ADBE': 'Technology',
                # Financial
                'JPM': 'Financial', 'BAC': 'Financial', 'WFC': 'Financial',
                'GS': 'Financial', 'MS': 'Financial', 'C': 'Financial',
                # Healthcare
                'JNJ': 'Healthcare', 'PFE': 'Healthcare', 'UNH': 'Healthcare',
                'ABBV': 'Healthcare', 'MRK': 'Healthcare', 'TMO': 'Healthcare',
                # Consumer
                'WMT': 'Consumer', 'HD': 'Consumer', 'PG': 'Consumer',
                'KO': 'Consumer', 'PEP': 'Consumer', 'NKE': 'Consumer',
                # Industrial
                'CAT': 'Industrial', 'BA': 'Industrial', 'GE': 'Industrial',
                'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy',
                # Communication
                'VZ': 'Communication', 'T': 'Communication', 'CMCSA': 'Communication',
                'DIS': 'Communication', 'NFLX': 'Communication',
                # Fintech
                'SHOP': 'Fintech', 'ROKU': 'Technology', 'SQ': 'Fintech', 'PYPL': 'Fintech'
            }
        }
        
        # Try to load from YAML config if provided
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    yaml_config = yaml.safe_load(f)
                
                # Update default config with YAML values
                if 'data' in yaml_config:
                    data_config = yaml_config['data']
                    if 'stocks' in data_config:
                        default_config['symbols'] = data_config['stocks']
                    if 'start_date' in data_config:
                        default_config['start_date'] = data_config['start_date']
                    if 'end_date' in data_config:
                        default_config['end_date'] = data_config['end_date']
                
                logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                logger.warning(f"Could not load config from {config_path}: {e}. Using defaults.")
        
        return default_config
    
    def _get_company_info(self) -> Dict:
        """Get company names and keywords for better news collection"""
        return {
            'AAPL': {'name': 'Apple Inc', 'keywords': ['apple', 'iphone', 'ipad', 'mac', 'ios', 'tim cook']},
            'MSFT': {'name': 'Microsoft', 'keywords': ['microsoft', 'windows', 'office', 'azure', 'xbox', 'satya nadella']},
            'GOOGL': {'name': 'Alphabet/Google', 'keywords': ['google', 'alphabet', 'youtube', 'android', 'search']},
            'AMZN': {'name': 'Amazon', 'keywords': ['amazon', 'aws', 'prime', 'alexa', 'jeff bezos', 'andy jassy']},
            'META': {'name': 'Meta/Facebook', 'keywords': ['meta', 'facebook', 'instagram', 'whatsapp', 'zuckerberg']},
            'NVDA': {'name': 'NVIDIA', 'keywords': ['nvidia', 'gpu', 'ai chip', 'jensen huang']},
            'TSLA': {'name': 'Tesla', 'keywords': ['tesla', 'elon musk', 'electric vehicle', 'model s', 'model 3']},
            'JPM': {'name': 'JPMorgan Chase', 'keywords': ['jpmorgan', 'chase', 'jamie dimon']},
            'BAC': {'name': 'Bank of America', 'keywords': ['bank of america', 'bofa']},
            'JNJ': {'name': 'Johnson & Johnson', 'keywords': ['johnson johnson', 'jnj', 'pharmaceutical']},
            'WMT': {'name': 'Walmart', 'keywords': ['walmart', 'retail', 'grocery']},
            'HD': {'name': 'Home Depot', 'keywords': ['home depot', 'hardware', 'construction']},
            'DIS': {'name': 'Disney', 'keywords': ['disney', 'marvel', 'streaming', 'parks']},
            'NFLX': {'name': 'Netflix', 'keywords': ['netflix', 'streaming', 'content']},
            'CRM': {'name': 'Salesforce', 'keywords': ['salesforce', 'crm', 'cloud software']},
            'SHOP': {'name': 'Shopify', 'keywords': ['shopify', 'e-commerce', 'online store']},
            'SQ': {'name': 'Block/Square', 'keywords': ['square', 'block', 'payment', 'bitcoin']},
            'PYPL': {'name': 'PayPal', 'keywords': ['paypal', 'payment', 'venmo']}
        }
    
    def _rate_limit(self, source: str, delay: float = 1.0):
        """Thread-safe rate limiting"""
        with self.request_lock:
            if source in self.last_request:
                elapsed = time.time() - self.last_request[source]
                if elapsed < delay:
                    time.sleep(delay - elapsed)
            self.last_request[source] = time.time()
    
    def collect_market_data(self, symbols: List[str] = None, use_parallel: bool = True) -> Dict[str, MarketData]:
        """Enhanced market data collection with parallel processing"""
        symbols = symbols or self.config['symbols']
        start_date = self.config['start_date']
        end_date = self.config['end_date']
        
        logger.info(f"Collecting market data for {len(symbols)} symbols")
        
        if use_parallel and len(symbols) > 1:
            return self._collect_market_data_parallel(symbols)
        else:
            return self._collect_market_data_sequential(symbols)
    
    def _collect_market_data_parallel(self, symbols: List[str]) -> Dict[str, MarketData]:
        """Parallel market data collection"""
        market_data = {}
        
        with ThreadPoolExecutor(max_workers=min(self.config['parallel_workers'], len(symbols))) as executor:
            future_to_symbol = {
                executor.submit(self._download_single_stock, symbol): symbol 
                for symbol in symbols
            }
            
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    result = future.result()
                    if result:
                        market_data[symbol] = result
                        logger.info(f"✅ {symbol}: {len(result.data)} days")
                    else:
                        logger.warning(f"❌ {symbol}: No data")
                except Exception as e:
                    logger.error(f"❌ {symbol}: {e}")
        
        logger.info(f"Market data collection complete: {len(market_data)}/{len(symbols)} symbols")
        return market_data
    
    def _collect_market_data_sequential(self, symbols: List[str]) -> Dict[str, MarketData]:
        """Sequential market data collection"""
        market_data = {}
        
        for symbol in symbols:
            try:
                result = self._download_single_stock(symbol)
                if result:
                    market_data[symbol] = result
                    logger.info(f"✅ {symbol}: {len(result.data)} days")
                else:
                    logger.warning(f"❌ {symbol}: No data")
            except Exception as e:
                logger.error(f"❌ {symbol}: {e}")
        
        return market_data
    
    def _download_single_stock(self, symbol: str) -> Optional[MarketData]:
        """Download and process single stock with proper error handling"""
        try:
            # Check cache first
            cache_file = self.cache_dir / f"{symbol}_market.parquet"
            if cache_file.exists():
                try:
                    data = pd.read_parquet(cache_file)
                    if not data.empty and len(data) > 100:  # Ensure sufficient data
                        tech_data = self._calculate_enhanced_technical_indicators(data)
                        sector = self.config['sector_mapping'].get(symbol, 'Unknown')
                        return MarketData(symbol, data, tech_data, sector)
                except Exception as e:
                    logger.warning(f"Cache read error for {symbol}: {e}")
            
            # Download fresh data
            self._rate_limit('yahoo_finance')
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=self.config['start_date'], end=self.config['end_date'])
            
            if data.empty:
                logger.warning(f"No data from yfinance for {symbol}")
                return None
            
            # Clean data
            data = self._clean_market_data(data)
            if data.empty:
                logger.warning(f"No data after cleaning for {symbol}")
                return None
            
            # Cache data
            try:
                data.to_parquet(cache_file)
            except Exception as e:
                logger.warning(f"Could not cache data for {symbol}: {e}")
            
            # Calculate enhanced technical indicators
            tech_data = self._calculate_enhanced_technical_indicators(data)
            sector = self.config['sector_mapping'].get(symbol, 'Unknown')
            
            return MarketData(symbol, data, tech_data, sector)
            
        except Exception as e:
            logger.error(f"Error downloading {symbol}: {e}")
            return None
    
    def _clean_market_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Enhanced data cleaning with better error handling"""
        if data.empty:
            return data
            
        original_len = len(data)
        
        try:
            # Remove weekends (if index has weekday attribute)
            if hasattr(data.index, 'weekday'):
                data = data[data.index.weekday < 5]
            
            # Handle missing values
            data = data.dropna()
            
            # Validate required columns exist
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in data.columns for col in required_cols):
                logger.warning(f"Missing required columns: {[col for col in required_cols if col not in data.columns]}")
                return pd.DataFrame()
            
            # Remove extreme outliers (stock splits, errors)
            for col in ['Open', 'High', 'Low', 'Close']:
                if col in data.columns and len(data) > 1:
                    pct_change = data[col].pct_change().abs()
                    outliers = pct_change > 0.5  # 50% change
                    if outliers.sum() > 0:
                        logger.debug(f"Removing {outliers.sum()} outliers from {col}")
                        data = data[~outliers]
            
            # Ensure positive volume
            if 'Volume' in data.columns:
                data = data[data['Volume'] > 0]
            
            # Validate OHLC relationships
            if len(data) > 0:
                valid_ohlc = (
                    (data['High'] >= data['Low']) &
                    (data['High'] >= data['Open']) &
                    (data['High'] >= data['Close']) &
                    (data['Low'] <= data['Open']) &
                    (data['Low'] <= data['Close'])
                )
                data = data[valid_ohlc]
            
            # Minimum data requirement
            if len(data) < 100:  # Need sufficient data for technical indicators
                logger.warning(f"Insufficient data after cleaning: {len(data)} rows")
                return pd.DataFrame()
            
            logger.debug(f"Data cleaning: {original_len} → {len(data)} rows")
            return data
            
        except Exception as e:
            logger.error(f"Error in data cleaning: {e}")
            return pd.DataFrame()
    
    def _calculate_enhanced_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate enhanced technical indicators with error handling"""
        tech = pd.DataFrame(index=data.index)
        
        if data.empty or len(data) < 50:  # Need minimum data for indicators
            return tech
        
        try:
            # Moving Averages
            for window in [5, 10, 20, 50, 200]:
                if len(data) >= window:
                    tech[f'SMA_{window}'] = data['Close'].rolling(window, min_periods=window//2).mean()
                    tech[f'EMA_{window}'] = data['Close'].ewm(span=window, min_periods=window//2).mean()
            
            # RSI with multiple periods
            for period in [14, 21]:
                if len(data) >= period * 2:
                    delta = data['Close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=period//2).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=period//2).mean()
                    rs = gain / loss.replace(0, np.nan)  # Avoid division by zero
                    tech[f'RSI_{period}'] = 100 - (100 / (1 + rs))
            
            # MACD
            if len(data) >= 26:
                ema12 = data['Close'].ewm(span=12, min_periods=6).mean()
                ema26 = data['Close'].ewm(span=26, min_periods=13).mean()
                tech['MACD'] = ema12 - ema26
                tech['MACD_signal'] = tech['MACD'].ewm(span=9, min_periods=4).mean()
                tech['MACD_histogram'] = tech['MACD'] - tech['MACD_signal']
            
            # Bollinger Bands
            for window in [20, 50]:
                if len(data) >= window:
                    sma = data['Close'].rolling(window, min_periods=window//2).mean()
                    std = data['Close'].rolling(window, min_periods=window//2).std()
                    tech[f'BB_upper_{window}'] = sma + (std * 2)
                    tech[f'BB_lower_{window}'] = sma - (std * 2)
                    tech[f'BB_middle_{window}'] = sma
                    tech[f'BB_width_{window}'] = (tech[f'BB_upper_{window}'] - tech[f'BB_lower_{window}']) / tech[f'BB_middle_{window}']
                    tech[f'BB_position_{window}'] = (data['Close'] - tech[f'BB_lower_{window}']) / (tech[f'BB_upper_{window}'] - tech[f'BB_lower_{window}'])
            
            # Price momentum
            for period in [1, 3, 5, 10, 20]:
                if len(data) > period:
                    tech[f'Momentum_{period}'] = data['Close'] / data['Close'].shift(period) - 1
                    tech[f'Close_lag_{period}'] = data['Close'].shift(period)
                    tech[f'Volume_lag_{period}'] = data['Volume'].shift(period)
            
            # Volatility measures
            if len(data) > 5:
                returns = data['Close'].pct_change()
                for window in [5, 10, 20, 60]:
                    if len(data) >= window:
                        tech[f'Volatility_{window}d'] = returns.rolling(window, min_periods=window//2).std()
                        if window <= 20:  # Only for shorter windows to avoid issues
                            tech[f'Returns_skew_{window}d'] = returns.rolling(window, min_periods=window//2).skew()
                            tech[f'Returns_kurt_{window}d'] = returns.rolling(window, min_periods=window//2).kurt()
            
            # Price patterns
            tech['HL_ratio'] = (data['High'] - data['Low']) / data['Close']
            tech['OC_ratio'] = (data['Close'] - data['Open']) / data['Open']
            tech['HC_ratio'] = (data['High'] - data['Close']) / data['Close']
            tech['LC_ratio'] = (data['Close'] - data['Low']) / data['Close']
            
            # Volume indicators
            if len(data) >= 20:
                tech['Volume_SMA_20'] = data['Volume'].rolling(20, min_periods=10).mean()
                tech['Volume_ratio'] = data['Volume'] / tech['Volume_SMA_20']
                tech['VWAP'] = (data['Close'] * data['Volume']).rolling(20, min_periods=10).sum() / data['Volume'].rolling(20, min_periods=10).sum()
            
        except Exception as e:
            logger.warning(f"Error calculating technical indicators: {e}")
        
        # Fill NaN values
        tech = tech.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        return tech
    
    def collect_news_data(self, symbols: List[str] = None) -> Dict[str, List[NewsArticle]]:
        """Enhanced news data collection with multiple sources"""
        symbols = symbols or self.config['symbols']
        
        logger.info(f"Collecting news data for {len(symbols)} symbols")
        
        news_data = {}
        
        for symbol in symbols:
            articles = []
            
            # Try multiple sources
            for source in self.config['news_sources']:
                try:
                    if source == 'yahoo_finance':
                        source_articles = self._get_yahoo_news(symbol)
                    elif source == 'newsapi':
                        source_articles = self._get_newsapi_news(symbol)
                    elif source == 'alphavantage':
                        source_articles = self._get_alphavantage_news(symbol)
                    elif source == 'mock':
                        source_articles = self._generate_enhanced_mock_news(symbol)
                    else:
                        continue
                    
                    articles.extend(source_articles)
                    logger.info(f"{source}: {len(source_articles)} articles for {symbol}")
                    
                except Exception as e:
                    logger.warning(f"{source} failed for {symbol}: {e}")
            
            # Process articles
            if articles:
                # Remove duplicates
                unique_articles = self._deduplicate_articles(articles)
                
                # Filter for relevance
                relevant_articles = self._filter_relevant_articles(unique_articles, symbol)
                
                # Limit per day
                limited_articles = self._limit_articles_per_day(relevant_articles)
                
                news_data[symbol] = limited_articles
                logger.info(f"Final: {len(limited_articles)} articles for {symbol}")
            else:
                news_data[symbol] = []
        
        total_articles = sum(len(articles) for articles in news_data.values())
        logger.info(f"News collection complete: {total_articles} total articles")
        
        return news_data
    
    def _get_yahoo_news(self, symbol: str) -> List[NewsArticle]:
        """Enhanced Yahoo Finance news collection with error handling"""
        articles = []
        
        try:
            self._rate_limit('yahoo_finance')
            ticker = yf.Ticker(symbol)
            news_items = ticker.news
            
            for item in news_items[:30]:  # Get more items
                try:
                    title = item.get('title', 'No title')
                    url = item.get('link', '')
                    
                    # Get publish time
                    if 'providerPublishTime' in item:
                        pub_time = datetime.fromtimestamp(item['providerPublishTime'])
                    else:
                        pub_time = datetime.now()
                    
                    content = item.get('summary', title)
                    
                    article = NewsArticle(
                        title=title,
                        content=content,
                        date=pub_time,
                        source='yahoo_finance',
                        url=url,
                        relevance_score=0.85  # High relevance for symbol-specific news
                    )
                    
                    articles.append(article)
                    
                except Exception as e:
                    logger.debug(f"Error processing Yahoo news item: {e}")
                    continue
                    
        except Exception as e:
            logger.warning(f"Yahoo Finance news error for {symbol}: {e}")
        
        return articles
    
    def _get_newsapi_news(self, symbol: str) -> List[NewsArticle]:
        """Get news from NewsAPI with proper error handling"""
        articles = []
        
        api_key = self.config['api_keys'].get('newsapi')
        if not api_key:
            logger.debug("NewsAPI key not provided, skipping NewsAPI collection")
            return articles
        
        try:
            self._rate_limit('newsapi', 2.0)
            
            # Get company info
            company_info = self.company_info.get(symbol, {'name': symbol, 'keywords': [symbol]})
            query = f"{symbol} OR {company_info['name']}"
            
            # NewsAPI only allows last 30 days for free accounts
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': query,
                'from': start_date.strftime('%Y-%m-%d'),
                'to': end_date.strftime('%Y-%m-%d'),
                'language': 'en',
                'sortBy': 'relevancy',
                'pageSize': 50,
                'apiKey': api_key
            }
            
            response = self.session.get(url, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('status') == 'ok':
                for item in data.get('articles', []):
                    try:
                        title = item.get('title', '')
                        content = item.get('description', '') or item.get('content', '')
                        pub_date = pd.to_datetime(item.get('publishedAt'))
                        url_link = item.get('url', '')
                        
                        if title and content:
                            article = NewsArticle(
                                title=title,
                                content=content,
                                date=pub_date.to_pydatetime(),
                                source='newsapi',
                                url=url_link,
                                relevance_score=0.8
                            )
                            articles.append(article)
                    except Exception as e:
                        logger.debug(f"Error processing NewsAPI item: {e}")
                        continue
                        
        except Exception as e:
            logger.warning(f"NewsAPI error for {symbol}: {e}")
        
        return articles
    
    def _get_alphavantage_news(self, symbol: str) -> List[NewsArticle]:
        """Get news from Alpha Vantage with proper error handling"""
        articles = []
        
        api_key = self.config['api_keys'].get('alphavantage')
        if not api_key:
            logger.debug("Alpha Vantage API key not provided, skipping Alpha Vantage collection")
            return articles
        
        try:
            self._rate_limit('alphavantage', 3.0)
            
            url = "https://www.alphavantage.co/query"
            params = {
                'function': 'NEWS_SENTIMENT',
                'tickers': symbol,
                'apikey': api_key,
                'limit': 50
            }
            
            response = self.session.get(url, params=params, timeout=20)
            response.raise_for_status()
            
            data = response.json()
            
            if 'feed' in data:
                for item in data['feed']:
                    try:
                        title = item.get('title', '')
                        content = item.get('summary', '')
                        pub_date = pd.to_datetime(item.get('time_published'))
                        url_link = item.get('url', '')
                        
                        # Get relevance and sentiment from Alpha Vantage
                        relevance = 0.7
                        sentiment = 0.0
                        
                        ticker_sentiment = item.get('ticker_sentiment', [])
                        for ticker_info in ticker_sentiment:
                            if ticker_info.get('ticker') == symbol:
                                relevance = float(ticker_info.get('relevance_score', 0.7))
                                sentiment = float(ticker_info.get('ticker_sentiment_score', 0.0))
                                break
                        
                        if title and content:
                            article = NewsArticle(
                                title=title,
                                content=content,
                                date=pub_date.to_pydatetime(),
                                source='alphavantage',
                                url=url_link,
                                relevance_score=relevance,
                                sentiment_score=sentiment
                            )
                            articles.append(article)
                    except Exception as e:
                        logger.debug(f"Error processing Alpha Vantage item: {e}")
                        continue
                        
        except Exception as e:
            logger.warning(f"Alpha Vantage error for {symbol}: {e}")
        
        return articles
    
    def _generate_enhanced_mock_news(self, symbol: str) -> List[NewsArticle]:
        """Generate enhanced mock historical news"""
        articles = []
        
        try:
            company_info = self.company_info.get(symbol, {'name': symbol, 'keywords': []})
            company_name = company_info['name']
            sector = self.config['sector_mapping'].get(symbol, 'Technology')
            
            # Enhanced templates by sector
            base_templates = [
                f"{company_name} reports quarterly earnings results",
                f"{company_name} announces strategic partnership",
                f"Analysts update {company_name} price target and rating",
                f"{company_name} CEO discusses future growth strategy",
                f"Market reacts to {company_name} latest developments",
                f"{company_name} shows strong performance in key metrics",
                f"Institutional investors adjust {company_name} positions",
                f"{company_name} navigates current market challenges",
                f"{company_name} innovation drives competitive advantage",
                f"Investment community analyzes {company_name} prospects"
            ]
            
            # Sector-specific templates
            if sector == 'Technology':
                sector_templates = [
                    f"{company_name} advances AI and machine learning capabilities",
                    f"{company_name} cloud services see strong adoption",
                    f"{company_name} cybersecurity solutions gain traction"
                ]
            elif sector == 'Financial':
                sector_templates = [
                    f"{company_name} net interest margin improves",
                    f"{company_name} credit quality remains strong",
                    f"{company_name} digital banking initiatives expand"
                ]
            elif sector == 'Healthcare':
                sector_templates = [
                    f"{company_name} drug pipeline shows promise",
                    f"{company_name} regulatory approval received",
                    f"{company_name} clinical trial results positive"
                ]
            else:
                sector_templates = []
            
            all_templates = base_templates + sector_templates
            
            start_dt = pd.to_datetime(self.config['start_date'])
            end_dt = pd.to_datetime(self.config['end_date'])
            
            # Generate more sophisticated distribution
            current_date = start_dt
            while current_date <= end_dt:
                if current_date.weekday() < 5:  # Weekdays only
                    # Varying probability by day of week and time of year
                    base_prob = 0.08  # 8% base chance
                    
                    # Higher probability on certain days
                    if current_date.weekday() in [1, 2, 3]:  # Tue, Wed, Thu
                        base_prob *= 1.5
                    
                    # Earnings season effect (quarterly spikes)
                    month = current_date.month
                    if month in [1, 4, 7, 10]:  # Earnings months
                        base_prob *= 2.0
                    
                    if random.random() < base_prob:
                        template = random.choice(all_templates)
                        
                        # Add more realistic content
                        content_additions = [
                            f"The announcement comes as {company_name} continues to navigate the evolving market landscape.",
                            f"Industry analysts are closely monitoring {company_name}'s strategic initiatives and market position.",
                            f"This development aligns with {company_name}'s long-term growth strategy and operational excellence.",
                            f"Investors and stakeholders are evaluating the implications for {company_name}'s future performance."
                        ]
                        
                        full_content = f"{template} {random.choice(content_additions)}"
                        
                        # Sentiment based on template keywords
                        sentiment_score = 0.0
                        if any(word in template.lower() for word in ['strong', 'improves', 'positive', 'advances']):
                            sentiment_score = random.uniform(0.2, 0.8)
                        elif any(word in template.lower() for word in ['challenges', 'concerns']):
                            sentiment_score = random.uniform(-0.6, -0.1)
                        else:
                            sentiment_score = random.uniform(-0.2, 0.2)
                        
                        article = NewsArticle(
                            title=template,
                            content=full_content,
                            date=current_date.to_pydatetime(),
                            source='mock_historical',
                            url=f"https://mock-news.com/{symbol.lower()}-{current_date.strftime('%Y%m%d')}",
                            relevance_score=random.uniform(0.7, 0.95),
                            sentiment_score=sentiment_score
                        )
                        
                        articles.append(article)
                
                current_date += timedelta(days=1)
            
        except Exception as e:
            logger.warning(f"Error generating mock news for {symbol}: {e}")
        
        return articles
    
    def _deduplicate_articles(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """Enhanced deduplication with error handling"""
        if not articles:
            return articles
        
        seen_urls = set()
        seen_titles = set()
        unique_articles = []
        
        for article in articles:
            try:
                # URL deduplication
                if article.url and article.url in seen_urls:
                    continue
                
                # Title deduplication (more sophisticated)
                title_key = re.sub(r'[^\w\s]', '', str(article.title).lower())
                title_key = ' '.join(title_key.split()[:10])  # First 10 words
                
                if title_key in seen_titles and len(title_key) > 10:
                    continue
                
                if article.url:
                    seen_urls.add(article.url)
                if len(title_key) > 5:
                    seen_titles.add(title_key)
                
                unique_articles.append(article)
                
            except Exception as e:
                logger.debug(f"Error in deduplication: {e}")
                continue
        
        logger.debug(f"Deduplicated {len(articles)} -> {len(unique_articles)} articles")
        return unique_articles
    
    def _filter_relevant_articles(self, articles: List[NewsArticle], symbol: str) -> List[NewsArticle]:
        """Enhanced relevance filtering with error handling"""
        if not articles:
            return articles
        
        company_keywords = self.company_info.get(symbol, {'keywords': [symbol.lower()]})
        keywords = [symbol.lower()] + [kw.lower() for kw in company_keywords.get('keywords', [])]
        
        relevant_articles = []
        
        for article in articles:
            try:
                text = (str(article.title) + " " + str(article.content)).lower()
                
                # Count keyword matches
                keyword_matches = sum(1 for keyword in keywords if keyword in text)
                
                # Boost relevance based on matches
                relevance_boost = min(keyword_matches * 0.1, 0.3)
                article.relevance_score = min(article.relevance_score + relevance_boost, 1.0)
                
                # Filter by minimum threshold
                if article.relevance_score >= 0.3:  # Lower threshold for more coverage
                    relevant_articles.append(article)
                    
            except Exception as e:
                logger.debug(f"Error in relevance filtering: {e}")
                continue
        
        logger.debug(f"Filtered {len(articles)} -> {len(relevant_articles)} relevant articles for {symbol}")
        return relevant_articles
    
    def _limit_articles_per_day(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """Enhanced daily limiting with quality prioritization"""
        if not articles:
            return articles
            
        max_per_day = self.config['max_articles_per_day']
        
        # Group by date
        daily_articles = {}
        for article in articles:
            try:
                date_key = article.date.date()
                if date_key not in daily_articles:
                    daily_articles[date_key] = []
                daily_articles[date_key].append(article)
            except Exception as e:
                logger.debug(f"Error grouping articles by date: {e}")
                continue
        
        # Limit each day with quality scoring
        limited_articles = []
        for date_key, day_articles in daily_articles.items():
            try:
                # Sort by composite score (relevance + sentiment confidence + source quality)
                for article in day_articles:
                    source_weight = {
                        'yahoo_finance': 1.0, 
                        'alphavantage': 0.95, 
                        'newsapi': 0.9, 
                        'mock_historical': 0.6
                    }.get(article.source, 0.5)
                    
                    sentiment_confidence = abs(article.sentiment_score or 0) if article.sentiment_score else 0
                    
                    article.quality_score = (
                        article.relevance_score * 0.5 + 
                        source_weight * 0.3 + 
                        sentiment_confidence * 0.2
                    )
                
                # Sort by quality score
                day_articles.sort(key=lambda x: getattr(x, 'quality_score', x.relevance_score), reverse=True)
                
                # Take top articles
                limited_articles.extend(day_articles[:max_per_day])
                
            except Exception as e:
                logger.debug(f"Error in daily limiting: {e}")
                limited_articles.extend(day_articles[:max_per_day])
        
        # Sort by date
        try:
            limited_articles.sort(key=lambda x: x.date)
        except:
            pass
        
        logger.debug(f"Limited articles: {len(articles)} -> {len(limited_articles)} (max {max_per_day}/day)")
        return limited_articles
    
    def create_combined_dataset(self, market_data: Dict[str, MarketData], 
                               news_data: Dict[str, List[NewsArticle]],
                               save_path: str = None) -> pd.DataFrame:
        """Enhanced combined dataset creation with error handling"""
        logger.info("Creating enhanced combined dataset")
        
        combined_data = []
        
        for symbol in market_data.keys():
            try:
                market = market_data[symbol]
                news = news_data.get(symbol, [])
                
                # Start with market data
                df = market.data.copy()
                df['symbol'] = symbol
                df['sector'] = market.sector
                
                # Add technical indicators
                for col in market.technical_indicators.columns:
                    df[col] = market.technical_indicators[col]
                
                # Add enhanced news features
                df = self._add_enhanced_news_features(df, news)
                
                combined_data.append(df)
                logger.info(f"Processed {symbol}: {len(df)} rows")
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                continue
        
        if not combined_data:
            logger.error("No data to combine")
            return pd.DataFrame()
        
        # Combine all symbols
        final_df = pd.concat(combined_data, ignore_index=False)
        final_df = final_df.sort_index()
        
        # Add target variables
        final_df = self._add_enhanced_target_variables(final_df)
        
        # Clean up
        final_df = final_df.replace([np.inf, -np.inf], np.nan)
        final_df = final_df.fillna(method='ffill').fillna(0)
        
        # Remove rows with missing targets
        target_cols = [col for col in final_df.columns if col.startswith('target_')]
        if target_cols:
            final_df = final_df.dropna(subset=target_cols)
        
        logger.info(f"Final enhanced dataset: {final_df.shape}")
        
        if save_path:
            try:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                final_df.to_parquet(save_path)
                
                # Save metadata
                metadata = {
                    'creation_time': datetime.now().isoformat(),
                    'symbols': sorted(final_df['symbol'].unique().tolist()),
                    'sectors': sorted(final_df['sector'].unique().tolist()),
                    'shape': final_df.shape,
                    'date_range': {
                        'start': final_df.index.min().isoformat(),
                        'end': final_df.index.max().isoformat()
                    },
                    'feature_groups': {
                        'market': len([col for col in final_df.columns if col in ['Open', 'High', 'Low', 'Close', 'Volume']]),
                        'technical': len([col for col in final_df.columns if any(tech in col for tech in ['SMA', 'EMA', 'RSI', 'MACD', 'BB'])]),
                        'news': len([col for col in final_df.columns if 'news' in col]),
                        'targets': len([col for col in final_df.columns if col.startswith(('target_', 'return_', 'direction_'))])
                    }
                }
                
                metadata_path = Path(save_path).with_suffix('.json')
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)
                
                logger.info(f"Saved to {save_path} with metadata")
            except Exception as e:
                logger.warning(f"Could not save: {e}")
        
        return final_df
    
    def _add_enhanced_news_features(self, df: pd.DataFrame, news: List[NewsArticle]) -> pd.DataFrame:
        """Add enhanced news features with error handling"""
        
        # Initialize enhanced news columns
        news_columns = [
            'news_count', 'news_count_1d', 'news_count_7d', 'news_count_30d',
            'avg_relevance', 'avg_sentiment', 'sentiment_std', 'sentiment_positive_ratio',
            'latest_news_age', 'news_recency_score', 'news_momentum_7d',
            'news_word_count_avg', 'news_source_diversity'
        ]
        
        for col in news_columns:
            df[col] = 0.0
        
        if not news:
            return df
        
        # Sort news by date
        try:
            news_sorted = sorted(news, key=lambda x: x.date)
        except:
            logger.warning("Could not sort news by date")
            return df
        
        # For each trading day, calculate enhanced news features
        for date_idx in df.index:
            try:
                row_date = date_idx.date()
                
                # Find news up to this date
                historical_news = [
                    article for article in news_sorted 
                    if article.date.date() <= row_date
                ]
                
                if historical_news:
                    # Calculate time windows
                    news_1d = [a for a in historical_news if (row_date - a.date.date()).days <= 1]
                    news_7d = [a for a in historical_news if (row_date - a.date.date()).days <= 7]
                    news_30d = [a for a in historical_news if (row_date - a.date.date()).days <= 30]
                    
                    # Basic counts
                    df.loc[date_idx, 'news_count_1d'] = len(news_1d)
                    df.loc[date_idx, 'news_count_7d'] = len(news_7d)
                    df.loc[date_idx, 'news_count_30d'] = len(news_30d)
                    df.loc[date_idx, 'news_count'] = len(news_7d)
                    
                    # Relevance features
                    if news_7d:
                        relevance_scores = [a.relevance_score for a in news_7d]
                        df.loc[date_idx, 'avg_relevance'] = np.mean(relevance_scores)
                    
                    # Enhanced sentiment features
                    if news_7d:
                        sentiment_scores = [a.sentiment_score for a in news_7d if a.sentiment_score is not None]
                        if sentiment_scores:
                            df.loc[date_idx, 'avg_sentiment'] = np.mean(sentiment_scores)
                            df.loc[date_idx, 'sentiment_std'] = np.std(sentiment_scores)
                            df.loc[date_idx, 'sentiment_positive_ratio'] = np.mean([s > 0 for s in sentiment_scores])
                    
                    # Latest news age and recency
                    latest_date = max(a.date.date() for a in historical_news)
                    days_since = (row_date - latest_date).days
                    df.loc[date_idx, 'latest_news_age'] = days_since
                    
                    # Recency score
                    df.loc[date_idx, 'news_recency_score'] = np.exp(-days_since * 0.1)
                    
                    # Content features
                    if news_7d:
                        word_counts = [a.word_count for a in news_7d if a.word_count > 0]
                        if word_counts:
                            df.loc[date_idx, 'news_word_count_avg'] = np.mean(word_counts)
                    
                    # Source diversity
                    if news_30d:
                        unique_sources = len(set(a.source for a in news_30d))
                        df.loc[date_idx, 'news_source_diversity'] = unique_sources
                        
            except Exception as e:
                logger.debug(f"Error processing news features for {date_idx}: {e}")
                continue
        
        # Add rolling and lag features
        try:
            df['news_momentum_7d'] = df['news_count_7d'].diff()
        except:
            df['news_momentum_7d'] = 0
        
        return df
    
    def _add_enhanced_target_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add enhanced target variables with error handling"""
        
        for horizon in [5, 30, 90]:
            try:
                # Future prices by symbol
                df[f'target_{horizon}d'] = df.groupby('symbol')['Close'].shift(-horizon)
                
                # Future returns
                df[f'return_{horizon}d'] = (
                    df[f'target_{horizon}d'] / df['Close'] - 1
                )
                
                # Direction (binary)
                df[f'direction_{horizon}d'] = (
                    df[f'return_{horizon}d'] > 0
                ).astype(int)
                
            except Exception as e:
                logger.warning(f"Error creating target variables for horizon {horizon}: {e}")
        
        return df

# Enhanced test function
def enhanced_test():
    """Test the enhanced data collection system"""
    print("🧪 Testing Enhanced Data Collection System")
    print("="*60)
    
    try:
        # Initialize
        collector = DataCollector()
        print("✅ Enhanced collector initialized")
        print(f"📊 Stock universe: {len(collector.config['symbols'])} stocks")
        
        # Test with a subset for faster testing
        test_symbols = ['AAPL', 'MSFT', 'JPM']
        print(f"🧪 Testing with: {test_symbols}")
        
        # Test market data
        print("\n📈 Testing enhanced market data collection...")
        market_data = collector.collect_market_data(symbols=test_symbols, use_parallel=False)
        
        if market_data:
            print(f"✅ Market data collected for {len(market_data)} symbols")
            for symbol, data in market_data.items():
                print(f"  {symbol} ({data.sector}): {len(data.data)} days, {len(data.technical_indicators.columns)} indicators")
        else:
            print("❌ No market data collected")
            return False
        
        # Test news data
        print("\n📰 Testing enhanced news data collection...")
        news_data = collector.collect_news_data(symbols=test_symbols)
        
        if news_data:
            total_articles = sum(len(articles) for articles in news_data.values())
            print(f"✅ News data collected: {total_articles} total articles")
            
            for symbol, articles in news_data.items():
                if articles:
                    sources = set(a.source for a in articles)
                    avg_sentiment = np.mean([a.sentiment_score for a in articles if a.sentiment_score is not None])
                    print(f"  {symbol}: {len(articles)} articles, sources: {sources}, avg sentiment: {avg_sentiment:.3f}")
                else:
                    print(f"  {symbol}: No articles")
        else:
            print("❌ No news data collected")
            return False
        
        # Test combined dataset
        print("\n🔄 Testing enhanced combined dataset creation...")
        combined_df = collector.create_combined_dataset(
            market_data, news_data,
            save_path='data/processed/enhanced_dataset.parquet'
        )
        
        if not combined_df.empty:
            print(f"✅ Enhanced dataset created: {combined_df.shape}")
            
            # Enhanced feature summary
            feature_groups = {
                'Market': [col for col in combined_df.columns if col in ['Open', 'High', 'Low', 'Close', 'Volume']],
                'Technical': [col for col in combined_df.columns if any(tech in col for tech in ['SMA', 'EMA', 'RSI', 'MACD', 'BB'])],
                'News': [col for col in combined_df.columns if 'news' in col],
                'Targets': [col for col in combined_df.columns if col.startswith(('target_', 'return_', 'direction_'))]
            }
            
            print(f"\n📊 Enhanced Feature Summary:")
            for ftype, features in feature_groups.items():
                print(f"  {ftype}: {len(features)} features")
            
            print(f"\n🏢 Sectors: {sorted(combined_df['sector'].unique())}")
            print(f"📅 Date range: {combined_df.index.min().date()} to {combined_df.index.max().date()}")
            
            # Sample data
            print(f"\n📋 Sample Enhanced Data:")
            sample_cols = ['symbol', 'sector', 'Close', 'news_count', 'avg_sentiment']
            available_cols = [col for col in sample_cols if col in combined_df.columns]
            print(combined_df[available_cols].head(3))
            
            print(f"\n🎉 ALL ENHANCED TESTS PASSED!")
            return True
        else:
            print("❌ Enhanced dataset creation failed")
            return False
            
    except Exception as e:
        print(f"❌ Enhanced test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# Main execution
if __name__ == "__main__":
    print("🚀 ENHANCED DATA COLLECTION SYSTEM")
    print("="*60)
    
    # Run enhanced test
    success = enhanced_test()
    
    if success:
        print(f"\n✅ ENHANCED SYSTEM IS READY!")
        print(f"\n📖 Enhanced Usage:")
        print(f"```python")
        print(f"from src.data_loader import DataCollector")
        print(f"")
        print(f"collector = DataCollector(config_path='configs/data_config.yaml')")
        print(f"")
        print(f"# Collect data for all stocks")
        print(f"market_data = collector.collect_market_data(use_parallel=True)")
        print(f"news_data = collector.collect_news_data()")
        print(f"")
        print(f"# Create enhanced dataset")
        print(f"dataset = collector.create_combined_dataset(")
        print(f"    market_data, news_data,")
        print(f"    save_path='data/processed/enhanced_dataset.parquet'")
        print(f")")
        print(f"```")
        
    else:
        print(f"\n❌ ENHANCED SYSTEM NEEDS FIXES!")