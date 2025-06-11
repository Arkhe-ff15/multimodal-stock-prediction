"""
src/sentiment.py - Enhanced FinBERT Sentiment Analysis
===================================================

✅ COMPLETE IMPLEMENTATION:
1. FinBERT integration with fallback to keyword-based analysis
2. Robust text preprocessing and quality filtering
3. Batch processing with memory management
4. Multi-horizon sentiment feature creation
5. Comprehensive caching and error handling
6. Statistical validation and confidence scoring

Designed to work seamlessly with the modular data system.
"""

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta
import re
import warnings
from pathlib import Path
import json
import pickle
import time
import gc

# Try to import FinBERT components with comprehensive error handling
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    FINBERT_AVAILABLE = True
except ImportError:
    FINBERT_AVAILABLE = False

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

@dataclass
class SentimentConfig:
    """Configuration for FinBERT sentiment analysis"""
    model_name: str = "ProsusAI/finbert"
    batch_size: int = 16
    max_length: int = 512
    confidence_threshold: float = 0.6  # Relaxed for better coverage
    relevance_threshold: float = 0.7   # Relaxed for better coverage
    quality_threshold: float = 0.5     # Relaxed for better coverage
    cache_results: bool = True
    device: str = "auto"
    
    # Quality filters
    min_text_length: int = 5           # Reduced minimum
    max_text_length: int = 2000
    filter_low_quality: bool = True
    
    # Processing parameters
    use_confidence_weighting: bool = True
    enable_batch_processing: bool = True
    max_memory_mb: int = 1000

@dataclass
class SentimentResult:
    """Individual sentiment analysis result"""
    text: str
    sentiment_score: float      # [-1, 1] normalized score
    confidence: float           # [0, 1] confidence in prediction
    label: str                  # 'positive', 'negative', 'neutral'
    raw_scores: Dict[str, float]  # Raw probabilities from model
    processing_time: float
    source: str = ""
    relevance_score: float = 1.0
    word_count: int = 0

class FinBERTSentimentAnalyzer:
    """
    Enhanced FinBERT sentiment analyzer with comprehensive error handling
    
    Features:
    - Automatic fallback to keyword-based analysis
    - Memory-efficient batch processing
    - Multi-tier quality assessment
    - Comprehensive caching system
    - Statistical validation
    """
    
    def __init__(self, config: SentimentConfig = None, cache_dir: str = "data/sentiment"):
        """Initialize with robust error handling and fallbacks"""
        self.config = config or SentimentConfig()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup device
        self.device = self._setup_device()
        logger.info(f"💻 Device: {self.device}")
        
        # Initialize models
        self.model_loaded = self._load_finbert_model()
        
        # Processing statistics
        self.processing_stats = {
            'total_processed': 0,
            'high_quality': 0,
            'medium_quality': 0,
            'low_quality': 0,
            'processing_times': [],
            'errors_handled': 0,
            'cache_hits': 0,
            'model_predictions': 0,
            'fallback_predictions': 0
        }
        
        # Enhanced keyword dictionaries
        self._initialize_keyword_dictionaries()
        
        logger.info(f"🧠 FinBERT Analyzer initialized (Model available: {self.model_loaded})")
    
    def _setup_device(self) -> torch.device:
        """Setup device with intelligent selection"""
        try:
            if self.config.device == "auto":
                if torch.cuda.is_available():
                    # Check GPU memory
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    if gpu_memory > 4:  # At least 4GB
                        device = torch.device("cuda")
                        logger.info(f"🚀 Using CUDA (GPU memory: {gpu_memory:.1f}GB)")
                        return device
                    else:
                        logger.warning(f"⚠️ Limited GPU memory ({gpu_memory:.1f}GB), using CPU")
                
                return torch.device("cpu")
            else:
                return torch.device(self.config.device)
                
        except Exception as e:
            logger.warning(f"⚠️ Device setup failed: {e}, using CPU")
            return torch.device("cpu")
    
    def _load_finbert_model(self) -> bool:
        """Load FinBERT model with comprehensive error handling"""
        if not FINBERT_AVAILABLE:
            logger.warning("⚠️ Transformers library not available, using keyword analysis only")
            return False
        
        try:
            logger.info(f"📥 Loading FinBERT model: {self.config.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                cache_dir=str(self.cache_dir / "transformers_cache")
            )
            
            # Load model
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.config.model_name,
                cache_dir=str(self.cache_dir / "transformers_cache")
            )
            
            # Move to device with error handling
            try:
                self.model.to(self.device)
                logger.info(f"📱 Model moved to {self.device}")
            except RuntimeError as e:
                logger.warning(f"⚠️ Could not move to {self.device}: {e}, using CPU")
                self.device = torch.device("cpu")
                self.model.to(self.device)
            
            self.model.eval()
            
            # Get label mappings
            self.id2label = self.model.config.id2label
            self.label2id = self.model.config.label2id
            
            logger.info(f"✅ FinBERT loaded successfully. Labels: {self.id2label}")
            return True
            
        except Exception as e:
            logger.error(f"❌ FinBERT loading failed: {e}")
            logger.info("🔄 Falling back to keyword-based analysis")
            return False
    
    def _initialize_keyword_dictionaries(self):
        """Initialize comprehensive keyword dictionaries for fallback analysis"""
        self.positive_keywords = {
            # Financial performance
            'growth', 'profit', 'revenue', 'earnings', 'gain', 'rise', 'increase',
            'beat', 'exceed', 'outperform', 'strong', 'robust', 'solid',
            
            # Market sentiment
            'bull', 'bullish', 'up', 'surge', 'rally', 'boost', 'soar',
            'optimistic', 'positive', 'confident', 'momentum',
            
            # Corporate actions
            'upgrade', 'buy', 'acquisition', 'merger', 'expansion', 'launch',
            'dividend', 'buyback', 'partnership', 'breakthrough',
            
            # Performance indicators
            'record', 'milestone', 'achievement', 'success', 'leader',
            'improved', 'better', 'higher', 'stronger'
        }
        
        self.negative_keywords = {
            # Financial performance
            'loss', 'losses', 'decline', 'fall', 'decrease', 'drop',
            'miss', 'missed', 'weak', 'poor', 'disappointing',
            
            # Market sentiment
            'bear', 'bearish', 'down', 'plunge', 'crash', 'tumble',
            'pessimistic', 'negative', 'concern', 'worry', 'fear',
            
            # Corporate issues
            'downgrade', 'sell', 'lawsuit', 'investigation', 'fraud',
            'bankruptcy', 'layoffs', 'restructuring', 'closure',
            
            # Risk indicators
            'risk', 'volatile', 'uncertainty', 'crisis', 'problem',
            'challenge', 'threat', 'warning', 'alert'
        }
        
        # Create weighted keyword scores
        self.keyword_weights = {
            # High impact financial terms
            **{word: 2.0 for word in ['profit', 'loss', 'earnings', 'revenue', 'beat', 'miss']},
            # Medium impact terms
            **{word: 1.5 for word in ['growth', 'decline', 'strong', 'weak', 'upgrade', 'downgrade']},
            # Standard impact terms
            **{word: 1.0 for word in self.positive_keywords | self.negative_keywords}
        }
    
    def preprocess_text(self, text: str) -> str:
        """Enhanced text preprocessing for financial content"""
        if not text or not isinstance(text, str):
            return ""
        
        try:
            # Convert to lowercase for processing
            text = text.lower()
            
            # Remove URLs and email addresses
            text = re.sub(r'http[s]?://\S+', '', text)
            text = re.sub(r'\S+@\S+', '', text)
            
            # Clean financial symbols and numbers
            text = re.sub(r'\$\d+(?:\.\d+)?[kmb]?', ' AMOUNT ', text)  # Replace amounts
            text = re.sub(r'\b\d+(?:\.\d+)?%', ' PERCENT ', text)      # Replace percentages
            
            # Remove excessive whitespace and special characters
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'[^\w\s\$\%\.\,\!\?\-\(\)]', ' ', text)
            
            # Truncate to max length
            if len(text) > self.config.max_text_length:
                text = text[:self.config.max_text_length]
            
            return text.strip()
            
        except Exception as e:
            logger.warning(f"⚠️ Text preprocessing failed: {e}")
            return str(text)[:self.config.max_text_length] if text else ""
    
    def analyze_sentiment_single(self, text: str, source: str = "", 
                                relevance_score: float = 1.0) -> SentimentResult:
        """Analyze sentiment for single text with model selection"""
        start_time = time.time()
        
        # Preprocess text
        cleaned_text = self.preprocess_text(text)
        word_count = len(cleaned_text.split())
        
        # Quality check
        if len(cleaned_text) < self.config.min_text_length:
            return self._create_empty_result(text, source, relevance_score, start_time, word_count)
        
        # Choose analysis method
        if self.model_loaded:
            result = self._analyze_with_finbert(cleaned_text, text, source, relevance_score, start_time, word_count)
            self.processing_stats['model_predictions'] += 1
        else:
            result = self._analyze_with_keywords(cleaned_text, text, source, relevance_score, start_time, word_count)
            self.processing_stats['fallback_predictions'] += 1
        
        # Update statistics
        self._update_processing_stats(result)
        
        return result
    
    def _analyze_with_finbert(self, cleaned_text: str, original_text: str,
                             source: str, relevance_score: float, 
                             start_time: float, word_count: int) -> SentimentResult:
        """Analyze using FinBERT model"""
        try:
            # Tokenize
            inputs = self.tokenizer(
                cleaned_text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=self.config.max_length
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = F.softmax(logits, dim=-1)
            
            # Extract results
            probs = probabilities.cpu().numpy()[0]
            predicted_id = np.argmax(probs)
            predicted_label = self.id2label[predicted_id]
            confidence = float(np.max(probs))
            
            # Create raw scores dictionary
            raw_scores = {self.id2label[i]: float(probs[i]) for i in range(len(probs))}
            
            # Calculate normalized sentiment score [-1, 1]
            sentiment_score = self._calculate_sentiment_score(probs, predicted_label)
            
            processing_time = time.time() - start_time
            
            return SentimentResult(
                text=original_text,
                sentiment_score=float(sentiment_score),
                confidence=confidence,
                label=predicted_label,
                raw_scores=raw_scores,
                processing_time=processing_time,
                source=source,
                relevance_score=relevance_score,
                word_count=word_count
            )
            
        except Exception as e:
            logger.warning(f"⚠️ FinBERT analysis failed: {e}")
            self.processing_stats['errors_handled'] += 1
            return self._analyze_with_keywords(cleaned_text, original_text, source, 
                                             relevance_score, start_time, word_count)
    
    def _calculate_sentiment_score(self, probs: np.ndarray, predicted_label: str) -> float:
        """Calculate normalized sentiment score from FinBERT probabilities"""
        try:
            # Standard FinBERT mapping: 0=negative, 1=neutral, 2=positive
            if len(probs) == 3:
                sentiment_score = probs[2] - probs[0]  # positive - negative
            else:
                # Fallback for different label schemes
                if predicted_label.lower() in ['positive', 'pos']:
                    sentiment_score = np.max(probs)
                elif predicted_label.lower() in ['negative', 'neg']:
                    sentiment_score = -np.max(probs)
                else:
                    sentiment_score = 0.0
            
            # Clip to [-1, 1] range
            return np.clip(sentiment_score, -1, 1)
            
        except Exception:
            return 0.0
    
    def _analyze_with_keywords(self, cleaned_text: str, original_text: str,
                              source: str, relevance_score: float,
                              start_time: float, word_count: int) -> SentimentResult:
        """Enhanced keyword-based sentiment analysis"""
        try:
            words = cleaned_text.lower().split()
            
            # Calculate weighted sentiment scores
            pos_score = 0.0
            neg_score = 0.0
            
            for word in words:
                if word in self.positive_keywords:
                    weight = self.keyword_weights.get(word, 1.0)
                    pos_score += weight
                elif word in self.negative_keywords:
                    weight = self.keyword_weights.get(word, 1.0)
                    neg_score += weight
            
            # Calculate final sentiment score
            total_score = pos_score + neg_score
            if total_score > 0:
                sentiment_score = (pos_score - neg_score) / (pos_score + neg_score)
                confidence = min(total_score / (word_count * 0.1), 0.95)  # Cap at 0.95
            else:
                sentiment_score = 0.0
                confidence = 0.1  # Low confidence for neutral
            
            # Determine label
            if sentiment_score > 0.1:
                label = 'positive'
            elif sentiment_score < -0.1:
                label = 'negative'
            else:
                label = 'neutral'
            
            # Create raw scores
            raw_scores = {
                'positive': max(0, sentiment_score),
                'negative': max(0, -sentiment_score),
                'neutral': 1 - abs(sentiment_score)
            }
            
            processing_time = time.time() - start_time
            
            return SentimentResult(
                text=original_text,
                sentiment_score=float(sentiment_score),
                confidence=confidence,
                label=label,
                raw_scores=raw_scores,
                processing_time=processing_time,
                source=source,
                relevance_score=relevance_score,
                word_count=word_count
            )
            
        except Exception as e:
            logger.warning(f"⚠️ Keyword analysis failed: {e}")
            self.processing_stats['errors_handled'] += 1
            return self._create_empty_result(original_text, source, relevance_score, start_time, word_count)
    
    def _create_empty_result(self, text: str, source: str, relevance_score: float,
                            start_time: float, word_count: int) -> SentimentResult:
        """Create empty/neutral sentiment result"""
        return SentimentResult(
            text=text,
            sentiment_score=0.0,
            confidence=0.0,
            label='neutral',
            raw_scores={'positive': 0.33, 'negative': 0.33, 'neutral': 0.34},
            processing_time=time.time() - start_time,
            source=source,
            relevance_score=relevance_score,
            word_count=word_count
        )
    
    def _update_processing_stats(self, result: SentimentResult):
        """Update processing statistics"""
        self.processing_stats['total_processed'] += 1
        self.processing_stats['processing_times'].append(result.processing_time)
        
        # Classify quality
        if result.confidence >= 0.8:
            self.processing_stats['high_quality'] += 1
        elif result.confidence >= 0.5:
            self.processing_stats['medium_quality'] += 1
        else:
            self.processing_stats['low_quality'] += 1
    
    def analyze_sentiment_batch(self, texts: List[str], sources: List[str] = None,
                               relevance_scores: List[float] = None) -> List[SentimentResult]:
        """Batch process sentiment analysis with memory management"""
        if sources is None:
            sources = [""] * len(texts)
        if relevance_scores is None:
            relevance_scores = [1.0] * len(texts)
        
        logger.info(f"📊 Processing {len(texts)} texts in batches...")
        
        results = []
        batch_size = self.config.batch_size
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_sources = sources[i:i+batch_size]
            batch_relevance = relevance_scores[i:i+batch_size]
            
            # Process batch
            batch_results = []
            for text, source, relevance in zip(batch_texts, batch_sources, batch_relevance):
                try:
                    result = self.analyze_sentiment_single(text, source, relevance)
                    batch_results.append(result)
                except Exception as e:
                    logger.warning(f"⚠️ Batch processing error: {e}")
                    batch_results.append(self._create_empty_result(text, source, relevance, time.time(), 0))
            
            results.extend(batch_results)
            
            # Progress logging
            progress = min((i + batch_size) / len(texts) * 100, 100)
            if i % (batch_size * 4) == 0:  # Log every 4 batches
                logger.info(f"📊 Progress: {progress:.1f}% ({len(results)}/{len(texts)})")
            
            # Memory cleanup
            if torch.cuda.is_available() and i % (batch_size * 2) == 0:
                torch.cuda.empty_cache()
            
            if i % (batch_size * 8) == 0:  # Garbage collect every 8 batches
                gc.collect()
        
        logger.info(f"✅ Batch processing complete: {len(results)} results")
        return results
    
    def process_news_data(self, news_data: Dict[str, List], symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """Process news data through sentiment analysis"""
        logger.info("📰 Processing news data through sentiment analysis...")
        
        all_sentiment_data = {}
        
        for symbol in symbols:
            if symbol not in news_data or not news_data[symbol]:
                logger.info(f"📰 No news data for {symbol}, creating empty sentiment")
                all_sentiment_data[symbol] = pd.DataFrame()
                continue
            
            logger.info(f"📊 Processing {len(news_data[symbol])} articles for {symbol}")
            
            # Check cache
            cache_file = self.cache_dir / f"{symbol}_sentiment_results.pkl"
            if self.config.cache_results and cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        cached_results = pickle.load(f)
                    logger.info(f"📂 Loaded cached sentiment for {symbol}")
                    all_sentiment_data[symbol] = cached_results
                    self.processing_stats['cache_hits'] += 1
                    continue
                except Exception as e:
                    logger.warning(f"⚠️ Cache load failed for {symbol}: {e}")
            
            try:
                # Prepare texts and metadata
                texts, sources, relevance_scores, metadata = self._prepare_articles_for_processing(
                    news_data[symbol], symbol
                )
                
                if not texts:
                    logger.warning(f"⚠️ No valid texts for {symbol}")
                    all_sentiment_data[symbol] = pd.DataFrame()
                    continue
                
                # Process sentiment
                sentiment_results = self.analyze_sentiment_batch(texts, sources, relevance_scores)
                
                # Create DataFrame
                sentiment_df = self._create_sentiment_dataframe(
                    sentiment_results, metadata, symbol
                )
                
                # Apply quality filtering
                if self.config.filter_low_quality:
                    sentiment_df = self._apply_quality_filters(sentiment_df)
                
                # Cache results
                if self.config.cache_results:
                    try:
                        with open(cache_file, 'wb') as f:
                            pickle.dump(sentiment_df, f)
                        logger.info(f"💾 Cached sentiment results for {symbol}")
                    except Exception as e:
                        logger.warning(f"⚠️ Cache save failed for {symbol}: {e}")
                
                all_sentiment_data[symbol] = sentiment_df
                logger.info(f"✅ Processed sentiment for {symbol}: {len(sentiment_df)} results")
                
            except Exception as e:
                logger.error(f"❌ Sentiment processing failed for {symbol}: {e}")
                all_sentiment_data[symbol] = pd.DataFrame()
        
        return all_sentiment_data
    
    def _prepare_articles_for_processing(self, articles: List[Any], symbol: str) -> Tuple[List[str], List[str], List[float], List[Dict]]:
        """Prepare articles for sentiment processing"""
        texts = []
        sources = []
        relevance_scores = []
        metadata = []
        
        for article in articles:
            try:
                # Extract text content
                if hasattr(article, 'title') and hasattr(article, 'content'):
                    combined_text = f"{article.title}. {article.content}"
                    article_source = getattr(article, 'source', 'unknown')
                    article_date = getattr(article, 'date', datetime.now())
                    article_relevance = getattr(article, 'relevance_score', 1.0)
                    article_url = getattr(article, 'url', '')
                    article_title = article.title
                elif isinstance(article, dict):
                    combined_text = f"{article.get('title', '')}. {article.get('content', '')}"
                    article_source = article.get('source', 'unknown')
                    article_date = article.get('date', datetime.now())
                    article_relevance = article.get('relevance_score', 1.0)
                    article_url = article.get('url', '')
                    article_title = article.get('title', '')
                else:
                    combined_text = str(article)
                    article_source = 'unknown'
                    article_date = datetime.now()
                    article_relevance = 1.0
                    article_url = ''
                    article_title = combined_text[:100]
                
                if len(combined_text.strip()) >= self.config.min_text_length:
                    texts.append(combined_text)
                    sources.append(article_source)
                    relevance_scores.append(article_relevance)
                    
                    metadata.append({
                        'date': article_date,
                        'source': article_source,
                        'url': article_url,
                        'title': article_title,
                        'word_count': len(combined_text.split())
                    })
                
            except Exception as e:
                logger.warning(f"⚠️ Error preparing article for {symbol}: {e}")
                continue
        
        return texts, sources, relevance_scores, metadata
    
    def _create_sentiment_dataframe(self, sentiment_results: List[SentimentResult], 
                                   metadata: List[Dict], symbol: str) -> pd.DataFrame:
        """Create DataFrame from sentiment results"""
        sentiment_data = []
        
        for i, result in enumerate(sentiment_results):
            if i < len(metadata):
                meta = metadata[i]
                
                sentiment_data.append({
                    'date': meta['date'],
                    'symbol': symbol,
                    'source': result.source,
                    'sentiment_score': result.sentiment_score,
                    'confidence': result.confidence,
                    'label': result.label,
                    'relevance_score': result.relevance_score,
                    'processing_time': result.processing_time,
                    'word_count': result.word_count,
                    'positive_prob': result.raw_scores.get('positive', 0.0),
                    'negative_prob': result.raw_scores.get('negative', 0.0),
                    'neutral_prob': result.raw_scores.get('neutral', 0.0),
                    'url': meta['url'],
                    'title': meta['title'],
                    'original_word_count': meta['word_count']
                })
        
        if sentiment_data:
            df = pd.DataFrame(sentiment_data)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            return df
        else:
            return pd.DataFrame()
    
    def _apply_quality_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply intelligent quality filters"""
        if df.empty:
            return df
        
        original_count = len(df)
        
        try:
            # Multi-tier quality filtering
            high_quality = (
                (df['confidence'] >= self.config.confidence_threshold) &
                (df['relevance_score'] >= self.config.relevance_threshold) &
                (df['word_count'] >= 10)
            )
            
            medium_quality = (
                (df['confidence'] >= 0.4) &
                (df['relevance_score'] >= 0.5) &
                (df['word_count'] >= 5)
            )
            
            # Keep high quality articles, and medium quality if we don't have enough high quality
            df_high = df[high_quality].copy()
            
            if len(df_high) < original_count * 0.3:  # If less than 30% are high quality
                df_filtered = df[medium_quality].copy()
                logger.info(f"📊 Using medium quality filter ({len(df_filtered)} articles)")
            else:
                df_filtered = df_high
                logger.info(f"📊 Using high quality filter ({len(df_filtered)} articles)")
            
            # Ensure minimum retention
            if len(df_filtered) < max(5, original_count * 0.2):
                df_sorted = df.sort_values('confidence', ascending=False)
                keep_count = max(5, int(original_count * 0.2))
                df_filtered = df_sorted.head(keep_count).copy()
                logger.info(f"📊 Minimum retention applied ({len(df_filtered)} articles)")
            
            # Mark quality tiers
            df_filtered['quality_tier'] = 'high'
            df_filtered.loc[df_filtered['confidence'] < self.config.confidence_threshold, 'quality_tier'] = 'medium'
            df_filtered.loc[df_filtered['confidence'] < 0.4, 'quality_tier'] = 'low'
            
            logger.info(f"📊 Quality filtering: {original_count} → {len(df_filtered)} articles retained")
            return df_filtered
            
        except Exception as e:
            logger.error(f"❌ Quality filtering failed: {e}")
            df['quality_tier'] = 'unknown'
            return df
    
    def create_sentiment_features(self, sentiment_data: Dict[str, pd.DataFrame],
                                 horizons: List[int] = [5, 30, 90]) -> Dict[str, pd.DataFrame]:
        """Create comprehensive sentiment features for different horizons"""
        logger.info(f"🔧 Creating sentiment features for horizons: {horizons}")
        
        all_features = {}
        
        for symbol, sentiment_df in sentiment_data.items():
            if sentiment_df.empty:
                logger.warning(f"⚠️ No sentiment data for {symbol}")
                all_features[symbol] = pd.DataFrame()
                continue
            
            try:
                logger.info(f"🔧 Creating features for {symbol}")
                
                # Get date range
                start_date = sentiment_df['date'].min()
                end_date = sentiment_df['date'].max()
                
                # Create daily feature grid
                date_range = pd.date_range(start_date, end_date, freq='D')
                daily_features = []
                
                for current_date in date_range:
                    row = {'date': current_date, 'symbol': symbol}
                    
                    for horizon in horizons:
                        # Get sentiment within lookback window
                        lookback_start = current_date - timedelta(days=horizon)
                        window_sentiment = sentiment_df[
                            (sentiment_df['date'] >= lookback_start) & 
                            (sentiment_df['date'] <= current_date)
                        ]
                        
                        if len(window_sentiment) > 0:
                            # Basic sentiment statistics
                            row.update(self._calculate_basic_sentiment_stats(window_sentiment, horizon))
                            
                            # Quality-weighted features
                            row.update(self._calculate_weighted_sentiment_stats(window_sentiment, horizon))
                            
                            # Source-specific features
                            row.update(self._calculate_source_specific_stats(window_sentiment, horizon))
                            
                            # Temporal pattern features
                            row.update(self._calculate_temporal_pattern_stats(window_sentiment, horizon, current_date))
                        else:
                            # Fill with zeros for no sentiment data
                            row.update(self._create_zero_sentiment_features(horizon))
                    
                    daily_features.append(row)
                
                # Create DataFrame
                features_df = pd.DataFrame(daily_features)
                features_df['date'] = pd.to_datetime(features_df['date'])
                features_df = features_df.set_index('date')
                
                all_features[symbol] = features_df
                logger.info(f"✅ Created sentiment features for {symbol}: {features_df.shape}")
                
            except Exception as e:
                logger.error(f"❌ Feature creation failed for {symbol}: {e}")
                all_features[symbol] = pd.DataFrame()
        
        return all_features
    
    def _calculate_basic_sentiment_stats(self, sentiment_df: pd.DataFrame, horizon: int) -> Dict[str, float]:
        """Calculate basic sentiment statistics"""
        return {
            f'sentiment_mean_{horizon}d': sentiment_df['sentiment_score'].mean(),
            f'sentiment_std_{horizon}d': sentiment_df['sentiment_score'].std(),
            f'sentiment_count_{horizon}d': len(sentiment_df),
            f'sentiment_positive_ratio_{horizon}d': (sentiment_df['sentiment_score'] > 0.1).mean(),
            f'sentiment_negative_ratio_{horizon}d': (sentiment_df['sentiment_score'] < -0.1).mean(),
            f'sentiment_neutral_ratio_{horizon}d': (abs(sentiment_df['sentiment_score']) <= 0.1).mean()
        }
    
    def _calculate_weighted_sentiment_stats(self, sentiment_df: pd.DataFrame, horizon: int) -> Dict[str, float]:
        """Calculate confidence-weighted sentiment statistics"""
        weights = sentiment_df['confidence']
        if weights.sum() > 0:
            weighted_mean = np.average(sentiment_df['sentiment_score'], weights=weights)
            weighted_std = np.sqrt(np.average((sentiment_df['sentiment_score'] - weighted_mean)**2, weights=weights))
        else:
            weighted_mean = 0.0
            weighted_std = 0.0
        
        return {
            f'sentiment_weighted_mean_{horizon}d': weighted_mean,
            f'sentiment_weighted_std_{horizon}d': weighted_std,
            f'sentiment_avg_confidence_{horizon}d': sentiment_df['confidence'].mean(),
            f'sentiment_avg_relevance_{horizon}d': sentiment_df['relevance_score'].mean()
        }
    
    def _calculate_source_specific_stats(self, sentiment_df: pd.DataFrame, horizon: int) -> Dict[str, float]:
        """Calculate source-specific sentiment statistics"""
        stats = {}
        
        # Define important sources
        important_sources = ['sec_edgar', 'federal_reserve', 'investor_relations', 'bloomberg', 'reuters']
        
        for source in important_sources:
            source_data = sentiment_df[sentiment_df['source'].str.contains(source, case=False, na=False)]
            if len(source_data) > 0:
                stats[f'{source}_sentiment_{horizon}d'] = source_data['sentiment_score'].mean()
                stats[f'{source}_count_{horizon}d'] = len(source_data)
                stats[f'{source}_confidence_{horizon}d'] = source_data['confidence'].mean()
            else:
                stats[f'{source}_sentiment_{horizon}d'] = 0.0
                stats[f'{source}_count_{horizon}d'] = 0
                stats[f'{source}_confidence_{horizon}d'] = 0.0
        
        # Overall source diversity
        stats[f'sentiment_source_diversity_{horizon}d'] = sentiment_df['source'].nunique()
        
        return stats
    
    def _calculate_temporal_pattern_stats(self, sentiment_df: pd.DataFrame, horizon: int, current_date: datetime) -> Dict[str, float]:
        """Calculate temporal pattern statistics"""
        stats = {}
        
        if len(sentiment_df) > 1:
            # Recent vs older sentiment trend
            mid_date = current_date - timedelta(days=horizon//2)
            recent_sentiment = sentiment_df[sentiment_df['date'] >= mid_date]
            older_sentiment = sentiment_df[sentiment_df['date'] < mid_date]
            
            if len(recent_sentiment) > 0 and len(older_sentiment) > 0:
                trend = recent_sentiment['sentiment_score'].mean() - older_sentiment['sentiment_score'].mean()
                stats[f'sentiment_trend_{horizon}d'] = trend
            else:
                stats[f'sentiment_trend_{horizon}d'] = 0.0
            
            # Sentiment momentum (acceleration)
            if horizon >= 10:
                quarter_days = horizon // 4
                periods = []
                for i in range(4):
                    period_start = current_date - timedelta(days=(i+1)*quarter_days)
                    period_end = current_date - timedelta(days=i*quarter_days)
                    period_data = sentiment_df[
                        (sentiment_df['date'] >= period_start) & 
                        (sentiment_df['date'] < period_end)
                    ]
                    if len(period_data) > 0:
                        periods.append(period_data['sentiment_score'].mean())
                
                if len(periods) >= 3:
                    # Calculate momentum as difference between recent and earlier periods
                    momentum = np.mean(periods[:2]) - np.mean(periods[2:])
                    stats[f'sentiment_momentum_{horizon}d'] = momentum
                else:
                    stats[f'sentiment_momentum_{horizon}d'] = 0.0
            else:
                stats[f'sentiment_momentum_{horizon}d'] = 0.0
        else:
            stats[f'sentiment_trend_{horizon}d'] = 0.0
            stats[f'sentiment_momentum_{horizon}d'] = 0.0
        
        return stats
    
    def _create_zero_sentiment_features(self, horizon: int) -> Dict[str, float]:
        """Create zero-filled features for periods with no sentiment data"""
        feature_names = [
            'sentiment_mean', 'sentiment_std', 'sentiment_positive_ratio',
            'sentiment_negative_ratio', 'sentiment_neutral_ratio',
            'sentiment_weighted_mean', 'sentiment_weighted_std',
            'sentiment_avg_confidence', 'sentiment_avg_relevance',
            'sentiment_trend', 'sentiment_momentum'
        ]
        
        zero_features = {f'{name}_{horizon}d': 0.0 for name in feature_names}
        zero_features[f'sentiment_count_{horizon}d'] = 0
        zero_features[f'sentiment_source_diversity_{horizon}d'] = 0
        
        # Source-specific zeros
        sources = ['sec_edgar', 'federal_reserve', 'investor_relations', 'bloomberg', 'reuters']
        for source in sources:
            zero_features[f'{source}_sentiment_{horizon}d'] = 0.0
            zero_features[f'{source}_count_{horizon}d'] = 0
            zero_features[f'{source}_confidence_{horizon}d'] = 0.0
        
        return zero_features
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics"""
        stats = self.processing_stats.copy()
        
        if stats['processing_times']:
            stats['avg_processing_time'] = np.mean(stats['processing_times'])
            stats['total_processing_time'] = sum(stats['processing_times'])
            stats['median_processing_time'] = np.median(stats['processing_times'])
        else:
            stats['avg_processing_time'] = 0.0
            stats['total_processing_time'] = 0.0
            stats['median_processing_time'] = 0.0
        
        if stats['total_processed'] > 0:
            stats['high_quality_ratio'] = stats['high_quality'] / stats['total_processed']
            stats['medium_quality_ratio'] = stats['medium_quality'] / stats['total_processed']
            stats['low_quality_ratio'] = stats['low_quality'] / stats['total_processed']
        else:
            stats['high_quality_ratio'] = 0.0
            stats['medium_quality_ratio'] = 0.0
            stats['low_quality_ratio'] = 0.0
        
        # Additional metadata
        stats['model_available'] = self.model_loaded
        stats['device_used'] = str(self.device)
        stats['config'] = {
            'model_name': self.config.model_name,
            'confidence_threshold': self.config.confidence_threshold,
            'batch_size': self.config.batch_size
        }
        
        return stats

# Testing and validation functions
def test_sentiment_analyzer():
    """Test the sentiment analyzer with sample data"""
    print("🧪 Testing Enhanced FinBERT Sentiment Analyzer")
    print("=" * 60)
    
    try:
        # Initialize analyzer
        config = SentimentConfig(batch_size=8, confidence_threshold=0.6)
        analyzer = FinBERTSentimentAnalyzer(config)
        
        # Test with financial news samples
        test_texts = [
            "Apple reported strong quarterly earnings, beating analyst expectations across all segments with record iPhone sales.",
            "Tesla's stock price plummeted after disappointing delivery numbers and supply chain concerns raised by management.",
            "Microsoft's cloud revenue continues to show impressive growth, driving strong performance in the enterprise segment.",
            "The Federal Reserve announced interest rate cuts to support economic recovery amid global uncertainty.",
            "Amazon's new AI initiatives position the company well for future growth in the technology sector."
        ]
        
        print(f"📊 Testing with {len(test_texts)} sample texts...")
        print(f"🤖 Model available: {analyzer.model_loaded}")
        
        # Batch processing test
        results = analyzer.analyze_sentiment_batch(test_texts)
        
        print("\n📈 Sentiment Analysis Results:")
        print("-" * 80)
        for i, (text, result) in enumerate(zip(test_texts, results)):
            print(f"\n{i+1}. Text: {text[:60]}...")
            print(f"   Sentiment: {result.sentiment_score:+.3f} ({result.label})")
            print(f"   Confidence: {result.confidence:.3f}")
            print(f"   Processing time: {result.processing_time:.3f}s")
        
        # Get statistics
        stats = analyzer.get_processing_statistics()
        print(f"\n📊 Processing Statistics:")
        print(f"   Total processed: {stats['total_processed']}")
        print(f"   High quality: {stats['high_quality_ratio']:.1%}")
        print(f"   Average processing time: {stats['avg_processing_time']:.3f}s")
        print(f"   Model predictions: {stats['model_predictions']}")
        print(f"   Fallback predictions: {stats['fallback_predictions']}")
        
        print("\n✅ Sentiment analyzer test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_sentiment_analyzer()