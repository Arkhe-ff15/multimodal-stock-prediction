"""
src/sentiment.py - FinBERT Sentiment Analyzer Implementation

Step 2: Process all news articles through FinBERT for accurate financial sentiment analysis
- Quality filtering and confidence thresholds
- Batch processing for efficiency
- Source-specific sentiment aggregation
- Caching for reproducibility
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta
import re
import warnings
from pathlib import Path
import json
from concurrent.futures import ThreadPoolExecutor
import pickle
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SentimentResult:
    """FinBERT sentiment analysis result"""
    text: str
    sentiment_score: float  # [-1, 1] normalized score
    confidence: float       # [0, 1] confidence in prediction
    label: str             # 'positive', 'negative', 'neutral'
    raw_scores: Dict[str, float]  # Raw probabilities
    processing_time: float
    source: str = ""
    relevance_score: float = 1.0

@dataclass
class SentimentConfig:
    """Configuration for FinBERT sentiment analysis"""
    model_name: str = "ProsusAI/finbert"
    batch_size: int = 16
    max_length: int = 512
    confidence_threshold: float = 0.7
    relevance_threshold: float = 0.85
    quality_threshold: float = 0.7
    cache_results: bool = True
    device: str = "auto"
    
    # Quality filters
    min_text_length: int = 10
    max_text_length: int = 2000
    filter_low_quality: bool = True

class FinBERTSentimentAnalyzer:
    """
    FinBERT-based sentiment analyzer for financial news
    
    Features:
    - Processes all news articles from Step 1
    - Quality filtering and confidence scoring
    - Source-specific sentiment aggregation
    - Multi-horizon sentiment features
    - Caching for reproducibility
    """
    
    def __init__(self, config: SentimentConfig = None, cache_dir: str = "data/sentiment"):
        """Initialize FinBERT sentiment analyzer"""
        self.config = config or SentimentConfig()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Set device
        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)
        
        logger.info(f"FinBERT Sentiment Analyzer initialized on device: {self.device}")
        
        # Load FinBERT model and tokenizer
        self._load_finbert_model()
        
        # Processing statistics
        self.processing_stats = {
            'total_processed': 0,
            'high_quality': 0,
            'low_confidence': 0,
            'processing_times': []
        }
    
    def _load_finbert_model(self):
        """Load FinBERT model and tokenizer"""
        try:
            logger.info(f"Loading FinBERT model: {self.config.model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.config.model_name)
            self.model.to(self.device)
            self.model.eval()
            
            # Get label mappings
            self.id2label = self.model.config.id2label
            self.label2id = self.model.config.label2id
            
            logger.info(f"FinBERT model loaded successfully. Labels: {self.id2label}")
            
        except Exception as e:
            logger.error(f"Error loading FinBERT model: {e}")
            raise RuntimeError(f"Failed to load FinBERT model: {e}")
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for FinBERT"""
        if not text or not isinstance(text, str):
            return ""
        
        # Remove URLs and email addresses
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        
        # Clean up whitespace and special characters
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\$\%\.\,\!\?\-\(\)]', '', text)
        
        # Truncate to max length
        if len(text) > self.config.max_text_length:
            text = text[:self.config.max_text_length]
        
        return text.strip()
    
    def analyze_sentiment_single(self, text: str, source: str = "", 
                                relevance_score: float = 1.0) -> SentimentResult:
        """Analyze sentiment for a single text using FinBERT"""
        start_time = time.time()
        
        # Preprocess text
        cleaned_text = self.preprocess_text(text)
        
        # Quality check
        if len(cleaned_text) < self.config.min_text_length:
            return SentimentResult(
                text=text,
                sentiment_score=0.0,
                confidence=0.0,
                label='neutral',
                raw_scores={},
                processing_time=0.0,
                source=source,
                relevance_score=relevance_score
            )
        
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
            # FinBERT typically has: 0=negative, 1=neutral, 2=positive
            if len(probs) == 3:
                sentiment_score = probs[2] - probs[0]  # positive - negative
            else:
                # Fallback for different label schemes
                if predicted_label == 'positive':
                    sentiment_score = confidence
                elif predicted_label == 'negative':
                    sentiment_score = -confidence
                else:
                    sentiment_score = 0.0
            
            # Clip to [-1, 1] range
            sentiment_score = np.clip(sentiment_score, -1, 1)
            
        except Exception as e:
            logger.error(f"Error in FinBERT sentiment analysis: {e}")
            return SentimentResult(
                text=text,
                sentiment_score=0.0,
                confidence=0.0,
                label='neutral',
                raw_scores={},
                processing_time=0.0,
                source=source,
                relevance_score=relevance_score
            )
        
        processing_time = time.time() - start_time
        
        return SentimentResult(
            text=text,
            sentiment_score=float(sentiment_score),
            confidence=confidence,
            label=predicted_label,
            raw_scores=raw_scores,
            processing_time=processing_time,
            source=source,
            relevance_score=relevance_score
        )
    
    def analyze_sentiment_batch(self, texts: List[str], sources: List[str] = None,
                               relevance_scores: List[float] = None) -> List[SentimentResult]:
        """Analyze sentiment for a batch of texts"""
        if sources is None:
            sources = [""] * len(texts)
        if relevance_scores is None:
            relevance_scores = [1.0] * len(texts)
        
        results = []
        batch_size = self.config.batch_size
        
        logger.info(f"Processing {len(texts)} texts in batches of {batch_size}")
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_sources = sources[i:i+batch_size]
            batch_relevance = relevance_scores[i:i+batch_size]
            
            batch_results = []
            for text, source, relevance in zip(batch_texts, batch_sources, batch_relevance):
                result = self.analyze_sentiment_single(text, source, relevance)
                batch_results.append(result)
                
                # Update statistics
                self.processing_stats['total_processed'] += 1
                if result.confidence >= self.config.confidence_threshold:
                    self.processing_stats['high_quality'] += 1
                else:
                    self.processing_stats['low_confidence'] += 1
                
                self.processing_stats['processing_times'].append(result.processing_time)
            
            results.extend(batch_results)
            
            # Progress logging
            progress = min((i + batch_size) / len(texts) * 100, 100)
            logger.info(f"Progress: {progress:.1f}% ({i + len(batch_results)}/{len(texts)})")
        
        return results
    
    def process_news_data(self, news_data: Dict[str, List], symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """Process news data from Step 1 through FinBERT"""
        logger.info("Processing news data through FinBERT sentiment analysis")
        
        all_sentiment_data = {}
        
        for symbol in symbols:
            if symbol not in news_data or not news_data[symbol]:
                logger.warning(f"No news data for {symbol}")
                all_sentiment_data[symbol] = pd.DataFrame()
                continue
            
            logger.info(f"Processing sentiment for {symbol} ({len(news_data[symbol])} articles)")
            
            # Check cache first
            cache_file = self.cache_dir / f"{symbol}_finbert_sentiment.pkl"
            if self.config.cache_results and cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        cached_results = pickle.load(f)
                    logger.info(f"Loaded cached sentiment results for {symbol}")
                    all_sentiment_data[symbol] = cached_results
                    continue
                except Exception as e:
                    logger.warning(f"Could not load cached results for {symbol}: {e}")
            
            # Prepare texts and metadata for FinBERT
            texts = []
            sources = []
            relevance_scores = []
            article_metadata = []
            
            for article in news_data[symbol]:
                # Combine title and content for richer sentiment analysis
                combined_text = f"{article.title}. {article.content}"
                texts.append(combined_text)
                sources.append(article.source)
                relevance_scores.append(getattr(article, 'relevance_score', 1.0))
                
                article_metadata.append({
                    'date': article.date,
                    'source': article.source,
                    'url': getattr(article, 'url', ''),
                    'original_title': article.title,
                    'word_count': len(combined_text.split())
                })
            
            # Process through FinBERT
            sentiment_results = self.analyze_sentiment_batch(texts, sources, relevance_scores)
            
            # Create DataFrame
            sentiment_data = []
            for i, result in enumerate(sentiment_results):
                metadata = article_metadata[i]
                
                sentiment_data.append({
                    'date': metadata['date'],
                    'symbol': symbol,
                    'source': result.source,
                    'sentiment_score': result.sentiment_score,
                    'confidence': result.confidence,
                    'label': result.label,
                    'relevance_score': result.relevance_score,
                    'processing_time': result.processing_time,
                    'word_count': metadata['word_count'],
                    'positive_prob': result.raw_scores.get('positive', 0.0),
                    'negative_prob': result.raw_scores.get('negative', 0.0),
                    'neutral_prob': result.raw_scores.get('neutral', 0.0),
                    'url': metadata['url'],
                    'title': metadata['original_title']
                })
            
            sentiment_df = pd.DataFrame(sentiment_data)
            if not sentiment_df.empty:
                sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
                sentiment_df = sentiment_df.sort_values('date')
            
            # Apply quality filtering
            if self.config.filter_low_quality:
                sentiment_df = self._apply_quality_filters(sentiment_df)
            
            # Cache results
            if self.config.cache_results:
                try:
                    with open(cache_file, 'wb') as f:
                        pickle.dump(sentiment_df, f)
                    logger.info(f"Cached sentiment results for {symbol}")
                except Exception as e:
                    logger.warning(f"Could not cache results for {symbol}: {e}")
            
            all_sentiment_data[symbol] = sentiment_df
            
            logger.info(f"Completed sentiment analysis for {symbol}: {len(sentiment_df)} high-quality results")
        
        return all_sentiment_data
    
    def _apply_quality_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply quality filters to sentiment results"""
        original_count = len(df)
        
        # Filter by confidence threshold
        df = df[df['confidence'] >= self.config.confidence_threshold]
        
        # Filter by relevance threshold
        df = df[df['relevance_score'] >= self.config.relevance_threshold]
        
        # Filter by minimum word count
        df = df[df['word_count'] >= self.config.min_text_length]
        
        logger.info(f"Quality filtering: {original_count} -> {len(df)} articles retained")
        
        return df
    
    def create_sentiment_features(self, sentiment_data: Dict[str, pd.DataFrame], 
                                horizons: List[int] = [5, 30, 90]) -> Dict[str, pd.DataFrame]:
        """Create sentiment features for different prediction horizons"""
        logger.info(f"Creating sentiment features for horizons: {horizons}")
        
        all_features = {}
        
        for symbol, sentiment_df in sentiment_data.items():
            if sentiment_df.empty:
                logger.warning(f"No sentiment data for {symbol}")
                all_features[symbol] = pd.DataFrame()
                continue
            
            logger.info(f"Creating sentiment features for {symbol}")
            
            # Get date range
            start_date = sentiment_df['date'].min()
            end_date = sentiment_df['date'].max()
            
            # Create daily feature grid
            daily_features = []
            current_date = start_date
            
            while current_date <= end_date:
                row = {'date': current_date, 'symbol': symbol}
                
                for horizon in horizons:
                    # Get sentiment within lookback window
                    lookback_start = current_date - timedelta(days=horizon)
                    window_sentiment = sentiment_df[
                        (sentiment_df['date'] >= lookback_start) & 
                        (sentiment_df['date'] <= current_date)
                    ]
                    
                    if len(window_sentiment) > 0:
                        # Basic aggregations
                        row[f'sentiment_mean_{horizon}d'] = window_sentiment['sentiment_score'].mean()
                        row[f'sentiment_std_{horizon}d'] = window_sentiment['sentiment_score'].std()
                        row[f'sentiment_count_{horizon}d'] = len(window_sentiment)
                        
                        # Confidence-weighted features
                        weights = window_sentiment['confidence']
                        if weights.sum() > 0:
                            row[f'sentiment_weighted_mean_{horizon}d'] = np.average(
                                window_sentiment['sentiment_score'], weights=weights
                            )
                            row[f'sentiment_weighted_std_{horizon}d'] = np.sqrt(
                                np.average((window_sentiment['sentiment_score'] - row[f'sentiment_weighted_mean_{horizon}d'])**2, weights=weights)
                            )
                        else:
                            row[f'sentiment_weighted_mean_{horizon}d'] = 0.0
                            row[f'sentiment_weighted_std_{horizon}d'] = 0.0
                        
                        # Sentiment distribution
                        row[f'sentiment_positive_ratio_{horizon}d'] = (window_sentiment['sentiment_score'] > 0.1).mean()
                        row[f'sentiment_negative_ratio_{horizon}d'] = (window_sentiment['sentiment_score'] < -0.1).mean()
                        row[f'sentiment_neutral_ratio_{horizon}d'] = (abs(window_sentiment['sentiment_score']) <= 0.1).mean()
                        
                        # Source-specific features
                        for source in ['sec_edgar', 'federal_reserve', 'investor_relations', 'bloomberg_twitter', 'yahoo_finance']:
                            source_sentiment = window_sentiment[window_sentiment['source'] == source]
                            if len(source_sentiment) > 0:
                                row[f'{source}_sentiment_{horizon}d'] = source_sentiment['sentiment_score'].mean()
                                row[f'{source}_count_{horizon}d'] = len(source_sentiment)
                                row[f'{source}_confidence_{horizon}d'] = source_sentiment['confidence'].mean()
                            else:
                                row[f'{source}_sentiment_{horizon}d'] = 0.0
                                row[f'{source}_count_{horizon}d'] = 0
                                row[f'{source}_confidence_{horizon}d'] = 0.0
                        
                        # Quality metrics
                        row[f'sentiment_avg_confidence_{horizon}d'] = window_sentiment['confidence'].mean()
                        row[f'sentiment_avg_relevance_{horizon}d'] = window_sentiment['relevance_score'].mean()
                        row[f'sentiment_source_diversity_{horizon}d'] = window_sentiment['source'].nunique()
                        
                        # Temporal patterns
                        if len(window_sentiment) > 1:
                            # Sentiment trend (recent vs older)
                            recent_sentiment = window_sentiment[window_sentiment['date'] >= (current_date - timedelta(days=horizon//2))]
                            older_sentiment = window_sentiment[window_sentiment['date'] < (current_date - timedelta(days=horizon//2))]
                            
                            if len(recent_sentiment) > 0 and len(older_sentiment) > 0:
                                row[f'sentiment_trend_{horizon}d'] = recent_sentiment['sentiment_score'].mean() - older_sentiment['sentiment_score'].mean()
                            else:
                                row[f'sentiment_trend_{horizon}d'] = 0.0
                        else:
                            row[f'sentiment_trend_{horizon}d'] = 0.0
                    
                    else:
                        # No sentiment data for this window - fill with zeros
                        for feature_suffix in [
                            'mean', 'std', 'weighted_mean', 'weighted_std', 'positive_ratio', 
                            'negative_ratio', 'neutral_ratio', 'avg_confidence', 'avg_relevance', 'trend'
                        ]:
                            row[f'sentiment_{feature_suffix}_{horizon}d'] = 0.0
                        
                        row[f'sentiment_count_{horizon}d'] = 0
                        row[f'sentiment_source_diversity_{horizon}d'] = 0
                        
                        # Source-specific zeros
                        for source in ['sec_edgar', 'federal_reserve', 'investor_relations', 'bloomberg_twitter', 'yahoo_finance']:
                            row[f'{source}_sentiment_{horizon}d'] = 0.0
                            row[f'{source}_count_{horizon}d'] = 0
                            row[f'{source}_confidence_{horizon}d'] = 0.0
                
                daily_features.append(row)
                current_date += timedelta(days=1)
            
            # Create DataFrame
            features_df = pd.DataFrame(daily_features)
            features_df['date'] = pd.to_datetime(features_df['date'])
            features_df = features_df.set_index('date')
            
            all_features[symbol] = features_df
            
            logger.info(f"Created sentiment features for {symbol}: {features_df.shape}")
        
        return all_features
    
    def get_processing_statistics(self) -> Dict:
        """Get processing statistics"""
        stats = self.processing_stats.copy()
        
        if stats['processing_times']:
            stats['avg_processing_time'] = np.mean(stats['processing_times'])
            stats['total_processing_time'] = sum(stats['processing_times'])
        else:
            stats['avg_processing_time'] = 0.0
            stats['total_processing_time'] = 0.0
        
        if stats['total_processed'] > 0:
            stats['high_quality_ratio'] = stats['high_quality'] / stats['total_processed']
            stats['low_confidence_ratio'] = stats['low_confidence'] / stats['total_processed']
        else:
            stats['high_quality_ratio'] = 0.0
            stats['low_confidence_ratio'] = 0.0
        
        return stats
    
    def save_sentiment_features(self, sentiment_features: Dict[str, pd.DataFrame], 
                               save_path: str = "data/processed/sentiment_features.parquet"):
        """Save sentiment features to file"""
        if not sentiment_features:
            logger.warning("No sentiment features to save")
            return
        
        # Combine all symbols into single DataFrame
        all_features = []
        for symbol, features_df in sentiment_features.items():
            if not features_df.empty:
                all_features.append(features_df)
        
        if all_features:
            combined_features = pd.concat(all_features, ignore_index=False)
            
            # Save to parquet
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            combined_features.to_parquet(save_path)
            
            logger.info(f"Saved sentiment features to {save_path}: {combined_features.shape}")
            
            # Save metadata
            metadata = {
                'creation_time': datetime.now().isoformat(),
                'model_used': self.config.model_name,
                'symbols': list(sentiment_features.keys()),
                'shape': combined_features.shape,
                'features_per_horizon': len([col for col in combined_features.columns if '_5d' in col]),
                'processing_stats': self.get_processing_statistics()
            }
            
            metadata_path = Path(save_path).with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            logger.info(f"Saved sentiment metadata to {metadata_path}")
        else:
            logger.warning("No valid sentiment features to save")

def test_finbert_analyzer():
    """Test the FinBERT sentiment analyzer"""
    print("🧪 Testing FinBERT Sentiment Analyzer")
    print("="*50)
    
    try:
        # Initialize analyzer
        config = SentimentConfig(
            confidence_threshold=0.7,
            cache_results=True
        )
        analyzer = FinBERTSentimentAnalyzer(config)
        print("✅ FinBERT analyzer initialized")
        
        # Test with sample financial texts
        sample_texts = [
            "Apple reported strong quarterly earnings, beating expectations across all segments.",
            "Tesla's new model received positive reviews from automotive experts and analysts.",
            "Microsoft's cloud revenue continues to grow at impressive rates this quarter.",
            "Market volatility increases due to economic uncertainty and geopolitical tensions.",
            "Amazon announced significant investments in artificial intelligence and machine learning."
        ]
        
        print("\n🔍 Testing sentiment analysis...")
        results = analyzer.analyze_sentiment_batch(sample_texts)
        
        print("\nFinBERT Sentiment Results:")
        for i, result in enumerate(results):
            print(f"\n{i+1}. Text: {sample_texts[i][:60]}...")
            print(f"   Sentiment: {result.sentiment_score:.3f} ({result.label})")
            print(f"   Confidence: {result.confidence:.3f}")
            print(f"   Raw scores: {result.raw_scores}")
        
        # Test feature creation
        print("\n📊 Testing sentiment feature creation...")
        
        # Create mock sentiment DataFrame
        dates = pd.date_range('2023-01-01', '2023-01-10', freq='D')
        mock_sentiment_data = {
            'AAPL': pd.DataFrame({
                'date': dates,
                'symbol': 'AAPL',
                'source': 'test',
                'sentiment_score': np.random.normal(0, 0.3, len(dates)),
                'confidence': np.random.uniform(0.7, 0.95, len(dates)),
                'label': ['positive' if s > 0 else 'negative' if s < 0 else 'neutral' 
                         for s in np.random.normal(0, 0.3, len(dates))],
                'relevance_score': np.random.uniform(0.8, 1.0, len(dates)),
                'processing_time': np.random.uniform(0.1, 0.3, len(dates)),
                'word_count': np.random.randint(50, 200, len(dates))
            })
        }
        
        # Create features
        sentiment_features = analyzer.create_sentiment_features(mock_sentiment_data, horizons=[5, 10])
        
        if sentiment_features and 'AAPL' in sentiment_features:
            features_df = sentiment_features['AAPL']
            print(f"✅ Created sentiment features: {features_df.shape}")
            print(f"Sample features: {list(features_df.columns)[:5]}")
        
        # Get statistics
        stats = analyzer.get_processing_statistics()
        print(f"\n📈 Processing Statistics:")
        print(f"  Total processed: {stats['total_processed']}")
        print(f"  High quality ratio: {stats['high_quality_ratio']:.3f}")
        print(f"  Average processing time: {stats['avg_processing_time']:.3f}s")
        
        print("\n✅ FinBERT sentiment analyzer test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ FinBERT test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_finbert_analyzer()