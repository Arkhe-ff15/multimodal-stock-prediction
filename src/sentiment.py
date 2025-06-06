"""
src/sentiment.py

FinBERT-based sentiment analysis with quality filtering and overfitting prevention
Integrates with temporal decay processing for horizon-specific sentiment features
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SentimentResult:
    """Sentiment analysis result"""
    text: str
    sentiment_score: float  # [-1, 1] where -1=very negative, 1=very positive
    confidence: float  # [0, 1] confidence in prediction
    label: str  # 'positive', 'negative', 'neutral'
    raw_scores: Dict[str, float]  # Raw logits/probabilities
    processing_time: float
    quality_metrics: Dict[str, float]

@dataclass
class SentimentConfig:
    """Configuration for sentiment analysis"""
    model_name: str = "ProsusAI/finbert"
    batch_size: int = 16
    max_length: int = 512
    confidence_threshold: float = 0.7
    relevance_threshold: float = 0.85
    quality_filters: Dict[str, bool] = None
    cache_results: bool = True
    device: str = "auto"  # "auto", "cpu", "cuda"
    
    def __post_init__(self):
        if self.quality_filters is None:
            self.quality_filters = {
                'min_length': True,
                'language_filter': True,
                'relevance_check': True,
                'confidence_filter': True,
                'duplicate_filter': True
            }

class FinBERTSentimentAnalyzer:
    """
    Advanced FinBERT sentiment analyzer with quality filtering and overfitting prevention
    """
    
    def __init__(self, config: SentimentConfig = None, cache_dir: str = "data/sentiment"):
        """
        Initialize FinBERT sentiment analyzer
        
        Args:
            config: Sentiment analysis configuration
            cache_dir: Directory for caching results
        """
        self.config = config or SentimentConfig()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Set device
        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)
        
        logger.info(f"Using device: {self.device}")
        
        # Load model and tokenizer
        self._load_model()
        
        # Quality metrics tracking
        self.quality_stats = {
            'total_processed': 0,
            'filtered_out': 0,
            'low_confidence': 0,
            'low_relevance': 0,
            'processing_times': []
        }
        
        # Company/symbol mappings for relevance checking
        self.symbol_mappings = {
            'AAPL': ['apple', 'iphone', 'ipad', 'mac', 'ios', 'tim cook', 'cupertino'],
            'MSFT': ['microsoft', 'windows', 'office', 'azure', 'xbox', 'satya nadella'],
            'GOOGL': ['google', 'alphabet', 'youtube', 'android', 'search', 'sundar pichai'],
            'AMZN': ['amazon', 'aws', 'prime', 'alexa', 'jeff bezos', 'andy jassy'],
            'TSLA': ['tesla', 'elon musk', 'electric vehicle', 'model s', 'model 3', 'model y', 'spacex']
        }
        
        logger.info("FinBERT sentiment analyzer initialized successfully")
    
    def _load_model(self):
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
            
            logger.info(f"Model loaded successfully. Labels: {self.id2label}")
            
        except Exception as e:
            logger.error(f"Error loading FinBERT model: {e}")
            raise
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for sentiment analysis
        
        Args:
            text: Raw text input
            
        Returns:
            Cleaned and preprocessed text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep financial symbols ($, %, etc.)
        text = re.sub(r'[^\w\s\$\%\.\,\!\?\-]', '', text)
        
        # Truncate to max length (leave room for special tokens)
        max_text_length = self.config.max_length - 10
        if len(text) > max_text_length:
            text = text[:max_text_length]
        
        return text.strip()
    
    def check_relevance(self, text: str, symbol: str) -> float:
        """
        Check relevance of text to specific stock symbol
        
        Args:
            text: Text to analyze
            symbol: Stock symbol (e.g., 'AAPL')
            
        Returns:
            Relevance score [0, 1]
        """
        if not text or symbol not in self.symbol_mappings:
            return 0.0
        
        text_lower = text.lower()
        keywords = self.symbol_mappings[symbol]
        
        # Count keyword matches
        matches = sum(1 for keyword in keywords if keyword in text_lower)
        
        # Calculate relevance score
        relevance_score = min(matches / max(len(keywords) * 0.3, 1), 1.0)
        
        # Boost if symbol appears directly
        if symbol.lower() in text_lower:
            relevance_score = min(relevance_score + 0.3, 1.0)
        
        return relevance_score
    
    def quality_filter(self, text: str, symbol: str) -> Tuple[bool, Dict[str, float]]:
        """
        Apply quality filters to determine if text should be processed
        
        Args:
            text: Text to filter
            symbol: Stock symbol for relevance check
            
        Returns:
            Tuple of (should_process: bool, quality_metrics: dict)
        """
        quality_metrics = {}
        should_process = True
        
        # Minimum length filter
        if self.config.quality_filters['min_length']:
            quality_metrics['length'] = len(text.split())
            if quality_metrics['length'] < 5:  # Minimum 5 words
                should_process = False
        
        # Language filter (basic English check)
        if self.config.quality_filters['language_filter']:
            english_words = {'the', 'and', 'is', 'in', 'to', 'of', 'a', 'that', 'it', 'with', 'for', 'as', 'was', 'on', 'are'}
            words = set(text.lower().split())
            english_ratio = len(words.intersection(english_words)) / max(len(words), 1)
            quality_metrics['english_ratio'] = english_ratio
            if english_ratio < 0.1:  # Less than 10% common English words
                should_process = False
        
        # Relevance check
        if self.config.quality_filters['relevance_check']:
            relevance_score = self.check_relevance(text, symbol)
            quality_metrics['relevance'] = relevance_score
            if relevance_score < self.config.relevance_threshold:
                should_process = False
        
        return should_process, quality_metrics
    
    def analyze_sentiment_single(self, text: str, symbol: str = None) -> SentimentResult:
        """
        Analyze sentiment for a single text
        
        Args:
            text: Text to analyze
            symbol: Optional stock symbol for relevance checking
            
        Returns:
            SentimentResult object
        """
        start_time = datetime.now()
        
        # Preprocess text
        cleaned_text = self.preprocess_text(text)
        
        # Quality filtering
        if symbol and self.config.quality_filters:
            should_process, quality_metrics = self.quality_filter(cleaned_text, symbol)
            if not should_process:
                return SentimentResult(
                    text=text,
                    sentiment_score=0.0,
                    confidence=0.0,
                    label='neutral',
                    raw_scores={},
                    processing_time=0.0,
                    quality_metrics=quality_metrics
                )
        else:
            quality_metrics = {}
        
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
            
            # Calculate sentiment score [-1, 1]
            # Assuming labels are: 0=negative, 1=neutral, 2=positive
            if len(probs) == 3:
                sentiment_score = probs[2] - probs[0]  # positive - negative
            else:
                # Fallback for different label schemes
                sentiment_score = 2 * (probs[predicted_id] - 0.5) if predicted_label == 'positive' else \
                                -2 * (probs[predicted_id] - 0.5) if predicted_label == 'negative' else 0.0
            
            # Clip to [-1, 1] range
            sentiment_score = np.clip(sentiment_score, -1, 1)
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return SentimentResult(
                text=text,
                sentiment_score=0.0,
                confidence=0.0,
                label='neutral',
                raw_scores={},
                processing_time=0.0,
                quality_metrics=quality_metrics
            )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return SentimentResult(
            text=text,
            sentiment_score=float(sentiment_score),
            confidence=confidence,
            label=predicted_label,
            raw_scores=raw_scores,
            processing_time=processing_time,
            quality_metrics=quality_metrics
        )
    
    def analyze_sentiment_batch(self, texts: List[str], symbol: str = None, 
                                show_progress: bool = True) -> List[SentimentResult]:
        """
        Analyze sentiment for a batch of texts
        
        Args:
            texts: List of texts to analyze
            symbol: Optional stock symbol for relevance checking
            show_progress: Whether to show progress bar
            
        Returns:
            List of SentimentResult objects
        """
        logger.info(f"Analyzing sentiment for {len(texts)} texts")
        
        results = []
        batch_size = self.config.batch_size
        
        # Process in batches to manage memory
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_results = []
            
            for text in batch_texts:
                result = self.analyze_sentiment_single(text, symbol)
                batch_results.append(result)
                
                # Update quality stats
                self.quality_stats['total_processed'] += 1
                if result.confidence < self.config.confidence_threshold:
                    self.quality_stats['low_confidence'] += 1
                if symbol and result.quality_metrics.get('relevance', 1.0) < self.config.relevance_threshold:
                    self.quality_stats['low_relevance'] += 1
                
                self.quality_stats['processing_times'].append(result.processing_time)
            
            results.extend(batch_results)
            
            if show_progress:
                progress = min((i + batch_size) / len(texts) * 100, 100)
                logger.info(f"Progress: {progress:.1f}% ({i + len(batch_texts)}/{len(texts)})")
        
        # Filter results by confidence if enabled
        if self.config.quality_filters.get('confidence_filter', False):
            high_quality_results = [r for r in results if r.confidence >= self.config.confidence_threshold]
            logger.info(f"Filtered by confidence: {len(results)} -> {len(high_quality_results)} results")
            results = high_quality_results
        
        return results
    
    def process_news_data(self, news_articles: List, symbol: str, 
                            save_cache: bool = True) -> pd.DataFrame:
        """
        Process news articles and extract sentiment features
        
        Args:
            news_articles: List of NewsArticle objects
            symbol: Stock symbol
            save_cache: Whether to save results to cache
            
        Returns:
            DataFrame with sentiment features
        """
        logger.info(f"Processing {len(news_articles)} news articles for {symbol}")
        
        # Check cache first
        cache_file = self.cache_dir / f"{symbol}_sentiment_results.json"
        if self.config.cache_results and cache_file.exists():
            logger.info(f"Loading cached sentiment results for {symbol}")
            with open(cache_file, 'r') as f:
                cached_results = json.load(f)
            
            # Convert back to results format
            sentiment_results = []
            for item in cached_results:
                sentiment_results.append(SentimentResult(**item))
        else:
            # Extract texts for sentiment analysis
            texts = []
            article_metadata = []
            
            for article in news_articles:
                # Combine title and content for richer sentiment analysis
                combined_text = f"{article.title}. {article.content}"
                texts.append(combined_text)
                article_metadata.append({
                    'date': article.date,
                    'source': article.source,
                    'url': article.url,
                    'original_relevance': getattr(article, 'relevance_score', 0.0)
                })
            
            # Perform sentiment analysis
            sentiment_results = self.analyze_sentiment_batch(texts, symbol)
            
            # Cache results
            if save_cache and self.config.cache_results:
                cache_data = []
                for result in sentiment_results:
                    cache_data.append({
                        'text': result.text,
                        'sentiment_score': result.sentiment_score,
                        'confidence': result.confidence,
                        'label': result.label,
                        'raw_scores': result.raw_scores,
                        'processing_time': result.processing_time,
                        'quality_metrics': result.quality_metrics
                    })
                
                with open(cache_file, 'w') as f:
                    json.dump(cache_data, f, indent=2)
                logger.info(f"Cached sentiment results for {symbol}")
        
        # Create DataFrame
        sentiment_data = []
        for i, result in enumerate(sentiment_results):
            if i < len(article_metadata):
                metadata = article_metadata[i]
                sentiment_data.append({
                    'date': metadata['date'],
                    'symbol': symbol,
                    'sentiment_score': result.sentiment_score,
                    'confidence': result.confidence,
                    'label': result.label,
                    'source': metadata['source'],
                    'relevance_score': result.quality_metrics.get('relevance', metadata.get('original_relevance', 0.0)),
                    'processing_time': result.processing_time,
                    'text_length': len(result.text.split()),
                    'positive_prob': result.raw_scores.get('positive', 0.0),
                    'negative_prob': result.raw_scores.get('negative', 0.0),
                    'neutral_prob': result.raw_scores.get('neutral', 0.0)
                })
        
        df = pd.DataFrame(sentiment_data)
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
        
        logger.info(f"Created sentiment DataFrame with {len(df)} rows for {symbol}")
        return df
    
    def create_sentiment_features(self, sentiment_df: pd.DataFrame, 
                                horizons: List[int] = [5, 30, 90]) -> pd.DataFrame:
        """
        Create aggregated sentiment features for different horizons
        
        Args:
            sentiment_df: DataFrame with individual sentiment scores
            horizons: List of horizon days for aggregation
            
        Returns:
            DataFrame with aggregated sentiment features
        """
        if sentiment_df.empty:
            return pd.DataFrame()
        
        logger.info(f"Creating sentiment features for horizons: {horizons}")
        
        # Get date range
        start_date = sentiment_df['date'].min()
        end_date = sentiment_df['date'].max()
        
        # Create daily features
        daily_features = []
        current_date = start_date
        
        while current_date <= end_date:
            row = {'date': current_date}
            
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
                    row[f'sentiment_positive_ratio_{horizon}d'] = (window_sentiment['sentiment_score'] > 0.1).mean()
                    row[f'sentiment_negative_ratio_{horizon}d'] = (window_sentiment['sentiment_score'] < -0.1).mean()
                    
                    # Confidence-weighted features
                    weights = window_sentiment['confidence']
                    if weights.sum() > 0:
                        row[f'sentiment_weighted_mean_{horizon}d'] = np.average(
                            window_sentiment['sentiment_score'], weights=weights
                        )
                    else:
                        row[f'sentiment_weighted_mean_{horizon}d'] = 0.0
                    
                    # Source diversity
                    row[f'sentiment_source_count_{horizon}d'] = window_sentiment['source'].nunique()
                    
                    # Quality metrics
                    row[f'sentiment_avg_confidence_{horizon}d'] = window_sentiment['confidence'].mean()
                    row[f'sentiment_avg_relevance_{horizon}d'] = window_sentiment['relevance_score'].mean()
                    
                else:
                    # No sentiment data for this window
                    for feature in [f'sentiment_mean_{horizon}d', f'sentiment_std_{horizon}d', 
                                    f'sentiment_weighted_mean_{horizon}d', f'sentiment_avg_confidence_{horizon}d',
                                    f'sentiment_avg_relevance_{horizon}d']:
                        row[feature] = 0.0
                    
                    for feature in [f'sentiment_count_{horizon}d', f'sentiment_source_count_{horizon}d']:
                        row[feature] = 0
                    
                    for feature in [f'sentiment_positive_ratio_{horizon}d', f'sentiment_negative_ratio_{horizon}d']:
                        row[feature] = 0.0
            
            daily_features.append(row)
            current_date += timedelta(days=1)
        
        features_df = pd.DataFrame(daily_features)
        features_df['date'] = pd.to_datetime(features_df['date'])
        features_df = features_df.set_index('date')
        
        logger.info(f"Created sentiment features with {len(features_df)} rows and {len(features_df.columns)} features")
        return features_df
    
    def plot_sentiment_analysis(self, sentiment_df: pd.DataFrame, symbol: str, 
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Create comprehensive visualization of sentiment analysis results
        """
        if sentiment_df.empty:
            logger.warning("Empty sentiment DataFrame, cannot create plots")
            return None
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Sentiment Analysis Results - {symbol}', fontsize=16, fontweight='bold')
        
        # Plot 1: Sentiment over time
        ax1 = axes[0, 0]
        daily_sentiment = sentiment_df.groupby(sentiment_df['date'].dt.date)['sentiment_score'].mean()
        ax1.plot(daily_sentiment.index, daily_sentiment.values, alpha=0.7, linewidth=2)
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax1.set_title('Daily Average Sentiment')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Sentiment Score')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Sentiment distribution
        ax2 = axes[0, 1]
        ax2.hist(sentiment_df['sentiment_score'], bins=30, alpha=0.7, edgecolor='black')
        ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Neutral')
        ax2.set_title('Sentiment Score Distribution')
        ax2.set_xlabel('Sentiment Score')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Confidence vs Sentiment
        ax3 = axes[0, 2]
        scatter = ax3.scatter(sentiment_df['sentiment_score'], sentiment_df['confidence'], 
                                alpha=0.6, c=sentiment_df['relevance_score'], cmap='viridis')
        ax3.set_title('Confidence vs Sentiment')
        ax3.set_xlabel('Sentiment Score')
        ax3.set_ylabel('Confidence')
        ax3.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax3, label='Relevance Score')
        
        # Plot 4: Quality metrics
        ax4 = axes[1, 0]
        quality_data = [
            sentiment_df['confidence'].mean(),
            sentiment_df['relevance_score'].mean(),
            (sentiment_df['confidence'] >= self.config.confidence_threshold).mean(),
            (sentiment_df['relevance_score'] >= self.config.relevance_threshold).mean()
        ]
        quality_labels = ['Avg Confidence', 'Avg Relevance', 'High Confidence %', 'High Relevance %']
        bars = ax4.bar(quality_labels, quality_data, alpha=0.7)
        ax4.set_title('Quality Metrics')
        ax4.set_ylabel('Score / Percentage')
        ax4.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, quality_data):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # Plot 5: Source breakdown
        ax5 = axes[1, 1]
        source_counts = sentiment_df['source'].value_counts()
        ax5.pie(source_counts.values, labels=source_counts.index, autopct='%1.1f%%')
        ax5.set_title('News Sources Distribution')
        
        # Plot 6: Sentiment by source
        ax6 = axes[1, 2]
        source_sentiment = sentiment_df.groupby('source')['sentiment_score'].mean()
        bars = ax6.bar(source_sentiment.index, source_sentiment.values, alpha=0.7)
        ax6.set_title('Average Sentiment by Source')
        ax6.set_xlabel('Source')
        ax6.set_ylabel('Average Sentiment')
        ax6.tick_params(axis='x', rotation=45)
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def get_quality_report(self) -> Dict:
        """Get quality statistics report"""
        if self.quality_stats['total_processed'] == 0:
            return {'error': 'No texts processed yet'}
        
        avg_processing_time = np.mean(self.quality_stats['processing_times']) if self.quality_stats['processing_times'] else 0
        
        return {
            'total_processed': self.quality_stats['total_processed'],
            'low_confidence_rate': self.quality_stats['low_confidence'] / self.quality_stats['total_processed'],
            'low_relevance_rate': self.quality_stats['low_relevance'] / self.quality_stats['total_processed'],
            'avg_processing_time': avg_processing_time,
            'processing_speed': self.quality_stats['total_processed'] / sum(self.quality_stats['processing_times']) if self.quality_stats['processing_times'] else 0
        }

# Example usage and testing
if __name__ == "__main__":
    # Initialize sentiment analyzer
    config = SentimentConfig(
        batch_size=8,
        confidence_threshold=0.7,
        relevance_threshold=0.85
    )
    
    analyzer = FinBERTSentimentAnalyzer(config)
    
    # Test with sample texts
    sample_texts = [
        "Apple reported strong quarterly earnings, beating expectations across all segments.",
        "Tesla's new model received positive reviews from automotive experts.",
        "Microsoft's cloud revenue continues to grow at impressive rates.",
        "Market volatility increases due to economic uncertainty.",
        "The weather is nice today."  # Low relevance text
    ]
    
    print("🔍 Testing sentiment analysis...")
    
    # Analyze sample texts
    results = analyzer.analyze_sentiment_batch(sample_texts, symbol='AAPL')
    
    # Display results
    print("\nSentiment Analysis Results:")
    for i, result in enumerate(results):
        print(f"\nText {i+1}: {sample_texts[i][:50]}...")
        print(f"  Sentiment: {result.sentiment_score:.3f} ({result.label})")
        print(f"  Confidence: {result.confidence:.3f}")
        print(f"  Relevance: {result.quality_metrics.get('relevance', 'N/A')}")
    
    # Quality report
    quality_report = analyzer.get_quality_report()
    print(f"\nQuality Report:")
    for key, value in quality_report.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    
    print("\n✅ Sentiment analysis testing complete!")
    print("Next steps:")
    print("1. Integrate with news data collection")
    print("2. Apply temporal decay processing")
    print("3. Create model-ready features")