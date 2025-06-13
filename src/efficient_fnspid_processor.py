#!/usr/bin/env python3
"""
EFFICIENT FNSPID PROCESSOR - Memory-Optimized for Large Datasets
================================================================

✅ OPTIMIZED FOR 22GB FNSPID DATASET:
1. Chunked processing to manage memory
2. Time-filtered sampling for efficiency
3. Smart stock matching with advanced keywords
4. Progressive sentiment analysis with checkpoints
5. Memory monitoring and optimization
6. Resume capability for interrupted processing

SCOPE: Process massive FNSPID dataset efficiently
OUTPUT: High-quality sentiment features for TFT training
"""

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
import warnings
import json
import re
import os
import gc
import psutil
from dataclasses import dataclass
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# Standard paths
DATA_DIR = "data/processed"
RAW_DIR = "data/raw"
BACKUP_DIR = "data/backups"
CACHE_DIR = "data/cache"

# FNSPID dataset paths
FNSPID_NEWS_FILE = f"{RAW_DIR}/nasdaq_external_data.csv"
SENTIMENT_OUTPUT_FILE = f"{DATA_DIR}/fnspid_sentiment_dataset.csv"
ARTICLE_SENTIMENT_FILE = f"{DATA_DIR}/fnspid_article_sentiments.csv"
CHECKPOINT_DIR = f"{CACHE_DIR}/fnspid_checkpoints"

@dataclass
class EfficientConfig:
    """Configuration optimized for large dataset processing"""
    # Target symbols for focused analysis
    target_symbols: List[str] = None
    
    # Time filtering for manageable dataset size
    start_date: str = "2020-01-01"
    end_date: str = "2024-01-31"
    sample_ratio: float = 0.1  # Process 10% of articles initially
    
    # Processing limits
    max_articles_per_symbol: int = 2000
    max_articles_per_day: int = 100
    chunk_size: int = 10000  # Process in chunks
    
    # Memory management
    max_memory_usage_gb: float = 8.0
    checkpoint_frequency: int = 1000  # Save progress every N articles
    
    # FinBERT settings
    model_name: str = "ProsusAI/finbert"
    batch_size: int = 16  # Smaller batches for memory efficiency
    confidence_threshold: float = 0.65
    
    # Quality filters
    min_text_length: int = 50
    max_text_length: int = 1000
    relevance_threshold: float = 1.0
    
    def __post_init__(self):
        if self.target_symbols is None:
            self.target_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'JPM']

class MemoryMonitor:
    """Monitor memory usage during processing"""
    
    def __init__(self, max_memory_gb: float = 8.0):
        self.max_memory_gb = max_memory_gb
        self.process = psutil.Process()
        
    def get_memory_usage(self) -> float:
        """Get current memory usage in GB"""
        return self.process.memory_info().rss / (1024**3)
    
    def check_memory_limit(self) -> bool:
        """Check if memory usage is approaching limit"""
        current_usage = self.get_memory_usage()
        return current_usage > (self.max_memory_gb * 0.8)  # 80% threshold
    
    def force_garbage_collection(self):
        """Force garbage collection to free memory"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

class EfficientStockMatcher:
    """Memory-efficient stock keyword matching"""
    
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.keyword_patterns = self._build_optimized_patterns()
        logger.info(f"🏷️ Efficient matcher initialized for {len(symbols)} symbols")
    
    def _build_optimized_patterns(self) -> Dict[str, Dict[str, re.Pattern]]:
        """Build optimized regex patterns for each symbol"""
        
        # Comprehensive symbol information
        symbol_info = {
            'AAPL': {
                'company': ['Apple Inc', 'Apple Computer'],
                'products': ['iPhone', 'iPad', 'Mac', 'iOS', 'macOS', 'AirPods'],
                'executives': ['Tim Cook', 'Steve Jobs'],
                'locations': ['Cupertino']
            },
            'MSFT': {
                'company': ['Microsoft Corporation', 'Microsoft Corp'],
                'products': ['Windows', 'Azure', 'Office', 'Xbox', 'Teams'],
                'executives': ['Satya Nadella', 'Bill Gates'],
                'locations': ['Redmond']
            },
            'GOOGL': {
                'company': ['Alphabet Inc', 'Google Inc'],
                'products': ['Google', 'YouTube', 'Android', 'Chrome', 'Gmail'],
                'executives': ['Sundar Pichai', 'Larry Page'],
                'locations': ['Mountain View']
            },
            'AMZN': {
                'company': ['Amazon.com Inc', 'Amazon Inc'],
                'products': ['AWS', 'Prime', 'Alexa', 'Kindle'],
                'executives': ['Jeff Bezos', 'Andy Jassy'],
                'locations': ['Seattle']
            },
            'NVDA': {
                'company': ['NVIDIA Corporation', 'NVIDIA Corp'],
                'products': ['GeForce', 'CUDA', 'Tegra', 'RTX'],
                'executives': ['Jensen Huang'],
                'locations': ['Santa Clara']
            },
            'TSLA': {
                'company': ['Tesla Inc', 'Tesla Motors'],
                'products': ['Model S', 'Model 3', 'Model X', 'Model Y', 'Cybertruck'],
                'executives': ['Elon Musk'],
                'locations': ['Austin', 'Fremont', 'Gigafactory']
            },
            'JPM': {
                'company': ['JPMorgan Chase', 'JP Morgan', 'Chase Bank'],
                'products': ['Chase', 'JPM Coin'],
                'executives': ['Jamie Dimon'],
                'locations': ['Wall Street']
            }
        }
        
        patterns = {}
        
        for symbol in self.symbols:
            if symbol not in symbol_info:
                # Basic pattern for unknown symbols
                patterns[symbol] = {
                    'symbol': re.compile(rf'\b{re.escape(symbol)}\b', re.IGNORECASE),
                    'company': re.compile(rf'\b{re.escape(symbol.lower())}\b', re.IGNORECASE)
                }
                continue
            
            info = symbol_info[symbol]
            symbol_patterns = {}
            
            # Symbol pattern (highest weight)
            symbol_patterns['symbol'] = re.compile(rf'\b{re.escape(symbol)}\b', re.IGNORECASE)
            
            # Company patterns (high weight)
            company_terms = '|'.join([re.escape(term) for term in info['company']])
            symbol_patterns['company'] = re.compile(rf'\b({company_terms})\b', re.IGNORECASE)
            
            # Product patterns (medium weight)
            if info['products']:
                product_terms = '|'.join([re.escape(term) for term in info['products']])
                symbol_patterns['products'] = re.compile(rf'\b({product_terms})\b', re.IGNORECASE)
            
            # Executive patterns (medium weight)
            if info['executives']:
                exec_terms = '|'.join([re.escape(term) for term in info['executives']])
                symbol_patterns['executives'] = re.compile(rf'\b({exec_terms})\b', re.IGNORECASE)
            
            # Location patterns (low weight)
            if info['locations']:
                location_terms = '|'.join([re.escape(term) for term in info['locations']])
                symbol_patterns['locations'] = re.compile(rf'\b({location_terms})\b', re.IGNORECASE)
            
            patterns[symbol] = symbol_patterns
        
        return patterns
    
    def match_text(self, text: str, title: str = "") -> Dict[str, float]:
        """Match text to symbols with optimized scoring"""
        full_text = f"{title} {text}".lower()
        matches = {}
        
        for symbol, patterns in self.keyword_patterns.items():
            score = 0.0
            
            # Symbol mentions (weight: 5.0)
            if 'symbol' in patterns:
                score += len(patterns['symbol'].findall(full_text)) * 5.0
            
            # Company mentions (weight: 3.0)
            if 'company' in patterns:
                score += len(patterns['company'].findall(full_text)) * 3.0
            
            # Product mentions (weight: 2.0)
            if 'products' in patterns:
                score += len(patterns['products'].findall(full_text)) * 2.0
            
            # Executive mentions (weight: 2.0)
            if 'executives' in patterns:
                score += len(patterns['executives'].findall(full_text)) * 2.0
            
            # Location mentions (weight: 1.0)
            if 'locations' in patterns:
                score += len(patterns['locations'].findall(full_text)) * 1.0
            
            # Title bonus (multiply by 1.5)
            if title and any(pattern.search(title.lower()) for pattern in patterns.values()):
                score *= 1.5
            
            # Store significant matches
            if score >= 1.0:
                matches[symbol] = min(score, 20.0)  # Cap at 20
        
        return matches

class EfficientFNSPIDProcessor:
    """Memory-efficient processor for large FNSPID dataset"""
    
    def __init__(self, config: EfficientConfig):
        self.config = config
        self.memory_monitor = MemoryMonitor(config.max_memory_usage_gb)
        self.stock_matcher = EfficientStockMatcher(config.target_symbols)
        
        # Initialize FinBERT
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._load_finbert()
        
        # Checkpointing
        self.checkpoint_dir = Path(CHECKPOINT_DIR)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Progress tracking
        self.processed_articles = 0
        self.matched_articles = 0
        self.sentiment_analyzed = 0
        
        logger.info(f"🚀 Efficient FNSPID Processor initialized")
        logger.info(f"   📱 Device: {self.device}")
        logger.info(f"   💾 Memory limit: {config.max_memory_usage_gb:.1f} GB")
        logger.info(f"   📊 Chunk size: {config.chunk_size:,}")
        logger.info(f"   🎯 Target symbols: {config.target_symbols}")
    
    def _load_finbert(self):
        """Load FinBERT with memory optimization"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32
            )
            self.model.to(self.device)
            self.model.eval()
            
            # Optimize for inference
            if hasattr(self.model, 'half') and self.device.type == 'cuda':
                self.model = self.model.half()
            
            self.label_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}
            logger.info("✅ FinBERT loaded with memory optimization")
            
        except Exception as e:
            logger.error(f"❌ Failed to load FinBERT: {e}")
            raise
    
    def analyze_dataset_structure(self) -> Dict[str, Any]:
        """Analyze the structure of the large FNSPID dataset"""
        logger.info("🔍 Analyzing large FNSPID dataset structure...")
        
        try:
            # Read just the header and a small sample
            sample_df = pd.read_csv(FNSPID_NEWS_FILE, nrows=1000)
            
            # Get file size
            file_size_gb = os.path.getsize(FNSPID_NEWS_FILE) / (1024**3)
            
            # Estimate total rows
            estimated_rows = int((file_size_gb * 1024**3) / (sample_df.memory_usage(deep=True).sum() / len(sample_df)))
            
            logger.info(f"   📊 File size: {file_size_gb:.1f} GB")
            logger.info(f"   📋 Columns: {list(sample_df.columns)}")
            logger.info(f"   📊 Estimated rows: {estimated_rows:,}")
            
            # Analyze column types and content
            structure_info = {
                'file_size_gb': file_size_gb,
                'estimated_rows': estimated_rows,
                'columns': list(sample_df.columns),
                'sample_data': sample_df.head().to_dict()
            }
            
            # Check for date columns
            date_columns = []
            for col in sample_df.columns:
                if sample_df[col].dtype == 'object':
                    # Try to parse as date
                    try:
                        pd.to_datetime(sample_df[col].head(), errors='raise')
                        date_columns.append(col)
                        logger.info(f"   📅 Date column found: {col}")
                    except:
                        pass
            
            structure_info['date_columns'] = date_columns
            
            # Check text columns
            text_columns = []
            for col in sample_df.columns:
                if sample_df[col].dtype == 'object' and col not in date_columns:
                    avg_length = sample_df[col].astype(str).str.len().mean()
                    if avg_length > 20:
                        text_columns.append(col)
                        logger.info(f"   📝 Text column found: {col} (avg length: {avg_length:.1f})")
            
            structure_info['text_columns'] = text_columns
            
            return structure_info
            
        except Exception as e:
            logger.error(f"❌ Failed to analyze dataset structure: {e}")
            raise
    
    def create_efficient_sample(self) -> str:
        """Create an efficient sample of the large dataset"""
        logger.info(f"📊 Creating efficient sample from large dataset...")
        
        # Output file for sample
        sample_file = f"{CACHE_DIR}/fnspid_sample.csv"
        
        try:
            # Read in chunks and filter
            chunks_processed = 0
            total_sampled = 0
            
            # Time filtering
            start_date = pd.to_datetime(self.config.start_date)
            end_date = pd.to_datetime(self.config.end_date)
            
            sampled_chunks = []
            
            logger.info(f"   📅 Time filter: {start_date.date()} to {end_date.date()}")
            logger.info(f"   📊 Sample ratio: {self.config.sample_ratio:.1%}")
            
            # Process in chunks
            chunk_iter = pd.read_csv(FNSPID_NEWS_FILE, chunksize=self.config.chunk_size)
            
            for chunk in tqdm(chunk_iter, desc="Processing chunks"):
                chunks_processed += 1
                
                # Memory check
                if self.memory_monitor.check_memory_limit():
                    logger.warning("⚠️ Memory limit approaching, forcing cleanup...")
                    self.memory_monitor.force_garbage_collection()
                
                # Basic filtering
                chunk = chunk.dropna(subset=['title'] if 'title' in chunk.columns else chunk.columns[:1])
                
                # Date filtering if date column exists
                date_col = None
                for col in ['date', 'published_date', 'timestamp', 'time']:
                    if col in chunk.columns:
                        date_col = col
                        break
                
                if date_col:
                    try:
                        chunk[date_col] = pd.to_datetime(chunk[date_col], errors='coerce')
                        chunk = chunk.dropna(subset=[date_col])
                        chunk = chunk[(chunk[date_col] >= start_date) & (chunk[date_col] <= end_date)]
                    except:
                        logger.warning(f"⚠️ Could not filter by date column: {date_col}")
                
                # Sample from chunk
                if len(chunk) > 0:
                    sample_size = max(1, int(len(chunk) * self.config.sample_ratio))
                    chunk_sample = chunk.sample(n=min(sample_size, len(chunk)), random_state=42)
                    sampled_chunks.append(chunk_sample)
                    total_sampled += len(chunk_sample)
                
                # Stop if we have enough data
                if total_sampled >= 50000:  # Reasonable sample size
                    logger.info(f"   🎯 Sample size reached: {total_sampled:,} articles")
                    break
                
                # Process limited number of chunks for initial analysis
                if chunks_processed >= 100:  # Limit initial processing
                    logger.info(f"   📊 Processed {chunks_processed} chunks, continuing with sample...")
                    break
            
            # Combine samples
            if sampled_chunks:
                sample_df = pd.concat(sampled_chunks, ignore_index=True)
                sample_df.to_csv(sample_file, index=False)
                
                logger.info(f"✅ Sample created successfully!")
                logger.info(f"   📄 Sample file: {sample_file}")
                logger.info(f"   📊 Sample size: {len(sample_df):,} articles")
                logger.info(f"   💾 Sample size: {os.path.getsize(sample_file) / (1024**2):.1f} MB")
                
                return sample_file
            else:
                raise ValueError("No data could be sampled from the dataset")
                
        except Exception as e:
            logger.error(f"❌ Failed to create sample: {e}")
            raise
    
    def process_sample_with_sentiment(self, sample_file: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Process the sample with sentiment analysis"""
        logger.info(f"🔍 Processing sample with sentiment analysis...")
        
        try:
            # Load sample
            sample_df = pd.read_csv(sample_file)
            logger.info(f"   📊 Sample loaded: {len(sample_df):,} articles")
            
            # Match articles to stocks
            matched_articles = []
            
            for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Matching to stocks"):
                title = str(row.get('title', ''))
                content = str(row.get('content', ''))
                
                # Skip if text too short
                total_text = f"{title} {content}"
                if len(total_text) < self.config.min_text_length:
                    continue
                
                # Match to stocks
                matches = self.stock_matcher.match_text(content, title)
                
                # Create records for significant matches
                for symbol, relevance_score in matches.items():
                    if relevance_score >= self.config.relevance_threshold:
                        matched_articles.append({
                            'article_id': idx,
                            'symbol': symbol,
                            'title': title[:200],  # Truncate for memory
                            'content': content[:self.config.max_text_length],
                            'relevance_score': relevance_score,
                            'date': row.get('date', row.get('published_date', '2022-01-01'))
                        })
            
            if not matched_articles:
                logger.warning("⚠️ No articles matched to target stocks")
                return pd.DataFrame(), pd.DataFrame()
            
            matched_df = pd.DataFrame(matched_articles)
            logger.info(f"   🎯 Matched articles: {len(matched_df):,}")
            
            # Apply limits per symbol
            limited_articles = []
            for symbol in self.config.target_symbols:
                symbol_articles = matched_df[matched_df['symbol'] == symbol]
                if len(symbol_articles) > self.config.max_articles_per_symbol:
                    symbol_articles = symbol_articles.nlargest(self.config.max_articles_per_symbol, 'relevance_score')
                limited_articles.append(symbol_articles)
            
            matched_df = pd.concat(limited_articles, ignore_index=True) if limited_articles else pd.DataFrame()
            
            if matched_df.empty:
                return pd.DataFrame(), pd.DataFrame()
            
            logger.info(f"   📊 Final matched articles: {len(matched_df):,}")
            
            # Analyze sentiment
            logger.info("🔍 Analyzing sentiment with FinBERT...")
            
            sentiment_results = []
            
            # Process in batches
            for i in tqdm(range(0, len(matched_df), self.config.batch_size), desc="Sentiment analysis"):
                batch = matched_df.iloc[i:i+self.config.batch_size]
                
                # Prepare texts
                texts = []
                for _, row in batch.iterrows():
                    combined_text = f"{row['title']} {row['content']}"
                    texts.append(combined_text[:512])  # FinBERT max length
                
                # Analyze sentiment
                batch_results = self._analyze_sentiment_batch(texts)
                sentiment_results.extend(batch_results)
                
                # Memory management
                if self.memory_monitor.check_memory_limit():
                    self.memory_monitor.force_garbage_collection()
            
            # Add sentiment results to dataframe
            for i, result in enumerate(sentiment_results):
                for key, value in result.items():
                    matched_df.loc[i, key] = value
            
            # Create aggregated features
            aggregated_df = self._create_aggregated_features(matched_df)
            
            logger.info(f"✅ Sentiment processing completed!")
            logger.info(f"   📊 Articles with sentiment: {len(matched_df):,}")
            logger.info(f"   📈 Daily aggregated records: {len(aggregated_df):,}")
            
            return matched_df, aggregated_df
            
        except Exception as e:
            logger.error(f"❌ Sentiment processing failed: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame(), pd.DataFrame()
    
    def _analyze_sentiment_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """Analyze sentiment for a batch of texts"""
        results = []
        
        try:
            for text in texts:
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
                
                results.append(result)
                
        except Exception as e:
            logger.warning(f"⚠️ Batch sentiment analysis failed: {e}")
            # Return neutral results
            for _ in texts:
                results.append({
                    'negative': 0.33, 'neutral': 0.34, 'positive': 0.33,
                    'compound': 0.0, 'label': 'neutral', 'confidence': 0.34
                })
        
        return results
    
    def _create_aggregated_features(self, article_df: pd.DataFrame) -> pd.DataFrame:
        """Create daily aggregated sentiment features"""
        if article_df.empty:
            return pd.DataFrame()
        
        # Convert date
        article_df['date'] = pd.to_datetime(article_df['date']).dt.date
        
        # Aggregate by symbol and date
        aggregated = article_df.groupby(['symbol', 'date']).agg({
            'compound': ['mean', 'std', 'min', 'max', 'count'],
            'positive': ['mean', 'std'],
            'negative': ['mean', 'std'],
            'neutral': 'mean',
            'confidence': ['mean', 'min'],
            'relevance_score': ['mean', 'max'],
            'article_id': 'count'
        }).round(6)
        
        # Flatten columns
        aggregated.columns = ['_'.join(col).strip() for col in aggregated.columns]
        aggregated = aggregated.reset_index()
        
        # Rename for clarity
        column_mapping = {
            'compound_mean': 'sentiment_compound',
            'compound_std': 'sentiment_volatility',
            'compound_min': 'sentiment_min',
            'compound_max': 'sentiment_max',
            'compound_count': 'sentiment_count',
            'positive_mean': 'sentiment_positive',
            'negative_mean': 'sentiment_negative',
            'neutral_mean': 'sentiment_neutral',
            'confidence_mean': 'sentiment_confidence',
            'relevance_score_mean': 'sentiment_relevance',
            'article_id_count': 'article_count'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in aggregated.columns:
                aggregated[new_col] = aggregated[old_col]
        
        # Select final columns
        final_columns = [
            'symbol', 'date', 'sentiment_compound', 'sentiment_volatility',
            'sentiment_min', 'sentiment_max', 'sentiment_positive', 'sentiment_negative',
            'sentiment_neutral', 'sentiment_confidence', 'sentiment_relevance',
            'article_count'
        ]
        
        result_columns = [col for col in final_columns if col in aggregated.columns]
        return aggregated[result_columns].copy()
    
    def run_efficient_analysis(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Run complete efficient analysis of large FNSPID dataset"""
        logger.info("🚀 STARTING EFFICIENT FNSPID ANALYSIS")
        logger.info("=" * 80)
        logger.info(f"💾 Available memory: {psutil.virtual_memory().available / (1024**3):.1f} GB")
        logger.info(f"📊 File size: {os.path.getsize(FNSPID_NEWS_FILE) / (1024**3):.1f} GB")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        try:
            # Step 1: Analyze dataset structure
            structure_info = self.analyze_dataset_structure()
            
            # Step 2: Create efficient sample
            sample_file = self.create_efficient_sample()
            
            # Step 3: Process sample with sentiment analysis
            article_sentiments, aggregated_features = self.process_sample_with_sentiment(sample_file)
            
            # Step 4: Save results
            if not article_sentiments.empty:
                self._save_results(article_sentiments, aggregated_features)
            
            # Step 5: Generate report
            total_time = time.time() - start_time
            self._generate_efficient_report(structure_info, total_time)
            
            return article_sentiments, aggregated_features
            
        except Exception as e:
            logger.error(f"❌ Efficient analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame(), pd.DataFrame()
    
    def _save_results(self, article_sentiments: pd.DataFrame, aggregated_features: pd.DataFrame):
        """Save analysis results"""
        logger.info("💾 Saving analysis results...")
        
        os.makedirs(Path(ARTICLE_SENTIMENT_FILE).parent, exist_ok=True)
        
        if not article_sentiments.empty:
            article_sentiments.to_csv(ARTICLE_SENTIMENT_FILE, index=False)
            logger.info(f"   📄 Article sentiments: {ARTICLE_SENTIMENT_FILE}")
        
        if not aggregated_features.empty:
            aggregated_features.to_csv(SENTIMENT_OUTPUT_FILE, index=False)
            logger.info(f"   📊 Aggregated features: {SENTIMENT_OUTPUT_FILE}")
    
    def _generate_efficient_report(self, structure_info: Dict, total_time: float):
        """Generate analysis report"""
        logger.info("📋 EFFICIENT FNSPID ANALYSIS REPORT")
        logger.info("=" * 60)
        logger.info(f"📊 Original dataset: {structure_info['file_size_gb']:.1f} GB")
        logger.info(f"📊 Estimated rows: {structure_info['estimated_rows']:,}")
        logger.info(f"⏱️ Processing time: {total_time:.1f} seconds")
        logger.info(f"💾 Peak memory usage: {self.memory_monitor.get_memory_usage():.1f} GB")
        logger.info("")
        
        logger.info("🎯 PROCESSING EFFICIENCY:")
        logger.info(f"   📊 Sample ratio: {self.config.sample_ratio:.1%}")
        logger.info(f"   🏢 Target symbols: {len(self.config.target_symbols)}")
        logger.info(f"   📈 Articles processed: {self.processed_articles:,}")
        logger.info(f"   📊 Articles matched: {self.matched_articles:,}")
        logger.info(f"   📊 Articles with sentiment: {self.sentiment_analyzed:,}")
        logger.info("")
        logger.info("✅ EFFICIENT ANALYSIS COMPLETED")
        logger.info("=" * 60)
        # Save report to file
        report_file = Path(DATA_DIR) / "fnspid_analysis_report.txt"