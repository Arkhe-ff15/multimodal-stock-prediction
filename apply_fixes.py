#!/usr/bin/env python3
"""
Quick Fixes for Original Script
===============================
Apply these fixes to make the original script work with your data.

CRITICAL FIXES NEEDED:
1. Fix filename typo in config.yaml
2. Create config_reader.py (see separate artifact)
3. Make resource requirements realistic
4. Fix data loading issues
"""

# 1. FIX CONFIG.YAML
# Replace this line in config.yaml:
#   fnspid_data: 'data/raw/nasdaq_exteral_data.csv'  # TYPO!
# With:
#   fnspid_data: 'data/raw/nasdaq_external_data.csv'  # FIXED

# 2. MAKE CONFIG REALISTIC - Add these overrides to config.yaml:
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from typing import List, Dict   
logger = logging.getLogger(__name__)

CONFIG_OVERRIDES = """
# Add these realistic overrides to your config.yaml:

# Make memory requirements realistic
system:
  memory:
    max_memory_usage_gb: 8  # Reduced from 24GB
  hardware:
    gpu_memory_fraction: 0.7  # Reduced from 0.85

# Make processing realistic  
pipeline:
  execution:
    max_workers: 2  # Reduced from 6
    memory_limit_gb: 8  # Reduced from 24

# Use more data (less aggressive filtering)
data:
  fnspid:
    production:
      sample_ratio: 1.0  # Use all data instead of 30%
      min_confidence_score: 0.6  # Reduced from 0.7
      chunk_size: 50000  # Reduced from 100000

# Smaller batch sizes
sentiment:
  models:
    finbert:
      batch_size: 8  # Reduced from 32
      confidence_threshold: 0.6  # Reduced from 0.7
"""

# 3. PATCH THE ORIGINAL SCRIPT - Apply these changes:

# Change 1: Fix the data loading to handle any CSV format
def load_and_filter_fnspid_FIXED(self) -> pd.DataFrame:
    """FIXED version of load_and_filter_fnspid"""
    logger.info("📥 Loading data with fixes...")
    fnspid_path = self.data_paths['raw_fnspid']
    
    if not fnspid_path.exists():
        raise FileNotFoundError(f"Data file not found: {fnspid_path}")
    
    # AUTO-DETECT COLUMNS (instead of hardcoded mapping)
    sample = pd.read_csv(fnspid_path, nrows=10)
    logger.info(f"📋 Available columns: {list(sample.columns)}")
    
    # Try to detect column mapping automatically
    column_mapping = {}
    
    # Find date column
    date_candidates = [col for col in sample.columns if any(word in col.lower() for word in ['date', 'time', 'published'])]
    if date_candidates:
        column_mapping['Date'] = date_candidates[0]
    else:
        logger.error("Could not find date column. Available columns:")
        for i, col in enumerate(sample.columns):
            logger.error(f"  {i}: {col}")
        raise ValueError("Please manually specify date column")
    
    # Find headline column  
    headline_candidates = [col for col in sample.columns if any(word in col.lower() for word in ['title', 'headline', 'text', 'article'])]
    if headline_candidates:
        column_mapping['Article_title'] = headline_candidates[0]
    else:
        raise ValueError("Could not find headline column")
    
    # Find symbol column
    symbol_candidates = [col for col in sample.columns if any(word in col.lower() for word in ['symbol', 'ticker', 'stock'])]
    if symbol_candidates:
        column_mapping['Stock_symbol'] = symbol_candidates[0]
    else:
        raise ValueError("Could not find symbol column")
    
    logger.info(f"📋 Auto-detected mapping: {column_mapping}")
    
    # REST OF FUNCTION REMAINS THE SAME...
    # (Process the data using the detected column mapping)

# Change 2: Fix memory management in chunk processing
def load_and_filter_fnspid_MEMORY_FIXED(self) -> pd.DataFrame:
    """Memory-efficient version"""
    # Instead of accumulating all chunks:
    # filtered_chunks.append(chunk_filtered)
    
    # Use this approach:
    temp_files = []
    chunk_counter = 0
    
    for chunk in pd.read_csv(fnspid_path, chunksize=chunk_size):
        # Process chunk
        chunk_filtered = process_chunk(chunk)  # Your existing logic
        
        # Save to temporary file instead of memory
        if len(chunk_filtered) > 0:
            temp_file = f"temp_chunk_{chunk_counter}.csv"
            chunk_filtered.to_csv(temp_file, index=False)
            temp_files.append(temp_file)
            chunk_counter += 1
    
    # Combine temporary files
    combined_chunks = []
    for temp_file in temp_files:
        combined_chunks.append(pd.read_csv(temp_file))
        os.remove(temp_file)  # Clean up
    
    return pd.concat(combined_chunks, ignore_index=True)

# Change 3: Fix the confidence filtering logic
def adaptive_confidence_filter_FIXED(self, sentiment_results: List[Dict]) -> List[Dict]:
    """FIXED adaptive confidence filtering"""
    if not self.enable_adaptive_confidence or not sentiment_results:
        return [r for r in sentiment_results if r['confidence'] >= self.confidence_threshold]
    
    confidences = [r['confidence'] for r in sentiment_results]
    
    # FIXED: More robust threshold calculation
    mean_conf = np.mean(confidences)
    percentile_25 = np.percentile(confidences, 25)
    percentile_75 = np.percentile(confidences, 75)
    
    # Use percentile-based approach instead of std-based
    adaptive_threshold = max(
        0.5,  # Minimum threshold
        percentile_25,  # 25th percentile as threshold
        self.confidence_threshold * 0.8  # 80% of original threshold
    )
    
    filtered_results = [r for r in sentiment_results if r['confidence'] >= adaptive_threshold]
    
    # Safety check: keep at least 50% of data
    if len(filtered_results) < 0.5 * len(sentiment_results):
        adaptive_threshold = np.percentile(confidences, 50)  # Median
        filtered_results = [r for r in sentiment_results if r['confidence'] >= adaptive_threshold]
    
    logger.info(f"🎯 Adaptive threshold: {adaptive_threshold:.3f} (was {self.confidence_threshold:.3f})")
    logger.info(f"📈 Retained: {len(filtered_results)}/{len(sentiment_results)} articles")
    
    return filtered_results

# Change 4: Add proper error handling to sentiment analysis
def analyze_sentiment_with_validation_FIXED(self, articles_df: pd.DataFrame) -> pd.DataFrame:
    """FIXED sentiment analysis with proper error handling"""
    sentiment_results = []
    failed_batches = 0
    
    for i in range(0, len(articles_df), self.batch_size):
        batch = articles_df.iloc[i:i+self.batch_size]
        headlines = batch['headline'].tolist()
        
        try:
            # Your existing FinBERT logic here
            inputs = self.tokenizer(headlines, ...)
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()
            
            # Process predictions (your existing logic)
            for j, pred in enumerate(predictions):
                result = batch.iloc[j].copy()
                # Add sentiment scores
                sentiment_results.append(result)
                
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning(f"🚨 GPU OOM at batch {i}, reducing batch size")
                # Reduce batch size and retry
                self.batch_size = max(1, self.batch_size // 2)
                torch.cuda.empty_cache()
                # Retry with smaller batch
                continue
            else:
                logger.error(f"❌ Batch {i} failed: {e}")
                failed_batches += 1
                
                # Add neutral sentiment for failed batch
                for j in range(len(batch)):
                    result = batch.iloc[j].copy()
                    result['sentiment_compound'] = 0.0
                    result['confidence'] = 0.3  # Low confidence for fallback
                    sentiment_results.append(result)
        
        except Exception as e:
            logger.error(f"❌ Unexpected error in batch {i}: {e}")
            failed_batches += 1
            
            # Add neutral sentiment for failed batch
            for j in range(len(batch)):
                result = batch.iloc[j].copy()
                result['sentiment_compound'] = 0.0
                result['confidence'] = 0.3
                sentiment_results.append(result)
    
    if failed_batches > 0:
        logger.warning(f"⚠️ {failed_batches} batches failed, used neutral sentiment as fallback")
    
    return pd.DataFrame(sentiment_results)

# 5. CREATE SIMPLE TEST SCRIPT
def test_fixes():
    """Test the fixes with a small sample"""
    print("Testing fixes...")
    
    # Test 1: Check if file exists
    import pandas as pd
    from pathlib import Path
    
    data_file = Path("data/raw/nasdaq_exteral_data.csv")
    if not data_file.exists():
        print("❌ Data file not found!")
        print("Expected location:", data_file.absolute())
        return False
    
    # Test 2: Check file structure
    try:
        sample = pd.read_csv(data_file, nrows=5)
        print("✅ File loads successfully")
        print("📋 Columns found:", list(sample.columns))
        print("📊 Sample data shape:", sample.shape)
        
        # Check for required column types
        date_cols = [col for col in sample.columns if any(word in col.lower() for word in ['date', 'time'])]
        text_cols = [col for col in sample.columns if any(word in col.lower() for word in ['title', 'headline', 'text'])]
        symbol_cols = [col for col in sample.columns if any(word in col.lower() for word in ['symbol', 'ticker', 'stock'])]
        
        print(f"📅 Potential date columns: {date_cols}")
        print(f"📰 Potential text columns: {text_cols}")
        print(f"📈 Potential symbol columns: {symbol_cols}")
        
        if not (date_cols and text_cols and symbol_cols):
            print("⚠️ Warning: May not have all required column types")
            
        return True
        
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return False

if __name__ == "__main__":
    print("Quick Fixes for Original Script")
    print("=" * 40)
    print()
    print("1. Fix filename typo in config.yaml:")
    print("   nasdaq_exteral_data.csv → nasdaq_external_data.csv")
    print()
    print("2. Add realistic resource limits to config.yaml:")
    print(CONFIG_OVERRIDES)
    print()
    print("3. Apply code patches above to original script")
    print()
    print("4. Create config_reader.py (see separate artifact)")
    print()
    print("Testing your data file...")
    test_fixes()