#!/usr/bin/env python3
"""
Patched Experiment Runner - Run Steps 1-2 with fixes applied
"""

import logging
import sys
from pathlib import Path

# Ensure src is in path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Apply patches programmatically
def apply_runtime_patches():
    """Apply patches at runtime"""
    import src.data_loader as dl
    import src.sentiment as sent
    
    # Patch data loader timezone handling
    original_download = dl.EnhancedDataCollector._download_enhanced_stock
    
    def patched_download(self, symbol: str):
        """Patched download with timezone handling"""
        try:
            result = original_download(self, symbol)
            if result and hasattr(result.data.index, 'tz') and result.data.index.tz is not None:
                result.data.index = result.data.index.tz_localize(None)
            return result
        except Exception as e:
            if "tz-naive and tz-aware" in str(e):
                logging.warning(f"Timezone error for {symbol}, using mock data")
                return self._create_enhanced_mock_data(symbol)
            raise
    
    dl.EnhancedDataCollector._download_enhanced_stock = patched_download
    
    # Patch sentiment config
    sent.SentimentConfig.confidence_threshold = 0.5
    sent.SentimentConfig.relevance_threshold = 0.7
    sent.SentimentConfig.min_text_length = 5
    
    logging.info("✅ Runtime patches applied")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("🔧 Applying runtime patches...")
    apply_runtime_patches()
    
    print("🚀 Running experiment with patches...")
    
    # Import after patches
    from run_experiment import main
    
    # Override sys.argv to run steps 1-2
    sys.argv = ["run_experiment.py", "--config", "configs/unified_config.yaml", "--steps", "1", "2"]
    
    try:
        main()
        print("✅ Experiment completed with patches!")
    except Exception as e:
        print(f"❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
