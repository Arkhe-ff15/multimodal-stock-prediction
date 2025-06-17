# Quick diagnostic script to find the 'stock' validation source

import sys
import os
sys.path.append(os.getcwd())

# Test our column mapping
print("=== FNSPID PROCESSOR COLUMN MAPPING ===")
try:
    from src.fnspid_processor import FNSPIDProcessor
    from config import PipelineConfig
    
    config = PipelineConfig()
    processor = FNSPIDProcessor(config)
    
    # Test column detection
    test_columns = ['Unnamed: 0', 'Date', 'Article_title', 'Stock_symbol', 'Url', 'Publisher']
    mapping = processor.detect_column_mapping(test_columns)
    print(f"✅ Column mapping: {mapping}")
    
    # Test validation requirements
    required = ['date', 'symbol', 'headline']
    print(f"✅ Required columns: {required}")
    
    missing = [col for col in required if col not in mapping]
    print(f"✅ Missing columns: {missing}")
    
except Exception as e:
    print(f"❌ FNSPID Processor Error: {e}")

print("\n=== DATA STANDARDS MODULE ===")
try:
    from src.data_standards import DataValidator
    print(f"✅ DataValidator imported successfully")
    
    # Check what DataValidator expects
    import inspect
    methods = [method for method in dir(DataValidator) if not method.startswith('_')]
    print(f"✅ DataValidator methods: {methods}")
    
    # Try to find validation requirements
    for method_name in methods:
        method = getattr(DataValidator, method_name)
        if hasattr(method, '__doc__') and method.__doc__:
            print(f"📋 {method_name}: {method.__doc__[:100]}...")
            
except Exception as e:
    print(f"❌ DataValidator Error: {e}")

print("\n=== PIPELINE CONFIG ===")
try:
    from config import PipelineConfig
    config = PipelineConfig()
    print(f"✅ Config symbols: {config.symbols}")
    print(f"✅ Config date range: {config.start_date} to {config.end_date}")
    
except Exception as e:
    print(f"❌ Config Error: {e}")

print("\n=== DIAGNOSIS COMPLETE ===")
print("Look for any 'stock' references in the output above!")