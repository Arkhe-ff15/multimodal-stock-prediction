#!/usr/bin/env python3
"""
Test Dependencies - Check if all required packages are available
================================================================
"""

def test_dependencies():
    """Test all critical dependencies"""
    print("🧪 TESTING CRITICAL DEPENDENCIES")
    print("=" * 50)
    
    dependencies = {
        # Core ML Dependencies
        'torch': 'PyTorch (neural networks)',
        'pytorch_lightning': 'PyTorch Lightning (training framework)', 
        'lightning.pytorch': 'Lightning 2.x (new import path)',
        
        # Financial ML Dependencies
        'pytorch_forecasting': 'PyTorch Forecasting (TFT models)',
        
        # NLP Dependencies
        'transformers': 'Transformers (FinBERT)',
        
        # Data Science Dependencies
        'pandas': 'Pandas (data manipulation)',
        'numpy': 'NumPy (numerical computing)',
        'scipy': 'SciPy (scientific computing)',
        'sklearn': 'Scikit-learn (ML utilities)',
        
        # Visualization
        'matplotlib': 'Matplotlib (plotting)',
        'seaborn': 'Seaborn (statistical plotting)',
        
        # Additional Dependencies
        'yaml': 'PyYAML (config files)',
        'joblib': 'Joblib (model persistence)',
        'psutil': 'psutil (memory monitoring)',
    }
    
    missing = []
    available = []
    warnings = []
    
    for package, description in dependencies.items():
        try:
            if package == 'lightning.pytorch':
                # Special case for Lightning 2.x
                import lightning.pytorch as pl
                available.append(f"✅ {package} - {description}")
                # Check version compatibility
                version = getattr(pl, '__version__', 'unknown')
                if version.startswith('2.'):
                    warnings.append(f"⚠️ Lightning 2.x detected ({version}) - may have TFT compatibility issues")
            else:
                __import__(package)
                available.append(f"✅ {package} - {description}")
        except ImportError as e:
            missing.append(f"❌ {package} - {description}")
            print(f"❌ Missing: {package} ({description})")
    
    print(f"\n📊 DEPENDENCY TEST RESULTS:")
    print(f"✅ Available: {len(available)}")
    print(f"❌ Missing: {len(missing)}")
    print(f"⚠️ Warnings: {len(warnings)}")
    
    if missing:
        print(f"\n❌ MISSING DEPENDENCIES:")
        for item in missing:
            print(f"   {item}")
        
        print(f"\n💡 INSTALL COMMAND:")
        missing_packages = [item.split(' - ')[0].replace('❌ ', '') for item in missing]
        install_cmd = f"pip install {' '.join(missing_packages)}"
        print(f"   {install_cmd}")
    
    if warnings:
        print(f"\n⚠️ WARNINGS:")
        for warning in warnings:
            print(f"   {warning}")
    
    if available:
        print(f"\n✅ AVAILABLE DEPENDENCIES:")
        for item in available[:5]:  # Show first 5
            print(f"   {item}")
        if len(available) > 5:
            print(f"   ... and {len(available) - 5} more")
    
    return len(missing) == 0

def test_specific_imports():
    """Test specific problematic imports"""
    print(f"\n🎯 TESTING SPECIFIC CRITICAL IMPORTS:")
    print("-" * 40)
    
    critical_tests = [
        ("PyTorch Lightning", "import pytorch_lightning as pl"),
        ("Lightning 2.x", "import lightning.pytorch as pl"),
        ("PyTorch Forecasting", "from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer"),
        ("FinBERT Dependencies", "from transformers import AutoTokenizer, AutoModelForSequenceClassification"),
        ("TFT Metrics", "from pytorch_forecasting.metrics import QuantileLoss, MAE, RMSE"),
        ("TFT Data", "from pytorch_forecasting.data import GroupNormalizer"),
    ]
    
    for test_name, import_cmd in critical_tests:
        try:
            exec(import_cmd)
            print(f"✅ {test_name}")
        except ImportError as e:
            print(f"❌ {test_name}: {e}")
        except Exception as e:
            print(f"⚠️ {test_name}: {e}")

if __name__ == "__main__":
    success = test_dependencies()
    test_specific_imports()
    
    if success:
        print(f"\n🎉 ALL DEPENDENCIES AVAILABLE!")
        print(f"✅ Ready to proceed with model training")
    else:
        print(f"\n❌ MISSING DEPENDENCIES DETECTED")
        print(f"🔧 Install missing packages before proceeding")
        exit(1)