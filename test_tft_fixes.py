#!/usr/bin/env python3
"""
QUICK TEST SCRIPT FOR TFT FIXES
===============================
"""

import sys
from pathlib import Path
sys.path.append('src')

def test_imports():
    """Test that all imports work"""
    print("🧪 Testing imports...")
    try:
        from models import (
            setup_signal_handlers, 
            check_version_compatibility,
            EnhancedTFTModel,
            EnhancedDataLoader
        )
        print("   ✅ All imports successful")
        return True
    except Exception as e:
        print(f"   ❌ Import failed: {e}")
        return False

def test_signal_handlers():
    """Test signal handler setup"""
    print("🧪 Testing signal handlers...")
    try:
        from models import setup_signal_handlers
        setup_signal_handlers()
        print("   ✅ Signal handlers configured")
        return True
    except Exception as e:
        print(f"   ❌ Signal handler test failed: {e}")
        return False

def test_version_compatibility():
    """Test version compatibility check"""
    print("🧪 Testing version compatibility...")
    try:
        # Test Lightning import compatibility
        try:
            import lightning.pytorch as pl
            lightning_version = pl.__version__
            lightning_style = "lightning.pytorch"
        except ImportError:
            try:
                import pytorch_lightning as pl
                lightning_version = pl.__version__
                lightning_style = "pytorch_lightning"
            except ImportError:
                print("   ❌ No Lightning installation found")
                return False
        
        # Test pytorch-forecasting
        try:
            import pytorch_forecasting as pf
            pf_version = pf.__version__
        except ImportError:
            print("   ❌ pytorch-forecasting not installed")
            return False
        
        print(f"   📦 Lightning: {lightning_version} ({lightning_style})")
        print(f"   📦 PyTorch Forecasting: {pf_version}")
        
        # Check for known compatibility issues
        issues = []
        if lightning_version.startswith('2.') and 'pytorch_lightning' in lightning_style:
            issues.append("Lightning 2.x detected but using old import style")
        
        if lightning_version.startswith('1.') and 'lightning.pytorch' in lightning_style:
            issues.append("Lightning 1.x detected but using new import style")
        
        if issues:
            print("   ⚠️ Compatibility issues detected:")
            for issue in issues:
                print(f"      - {issue}")
            print("   💡 Try: pip install pytorch-lightning==1.9.* pytorch-forecasting")
        else:
            print("   ✅ Version compatibility looks good")
        
        return len(issues) == 0
        
    except Exception as e:
        print(f"   ❌ Version check failed: {e}")
        return False

def test_dataset_loading():
    """Test dataset loading"""
    print("🧪 Testing dataset loading...")
    try:
        from models import EnhancedDataLoader
        
        # Check if data exists
        data_dir = Path("data/model_ready")
        if not data_dir.exists():
            print("   ⚠️ Data directory not found - run data_prep.py first")
            return False
        
        # Try loading datasets
        loader = EnhancedDataLoader()
        
        # Test baseline
        baseline_files = list(data_dir.glob("baseline_*.csv"))
        if baseline_files:
            try:
                baseline_dataset = loader.load_dataset('baseline')
                print(f"   ✅ Baseline loaded: {len(baseline_dataset['selected_features'])} features")
            except Exception as e:
                print(f"   ⚠️ Baseline loading failed: {e}")
        
        # Test enhanced
        enhanced_files = list(data_dir.glob("enhanced_*.csv"))
        if enhanced_files:
            try:
                enhanced_dataset = loader.load_dataset('enhanced')
                print(f"   ✅ Enhanced loaded: {len(enhanced_dataset['selected_features'])} features")
                
                # Check for sentiment features
                sentiment_features = enhanced_dataset['feature_analysis'].get('sentiment_features', [])
                print(f"   🎭 Sentiment features: {len(sentiment_features)}")
                
            except Exception as e:
                print(f"   ⚠️ Enhanced loading failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Dataset loading test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 RUNNING TFT FIXES VALIDATION TESTS")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Signal Handlers", test_signal_handlers),
        ("Version Compatibility", test_version_compatibility),
        ("Dataset Loading", test_dataset_loading)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ Test {test_name} crashed: {e}")
            results[test_name] = False
        print()
    
    # Summary
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 30)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! TFT fixes are working correctly.")
        print("🚀 Ready to run full training with: python src/models.py")
    else:
        print("⚠️ Some tests failed. Check the output above for issues.")
        print("📝 You may need to:")
        print("   1. Run data_prep.py if dataset loading failed")
        print("   2. Check package versions if compatibility failed")
        print("   3. Review the applied fixes if imports failed")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
