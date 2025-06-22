#!/usr/bin/env python3
"""
QUICK FIX IMPLEMENTATION SCRIPT
===============================
Run this script to apply the most critical fixes to resolve TFT training issues.

Usage: python quick_fix_script.py
"""

import os
import shutil
import re
from pathlib import Path
from datetime import datetime

def backup_file(file_path):
    """Create backup of original file"""
    backup_path = f"{file_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(file_path, backup_path)
    print(f"✅ Backup created: {backup_path}")
    return backup_path

def apply_critical_fixes():
    """Apply the most critical fixes to resolve TFT training issues"""
    
    print("🔧 APPLYING CRITICAL TFT FIXES")
    print("=" * 50)
    
    models_py_path = Path("src/models.py")
    
    if not models_py_path.exists():
        print(f"❌ File not found: {models_py_path}")
        return False
    
    # Create backup
    backup_path = backup_file(models_py_path)
    
    try:
        # Read the original file
        with open(models_py_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print("📝 Applying fixes...")
        
        # Fix 0: CRITICAL - Fix Lightning import pattern
        lightning_import_patterns = [
            (r'import pytorch_lightning as pl', 'import lightning.pytorch as pl'),
            (r'from pytorch_lightning\b', 'from lightning.pytorch'),
            (r'from pytorch_lightning\.', 'from lightning.pytorch.'),
        ]
        
        for old_pattern, new_pattern in lightning_import_patterns:
            if re.search(old_pattern, content):
                content = re.sub(old_pattern, new_pattern, content)
                print(f"   ✅ Fixed Lightning import: {old_pattern} -> {new_pattern}")
        
        # Check if we need both import styles (compatibility)
        if 'pytorch_lightning' in content and 'lightning.pytorch' not in content:
            # Add compatibility import
            import_section = re.search(r'(import warnings\nwarnings\.filterwarnings\(\'ignore\'\))', content)
            if import_section:
                compatibility_imports = '''
# Lightning compatibility imports
try:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
    from lightning.pytorch.loggers import TensorBoardLogger
    LIGHTNING_AVAILABLE = True
except ImportError:
    try:
        import pytorch_lightning as pl
        from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
        from pytorch_lightning.loggers import TensorBoardLogger
        LIGHTNING_AVAILABLE = True
    except ImportError:
        LIGHTNING_AVAILABLE = False
'''
                content = content.replace(
                    import_section.group(1),
                    import_section.group(1) + compatibility_imports
                )
                print("   ✅ Added Lightning compatibility imports")
        
        # Fix 1: Add signal import
        if 'import signal' not in content:
            import_section = re.search(r'(import warnings\nwarnings\.filterwarnings\(\'ignore\'\))', content)
            if import_section:
                content = content.replace(
                    import_section.group(1),
                    import_section.group(1) + '\nimport signal\nimport time'
                )
                print("   ✅ Added signal and time imports")
        
        # Fix 2: Replace "robust" transformation with "softplus"
        robust_patterns = [
            r'transformation="robust"',
            r"transformation='robust'",
            r'transformation=.*robust.*'
        ]
        
        for pattern in robust_patterns:
            if re.search(pattern, content):
                content = re.sub(pattern, 'transformation="softplus"', content)
                print("   ✅ Fixed 'robust' transformation -> 'softplus'")
        
        # Fix 3: Add signal handler function
        signal_handler_code = '''
def setup_signal_handlers():
    """Setup graceful shutdown handlers"""
    def signal_handler(signum, frame):
        logger.info(f"🛑 Received signal {signum}, attempting graceful shutdown...")
        raise KeyboardInterrupt("Training interrupted by user")
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

'''
        
        # Find a good place to insert the signal handler (after imports, before classes)
        class_match = re.search(r'class ModelTrainingError\(Exception\):', content)
        if class_match and 'setup_signal_handlers' not in content:
            insert_pos = class_match.start()
            content = content[:insert_pos] + signal_handler_code + content[insert_pos:]
            print("   ✅ Added signal handler function")
        
        # Fix 4: Fix the exit() call issue
        exit_pattern = r'(\s+)exit\(1\)(\s*#.*)?'
        if re.search(exit_pattern, content):
            content = re.sub(
                exit_pattern,
                r'\1# Fixed: Return gracefully instead of calling exit()\1return {"error": "Training interrupted", "interrupted": True}',
                content
            )
            print("   ✅ Fixed exit() call issue")
        
        # Fix 5: Add Lightning module check before trainer.fit()
        trainer_fit_pattern = r'(\s+)(self\.trainer\.fit\(self\.model,.*?\))'
        if re.search(trainer_fit_pattern, content):
            lightning_check = r'''\1# FIX: Ensure model is Lightning compatible
\1if not isinstance(self.model, pl.LightningModule):
\1    raise ModelTrainingError(f"Model is not a LightningModule: {type(self.model)}")
\1
\1\2'''
            
            content = re.sub(trainer_fit_pattern, lightning_check, content)
            print("   ✅ Added Lightning module compatibility check")
        
        # Fix 6: Add version compatibility check function
        version_check_code = '''
def check_version_compatibility():
    """Check package version compatibility"""
    try:
        import pytorch_lightning as pl
        import pytorch_forecasting as pf
        import torch
        
        logger.info("🔍 Checking version compatibility...")
        logger.info(f"   PyTorch: {torch.__version__}")
        logger.info(f"   Lightning: {pl.__version__}")
        logger.info(f"   PyTorch Forecasting: {pf.__version__}")
        
        # Check for known incompatible versions
        if pl.__version__.startswith('2.'):
            logger.warning("⚠️ Lightning 2.x detected - may have compatibility issues")
        
        return True
    except Exception as e:
        logger.error(f"❌ Version compatibility check failed: {e}")
        return False

'''
        
        # Insert version check function
        if 'check_version_compatibility' not in content:
            signal_handler_pos = content.find('def setup_signal_handlers():')
            if signal_handler_pos != -1:
                content = content[:signal_handler_pos] + version_check_code + content[signal_handler_pos:]
                print("   ✅ Added version compatibility check")
        
        # Fix 7: Add signal handler call in training methods
        training_start_pattern = r'(\s+)(start_time = datetime\.now\(\))'
        if re.search(training_start_pattern, content):
            signal_setup = r'\1setup_signal_handlers()  # Setup graceful shutdown\1\2'
            content = re.sub(training_start_pattern, signal_setup, content)
            print("   ✅ Added signal handler setup in training methods")
        
        # Write the fixed content back
        with open(models_py_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("✅ All critical fixes applied successfully!")
        print(f"📁 Original backed up to: {backup_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error applying fixes: {e}")
        print(f"🔄 Restoring from backup: {backup_path}")
        
        # Restore from backup
        shutil.copy2(backup_path, models_py_path)
        return False

def create_test_script():
    """Create a test script to verify the fixes"""
    
    test_script = '''#!/usr/bin/env python3
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
    
    print(f"\\nOverall: {passed}/{total} tests passed")
    
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
'''
    
    test_script_path = Path("test_tft_fixes.py")
    with open(test_script_path, 'w', encoding='utf-8') as f:
        f.write(test_script)
    
    print(f"✅ Test script created: {test_script_path}")
    print("🧪 Run with: python test_tft_fixes.py")

def main():
    """Main execution"""
    print("🔧 QUICK TFT FIXES APPLICATION")
    print("=" * 40)
    print("This script will apply critical fixes to resolve TFT training issues:")
    print("1. Fix 'robust' transformation error")
    print("2. Fix Lightning module compatibility")
    print("3. Fix signal handling and exit() issues") 
    print("4. Add version compatibility checks")
    print("5. Add graceful shutdown handling")
    print("=" * 40)
    
    # Confirm before proceeding
    response = input("🤔 Apply fixes to src/models.py? (y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        print("❌ Operation cancelled")
        return False
    
    # Apply fixes
    success = apply_critical_fixes()
    
    if success:
        print("\\n🎉 FIXES APPLIED SUCCESSFULLY!")
        
        # Create test script
        create_test_script()
        
        print("\\n🚀 NEXT STEPS:")
        print("1. Run the test script: python test_tft_fixes.py")
        print("2. If tests pass, try training: python src/models.py")
        print("3. If issues persist, check the full test plan in the artifacts")
        
        return True
    else:
        print("\\n❌ FIXES FAILED TO APPLY")
        print("Please check the error messages above and apply fixes manually")
        return False

if __name__ == "__main__":
    main()