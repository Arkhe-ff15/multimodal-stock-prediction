#!/usr/bin/env python3
"""
🔧 CACHE DISABLING FIX SCRIPT
============================
Fix the impossibly fast training by disabling feature caching.
"""

import os
import shutil
from pathlib import Path

def print_header(title, char="=", width=60):
    """Print formatted header"""
    print(f"\n{char * width}")
    print(f"🔧 {title}")
    print(f"{char * width}")

def backup_cache_files(project_root):
    """Backup cache files before deletion"""
    print_header("BACKING UP CACHE FILES")
    
    cache_dir = project_root / "data" / "processed"
    backup_dir = project_root / "cache_backup"
    
    if cache_dir.exists():
        print(f"📁 Cache directory: {cache_dir}")
        cache_files = list(cache_dir.glob("*.pkl"))
        
        if cache_files:
            print(f"💾 Found {len(cache_files)} cache files")
            
            # Create backup directory
            backup_dir.mkdir(exist_ok=True)
            print(f"📦 Creating backup: {backup_dir}")
            
            # Copy cache files to backup
            for cache_file in cache_files:
                backup_file = backup_dir / cache_file.name
                shutil.copy2(cache_file, backup_file)
                print(f"   ✅ Backed up: {cache_file.name}")
            
            return cache_files
        else:
            print("✅ No .pkl cache files found")
            return []
    else:
        print("✅ No cache directory found")
        return []

def delete_cache_files(cache_files):
    """Delete cache files to force fresh preprocessing"""
    print_header("DELETING CACHE FILES")
    
    if cache_files:
        print(f"🗑️ Deleting {len(cache_files)} cache files...")
        
        for cache_file in cache_files:
            try:
                cache_file.unlink()
                print(f"   ❌ Deleted: {cache_file.name}")
            except Exception as e:
                print(f"   ⚠️ Failed to delete {cache_file.name}: {e}")
        
        print("✅ Cache files deleted!")
        print("🎯 This will force fresh preprocessing and real training times")
    else:
        print("✅ No cache files to delete")

def find_cache_flags_in_code(project_root):
    """Find and suggest fixes for cache flags in code"""
    print_header("FINDING CACHE FLAGS IN CODE")
    
    models_file = project_root / "src" / "models.py"
    
    if not models_file.exists():
        print("❌ src/models.py not found")
        return []
    
    with open(models_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Look for joblib caching patterns
    cache_patterns = [
        r'joblib\.Memory',
        r'memory\.cache',
        r'@memory\.cache',
        r'cachedir',
        r'cache_size',
        r'compress.*=.*True',
        r'verbose.*=.*\d+'
    ]
    
    cache_flags_found = []
    lines = content.split('\n')
    
    for pattern in cache_patterns:
        import re
        matches = re.finditer(pattern, content, re.IGNORECASE)
        for match in matches:
            # Find the line number
            line_start = content[:match.start()].count('\n')
            line_content = lines[line_start] if line_start < len(lines) else ""
            
            cache_flags_found.append({
                'pattern': pattern,
                'line_num': line_start + 1,
                'line': line_content.strip()
            })
    
    if cache_flags_found:
        print("🚨 CACHE FLAGS FOUND IN CODE:")
        for i, flag in enumerate(cache_flags_found, 1):
            print(f"   {i}. Line {flag['line_num']}: {flag['line']}")
    else:
        print("✅ No obvious cache flags found in code")
    
    return cache_flags_found

def create_test_training_script(project_root):
    """Create a test script to verify training speed"""
    print_header("CREATING TEST TRAINING SCRIPT")
    
    test_script = project_root / "test_training_speed.py"
    
    script_content = '''#!/usr/bin/env python3
"""
🧪 TEST TRAINING SPEED
=====================
Verify that training now takes proper time (3-4 minutes per epoch).
"""

import time
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_single_epoch():
    """Test training speed for one epoch"""
    print("🧪 TESTING TRAINING SPEED")
    print("=" * 40)
    print("🎯 Expected: 3-4 minutes per epoch")
    print("🚨 If still fast: Cache still active!")
    print("=" * 40)
    
    try:
        # Import your training framework
        from enhanced_model_framework import EnhancedModelFramework
        
        # Initialize framework
        framework = EnhancedModelFramework()
        
        # Test LSTM training for just 1 epoch
        print("🚀 Testing LSTM training for 1 epoch...")
        
        start_time = time.time()
        
        # You'll need to modify this to train just 1 epoch
        # This is a template - adjust for your specific implementation
        print("⚠️ MANUAL IMPLEMENTATION NEEDED")
        print("   1. Load your dataset")
        print("   2. Create model")
        print("   3. Train for exactly 1 epoch")
        print("   4. Measure time")
        
        # Placeholder timing
        elapsed = time.time() - start_time
        
        print(f"⏱️ Training time: {elapsed:.1f} seconds")
        
        if elapsed < 60:
            print("🚨 STILL TOO FAST! Cache not fully disabled")
            print("   Expected: 180-240 seconds (3-4 minutes)")
        elif elapsed > 300:
            print("⚠️ Might be too slow - check configuration")
        else:
            print("✅ TRAINING SPEED LOOKS GOOD!")
            print("🎯 Ready for full training")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("💡 Implement the test manually in your training script")

if __name__ == "__main__":
    test_single_epoch()
'''
    
    with open(test_script, 'w') as f:
        f.write(script_content)
    
    test_script.chmod(0o755)  # Make executable
    print(f"📝 Created test script: {test_script}")
    print("🧪 Run: python test_training_speed.py")

def main():
    """Main fix function"""
    print_header("CACHE DISABLING FIX", "🔧")
    print("🎯 Goal: Disable caching to get proper 3-4 minute training times")
    print("🚨 Current: 5 seconds per epoch (WAY TOO FAST!)")
    
    # Find project root
    current_dir = Path.cwd()
    project_root = None
    
    for parent in [current_dir] + list(current_dir.parents):
        if (parent / "src" / "models.py").exists():
            project_root = parent
            break
    
    if not project_root:
        print("❌ Could not find project root")
        return
    
    print(f"📁 Project root: {project_root}")
    
    # Step 1: Backup cache files
    cache_files = backup_cache_files(project_root)
    
    # Step 2: Delete cache files
    if cache_files:
        response = input(f"\n🗑️ Delete {len(cache_files)} cache files? (y/N): ")
        if response.lower().startswith('y'):
            delete_cache_files(cache_files)
        else:
            print("⚠️ Cache files not deleted - training will still be fast")
    
    # Step 3: Find cache flags in code
    cache_flags = find_cache_flags_in_code(project_root)
    
    # Step 4: Create test script
    create_test_training_script(project_root)
    
    # Summary and next steps
    print_header("FIX SUMMARY")
    
    if cache_files:
        print("✅ Cache files backed up and deleted")
        print("🎯 This should fix the training speed issue")
    else:
        print("⚠️ No cache files found - issue might be elsewhere")
    
    print_header("NEXT STEPS")
    
    print("1. 🧪 Run: python test_training_speed.py")
    print("2. ⏱️ Verify training takes 3-4 minutes per epoch")
    print("3. 🚀 If still fast, check for other caching mechanisms")
    print("4. 🎯 Once fixed, retrain all models for valid results")
    
    print(f"\n💡 EXPECTED RESULTS AFTER FIX:")
    print("   📊 LSTM: 3-4 minutes per epoch (not 5 seconds)")
    print("   🎯 31 epochs: ~2 hours total (not 2.6 minutes)")
    print("   ✅ Academically valid training times")
    
    print(f"\n🔄 TO RESTORE CACHE (if needed):")
    print(f"   📦 Cache backup saved in: cache_backup/")
    print(f"   🔄 Copy files back to: data/processed/")

if __name__ == "__main__":
    main()