#!/usr/bin/env python3
"""
Check data paths and fix any mismatches
"""

def check_data_paths():
    """Check where data files are being created vs expected"""
    
    print("🔍 CHECKING DATA PATHS")
    print("=" * 30)
    
    # Check config paths
    from config import get_quick_test_config
    config = get_quick_test_config()
    
    print(f"📍 Expected core dataset path: {config.core_dataset_path}")
    print(f"📍 Path exists: {config.core_dataset_path.exists()}")
    
    # Check if file exists anywhere
    import glob
    from pathlib import Path
    
    # Search for dataset files
    dataset_files = []
    for pattern in ["**/combined_dataset.csv", "**/*dataset*.csv"]:
        files = list(Path(".").glob(pattern))
        dataset_files.extend(files)
    
    if dataset_files:
        print(f"\n📊 Found dataset files:")
        for file in dataset_files:
            size = file.stat().st_size / (1024*1024)  # MB
            print(f"   • {file} ({size:.1f} MB)")
    else:
        print("\n❌ No dataset files found")
    
    # Check if data.py creates the file in the right place
    print(f"\n🔍 Checking data.py behavior...")
    
    try:
        # Read data.py to see what path it uses
        with open("src/data.py", 'r') as f:
            content = f.read()
        
        # Look for dataset file references
        if 'COMBINED_DATASET' in content:
            import re
            matches = re.findall(r'COMBINED_DATASET\s*=\s*["\']([^"\']+)["\']', content)
            if matches:
                print(f"   📍 data.py uses path: {matches[0]}")
        
        # Look for to_csv calls
        csv_matches = re.findall(r'\.to_csv\(["\']([^"\']+)["\']', content)
        if csv_matches:
            print(f"   📍 data.py saves to: {csv_matches}")
            
    except Exception as e:
        print(f"   ❌ Error reading data.py: {e}")

def fix_path_mismatch():
    """Fix any path mismatches"""
    
    print(f"\n🔧 ATTEMPTING PATH FIX")
    print("=" * 30)
    
    from config import get_quick_test_config
    import shutil
    from pathlib import Path
    
    config = get_quick_test_config()
    expected_path = config.core_dataset_path
    
    # Look for dataset files that might need moving
    dataset_files = list(Path(".").glob("**/combined_dataset.csv"))
    
    if dataset_files:
        for file in dataset_files:
            if file != expected_path:
                print(f"📁 Found dataset at: {file}")
                print(f"📁 Expected at: {expected_path}")
                
                # Create directory if needed
                expected_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Copy file to expected location
                try:
                    shutil.copy2(file, expected_path)
                    print(f"✅ Copied dataset to expected location")
                    
                    # Verify
                    if expected_path.exists():
                        import pandas as pd
                        data = pd.read_csv(expected_path)
                        print(f"✅ Verified dataset: {data.shape} records")
                        return True
                    
                except Exception as e:
                    print(f"❌ Copy failed: {e}")
    
    # If no files found, try running data.py and capturing where it saves
    print("🔄 Running data.py to create dataset...")
    
    try:
        import subprocess
        import sys
        
        result = subprocess.run([
            sys.executable, 'src/data.py'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ data.py completed successfully")
            
            # Check again for created files
            dataset_files = list(Path(".").glob("**/combined_dataset.csv"))
            if dataset_files:
                print(f"📊 Dataset created at: {dataset_files[0]}")
                
                # Move to expected location if different
                if dataset_files[0] != expected_path:
                    expected_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(dataset_files[0], expected_path)
                    print(f"✅ Moved dataset to expected location")
                
                return True
            else:
                print("❌ data.py ran but no dataset found")
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                return False
        else:
            print(f"❌ data.py failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error running data.py: {e}")
        return False

def main():
    """Main execution"""
    print("🚀 DATA PATHS DIAGNOSTIC")
    print("=" * 35)
    
    # Check current state
    check_data_paths()
    
    # Try to fix any issues
    success = fix_path_mismatch()
    
    if success:
        print(f"\n🎉 SUCCESS! Dataset is now in the correct location.")
        print("\n🚀 TRY PIPELINE AGAIN:")
        print("python src/pipeline_orchestrator.py --config-type quick_test")
    else:
        print(f"\n⚠️ Could not automatically fix the issue.")
        print("💡 Try running data.py manually and check the output:")
        print("python src/data.py")

if __name__ == "__main__":
    main()
