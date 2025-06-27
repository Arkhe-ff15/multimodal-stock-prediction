#!/usr/bin/env python3
"""
Quick fix for the logger issue in models.py
"""

import sys
from pathlib import Path

def fix_logger_issue():
    models_path = Path("src/models.py")
    
    with open(models_path, 'r') as f:
        content = f.read()
    
    # Replace logger.info with print in the patch function
    content = content.replace(
        'logger.info("✅ Applied PyTorch Forecasting device handling patches")',
        'print("✅ Applied PyTorch Forecasting device handling patches")'
    )
    
    # Write back
    with open(models_path, 'w') as f:
        f.write(content)
    
    print("✅ Fixed logger issue!")

if __name__ == "__main__":
    fix_logger_issue()