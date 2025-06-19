#!/usr/bin/env python3
import sys
import os
from pathlib import Path

# Add src directory to Python path so we can import config_reader
script_dir = Path(__file__).parent
if 'src' in str(script_dir):
    # Running from src directory
    sys.path.insert(0, str(script_dir))
else:
    # Running from project root
    sys.path.insert(0, str(script_dir / 'src'))


"""Multi-Horizon Sentiment-Enhanced TFT - src module"""
