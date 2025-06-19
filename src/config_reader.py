#!/usr/bin/env python3
"""
Simple Config Reader - Replaces overcomplicated config.py
=========================================================
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logger.info(f"✅ Config loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"❌ Failed to load config: {e}")
        raise

def get_data_paths(config: Dict[str, Any]) -> Dict[str, Path]:
    """Get all data-related paths"""
    return {
        'raw_fnspid': Path(config['paths']['raw']['fnspid_data']),
        'core_dataset': Path(config['paths']['processed']['core_dataset']),
        'fnspid_daily_sentiment': Path(config['paths']['processed']['fnspid_daily_sentiment']),
        'temporal_decay_dataset': Path(config['paths']['processed']['temporal_decay_dataset']),
        'final_dataset': Path(config['paths']['processed']['final_dataset'])
    }