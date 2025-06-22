# Create: src/enhanced_model_framework.py
#!/usr/bin/env python3
"""
Enhanced Model Framework - Main Entry Point for Academic Training
================================================================
"""

import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import all components
from models import (
    EnhancedModelFramework,
    EnhancedDataLoader,
    MemoryMonitor,
    set_random_seeds
)

from evaluation import (
    AcademicModelEvaluator,
    StatisticalTestSuite,
    AcademicMetricsCalculator,
    ModelPredictor
)

from data_prep import AcademicDataPreparator

# Export main components
__all__ = [
    'EnhancedModelFramework',
    'EnhancedDataLoader', 
    'MemoryMonitor',
    'set_random_seeds',
    'AcademicModelEvaluator',
    'StatisticalTestSuite',
    'AcademicMetricsCalculator',
    'ModelPredictor',
    'AcademicDataPreparator'
]

def main():
    """Main training execution"""
    from models import main as models_main
    return models_main()

if __name__ == "__main__":
    exit(main())