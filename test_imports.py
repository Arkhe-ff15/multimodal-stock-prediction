#!/usr/bin/env python3
"""
Test script to verify import fixes are working
"""

print("🧪 TESTING PIPELINE IMPORTS")
print("=" * 40)

modules_to_test = [
    ('config', 'from config import PipelineConfig'),
    ('temporal_decay', 'from src.temporal_decay import *'),
    ('models', 'from src.models import *'),
    ('pipeline_orchestrator', 'from src.pipeline_orchestrator import *'),
    ('fnspid_processor', 'from src.fnspid_processor import *'),
    ('sentiment', 'from src.sentiment import *'),
]

all_passed = True

for module_name, import_statement in modules_to_test:
    try:
        exec(import_statement)
        print(f"✅ {module_name}")
    except Exception as e:
        print(f"❌ {module_name}: {e}")
        all_passed = False

if all_passed:
    print("\n🎉 ALL IMPORTS SUCCESSFUL!")
    print("Ready to run pipeline:")
    print("python src/pipeline_orchestrator.py --config-type quick_test")
else:
    print("\n⚠️ Some imports failed. Check the errors above.")
