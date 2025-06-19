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


"""
Pipeline Orchestrator - Fixed Simple Version
============================================
✅ FIXES APPLIED:
- Removed complex config.py dependency
- Simple YAML config reading
- Simplified orchestration logic
- Removed complex state management
- Each stage works independently
- Clear error handling

Usage:
    python src/pipeline_orchestrator.py
    python src/pipeline_orchestrator.py --stages fnspid temporal_decay
"""

import logging
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import argparse
import warnings
warnings.filterwarnings('ignore')

# Simple imports
from config_reader import load_config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimplePipelineOrchestrator:
    """Simple pipeline orchestrator without over-engineering"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize with simple config"""
        self.config = load_config(config_path)
        self.start_time = datetime.now()
        
        # Pipeline stages with their script paths
        self.stages = {
            'data': {
                'name': 'Data Collection',
                'script': 'src/data.py',
                'description': 'Collect and process stock market data',
                'required': True
            },
            'clean': {
                'name': 'Data Cleaning',
                'script': 'src/clean.py',
                'description': 'Clean and validate dataset',
                'required': False
            },
            'fnspid': {
                'name': 'FNSPID Processing',
                'script': 'src/fnspid_processor.py',
                'description': 'Process FNSPID news data and sentiment analysis',
                'required': True
            },
            'temporal_decay': {
                'name': 'Temporal Decay',
                'script': 'src/temporal_decay.py',
                'description': 'Calculate exponential temporal decay features',
                'required': True
            },
            'sentiment': {
                'name': 'Sentiment Integration',
                'script': 'src/sentiment.py',
                'description': 'Integrate sentiment features with market data',
                'required': True
            },
            'models': {
                'name': 'Model Training',
                'script': 'src/models.py',
                'description': 'Train LSTM and TFT models',
                'required': False
            },
            'evaluation': {
                'name': 'Model Evaluation',
                'script': 'src/evaluation.py',
                'description': 'Evaluate and compare model performance',
                'required': False
            }
        }
        
        # Track execution
        self.completed_stages = []
        self.failed_stages = []
        
        logger.info("🚀 Simple Pipeline Orchestrator initialized")
        logger.info(f"   📊 Available stages: {list(self.stages.keys())}")
    
    def check_dependencies(self) -> dict:
        """Check if required files exist"""
        logger.info("🔍 Checking dependencies...")
        
        dependencies = {
            'config_yaml': Path('config.yaml').exists(),
            'data_script': Path('src/data.py').exists(),
            'fnspid_raw': Path(self.config['paths']['raw']['fnspid_data']).exists(),
            'core_dataset': Path(self.config['paths']['processed']['core_dataset']).exists()
        }
        
        logger.info("📋 Dependency check:")
        for dep, status in dependencies.items():
            status_icon = "✅" if status else "❌"
            logger.info(f"   {status_icon} {dep}")
        
        return dependencies
    
    def run_stage(self, stage_name: str) -> bool:
        """Run a single pipeline stage"""
        if stage_name not in self.stages:
            logger.error(f"❌ Unknown stage: {stage_name}")
            return False
        
        stage = self.stages[stage_name]
        script_path = Path(stage['script'])
        
        if not script_path.exists():
            logger.error(f"❌ Script not found: {script_path}")
            return False
        
        logger.info(f"🚀 Running stage: {stage['name']}")
        logger.info(f"   📄 Script: {script_path}")
        logger.info(f"   📝 Description: {stage['description']}")
        
        try:
            # Run the script
            start_time = datetime.now()
            result = subprocess.run([
                sys.executable, str(script_path)
            ], capture_output=True, text=True, cwd='.')
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            if result.returncode == 0:
                logger.info(f"✅ {stage['name']} completed successfully ({execution_time:.1f}s)")
                if result.stdout:
                    logger.info("📄 Output:")
                    for line in result.stdout.strip().split('\n')[-10:]:  # Show last 10 lines
                        logger.info(f"   {line}")
                self.completed_stages.append(stage_name)
                return True
            else:
                logger.error(f"❌ {stage['name']} failed ({execution_time:.1f}s)")
                logger.error(f"   Error: {result.stderr}")
                if result.stdout:
                    logger.error(f"   Output: {result.stdout}")
                self.failed_stages.append(stage_name)
                return False
        
        except Exception as e:
            logger.error(f"❌ {stage['name']} execution failed: {e}")
            self.failed_stages.append(stage_name)
            return False
    
    def run_stages(self, stage_names: list = None, stop_on_error: bool = True) -> bool:
        """Run multiple pipeline stages"""
        
        # Default to all stages if none specified
        if stage_names is None:
            stage_names = list(self.stages.keys())
        
        logger.info(f"🚀 Running pipeline stages: {stage_names}")
        
        overall_success = True
        
        for stage_name in stage_names:
            success = self.run_stage(stage_name)
            
            if not success:
                overall_success = False
                if stop_on_error:
                    logger.error(f"🛑 Stopping pipeline due to {stage_name} failure")
                    break
                else:
                    logger.warning(f"⚠️ Continuing despite {stage_name} failure")
        
        return overall_success
    
    def run_data_pipeline(self) -> bool:
        """Run core data processing pipeline (data → fnspid → temporal_decay → sentiment)"""
        logger.info("📊 Running core data processing pipeline")
        
        core_stages = ['data', 'fnspid', 'temporal_decay', 'sentiment']
        return self.run_stages(core_stages, stop_on_error=True)
    
    def run_model_pipeline(self) -> bool:
        """Run model training and evaluation pipeline"""
        logger.info("🤖 Running model training pipeline")
        
        model_stages = ['models', 'evaluation']
        return self.run_stages(model_stages, stop_on_error=False)
    
    def run_full_pipeline(self) -> bool:
        """Run complete pipeline"""
        logger.info("🚀 Running complete pipeline")
        
        # First run data processing
        data_success = self.run_data_pipeline()
        
        if not data_success:
            logger.error("❌ Data pipeline failed - stopping")
            return False
        
        # Then run model training (can continue even if some fail)
        model_success = self.run_model_pipeline()
        
        return data_success and model_success
    
    def generate_summary_report(self):
        """Generate execution summary"""
        total_time = (datetime.now() - self.start_time).total_seconds()
        
        logger.info("\n" + "="*60)
        logger.info("📋 PIPELINE EXECUTION SUMMARY")
        logger.info("="*60)
        logger.info(f"🕐 Total execution time: {total_time:.1f} seconds")
        logger.info(f"✅ Completed stages: {len(self.completed_stages)}")
        logger.info(f"❌ Failed stages: {len(self.failed_stages)}")
        
        if self.completed_stages:
            logger.info(f"\n✅ Successfully completed:")
            for stage in self.completed_stages:
                stage_info = self.stages[stage]
                logger.info(f"   • {stage_info['name']}")
        
        if self.failed_stages:
            logger.info(f"\n❌ Failed stages:")
            for stage in self.failed_stages:
                stage_info = self.stages[stage]
                logger.info(f"   • {stage_info['name']}")
        
        # Check key outputs
        logger.info(f"\n📁 Key Output Files:")
        key_files = [
            ('Core Dataset', self.config['paths']['processed']['core_dataset']),
            ('Daily Sentiment', self.config['paths']['processed']['fnspid_daily_sentiment']),
            ('Temporal Decay', self.config['paths']['processed']['temporal_decay_dataset']),
            ('Final Dataset', self.config['paths']['processed']['final_dataset'])
        ]
        
        for name, path in key_files:
            exists = Path(path).exists()
            status = "✅" if exists else "❌"
            logger.info(f"   {status} {name}: {path}")
        
        # Model files
        models_dir = Path("models")
        if models_dir.exists():
            model_files = list(models_dir.glob("*.pth"))
            logger.info(f"\n🤖 Model Files: {len(model_files)}")
            for model_file in model_files:
                logger.info(f"   ✅ {model_file.name}")
        
        logger.info("="*60)

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(
        description='Simple Pipeline Orchestrator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/pipeline_orchestrator.py                    # Run full pipeline
  python src/pipeline_orchestrator.py --data-only        # Data processing only
  python src/pipeline_orchestrator.py --model-only       # Model training only
  python src/pipeline_orchestrator.py --stages data fnspid  # Specific stages
        """
    )
    
    parser.add_argument('--stages', nargs='+', 
                       choices=['data', 'clean', 'fnspid', 'temporal_decay', 'sentiment', 'models', 'evaluation'],
                       help='Specific stages to run')
    parser.add_argument('--data-only', action='store_true',
                       help='Run data processing pipeline only')
    parser.add_argument('--model-only', action='store_true',
                       help='Run model training pipeline only')
    parser.add_argument('--continue-on-error', action='store_true',
                       help='Continue pipeline even if stages fail')
    parser.add_argument('--check-deps', action='store_true',
                       help='Check dependencies and exit')
    
    args = parser.parse_args()
    
    print("🚀 SIMPLE PIPELINE ORCHESTRATOR")
    print("=" * 50)
    
    try:
        # Initialize orchestrator
        orchestrator = SimplePipelineOrchestrator()
        
        # Check dependencies
        deps = orchestrator.check_dependencies()
        
        if args.check_deps:
            print("\n✅ Dependency check complete")
            return
        
        # Warn about missing dependencies
        if not all(deps.values()):
            print("\n⚠️ Some dependencies missing - pipeline may fail")
        
        # Determine execution mode
        success = False
        
        if args.stages:
            # Run specific stages
            success = orchestrator.run_stages(args.stages, stop_on_error=not args.continue_on_error)
        elif args.data_only:
            # Data processing only
            success = orchestrator.run_data_pipeline()
        elif args.model_only:
            # Model training only
            success = orchestrator.run_model_pipeline()
        else:
            # Full pipeline
            success = orchestrator.run_full_pipeline()
        
        # Generate summary
        orchestrator.generate_summary_report()
        
        if success:
            print(f"\n🎉 Pipeline execution completed successfully!")
        else:
            print(f"\n❌ Pipeline execution completed with errors")
        
        print(f"\n🚀 Next Steps:")
        if 'models' in orchestrator.completed_stages:
            print("   📊 Check model performance in results/")
        if 'evaluation' in orchestrator.completed_stages:
            print("   🏆 Review model comparison results")
        print("   📁 All outputs saved in data/processed/ and models/")
        
    except Exception as e:
        print(f"\n❌ Pipeline orchestrator failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())