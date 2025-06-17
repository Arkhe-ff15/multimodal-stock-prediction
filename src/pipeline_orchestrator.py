#!/usr/bin/env python3
"""
PIPELINE ORCHESTRATOR - Central Controller
==========================================

Fixes: Interactive prompts, missing automation, error propagation
Result: Full pipeline automation with proper validation
"""

import logging
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PipelineConfig:
    """Centralized configuration for entire pipeline"""
    # Data paths
    data_dir: str = "data/processed"
    raw_dir: str = "data/raw"
    results_dir: str = "results"
    models_dir: str = "models/checkpoints"
    
    # Input files
    fnspid_raw_file: str = "nasdaq_external_data.csv"  # Fixed typo
    
    # Pipeline control
    symbols: List[str] = None
    start_date: str = "2018-01-01"
    end_date: str = "2024-01-31"
    
    # Processing parameters
    fnspid_sample_ratio: float = 0.15
    chunk_size: int = 50000
    max_epochs: int = 50
    batch_size: int = 64
    
    # Pipeline stages to run
    run_fnspid_processing: bool = True
    run_temporal_decay: bool = True
    run_sentiment_integration: bool = True
    run_model_training: bool = True
    run_evaluation: bool = True
    
    # Fallback strategies
    use_synthetic_sentiment: bool = False  # If FNSPID data unavailable
    
    def __post_init__(self):
        """Validate and setup configuration"""
        if self.symbols is None:
            self.symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'JPM']
        
        # Ensure directories exist
        for directory in [self.data_dir, self.results_dir, self.models_dir]:
            Path(directory).mkdir(parents=True, exist_ok=True)

class PipelineOrchestrator:
    """Central pipeline controller with proper error handling and validation"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.pipeline_state = {
            'started_at': datetime.now(),
            'current_stage': None,
            'completed_stages': [],
            'failed_stages': [],
            'data_artifacts': {},
            'model_results': {}
        }
        
        # Standard file paths
        self.file_paths = {
            'core_dataset': f"{config.data_dir}/combined_dataset.csv",
            'fnspid_daily_sentiment': f"{config.data_dir}/fnspid_daily_sentiment.csv",
            'temporal_decay_data': f"{config.data_dir}/sentiment_with_temporal_decay.csv",
            'enhanced_dataset': f"{config.data_dir}/combined_dataset_with_sentiment.csv",
            'pipeline_report': f"{config.results_dir}/pipeline_execution_report.json"
        }
        
        logger.info("🚀 Pipeline Orchestrator initialized")
        logger.info(f"   📊 Symbols: {config.symbols}")
        logger.info(f"   📅 Date range: {config.start_date} to {config.end_date}")
    
    def validate_dependencies(self) -> Dict[str, bool]:
        """Validate all required dependencies and files"""
        logger.info("🔍 Validating pipeline dependencies...")
        
        validation = {
            'core_dataset_exists': False,
            'fnspid_data_exists': False,
            'pytorch_forecasting_available': False,
            'ta_library_available': False,
            'transformers_available': False,
            'directories_ready': False
        }
        
        try:
            # Check core dataset
            validation['core_dataset_exists'] = os.path.exists(self.file_paths['core_dataset'])
            
            # Check FNSPID data
            fnspid_path = f"{self.config.raw_dir}/{self.config.fnspid_raw_file}"
            validation['fnspid_data_exists'] = os.path.exists(fnspid_path)
            
            # Check Python dependencies
            try:
                import pytorch_forecasting
                validation['pytorch_forecasting_available'] = True
            except ImportError:
                logger.warning("⚠️ pytorch-forecasting not available")
            
            try:
                import ta
                validation['ta_library_available'] = True
            except ImportError:
                logger.warning("⚠️ ta library not available")
                
            try:
                import transformers
                validation['transformers_available'] = True
            except ImportError:
                logger.warning("⚠️ transformers library not available")
            
            # Check directories
            validation['directories_ready'] = all(
                Path(directory).exists() 
                for directory in [self.config.data_dir, self.config.results_dir, self.config.models_dir]
            )
            
        except Exception as e:
            logger.error(f"❌ Dependency validation failed: {e}")
        
        # Report validation results
        logger.info("📋 Dependency Validation Results:")
        for check, status in validation.items():
            status_icon = "✅" if status else "❌"
            logger.info(f"   {status_icon} {check}")
        
        return validation
    
    def run_stage_fnspid_processing(self) -> bool:
        """Run FNSPID processing stage programmatically"""
        self.pipeline_state['current_stage'] = 'fnspid_processing'
        logger.info("📊 STAGE 1: FNSPID Processing")
        
        try:
            # Import and configure FNSPID processor
            from fnspid_processor import OptimizedFNSPIDProcessor, PipelineConfig as FNSPIDConfig
            
            # Create FNSPID configuration
            fnspid_config = FNSPIDConfig(
                target_symbols=self.config.symbols,
                start_date=self.config.start_date,
                end_date=self.config.end_date,
                sample_ratio=self.config.fnspid_sample_ratio,
                chunk_size=self.config.chunk_size
            )
            
            # Process FNSPID data programmatically
            processor = OptimizedFNSPIDProcessor(fnspid_config)
            filtered_articles, article_sentiment, daily_sentiment = processor.run_complete_pipeline()
            
            # Validate outputs
            if daily_sentiment.empty:
                logger.warning("⚠️ No FNSPID sentiment data generated")
                return False
            
            self.pipeline_state['data_artifacts']['fnspid_daily_sentiment'] = len(daily_sentiment)
            logger.info(f"✅ FNSPID processing completed: {len(daily_sentiment):,} records")
            return True
            
        except Exception as e:
            logger.error(f"❌ FNSPID processing failed: {e}")
            self.pipeline_state['failed_stages'].append('fnspid_processing')
            return False
    
    def run_stage_temporal_decay(self) -> bool:
        """Run temporal decay processing stage"""
        self.pipeline_state['current_stage'] = 'temporal_decay'
        logger.info("🔬 STAGE 2: Temporal Decay Processing")
        
        try:
            from temporal_decay import TemporalDecayProcessor, create_optimal_decay_parameters
            
            # Load sentiment data
            if not os.path.exists(self.file_paths['fnspid_daily_sentiment']):
                logger.error("❌ FNSPID sentiment data not found")
                return False
            
            sentiment_data = pd.read_csv(self.file_paths['fnspid_daily_sentiment'])
            
            # Create decay processor
            decay_params = create_optimal_decay_parameters()
            processor = TemporalDecayProcessor(decay_params)
            
            # Process temporal decay
            processed_data = processor.batch_process_all_symbols(sentiment_data)
            
            # Validate and save
            if processed_data.empty:
                logger.error("❌ No temporal decay data generated")
                return False
            
            processed_data.to_csv(self.file_paths['temporal_decay_data'], index=False)
            self.pipeline_state['data_artifacts']['temporal_decay_data'] = len(processed_data)
            
            logger.info(f"✅ Temporal decay completed: {len(processed_data):,} records")
            return True
            
        except Exception as e:
            logger.error(f"❌ Temporal decay processing failed: {e}")
            self.pipeline_state['failed_stages'].append('temporal_decay')
            return False
    
    def run_stage_sentiment_integration(self) -> bool:
        """Run sentiment integration stage"""
        self.pipeline_state['current_stage'] = 'sentiment_integration'
        logger.info("🔗 STAGE 3: Sentiment Integration")
        
        try:
            from sentiment import SentimentProcessor
            
            # Run sentiment integration programmatically
            processor = SentimentProcessor()
            success, summary = processor.run_complete_integration()
            
            if not success:
                logger.error(f"❌ Sentiment integration failed: {summary.get('error', 'Unknown error')}")
                return False
            
            self.pipeline_state['data_artifacts']['enhanced_dataset'] = summary['records']
            logger.info(f"✅ Sentiment integration completed: {summary['records']:,} records")
            return True
            
        except Exception as e:
            logger.error(f"❌ Sentiment integration failed: {e}")
            self.pipeline_state['failed_stages'].append('sentiment_integration')
            return False
    
    def run_stage_model_training(self) -> bool:
        """Run model training stage"""
        self.pipeline_state['current_stage'] = 'model_training'
        logger.info("🤖 STAGE 4: Model Training")
        
        try:
            from models import ModelTrainer
            
            # Configure model training
            config_overrides = {
                'max_epochs': self.config.max_epochs,
                'batch_size': self.config.batch_size,
                'early_stopping_patience': 15
            }
            
            # Train models
            trainer = ModelTrainer(config_overrides)
            results = trainer.train_all_models()
            
            self.pipeline_state['model_results'] = results
            logger.info(f"✅ Model training completed: {len(results)} models trained")
            return True
            
        except Exception as e:
            logger.error(f"❌ Model training failed: {e}")
            self.pipeline_state['failed_stages'].append('model_training')
            return False
    
    def run_stage_evaluation(self) -> bool:
        """Run evaluation stage"""
        self.pipeline_state['current_stage'] = 'evaluation'
        logger.info("📊 STAGE 5: Model Evaluation")
        
        try:
            from evaluation import integrate_with_models
            from models import ModelTrainer
            
            # Re-load trained models for evaluation
            trainer = ModelTrainer()
            evaluation_results = integrate_with_models(trainer)
            
            self.pipeline_state['evaluation_results'] = evaluation_results
            logger.info("✅ Model evaluation completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Model evaluation failed: {e}")
            self.pipeline_state['failed_stages'].append('evaluation')
            return False
    
    def run_full_pipeline(self) -> Dict[str, Any]:
        """Execute complete pipeline with proper error handling"""
        logger.info("🚀 STARTING FULL PIPELINE EXECUTION")
        logger.info("=" * 60)
        
        # Validate dependencies first
        validation = self.validate_dependencies()
        
        # Check if core dataset exists
        if not validation['core_dataset_exists']:
            logger.error("❌ Core dataset not found. Run data.py first.")
            return {'success': False, 'error': 'Missing core dataset'}
        
        pipeline_success = True
        
        # Execute pipeline stages
        stages = [
            ('fnspid_processing', self.run_stage_fnspid_processing, self.config.run_fnspid_processing),
            ('temporal_decay', self.run_stage_temporal_decay, self.config.run_temporal_decay),
            ('sentiment_integration', self.run_stage_sentiment_integration, self.config.run_sentiment_integration),
            ('model_training', self.run_stage_model_training, self.config.run_model_training),
            ('evaluation', self.run_stage_evaluation, self.config.run_evaluation)
        ]
        
        for stage_name, stage_func, should_run in stages:
            if should_run:
                logger.info(f"\n📍 Executing stage: {stage_name}")
                success = stage_func()
                
                if success:
                    self.pipeline_state['completed_stages'].append(stage_name)
                    logger.info(f"✅ Stage {stage_name} completed successfully")
                else:
                    pipeline_success = False
                    logger.error(f"❌ Stage {stage_name} failed")
                    
                    # Decide whether to continue or stop
                    if stage_name in ['fnspid_processing', 'temporal_decay']:
                        # These can use fallbacks
                        logger.warning(f"⚠️ Continuing with fallback for {stage_name}")
                    else:
                        # Critical failures
                        logger.error(f"🛑 Stopping pipeline due to {stage_name} failure")
                        break
            else:
                logger.info(f"⏭️ Skipping stage: {stage_name}")
        
        # Generate final report
        self.pipeline_state['completed_at'] = datetime.now()
        self.pipeline_state['total_duration'] = (
            self.pipeline_state['completed_at'] - self.pipeline_state['started_at']
        ).total_seconds()
        
        self._generate_pipeline_report()
        
        logger.info(f"\n🎉 PIPELINE EXECUTION COMPLETED")
        logger.info(f"   ✅ Successful stages: {len(self.pipeline_state['completed_stages'])}")
        logger.info(f"   ❌ Failed stages: {len(self.pipeline_state['failed_stages'])}")
        logger.info(f"   ⏱️ Total duration: {self.pipeline_state['total_duration']:.1f} seconds")
        
        return {
            'success': pipeline_success,
            'pipeline_state': self.pipeline_state,
            'report_path': self.file_paths['pipeline_report']
        }
    
    def _generate_pipeline_report(self):
        """Generate comprehensive pipeline execution report"""
        report = {
            'pipeline_execution': self.pipeline_state,
            'configuration': {
                'symbols': self.config.symbols,
                'date_range': f"{self.config.start_date} to {self.config.end_date}",
                'processing_parameters': {
                    'fnspid_sample_ratio': self.config.fnspid_sample_ratio,
                    'chunk_size': self.config.chunk_size,
                    'max_epochs': self.config.max_epochs
                }
            },
            'data_artifacts': self.pipeline_state.get('data_artifacts', {}),
            'model_results': self.pipeline_state.get('model_results', {}),
            'file_paths': self.file_paths
        }
        
        with open(self.file_paths['pipeline_report'], 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"📋 Pipeline report saved: {self.file_paths['pipeline_report']}")

def run_pipeline_with_config(config: PipelineConfig) -> Dict[str, Any]:
    """Convenience function to run pipeline with configuration"""
    orchestrator = PipelineOrchestrator(config)
    return orchestrator.run_full_pipeline()

if __name__ == "__main__":
    # Example usage
    config = PipelineConfig(
        symbols=['AAPL', 'MSFT', 'GOOGL'],
        fnspid_sample_ratio=0.1,  # Quick test
        max_epochs=10  # Quick test
    )
    
    result = run_pipeline_with_config(config)
    
    if result['success']:
        print("🎉 Pipeline completed successfully!")
    else:
        print(f"❌ Pipeline failed: {result.get('error', 'Unknown error')}")