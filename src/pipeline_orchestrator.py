#!/usr/bin/env python3
"""
PIPELINE ORCHESTRATOR - CONFIG-INTEGRATED CENTRAL CONTROLLER
============================================================

✅ FIXES APPLIED:
- Proper config.py integration
- Removed all interactive prompts
- Fixed stage handoffs and error propagation
- Standardized programmatic execution
- Proper import statements for fixed modules
- Comprehensive error handling

Author: Research Team
Version: 2.1 (Config-Integrated & Automated)
"""

import logging
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd

# ✅ FIXED: Proper config integration
from config import PipelineConfig, get_default_config, validate_environment

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConfigIntegratedPipelineOrchestrator:
    """
    ✅ FIXED: Central pipeline controller with proper config integration
    
    FIXES:
    - Uses centralized PipelineConfig
    - No interactive prompts
    - Proper stage handoffs
    - Standardized error handling
    - Automated execution flow
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.pipeline_state = {
            'started_at': datetime.now(),
            'current_stage': None,
            'completed_stages': [],
            'failed_stages': [],
            'data_artifacts': {},
            'model_results': {},
            'stage_reports': {}
        }
        
        logger.info("🚀 Config-Integrated Pipeline Orchestrator initialized")
        logger.info(f"   📊 Symbols: {config.symbols}")
        logger.info(f"   📅 Date range: {config.start_date} to {config.end_date}")
        logger.info(f"   🎯 Target horizons: {config.target_horizons}")
        logger.info(f"   📈 FNSPID sample ratio: {config.fnspid_sample_ratio}")
    
    def validate_dependencies(self) -> Dict[str, bool]:
        """✅ FIXED: Validate dependencies using config environment validation"""
        logger.info("🔍 Validating pipeline dependencies...")
        
        # Use config's environment validation
        validation = validate_environment()
        
        # Additional pipeline-specific validations
        validation.update({
            'core_dataset_exists': self.config.core_dataset_path.exists(),
            'fnspid_data_exists': self.config.fnspid_raw_path.exists(),
        })
        
        # Report validation results
        logger.info("📋 Dependency Validation Results:")
        for check, status in validation.items():
            status_icon = "✅" if status else "❌"
            logger.info(f"   {status_icon} {check}")
        
        return validation
    
    def run_stage_data_collection(self) -> bool:
        """✅ FIXED: Run data collection stage (data.py)"""
        self.pipeline_state['current_stage'] = 'data_collection'
        logger.info("📊 STAGE 1: Data Collection")
        
        try:
            # Check if core dataset already exists
            if self.config.core_dataset_path.exists():
                logger.info("✅ Core dataset already exists, skipping data collection")
                
                # Validate existing dataset
                core_data = pd.read_csv(self.config.core_dataset_path)
                self.pipeline_state['data_artifacts']['core_dataset'] = len(core_data)
                
                logger.info(f"   📊 Existing core dataset: {len(core_data):,} records")
                return True
            
            logger.info("🔄 Core dataset not found, would need to run data.py")
            logger.info("💡 Run: python src/data.py to create core dataset")
            
            # For now, return False to indicate manual step needed
            # In a full automation, this would programmatically call data.py
            return False
            
        except Exception as e:
            logger.error(f"❌ Data collection stage failed: {e}")
            self.pipeline_state['failed_stages'].append('data_collection')
            return False
    
    def run_stage_fnspid_processing(self) -> bool:
        """✅ FIXED: Run FNSPID processing stage using fixed fnspid_processor"""
        self.pipeline_state['current_stage'] = 'fnspid_processing'
        logger.info("📊 STAGE 2: FNSPID Processing")
        
        try:
            # ✅ Import fixed FNSPID processor
            from fnspid_processor import run_fnspid_processing_programmatic
            
            # Run FNSPID processing with config
            success, results = run_fnspid_processing_programmatic(self.config)
            
            if not success:
                if results.get('fallback_available', False):
                    logger.warning("⚠️ FNSPID processing failed, will use synthetic sentiment later")
                    self.config.use_synthetic_sentiment = True
                    return True  # Continue pipeline with synthetic fallback
                else:
                    logger.error(f"❌ FNSPID processing failed: {results.get('error', 'Unknown error')}")
                    return False
            
            self.pipeline_state['data_artifacts']['fnspid_daily_sentiment'] = results['processing_summary']['daily_sentiment_records']
            self.pipeline_state['stage_reports']['fnspid_processing'] = results
            
            logger.info(f"✅ FNSPID processing completed: {results['processing_summary']['daily_sentiment_records']:,} records")
            return True
            
        except ImportError as e:
            logger.error(f"❌ Could not import FNSPID processor: {e}")
            self.pipeline_state['failed_stages'].append('fnspid_processing')
            return False
        except Exception as e:
            logger.error(f"❌ FNSPID processing failed: {e}")
            self.pipeline_state['failed_stages'].append('fnspid_processing')
            return False
    
    def run_stage_temporal_decay(self) -> bool:
        """✅ FIXED: Run temporal decay processing stage using fixed temporal_decay"""
        self.pipeline_state['current_stage'] = 'temporal_decay'
        logger.info("🔬 STAGE 3: Temporal Decay Processing")
        
        try:
            # ✅ Import fixed temporal decay processor
            from temporal_decay import run_temporal_decay_processing_programmatic
            
            # Check if sentiment data is available
            if not self.config.fnspid_daily_sentiment_path.exists() and not self.config.use_synthetic_sentiment:
                logger.warning("⚠️ No sentiment data available for temporal decay")
                self.config.use_synthetic_sentiment = True
            
            # Run temporal decay processing with config
            success, results = run_temporal_decay_processing_programmatic(self.config)
            
            if not success:
                logger.error(f"❌ Temporal decay processing failed: {results.get('error', 'Unknown error')}")
                return False
            
            self.pipeline_state['data_artifacts']['temporal_decay_data'] = results['processing_summary']['output_records']
            self.pipeline_state['stage_reports']['temporal_decay'] = results
            
            logger.info(f"✅ Temporal decay completed: {results['processing_summary']['output_records']:,} records")
            logger.info(f"   📊 Validation score: {results['validation']['overall_score']:.0f}/100")
            return True
            
        except ImportError as e:
            logger.error(f"❌ Could not import temporal decay processor: {e}")
            self.pipeline_state['failed_stages'].append('temporal_decay')
            return False
        except Exception as e:
            logger.error(f"❌ Temporal decay processing failed: {e}")
            self.pipeline_state['failed_stages'].append('temporal_decay')
            return False
    
    def run_stage_sentiment_integration(self) -> bool:
        """✅ FIXED: Run sentiment integration stage using fixed sentiment"""
        self.pipeline_state['current_stage'] = 'sentiment_integration'
        logger.info("🔗 STAGE 4: Sentiment Integration")
        
        try:
            # ✅ Import fixed sentiment processor
            from sentiment import run_sentiment_integration_programmatic
            
            # Run sentiment integration with config
            success, results = run_sentiment_integration_programmatic(self.config)
            
            if not success:
                logger.error(f"❌ Sentiment integration failed: {results.get('error', 'Unknown error')}")
                return False
            
            self.pipeline_state['data_artifacts']['enhanced_dataset'] = results['records']
            self.pipeline_state['stage_reports']['sentiment_integration'] = results
            
            logger.info(f"✅ Sentiment integration completed: {results['records']:,} records")
            logger.info(f"   📈 Coverage: {results['coverage']:.1f}%")
            logger.info(f"   🆕 Features added: {results['features_added']}")
            return True
            
        except ImportError as e:
            logger.error(f"❌ Could not import sentiment processor: {e}")
            self.pipeline_state['failed_stages'].append('sentiment_integration')
            return False
        except Exception as e:
            logger.error(f"❌ Sentiment integration failed: {e}")
            self.pipeline_state['failed_stages'].append('sentiment_integration')
            return False
    
    def run_stage_model_training(self) -> bool:
        """✅ FIXED: Run model training stage"""
        self.pipeline_state['current_stage'] = 'model_training'
        logger.info("🤖 STAGE 5: Model Training")
        
        try:
            # Check if enhanced dataset exists
            if not self.config.enhanced_dataset_path.exists():
                logger.error("❌ Enhanced dataset not found for model training")
                return False
            
            # ✅ Import models with proper error handling
            try:
                from models import ModelTrainer
                
                # Configure model training using config
                config_overrides = {
                    'max_epochs': self.config.max_epochs,
                    'batch_size': self.config.batch_size,
                    'learning_rate': self.config.learning_rate,
                    'early_stopping_patience': self.config.early_stopping_patience,
                    'hidden_size': self.config.hidden_size,
                    'max_encoder_length': self.config.max_encoder_length
                }
                
                # Train models
                trainer = ModelTrainer(config_overrides)
                results = trainer.train_all_models()
                
                self.pipeline_state['model_results'] = results
                self.pipeline_state['stage_reports']['model_training'] = {
                    'models_trained': list(results.keys()),
                    'training_summary': {
                        model: {'training_time': result.get('training_time', 0),
                               'best_val_loss': result.get('best_val_loss', None)}
                        for model, result in results.items()
                    }
                }
                
                logger.info(f"✅ Model training completed: {len(results)} models trained")
                for model_name, result in results.items():
                    training_time = result.get('training_time', 0)
                    val_loss = result.get('best_val_loss', 'N/A')
                    logger.info(f"   🤖 {model_name}: {training_time:.1f}s, Val Loss: {val_loss}")
                
                return True
                
            except ImportError as e:
                logger.warning(f"⚠️ Model training dependencies missing: {e}")
                logger.warning("💡 Install with: pip install pytorch-forecasting")
                self.pipeline_state['failed_stages'].append('model_training')
                return False
                
        except Exception as e:
            logger.error(f"❌ Model training failed: {e}")
            self.pipeline_state['failed_stages'].append('model_training')
            return False
    
    def run_stage_evaluation(self) -> bool:
        """✅ FIXED: Run evaluation stage"""
        self.pipeline_state['current_stage'] = 'evaluation'
        logger.info("📊 STAGE 6: Model Evaluation")
        
        try:
            # Check if models were trained
            if not self.pipeline_state.get('model_results'):
                logger.warning("⚠️ No trained models found for evaluation")
                return False
            
            # ✅ Import evaluation with proper error handling
            try:
                from evaluation import integrate_with_models
                from models import ModelTrainer
                
                # Re-initialize trainer for evaluation
                trainer = ModelTrainer()
                evaluation_results = integrate_with_models(trainer)
                
                self.pipeline_state['evaluation_results'] = evaluation_results
                self.pipeline_state['stage_reports']['evaluation'] = evaluation_results
                
                logger.info("✅ Model evaluation completed")
                
                if 'model_results' in evaluation_results:
                    for model_name, results in evaluation_results['model_results'].items():
                        if 'metrics' in results:
                            metrics = results['metrics'].get('horizon_5d', {})
                            rmse = metrics.get('rmse', 'N/A')
                            r2 = metrics.get('r2', 'N/A')
                            logger.info(f"   📊 {model_name}: RMSE={rmse}, R²={r2}")
                
                return True
                
            except ImportError as e:
                logger.warning(f"⚠️ Evaluation dependencies missing: {e}")
                self.pipeline_state['failed_stages'].append('evaluation')
                return False
                
        except Exception as e:
            logger.error(f"❌ Model evaluation failed: {e}")
            self.pipeline_state['failed_stages'].append('evaluation')
            return False
    
    def run_full_pipeline(self) -> Dict[str, Any]:
        """✅ FIXED: Execute complete pipeline with proper error handling"""
        logger.info("🚀 STARTING FULL CONFIG-INTEGRATED PIPELINE EXECUTION")
        logger.info("=" * 70)
        
        # Validate dependencies first
        validation = self.validate_dependencies()
        
        pipeline_success = True
        continue_on_errors = self.config.skip_on_errors
        
        # Define pipeline stages with config control
        stages = [
            ('data_collection', self.run_stage_data_collection, self.config.run_data_collection),
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
                    self.pipeline_state['failed_stages'].append(stage_name)
                    logger.error(f"❌ Stage {stage_name} failed")
                    
                    # Decide whether to continue or stop
                    if stage_name in ['data_collection', 'fnspid_processing']:
                        # These can potentially use fallbacks
                        if continue_on_errors:
                            logger.warning(f"⚠️ Continuing pipeline despite {stage_name} failure")
                            continue
                        else:
                            logger.error(f"🛑 Stopping pipeline due to {stage_name} failure")
                            break
                    elif stage_name in ['temporal_decay', 'sentiment_integration']:
                        # These are critical for the innovation
                        logger.error(f"🛑 Stopping pipeline due to critical {stage_name} failure")
                        break
                    else:
                        # Model training and evaluation can be skipped
                        if continue_on_errors:
                            logger.warning(f"⚠️ Continuing without {stage_name}")
                            continue
                        else:
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
        
        # Show key outputs
        if self.pipeline_state['data_artifacts']:
            logger.info(f"\n📊 Data Artifacts Generated:")
            for artifact, count in self.pipeline_state['data_artifacts'].items():
                logger.info(f"   • {artifact}: {count:,} records")
        
        if self.pipeline_state['model_results']:
            logger.info(f"\n🤖 Models Trained:")
            for model_name in self.pipeline_state['model_results'].keys():
                logger.info(f"   • {model_name}")
        
        return {
            'success': pipeline_success,
            'pipeline_state': self.pipeline_state,
            'report_path': str(self.config.pipeline_report_path),
            'completed_stages': self.pipeline_state['completed_stages'],
            'failed_stages': self.pipeline_state['failed_stages'],
            'data_artifacts': self.pipeline_state['data_artifacts']
        }
    
    def _generate_pipeline_report(self):
        """✅ FIXED: Generate comprehensive pipeline execution report"""
        report = {
            'pipeline_execution': self.pipeline_state,
            'configuration': {
                'symbols': self.config.symbols,
                'date_range': f"{self.config.start_date} to {self.config.end_date}",
                'target_horizons': self.config.target_horizons,
                'processing_parameters': {
                    'fnspid_sample_ratio': self.config.fnspid_sample_ratio,
                    'fnspid_chunk_size': self.config.fnspid_chunk_size,
                    'max_epochs': self.config.max_epochs,
                    'batch_size': self.config.batch_size,
                    'use_synthetic_sentiment': self.config.use_synthetic_sentiment
                },
                'temporal_decay_params': self.config.temporal_decay_params
            },
            'data_artifacts': self.pipeline_state.get('data_artifacts', {}),
            'model_results': self.pipeline_state.get('model_results', {}),
            'stage_reports': self.pipeline_state.get('stage_reports', {}),
            'file_paths': {
                'core_dataset': str(self.config.core_dataset_path),
                'enhanced_dataset': str(self.config.enhanced_dataset_path),
                'fnspid_daily_sentiment': str(self.config.fnspid_daily_sentiment_path),
                'temporal_decay_data': str(self.config.temporal_decay_data_path)
            }
        }
        
        # Ensure report directory exists
        self.config.pipeline_report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.config.pipeline_report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"📋 Pipeline report saved: {self.config.pipeline_report_path}")

def run_pipeline_with_config(config: PipelineConfig) -> Dict[str, Any]:
    """✅ FIXED: Convenience function to run pipeline with configuration"""
    orchestrator = ConfigIntegratedPipelineOrchestrator(config)
    return orchestrator.run_full_pipeline()

def main():
    """✅ FIXED: Main execution with config options"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Config-Integrated Pipeline Orchestrator')
    parser.add_argument('--config-type', type=str, default='default',
                       choices=['default', 'quick_test', 'research'],
                       help='Configuration type to use')
    parser.add_argument('--skip-errors', action='store_true',
                       help='Continue pipeline on non-critical errors')
    parser.add_argument('--stages', nargs='+', 
                       choices=['data_collection', 'fnspid_processing', 'temporal_decay', 
                               'sentiment_integration', 'model_training', 'evaluation'],
                       help='Specific stages to run')
    
    args = parser.parse_args()
    
    print("🚀 CONFIG-INTEGRATED PIPELINE ORCHESTRATOR")
    print("=" * 60)
    
    try:
        # ✅ Load config based on type
        from config import get_default_config, get_quick_test_config, get_research_config
        
        if args.config_type == 'quick_test':
            config = get_quick_test_config()
        elif args.config_type == 'research':
            config = get_research_config()
        else:
            config = get_default_config()
        
        # Apply command line options
        if args.skip_errors:
            config.skip_on_errors = True
        
        # Set stages to run
        if args.stages:
            # Disable all stages first
            config.run_data_collection = 'data_collection' in args.stages
            config.run_fnspid_processing = 'fnspid_processing' in args.stages
            config.run_temporal_decay = 'temporal_decay' in args.stages
            config.run_sentiment_integration = 'sentiment_integration' in args.stages
            config.run_model_training = 'model_training' in args.stages
            config.run_evaluation = 'evaluation' in args.stages
        
        print(f"📊 Configuration: {args.config_type}")
        print(f"🎯 Symbols: {config.symbols}")
        print(f"📅 Date range: {config.start_date} to {config.end_date}")
        print(f"⚠️ Skip errors: {config.skip_on_errors}")
        
        if args.stages:
            print(f"🎯 Stages to run: {args.stages}")
        
        # Run pipeline
        result = run_pipeline_with_config(config)
        
        if result['success']:
            print(f"\n🎉 Pipeline completed successfully!")
            print(f"📋 Report: {result['report_path']}")
            print(f"✅ Completed stages: {result['completed_stages']}")
            
            if result['failed_stages']:
                print(f"❌ Failed stages: {result['failed_stages']}")
        else:
            print(f"❌ Pipeline failed")
            print(f"❌ Failed stages: {result['failed_stages']}")
    
    except Exception as e:
        print(f"❌ Pipeline orchestrator failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()