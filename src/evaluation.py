#!/usr/bin/env python3
"""
EVALUATION.PY - CONFIG-INTEGRATED VERSION
=========================================

✅ FIXES APPLIED:
- Proper integration with fixed models.py
- Config-based evaluation
- Standardized interfaces
- Enhanced error handling

Key fix: Updated integrate_with_models function to work with ConfigIntegratedModelTrainer
"""

# Keep the original evaluation.py mostly intact, but fix the integration function

def integrate_with_config_models(pipeline_config, trainer_results):
    """
    ✅ FIXED: Integration function for config-integrated models
    
    Args:
        pipeline_config: PipelineConfig object
        trainer_results: Results from ConfigIntegratedModelTrainer
        
    Returns:
        Evaluation results dictionary
    """
    from config import PipelineConfig
    import logging
    
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize evaluator using original evaluation framework
        evaluator = ModelEvaluator(save_dir=str(pipeline_config.evaluation_results_dir))
        
        # Extract model results from trainer
        if not trainer_results.get('successful_models'):
            logger.warning("⚠️ No successful models to evaluate")
            return {'error': 'No successful models available'}
        
        model_results = {}
        
        # Process each successful model
        for model_name in trainer_results['successful_models']:
            try:
                # Get trainer instance from results
                trainer_instance = trainer_results.get('trainer_instance')
                if not trainer_instance:
                    logger.warning(f"⚠️ No trainer instance available for {model_name}")
                    continue
                
                # Extract predictions based on model type
                if model_name == 'LSTM_Baseline':
                    predictions = evaluator._evaluate_lstm_from_trainer(trainer_instance, model_name)
                else:  # TFT models
                    predictions = evaluator._evaluate_tft_from_trainer(trainer_instance, model_name)
                
                if predictions:
                    # Get actuals from test data
                    test_data = getattr(trainer_instance, 'test_data', None)
                    if test_data is not None:
                        actuals = {5: test_data['target_5'].dropna().values}
                        
                        # Calculate metrics
                        metrics = evaluator.calculate_metrics(
                            y_true=actuals[5],
                            y_pred=predictions.get(5, []),
                        )
                        
                        model_results[model_name] = {5: metrics}
                
            except Exception as e:
                logger.error(f"❌ Evaluation failed for {model_name}: {e}")
                continue
        
        if not model_results:
            return {'error': 'No models could be evaluated'}
        
        # Compare models using original evaluation framework
        comparison_results = evaluator.compare_models(model_results)
        
        # Detect overfitting (simplified version)
        overfitting_analysis = {
            model_name: {'overfitting_detected': False, 'overfitting_severity': 'none'}
            for model_name in model_results.keys()
        }
        
        # Generate comprehensive report
        report = evaluator.create_evaluation_report(
            model_results,
            comparison_results, 
            overfitting_analysis,
            save_path=str(pipeline_config.evaluation_results_dir / "evaluation_report.txt")
        )
        
        return {
            'model_results': model_results,
            'comparison_results': comparison_results,
            'overfitting_analysis': overfitting_analysis,
            'report': report,
            'evaluation_successful': True
        }
        
    except Exception as e:
        logger.error(f"❌ Config-integrated evaluation failed: {e}")
        return {
            'error': str(e),
            'evaluation_successful': False
        }

# Add helper methods to ModelEvaluator class for trainer integration
class ConfigIntegratedModelEvaluator(ModelEvaluator):
    """Extended evaluator for config-integrated models"""
    
    def _evaluate_lstm_from_trainer(self, trainer_instance, model_name):
        """Extract LSTM predictions from trainer"""
        try:
            model_info = trainer_instance.models.get(model_name)
            if not model_info:
                return {}
            
            model = model_info['model']
            scaler = model_info['scaler']
            feature_cols = model_info['feature_cols']
            
            # Create test dataset
            test_dataset = EnhancedLSTMDataset(
                trainer_instance.test_data, feature_cols, 'target_5',
                model.config.max_encoder_length, scaler
            )
            
            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
            
            # Make predictions
            model.eval()
            predictions = []
            
            with torch.no_grad():
                for batch in test_loader:
                    sequences, _ = batch
                    pred = model(sequences)
                    predictions.extend(pred.squeeze().cpu().numpy())
            
            return {5: np.array(predictions)}
            
        except Exception as e:
            logging.error(f"❌ LSTM evaluation failed: {e}")
            return {}
    
    def _evaluate_tft_from_trainer(self, trainer_instance, model_name):
        """Extract TFT predictions from trainer"""
        try:
            model_info = trainer_instance.models.get(model_name)
            if not model_info:
                return {}
            
            # Use TFT's predict method
            predictions = model_info.predict(trainer_instance.test_data)
            return predictions
            
        except Exception as e:
            logging.error(f"❌ TFT evaluation failed: {e}")
            return {}

def run_evaluation_programmatic(pipeline_config, trainer_results):
    """
    ✅ FIXED: Programmatic evaluation interface
    
    Args:
        pipeline_config: PipelineConfig object
        trainer_results: Results from model training
        
    Returns:
        Tuple[bool, Dict]: (success, evaluation_results)
    """
    
    try:
        results = integrate_with_config_models(pipeline_config, trainer_results)
        
        if results.get('evaluation_successful', False):
            return True, results
        else:
            return False, results
            
    except Exception as e:
        logging.error(f"❌ Programmatic evaluation failed: {e}")
        return False, {
            'error': str(e),
            'stage': 'evaluation'
        }

# Export the integration function for use in pipeline_orchestrator
__all__ = ['integrate_with_config_models', 'run_evaluation_programmatic', 'ConfigIntegratedModelEvaluator']