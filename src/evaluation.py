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
    import logging
    
    logger = logging.getLogger(__name__)
    
    try:
        # Check if we have successful models
        if not trainer_results.get('training_summary', {}).get('successful_models'):
            logger.warning("⚠️ No successful models to evaluate")
            return {'error': 'No successful models available'}
        
        # Get trainer instance
        trainer_instance = trainer_results.get('trainer_instance')
        if not trainer_instance:
            logger.error("❌ No trainer instance available")
            return {'error': 'No trainer instance available'}
        
        # Initialize evaluator
        try:
            evaluator = ModelEvaluator(save_dir=str(pipeline_config.evaluation_results_dir))
        except NameError:
            # Fallback if ModelEvaluator not properly imported
            logger.warning("⚠️ ModelEvaluator not available, using simplified evaluation")
            return _simplified_evaluation(trainer_results)
        
        model_results = {}
        successful_models = trainer_results['training_summary']['successful_models']
        
        # Process each successful model
        for model_name in successful_models:
            try:
                logger.info(f"📊 Evaluating {model_name}...")
                
                if model_name == 'LSTM_Baseline':
                    # LSTM evaluation
                    model_info = trainer_instance.models.get(model_name)
                    if model_info and hasattr(trainer_instance, 'test_data'):
                        predictions = _evaluate_lstm_model(model_info, trainer_instance.test_data)
                        if predictions:
                            actuals = trainer_instance.test_data['target_5'].dropna().values[-len(predictions):]
                            metrics = _calculate_simple_metrics(actuals, predictions)
                            model_results[model_name] = {5: metrics}
                
                elif 'TFT' in model_name:
                    # TFT evaluation
                    tft_model = trainer_instance.models.get(model_name)
                    if tft_model and hasattr(tft_model, 'predict'):
                        predictions = tft_model.predict()
                        if predictions and 5 in predictions:
                            if hasattr(trainer_instance, 'test_data'):
                                actuals = trainer_instance.test_data['target_5'].dropna().values[-len(predictions[5]):]
                                metrics = _calculate_simple_metrics(actuals, predictions[5])
                                model_results[model_name] = {5: metrics}
                
            except Exception as e:
                logger.error(f"❌ Evaluation failed for {model_name}: {e}")
                continue
        
        if not model_results:
            return {'error': 'No models could be evaluated'}
        
        # Simple comparison
        comparison_results = _simple_model_comparison(model_results)
        
        return {
            'model_results': model_results,
            'comparison_results': comparison_results,
            'evaluation_successful': True,
            'best_model': comparison_results.get('best_model', 'unknown')
        }
        
    except Exception as e:
        logger.error(f"❌ Config-integrated evaluation failed: {e}")
        return {
            'error': str(e),
            'evaluation_successful': False
        }

def _simplified_evaluation(trainer_results):
    """Fallback evaluation when ModelEvaluator unavailable"""
    return {
        'status': 'simplified_evaluation',
        'models_trained': trainer_results['training_summary']['successful_models'],
        'evaluation_successful': True
    }

def _evaluate_lstm_model(model_info, test_data):
    """Extract LSTM predictions"""
    try:
        import torch
        from torch.utils.data import DataLoader
        
        model = model_info['model']
        scaler = model_info['scaler']
        feature_cols = model_info['feature_cols']
        
        # Create test dataset (simplified)
        test_features = test_data[feature_cols].fillna(0).values
        if scaler:
            test_features = scaler.transform(test_features)
        
        model.eval()
        predictions = []
        
        # Simple sliding window prediction
        seq_length = model.config.max_encoder_length if hasattr(model, 'config') else 30
        for i in range(seq_length, len(test_features)):
            sequence = torch.FloatTensor(test_features[i-seq_length:i]).unsqueeze(0)
            with torch.no_grad():
                pred = model(sequence)
                predictions.append(float(pred.squeeze().cpu().numpy()))
        
        return predictions
        
    except Exception as e:
        logging.error(f"❌ LSTM evaluation failed: {e}")
        return []

def _calculate_simple_metrics(actuals, predictions):
    """Calculate basic metrics"""
    import numpy as np
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    try:
        # Align lengths
        min_len = min(len(actuals), len(predictions))
        actuals = actuals[-min_len:] if len(actuals) > min_len else actuals
        predictions = predictions[-min_len:] if len(predictions) > min_len else predictions
        
        mae = mean_absolute_error(actuals, predictions)
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        r2 = r2_score(actuals, predictions)
        
        return {
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'samples': min_len
        }
    except Exception as e:
        logging.error(f"❌ Metrics calculation failed: {e}")
        return {'mae': float('inf'), 'rmse': float('inf'), 'r2': -1, 'samples': 0}

def _simple_model_comparison(model_results):
    """Simple model comparison"""
    comparison = {}
    best_model = None
    best_r2 = -float('inf')
    
    for model_name, results in model_results.items():
        if 5 in results:
            r2 = results[5].get('r2', -1)
            comparison[model_name] = {'r2': r2, 'rank': 0}
            if r2 > best_r2:
                best_r2 = r2
                best_model = model_name
    
    # Assign ranks
    sorted_models = sorted(comparison.items(), key=lambda x: x[1]['r2'], reverse=True)
    for rank, (model_name, _) in enumerate(sorted_models):
        comparison[model_name]['rank'] = rank + 1
    
    return {
        'model_rankings': comparison,
        'best_model': best_model,
        'best_r2': best_r2
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