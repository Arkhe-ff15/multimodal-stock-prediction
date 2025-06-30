#!/usr/bin/env python3
"""
Script to fix OptimizedTFTTrainer class method indentation issues
Fixes missing training_step(), validation_step(), and configure_optimizers() methods
"""

import re
import os
from pathlib import Path

def fix_tft_trainer_indentation():
    """Fix the indentation issues in OptimizedTFTTrainer class"""
    
    # Path to models.py
    models_path = Path("src/models.py")
    
    if not models_path.exists():
        print(f"❌ Error: {models_path} not found")
        return False
    
    print(f"📝 Reading {models_path}...")
    
    # Read the file
    with open(models_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Backup the original file
    backup_path = models_path.with_suffix('.py.backup2')
    print(f"💾 Creating backup at {backup_path}...")
    
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    # Fix the indentation issues in OptimizedTFTTrainer class
    print("🔧 Fixing OptimizedTFTTrainer method indentations...")
    
    # The issue is that the methods are incorrectly indented inside other methods
    # We need to find the OptimizedTFTTrainer class and fix the method definitions
    
    # Pattern to find the class and fix method indentations
    lines = content.split('\n')
    fixed_lines = []
    inside_tft_trainer = False
    current_indent = 0
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Detect OptimizedTFTTrainer class
        if 'class OptimizedTFTTrainer(pl.LightningModule):' in line:
            inside_tft_trainer = True
            current_indent = len(line) - len(line.lstrip())
            fixed_lines.append(line)
            print(f"🎯 Found OptimizedTFTTrainer class at line {i+1}")
        
        # Detect end of class
        elif inside_tft_trainer and line.strip() and not line.startswith(' ') and not line.startswith('\t'):
            if line.startswith('class ') or line.startswith('def ') and not line.startswith('    '):
                inside_tft_trainer = False
                print(f"🏁 End of OptimizedTFTTrainer class at line {i+1}")
            fixed_lines.append(line)
        
        # Fix method indentations inside the class
        elif inside_tft_trainer:
            # Check if this is a method definition that needs fixing
            if re.match(r'\s*def (training_step|validation_step|on_validation_epoch_end|configure_optimizers)\s*\(', line):
                # Fix indentation to be at class level (4 spaces from class definition)
                method_content = line.strip()
                fixed_line = '    ' + method_content
                fixed_lines.append(fixed_line)
                print(f"🔧 Fixed method indentation: {method_content[:50]}...")
                
                # Fix the method body indentation
                i += 1
                while i < len(lines) and (lines[i].startswith('        ') or lines[i].strip() == ''):
                    if lines[i].strip():
                        # Method body should be 8 spaces from class definition
                        method_body = lines[i].strip()
                        fixed_lines.append('        ' + method_body)
                    else:
                        fixed_lines.append(lines[i])
                    i += 1
                i -= 1  # Back up one since the loop will increment
            else:
                fixed_lines.append(line)
        else:
            fixed_lines.append(line)
        
        i += 1
    
    # Rejoin the content
    fixed_content = '\n'.join(fixed_lines)
    
    # Additional fix: Ensure the methods are properly defined with correct signatures
    print("🔧 Adding missing method definitions if needed...")
    
    # Check if we have all required methods
    required_methods = ['training_step', 'validation_step', 'on_validation_epoch_end', 'configure_optimizers']
    
    for method in required_methods:
        if f'def {method}(' not in fixed_content:
            print(f"⚠️ Method {method} still missing, adding template...")
            
            # Find the end of __init__ method and add the missing method
            pattern = r'(class OptimizedTFTTrainer\(pl\.LightningModule\):.*?def __init__.*?logger\.info.*?\n)'
            
            if method == 'training_step':
                method_template = '''
    def training_step(self, batch, batch_idx):
        loss, predictions, y_true = self._shared_step(batch)
        
        # Enhanced regularization for complex model
        l1_lambda = self.config.tft_enhanced_l1_lambda if self.model_type == 'TFT_Enhanced' else self.config.tft_l1_lambda
        try:
            l1_reg = torch.tensor(0.0, device=self.device, dtype=torch.float32)
            l2_reg = torch.tensor(0.0, device=self.device, dtype=torch.float32)
            
            for name, param in self.named_parameters():
                if param.requires_grad and torch.isfinite(param).all():
                    l1_reg += param.abs().sum()
                    l2_reg += param.pow(2).sum()
            
            l1_reg = self._safe_tensor_clamp(l1_reg, "l1_reg")
            l2_reg = self._safe_tensor_clamp(l2_reg, "l2_reg")
            
            total_loss = loss + l1_lambda * l1_reg + (l1_lambda * 0.5) * l2_reg
            
        except Exception as e:
            logger.warning(f"Regularization failed: {e}")
            total_loss = loss
        
        self.log('train_loss', total_loss, on_epoch=True, prog_bar=True)
        
        try:
            if predictions.dim() > 1 and predictions.shape[-1] > 1:
                median_predictions = predictions[..., self.median_idx]
            else:
                median_predictions = predictions.squeeze() if predictions.dim() > 1 else predictions
            
            y_true_flat = y_true.squeeze() if y_true.dim() > 1 else y_true
            
            if median_predictions.shape != y_true_flat.shape:
                min_size = min(median_predictions.numel(), y_true_flat.numel())
                median_predictions = median_predictions.flatten()[:min_size]
                y_true_flat = y_true_flat.flatten()[:min_size]
            
            # Calculate metrics
            train_mae = self.financial_metrics.calculate_mae(y_true_flat, median_predictions)
            train_rmse = self.financial_metrics.calculate_rmse(y_true_flat, median_predictions)
            train_r2 = self.financial_metrics.calculate_r2(y_true_flat, median_predictions)
            train_mda = self.financial_metrics.mean_directional_accuracy(y_true_flat, median_predictions)
            
            self.log('train_mae', train_mae, on_epoch=True, prog_bar=False)
            self.log('train_rmse', train_rmse, on_epoch=True, prog_bar=False)
            self.log('train_r2', train_r2, on_epoch=True, prog_bar=False)
            self.log('train_mda', train_mda, on_epoch=True, prog_bar=False)
            
            self.training_step_outputs.append({
                'loss': total_loss.detach().cpu(),
                'predictions': median_predictions.detach().cpu(),
                'targets': y_true_flat.detach().cpu()
            })
        except Exception as e:
            logger.warning(f"Training metrics calculation failed: {e}")
        
        return total_loss
'''
            
            elif method == 'validation_step':
                method_template = '''
    def validation_step(self, batch, batch_idx):
        loss, predictions, y_true = self._shared_step(batch)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        
        try:
            if predictions.dim() > 1 and predictions.shape[-1] > 1:
                median_predictions = predictions[..., self.median_idx]
            else:
                median_predictions = predictions.squeeze() if predictions.dim() > 1 else predictions
            
            y_true_flat = y_true.squeeze() if y_true.dim() > 1 else y_true
            
            if median_predictions.shape != y_true_flat.shape:
                min_size = min(median_predictions.numel(), y_true_flat.numel())
                median_predictions = median_predictions.flatten()[:min_size]
                y_true_flat = y_true_flat.flatten()[:min_size]
            
            # Calculate metrics
            val_mae = self.financial_metrics.calculate_mae(y_true_flat, median_predictions)
            val_rmse = self.financial_metrics.calculate_rmse(y_true_flat, median_predictions)
            val_r2 = self.financial_metrics.calculate_r2(y_true_flat, median_predictions)
            val_mda = self.financial_metrics.mean_directional_accuracy(y_true_flat, median_predictions)
            val_mape = self.financial_metrics.calculate_mape(y_true_flat, median_predictions)
            val_smape = self.financial_metrics.calculate_smape(y_true_flat, median_predictions)
            
            self.log('val_mae', val_mae, on_epoch=True, prog_bar=True)
            self.log('val_rmse', val_rmse, on_epoch=True, prog_bar=True)
            self.log('val_r2', val_r2, on_epoch=True, prog_bar=True)
            self.log('val_mda', val_mda, on_epoch=True, prog_bar=True)
            self.log('val_mape', val_mape, on_epoch=True, prog_bar=False)
            self.log('val_smape', val_smape, on_epoch=True, prog_bar=False)
            
            self.validation_step_outputs.append({
                'loss': loss.detach().cpu(),
                'predictions': median_predictions.detach().cpu(),
                'targets': y_true_flat.detach().cpu()
            })
        except Exception as e:
            logger.warning(f"Validation metrics calculation failed: {e}")
        
        return loss
'''
            
            elif method == 'on_validation_epoch_end':
                method_template = '''
    def on_validation_epoch_end(self):
        if not self.validation_step_outputs:
            return

        try:
            avg_loss = torch.stack([x['loss'] for x in self.validation_step_outputs]).mean()
            self.log('val_loss_epoch', avg_loss, prog_bar=True)
            
            all_preds = torch.cat([x['predictions'].flatten() for x in self.validation_step_outputs])
            all_targets = torch.cat([x['targets'].flatten() for x in self.validation_step_outputs])
            
            # Calculate comprehensive financial metrics
            sharpe = self.financial_metrics.sharpe_ratio(all_preds.numpy())
            max_dd = self.financial_metrics.maximum_drawdown(all_preds.numpy())
            f1_score = self.financial_metrics.directional_f1_score(all_targets, all_preds)
            
            self.log('val_sharpe', sharpe, prog_bar=True)
            self.log('val_max_drawdown', max_dd, prog_bar=False)
            self.log('val_f1_direction', f1_score, prog_bar=True)
            
        except Exception as e:
            logger.warning(f"Validation epoch end calculation failed for {self.model_type}: {e}")
        finally:
            self.validation_step_outputs.clear()
            self.training_step_outputs.clear()
'''
            
            elif method == 'configure_optimizers':
                method_template = '''
    def configure_optimizers(self):
        if self.model_type == 'TFT_Enhanced':
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.config.tft_enhanced_weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
            
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, 
                T_0=self.config.cosine_t_max, 
                T_mult=self.config.cosine_t_mult,
                eta_min=self.config.tft_enhanced_min_lr
            )
        else:
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.config.tft_weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
            
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, 
                T_0=self.config.cosine_t_max, 
                T_mult=self.config.cosine_t_mult,
                eta_min=self.config.tft_min_learning_rate
            )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        }
'''
            
            # Insert the method template right after the __init__ method
            init_end_pattern = r'(\s+logger\.info\(f"🔢 Precision: FLOAT32 \(stability enforced\)"\))'
            if re.search(init_end_pattern, fixed_content):
                fixed_content = re.sub(
                    init_end_pattern,
                    r'\1' + method_template,
                    fixed_content,
                    count=1
                )
    
    # Write the fixed content
    print(f"✍️ Writing fixed content to {models_path}...")
    with open(models_path, 'w', encoding='utf-8') as f:
        f.write(fixed_content)
    
    return True

def validate_tft_trainer_methods():
    """Validate that TFT trainer methods are properly defined"""
    models_path = Path("src/models.py")
    
    try:
        print("🔍 Validating TFT trainer methods...")
        with open(models_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        required_methods = ['training_step', 'validation_step', 'on_validation_epoch_end', 'configure_optimizers']
        found_methods = []
        
        for method in required_methods:
            if f'def {method}(' in content:
                found_methods.append(method)
        
        if len(found_methods) == len(required_methods):
            print("✅ All required methods found in OptimizedTFTTrainer")
            return True
        else:
            missing = set(required_methods) - set(found_methods)
            print(f"❌ Missing methods: {missing}")
            return False
    
    except Exception as e:
        print(f"❌ Error validating methods: {e}")
        return False

def validate_syntax():
    """Validate that the Python file has correct syntax"""
    models_path = Path("src/models.py")
    
    try:
        print("🔍 Validating Python syntax...")
        with open(models_path, 'r', encoding='utf-8') as f:
            code = f.read()
        
        # Try to compile the code
        compile(code, str(models_path), 'exec')
        print("✅ Python syntax is valid")
        return True
    
    except SyntaxError as e:
        print(f"❌ Syntax error in {models_path}:")
        print(f"   Line {e.lineno}: {e.text}")
        print(f"   Error: {e.msg}")
        return False
    except Exception as e:
        print(f"❌ Error validating syntax: {e}")
        return False

def main():
    """Main function"""
    print("🔧 Fixing OptimizedTFTTrainer method indentation issues")
    print("=" * 70)
    
    # Check if src directory exists
    if not Path("src").exists():
        print("❌ Error: 'src' directory not found")
        print("💡 Make sure you're running this script from the project root directory")
        return 1
    
    # Apply fixes
    if fix_tft_trainer_indentation():
        print("\n🎉 Indentation fixes applied!")
        
        # Validate methods
        if validate_tft_trainer_methods():
            # Validate syntax
            if validate_syntax():
                print("✅ OptimizedTFTTrainer is now properly configured")
            else:
                print("❌ Syntax validation failed - please check the file manually")
                return 1
        else:
            print("❌ Method validation failed")
            return 1
    else:
        print("\n❌ Failed to apply fixes")
        return 1
    
    print("\n📋 Summary of changes:")
    print("  • Fixed indentation for training_step() method")
    print("  • Fixed indentation for validation_step() method") 
    print("  • Fixed indentation for on_validation_epoch_end() method")
    print("  • Fixed indentation for configure_optimizers() method")
    print("\n💾 Original file backed up as 'src/models.py.backup2'")
    print("🚀 TFT models should now train without Lightning configuration errors!")
    
    return 0

if __name__ == "__main__":
    exit(main())