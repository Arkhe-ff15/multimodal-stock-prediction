#!/usr/bin/env python3
"""
file used for quick scripting fixes to the codebase
"""

def fix_models_sequential():
    """Apply manual fix to models_sequential.py"""
    
    print("🔧 Applying manual fix to models_sequential.py...")
    
    # Read the current sequential file
    with open('src/models_sequential.py', 'r') as f:
        content = f.read()
    
    # Find and fix the LSTM training method
    # The issue is the target variable needs to be defined before it's used
    
    # Look for the line that's causing the error and the surrounding context
    old_section = """# FIXED: Use consistent target instead of target_features[0]
        target = target_override if target_override else 'target_5'  # Allow override
        if target not in target_features:
            # Fallback to first available regression target
            regression_targets = [t for t in target_features if t != 'target_5_direction']
            if not regression_targets:
                raise ValueError("No regression targets found")
            target = regression_targets[0]
            logger.warning(f"target_5 not found, using {target} instead")
        
        logger.info(f"📊 Using {len(features)} features, target: {target}")"""
    
    new_section = """# FIXED: Use consistent target instead of target_features[0]
        target = target_override if target_override else 'target_5'  # Allow override
        if target not in target_features:
            # Fallback to first available regression target
            regression_targets = [t for t in target_features if t != 'target_5_direction']
            if not regression_targets:
                raise ValueError("No regression targets found")
            target = regression_targets[0]
            logger.warning(f"target_5 not found, using {target} instead")
        
        logger.info(f"📊 Using {len(features)} features, target: {target}")"""
    
    # If the exact text isn't found, let's try a more targeted approach
    if old_section not in content:
        # Find the method definition and fix it more surgically
        lines = content.split('\n')
        fixed_lines = []
        in_lstm_method = False
        target_defined = False
        
        for i, line in enumerate(lines):
            # Look for the train_lstm_baseline method
            if 'def train_lstm_baseline(self, target_override: str = None)' in line:
                in_lstm_method = True
                fixed_lines.append(line)
                continue
            
            # If we're in the LSTM method and we see the target features extraction
            if in_lstm_method and 'target_features = dataset[' in line and 'target_features' in line:
                fixed_lines.append(line)
                # Add target definition right after target_features
                fixed_lines.append('')
                fixed_lines.append('        # FIXED: Define target with override support')
                fixed_lines.append('        target = target_override if target_override else "target_5"')
                fixed_lines.append('        if target not in target_features:')
                fixed_lines.append('            regression_targets = [t for t in target_features if t != "target_5_direction"]')
                fixed_lines.append('            if not regression_targets:')
                fixed_lines.append('                raise ValueError("No regression targets found")')
                fixed_lines.append('            target = regression_targets[0]')
                fixed_lines.append('            logger.warning(f"target_5 not found, using {target} instead")')
                fixed_lines.append('')
                target_defined = True
                continue
            
            # Skip any existing target = lines that might be incomplete
            if in_lstm_method and target_defined and line.strip().startswith('target =') and 'target_override' in line:
                continue
                
            # Exit the method when we see the next method definition
            if in_lstm_method and line.strip().startswith('def ') and 'train_lstm_baseline' not in line:
                in_lstm_method = False
                target_defined = False
            
            fixed_lines.append(line)
        
        content = '\n'.join(fixed_lines)
    else:
        content = content.replace(old_section, new_section)
    
    # Similar fix for TFT methods - add target parameter handling
    tft_methods = ['train_tft_baseline', 'train_tft_enhanced']
    
    for method in tft_methods:
        # Look for the method and add target handling
        method_pattern = f'def {method}(self, target_override: str = None)'
        if method_pattern in content:
            # Find the target_col definition and fix it
            old_target_line = 'target_col = target_override if target_override else \'target_5\'  # Allow override'
            new_target_line = 'target_col = target_override if target_override else "target_5"  # Allow override'
            content = content.replace(old_target_line, new_target_line)
    
    # Write the fixed content back
    with open('src/models_sequential.py', 'w') as f:
        f.write(content)
    
    print("✅ Manual fix applied to models_sequential.py")
    print("🔍 Fixed target variable definition in LSTM training method")
    
    return True

if __name__ == "__main__":
    fix_models_sequential()