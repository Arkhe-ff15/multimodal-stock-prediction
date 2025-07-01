#!/bin/bash
"""
🚀 AUTOMATED RTX 4090 CONFIG TESTING SUITE
==========================================

Automatically creates and tests all optimized configs:
1. Debug Config (5 min) - Pipeline validation
2. Fast Config (45 min) - Quick model validation  
3. Balanced RTX 4090 (3-4 hours) - Production ready
4. Maximum Performance (6-8 hours) - Full RTX 4090 power
5. Research Grade (12-15 hours) - Academic excellence

Usage:
    chmod +x run_all_configs.sh
    ./run_all_configs.sh [options]

Options:
    --start-from CONFIG    Start from specific config (debug|fast|balanced|maximum|research)
    --only CONFIG          Run only specific config
    --skip-gpu-check       Skip GPU availability check
    --dry-run             Create configs but don't run training
    --help                Show this help

Example:
    ./run_all_configs.sh                    # Run all configs
    ./run_all_configs.sh --start-from fast  # Start from fast config
    ./run_all_configs.sh --only balanced    # Run only balanced config
"""

# =====================================
# CONFIGURATION AND SETUP
# =====================================

set -e  # Exit on any error
START_TIME=$(date +%s)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="logs/config_runs/$TIMESTAMP"
RESULTS_DIR="results/config_comparison/$TIMESTAMP"

# Create directories
mkdir -p "$LOG_DIR" "$RESULTS_DIR" "configs"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color

# Configuration settings
CONFIGS=("debug" "fast" "balanced" "maximum" "research")
START_FROM=""
ONLY_CONFIG=""
SKIP_GPU_CHECK=false
DRY_RUN=false

# =====================================
# UTILITY FUNCTIONS
# =====================================

log() {
    echo -e "${WHITE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_DIR/main.log"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}" | tee -a "$LOG_DIR/main.log"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}" | tee -a "$LOG_DIR/main.log"
}

log_error() {
    echo -e "${RED}❌ $1${NC}" | tee -a "$LOG_DIR/main.log"
}

log_info() {
    echo -e "${CYAN}ℹ️  $1${NC}" | tee -a "$LOG_DIR/main.log"
}

show_progress() {
    local current=$1
    local total=$2
    local config=$3
    local status=$4
    
    echo -e "\n${PURPLE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${WHITE}📊 PROGRESS: Config $current/$total - $config ($status)${NC}"
    echo -e "${PURPLE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
}

check_gpu() {
    if ! command -v nvidia-smi &> /dev/null; then
        log_error "nvidia-smi not found. GPU support required for optimal performance."
        return 1
    fi
    
    local gpu_memory=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    if [ "$gpu_memory" -lt 8000 ]; then
        log_warning "GPU memory ($gpu_memory MB) is less than recommended 8GB"
    else
        log_success "GPU detected with $gpu_memory MB memory"
    fi
    
    # Show current GPU status
    log_info "Current GPU status:"
    nvidia-smi | tee -a "$LOG_DIR/main.log"
}

parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --start-from)
                START_FROM="$2"
                shift 2
                ;;
            --only)
                ONLY_CONFIG="$2"
                shift 2
                ;;
            --skip-gpu-check)
                SKIP_GPU_CHECK=true
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --help)
                head -n 30 "$0" | tail -n +2 | sed 's/^"""//' | sed 's/"""$//'
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
}

# =====================================
# CONFIG CREATION FUNCTIONS
# =====================================

create_debug_config() {
    cat > configs/config_debug.yaml << 'EOF'
# DEBUG CONFIG - Ultra-fast pipeline validation (5 minutes total)
model:
  # MINIMAL LSTM (1-2 minutes)
  lstm_hidden_size: 16
  lstm_num_layers: 1
  lstm_max_epochs: 3
  lstm_sequence_length: 20
  lstm_learning_rate: 0.01
  lstm_attention_heads: 2
  
  # MINIMAL TFT BASELINE (2-3 minutes)
  tft_hidden_size: 24
  tft_max_epochs: 5
  tft_max_encoder_length: 20
  tft_max_prediction_length: 5
  tft_learning_rate: 0.02
  
  # MINIMAL TFT ENHANCED (3-5 minutes)
  tft_enhanced_hidden_size: 32
  tft_enhanced_max_epochs: 8
  tft_enhanced_learning_rate: 0.015
  
  # DEBUG SETTINGS
  batch_size: 32
  early_stopping_patience: 2
  num_workers: 2
  
training:
  target_rmse: 0.050
  target_mae: 0.040
  target_r2: 0.20
  target_directional_accuracy: 0.55
  target_sharpe_ratio: 1.5
EOF
}

create_fast_config() {
    cat > configs/config_fast.yaml << 'EOF'
# FAST CONFIG - Quick validation (45 minutes total)
model:
  # FAST LSTM (5-10 minutes)
  lstm_hidden_size: 32
  lstm_num_layers: 2
  lstm_max_epochs: 25
  lstm_sequence_length: 30
  lstm_learning_rate: 0.005
  lstm_attention_heads: 4
  
  # FAST TFT BASELINE (15-20 minutes)
  tft_hidden_size: 48
  tft_max_epochs: 50
  tft_max_encoder_length: 40
  tft_max_prediction_length: 8
  tft_learning_rate: 0.012
  tft_attention_head_size: 6
  
  # FAST TFT ENHANCED (20-25 minutes)
  tft_enhanced_hidden_size: 64
  tft_enhanced_max_epochs: 75
  tft_enhanced_learning_rate: 0.008
  tft_enhanced_attention_head_size: 8
  
  # FAST SETTINGS
  batch_size: 128
  early_stopping_patience: 8
  num_workers: 4
  pin_memory: true

training:
  target_rmse: 0.035
  target_mae: 0.025
  target_r2: 0.30
  target_directional_accuracy: 0.60
  target_sharpe_ratio: 2.0
EOF
}

create_balanced_config() {
    cat > configs/config_balanced.yaml << 'EOF'
# BALANCED RTX 4090 CONFIG - Production ready (3-4 hours total)
model:
  # BALANCED RTX 4090 LSTM (45 minutes)
  lstm_hidden_size: 80
  lstm_num_layers: 2
  lstm_max_epochs: 150
  lstm_sequence_length: 90
  lstm_attention_heads: 10
  lstm_learning_rate: 0.002
  
  # BALANCED RTX 4090 TFT BASELINE (90 minutes)
  tft_hidden_size: 128
  tft_hidden_continuous_size: 64
  tft_max_epochs: 300
  tft_max_encoder_length: 120
  tft_attention_head_size: 16
  tft_learning_rate: 0.006
  
  # BALANCED RTX 4090 TFT ENHANCED (150 minutes)
  tft_enhanced_hidden_size: 192
  tft_enhanced_hidden_continuous_size: 96
  tft_enhanced_max_epochs: 450
  tft_enhanced_attention_head_size: 24
  tft_enhanced_learning_rate: 0.004
  
  # RTX 4090 OPTIMIZED SETTINGS
  batch_size: 192
  num_workers: 6
  pin_memory: true
  mixed_precision: false
  early_stopping_patience: 35

training:
  target_rmse: 0.022
  target_mae: 0.016
  target_r2: 0.45
  target_directional_accuracy: 0.72
  target_sharpe_ratio: 3.2
EOF
}

create_maximum_config() {
    cat > configs/config_maximum.yaml << 'EOF'
# MAXIMUM PERFORMANCE RTX 4090 CONFIG - Full GPU power (6-8 hours total)
model:
  # MAXIMUM LSTM (90 minutes)
  lstm_hidden_size: 128
  lstm_num_layers: 3
  lstm_max_epochs: 200
  lstm_sequence_length: 120
  lstm_attention_heads: 16
  lstm_learning_rate: 0.001
  
  # MAXIMUM TFT BASELINE (180 minutes)
  tft_hidden_size: 192
  tft_hidden_continuous_size: 96
  tft_max_epochs: 400
  tft_max_encoder_length: 150
  tft_max_prediction_length: 30
  tft_attention_head_size: 24
  tft_learning_rate: 0.004
  
  # MAXIMUM TFT ENHANCED (300 minutes)
  tft_enhanced_hidden_size: 256
  tft_enhanced_hidden_continuous_size: 128
  tft_enhanced_max_epochs: 600
  tft_enhanced_lstm_layers: 4
  tft_enhanced_attention_head_size: 32
  tft_enhanced_learning_rate: 0.003
  
  # MAXIMUM GPU UTILIZATION
  batch_size: 256
  gradient_accumulation_steps: 1
  num_workers: 8
  pin_memory: true
  mixed_precision: true
  early_stopping_patience: 50

training:
  target_rmse: 0.020
  target_mae: 0.015
  target_r2: 0.50
  target_directional_accuracy: 0.75
  target_sharpe_ratio: 3.5
EOF
}

create_research_config() {
    cat > configs/config_research.yaml << 'EOF'
# RESEARCH GRADE CONFIG - Academic excellence (12-15 hours total)
model:
  # RESEARCH LSTM (180 minutes)
  lstm_hidden_size: 96
  lstm_num_layers: 4
  lstm_max_epochs: 300
  lstm_sequence_length: 180
  lstm_attention_heads: 12
  lstm_dropout: 0.2
  lstm_learning_rate: 0.0008
  
  # RESEARCH TFT BASELINE (360 minutes)
  tft_hidden_size: 160
  tft_hidden_continuous_size: 80
  tft_max_epochs: 500
  tft_max_encoder_length: 200
  tft_attention_head_size: 20
  tft_num_encoder_layers: 4
  tft_learning_rate: 0.003
  
  # RESEARCH TFT ENHANCED (600 minutes)
  tft_enhanced_hidden_size: 320
  tft_enhanced_hidden_continuous_size: 160
  tft_enhanced_max_epochs: 800
  tft_enhanced_lstm_layers: 5
  tft_enhanced_attention_head_size: 40
  tft_enhanced_num_encoder_layers: 5
  tft_enhanced_learning_rate: 0.002
  
  # RESEARCH SETTINGS
  batch_size: 512
  early_stopping_patience: 100
  gradient_clip_val: 0.1
  use_swa: true
  swa_start_epoch: 300
  num_workers: 8
  pin_memory: true

training:
  target_rmse: 0.018
  target_mae: 0.012
  target_r2: 0.55
  target_directional_accuracy: 0.78
  target_sharpe_ratio: 4.0
EOF
}

create_all_configs() {
    log_info "Creating all configuration files..."
    
    create_debug_config
    log_success "Created debug config (5 min runtime)"
    
    create_fast_config  
    log_success "Created fast config (45 min runtime)"
    
    create_balanced_config
    log_success "Created balanced config (3-4 hour runtime)"
    
    create_maximum_config
    log_success "Created maximum config (6-8 hour runtime)"
    
    create_research_config
    log_success "Created research config (12-15 hour runtime)"
    
    log_info "All configs saved in ./configs/ directory"
}

# =====================================
# TRAINING EXECUTION FUNCTIONS
# =====================================

run_single_config() {
    local config_name=$1
    local config_number=$2
    local total_configs=$3
    
    local config_file="configs/config_${config_name}.yaml"
    local log_file="$LOG_DIR/${config_name}.log"
    local start_time=$(date +%s)
    
    show_progress $config_number $total_configs $config_name "Starting"
    
    log_info "Starting $config_name config training..."
    log_info "Config file: $config_file"
    log_info "Log file: $log_file"
    
    # Create GPU monitoring in background
    nvidia-smi dmon -s pucvmet -d 30 > "$LOG_DIR/${config_name}_gpu.log" &
    local gpu_monitor_pid=$!
    
    # Run training with timeout and error handling
    local success=false
    local error_msg=""
    
    if timeout 36000 python src/models.py --config "$config_file" --model all > "$log_file" 2>&1; then
        success=true
        log_success "$config_name config completed successfully!"
    else
        local exit_code=$?
        if [ $exit_code -eq 124 ]; then
            error_msg="Training timed out after 10 hours"
        else
            error_msg="Training failed with exit code $exit_code"
        fi
        log_error "$config_name config failed: $error_msg"
    fi
    
    # Stop GPU monitoring
    kill $gpu_monitor_pid 2>/dev/null || true
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local hours=$((duration / 3600))
    local minutes=$(((duration % 3600) / 60))
    
    # Extract key metrics from logs
    extract_metrics "$config_name" "$log_file" "$duration" "$success" "$error_msg"
    
    show_progress $config_number $total_configs $config_name "Completed in ${hours}h ${minutes}m"
    
    return $([ "$success" = true ] && echo 0 || echo 1)
}

extract_metrics() {
    local config_name=$1
    local log_file=$2
    local duration=$3
    local success=$4
    local error_msg=$5
    
    local results_file="$RESULTS_DIR/${config_name}_results.json"
    
    # Extract metrics from log file
    local lstm_mda=$(grep -o "LSTM.*MDA.*[0-9]\+\.[0-9]\+" "$log_file" | tail -1 | grep -o "[0-9]\+\.[0-9]\+" || echo "0.000")
    local tft_baseline_mda=$(grep -o "TFT_Optimized_Baseline.*MDA.*[0-9]\+\.[0-9]\+" "$log_file" | tail -1 | grep -o "[0-9]\+\.[0-9]\+" || echo "0.000")
    local tft_enhanced_mda=$(grep -o "TFT_Optimized_Enhanced.*MDA.*[0-9]\+\.[0-9]\+" "$log_file" | tail -1 | grep -o "[0-9]\+\.[0-9]\+" || echo "0.000")
    
    # Create results JSON
    cat > "$results_file" << EOF
{
    "config_name": "$config_name",
    "success": $success,
    "error_message": "$error_msg",
    "duration_seconds": $duration,
    "duration_formatted": "$(($duration / 3600))h $((($duration % 3600) / 60))m",
    "metrics": {
        "lstm_mda": $lstm_mda,
        "tft_baseline_mda": $tft_baseline_mda,
        "tft_enhanced_mda": $tft_enhanced_mda
    },
    "hierarchy_achieved": $(echo "$tft_enhanced_mda > $tft_baseline_mda && $tft_baseline_mda > $lstm_mda" | bc -l),
    "timestamp": "$(date -Iseconds)"
}
EOF
    
    log_info "Results saved to $results_file"
}

# =====================================
# REPORTING FUNCTIONS
# =====================================

generate_final_report() {
    local report_file="$RESULTS_DIR/final_comparison_report.md"
    local total_time=$(($(date +%s) - START_TIME))
    
    cat > "$report_file" << EOF
# 🚀 RTX 4090 Config Comparison Report

**Generated:** $(date)  
**Total Runtime:** $((total_time / 3600))h $(((total_time % 3600) / 60))m  
**Hardware:** RTX 4090 (24GB VRAM)

## 📊 Results Summary

| Config | Status | Duration | LSTM MDA | TFT Baseline MDA | TFT Enhanced MDA | Hierarchy ✓ |
|--------|--------|----------|----------|------------------|------------------|-------------|
EOF

    # Add results for each config
    for config in "${CONFIGS[@]}"; do
        local results_file="$RESULTS_DIR/${config}_results.json"
        if [ -f "$results_file" ]; then
            local status=$(jq -r '.success' "$results_file")
            local duration=$(jq -r '.duration_formatted' "$results_file")
            local lstm_mda=$(jq -r '.metrics.lstm_mda' "$results_file")
            local tft_baseline_mda=$(jq -r '.metrics.tft_baseline_mda' "$results_file")
            local tft_enhanced_mda=$(jq -r '.metrics.tft_enhanced_mda' "$results_file")
            local hierarchy=$(jq -r '.hierarchy_achieved' "$results_file")
            
            local status_icon=$([ "$status" = "true" ] && echo "✅" || echo "❌")
            local hierarchy_icon=$([ "$hierarchy" = "1" ] && echo "✅" || echo "❌")
            
            echo "| $config | $status_icon | $duration | $lstm_mda | $tft_baseline_mda | $tft_enhanced_mda | $hierarchy_icon |" >> "$report_file"
        else
            echo "| $config | ⏭️ | Skipped | - | - | - | - |" >> "$report_file"
        fi
    done
    
    cat >> "$report_file" << EOF

## 🎯 Performance Analysis

### Best Performing Config
$(find_best_config)

### Hierarchy Achievement Rate
$(calculate_hierarchy_rate)

## 📁 Generated Files

- **Configs:** \`./configs/\`
- **Logs:** \`$LOG_DIR/\`  
- **Results:** \`$RESULTS_DIR/\`
- **GPU Monitoring:** \`$LOG_DIR/*_gpu.log\`

## 🚀 Next Steps

1. **Deploy best model:** Use the highest performing config for production
2. **Fine-tune:** Adjust hyperparameters based on best config
3. **Scale up:** Consider ensemble methods with top 2-3 configs

---
*Generated by automated RTX 4090 config testing suite*
EOF

    log_success "Final report generated: $report_file"
    log_info "Opening report preview..."
    head -50 "$report_file"
}

find_best_config() {
    local best_config=""
    local best_score=0
    
    for config in "${CONFIGS[@]}"; do
        local results_file="$RESULTS_DIR/${config}_results.json"
        if [ -f "$results_file" ]; then
            local enhanced_mda=$(jq -r '.metrics.tft_enhanced_mda' "$results_file")
            if (( $(echo "$enhanced_mda > $best_score" | bc -l) )); then
                best_score=$enhanced_mda
                best_config=$config
            fi
        fi
    done
    
    echo "**$best_config** (TFT Enhanced MDA: $best_score)"
}

calculate_hierarchy_rate() {
    local total=0
    local achieved=0
    
    for config in "${CONFIGS[@]}"; do
        local results_file="$RESULTS_DIR/${config}_results.json"
        if [ -f "$results_file" ]; then
            total=$((total + 1))
            local hierarchy=$(jq -r '.hierarchy_achieved' "$results_file")
            if [ "$hierarchy" = "1" ]; then
                achieved=$((achieved + 1))
            fi
        fi
    done
    
    if [ $total -gt 0 ]; then
        local rate=$((achieved * 100 / total))
        echo "$achieved/$total configs achieved TFT hierarchy ($rate%)"
    else
        echo "No configs completed"
    fi
}

# =====================================
# MAIN EXECUTION
# =====================================

main() {
    echo -e "${PURPLE}"
    cat << 'EOF'
🚀 ═══════════════════════════════════════════════════════════════════════════════
   AUTOMATED RTX 4090 FINANCIAL ML CONFIG TESTING SUITE
   Testing 5 optimized configurations from debug to research-grade
═══════════════════════════════════════════════════════════════════════════════
EOF
    echo -e "${NC}"
    
    # Parse command line arguments
    parse_arguments "$@"
    
    # Setup
    log_info "Starting automated config testing suite"
    log_info "Timestamp: $TIMESTAMP"
    log_info "Log directory: $LOG_DIR"
    log_info "Results directory: $RESULTS_DIR"
    
    # GPU check
    if [ "$SKIP_GPU_CHECK" = false ]; then
        check_gpu || exit 1
    fi
    
    # Create all configs
    create_all_configs
    
    if [ "$DRY_RUN" = true ]; then
        log_info "Dry run completed. Configs created but no training performed."
        exit 0
    fi
    
    # Determine which configs to run
    local configs_to_run=()
    if [ -n "$ONLY_CONFIG" ]; then
        configs_to_run=("$ONLY_CONFIG")
        log_info "Running only: $ONLY_CONFIG"
    elif [ -n "$START_FROM" ]; then
        local found=false
        for config in "${CONFIGS[@]}"; do
            if [ "$config" = "$START_FROM" ] || [ "$found" = true ]; then
                configs_to_run+=("$config")
                found=true
            fi
        done
        log_info "Starting from: $START_FROM"
    else
        configs_to_run=("${CONFIGS[@]}")
        log_info "Running all configs"
    fi
    
    # Run selected configs
    local total_configs=${#configs_to_run[@]}
    local current=1
    local successful=0
    local failed=0
    
    for config in "${configs_to_run[@]}"; do
        if run_single_config "$config" $current $total_configs; then
            successful=$((successful + 1))
        else
            failed=$((failed + 1))
        fi
        current=$((current + 1))
        
        # Cleanup between runs
        log_info "Cleaning up GPU memory..."
        python3 -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None" 2>/dev/null || true
        sleep 10
    done
    
    # Generate final report
    log_info "Generating final comparison report..."
    generate_final_report
    
    # Final summary
    local total_time=$(($(date +%s) - START_TIME))
    echo -e "\n${GREEN}🎉 AUTOMATED TESTING COMPLETED!${NC}"
    echo -e "${WHITE}📊 Total time: $((total_time / 3600))h $(((total_time % 3600) / 60))m${NC}"
    echo -e "${WHITE}✅ Successful: $successful${NC}"
    echo -e "${WHITE}❌ Failed: $failed${NC}"
    echo -e "${WHITE}📁 Results: $RESULTS_DIR/final_comparison_report.md${NC}"
    
    if [ $successful -gt 0 ]; then
        echo -e "\n${CYAN}🚀 Best performing models are ready for deployment!${NC}"
        echo -e "${CYAN}📈 Check the final report for detailed analysis${NC}"
    fi
}

# Run main function with all arguments
main "$@"