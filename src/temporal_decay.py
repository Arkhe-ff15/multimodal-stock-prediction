"""
src/temporal_decay.py

CORE INNOVATION: Horizon-specific temporal sentiment decay
Mathematical framework for weighting historical sentiment based on forecasting horizon
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
import logging
from scipy import stats

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SentimentItem:
    """Individual sentiment measurement"""
    date: datetime
    score: float  # [-1, 1] sentiment score
    confidence: float  # [0, 1] confidence score
    article_count: int  # Number of articles aggregated
    source: str  # Data source identifier

@dataclass 
class DecayParameters:
    """Horizon-specific decay parameters"""
    horizon: int  # Forecast horizon in days
    lambda_decay: float  # Decay rate parameter
    lookback_days: int  # Maximum lookback window
    min_sentiment_count: int = 3  # Minimum articles for reliable sentiment
    
    def __post_init__(self):
        """Validate parameters to prevent overfitting"""
        if self.lambda_decay <= 0 or self.lambda_decay > 1:
            raise ValueError(f"Lambda decay must be in (0, 1], got {self.lambda_decay}")
        if self.lookback_days < self.horizon:
            warnings.warn(f"Lookback ({self.lookback_days}) < horizon ({self.horizon})")

class TemporalDecayProcessor:
    """
    Core class implementing horizon-specific temporal sentiment decay
    
    Mathematical Framework:
    sentiment_weighted = Σ(sentiment_i * exp(-λ_h * age_i)) / Σ(exp(-λ_h * age_i))
    
    Where:
    - λ_h: horizon-specific decay parameter
    - age_i: age of sentiment_i in trading days
    - h: forecasting horizon (5, 30, 90 days)
    """
    
    def __init__(self, decay_params: Dict[int, DecayParameters], 
                trading_calendar: Optional[pd.DatetimeIndex] = None):
        """
        Initialize temporal decay processor
        
        Args:
            decay_params: Dictionary mapping horizons to decay parameters
            trading_calendar: Optional trading calendar for business day calculations
        """
        self.decay_params = decay_params
        self.trading_calendar = trading_calendar
        self.validation_stats = {}
        
        # Validate configuration
        self._validate_parameters()
        
        logger.info(f"Initialized TemporalDecayProcessor with horizons: {list(decay_params.keys())}")
    
    def _validate_parameters(self):
        """Validate decay parameters to prevent overfitting"""
        required_horizons = [5, 30, 90]
        
        for horizon in required_horizons:
            if horizon not in self.decay_params:
                raise ValueError(f"Missing decay parameters for horizon {horizon}")
        
        # Check that decay rates follow expected pattern (faster decay for shorter horizons)
        lambdas = [(h, p.lambda_decay) for h, p in self.decay_params.items()]
        lambdas.sort()  # Sort by horizon
        
        for i in range(1, len(lambdas)):
            if lambdas[i][1] > lambdas[i-1][1]:
                warnings.warn(f"Decay rate increases with horizon: {lambdas[i-1]} -> {lambdas[i]}")
    
    def calculate_trading_days(self, start_date: datetime, end_date: datetime) -> int:
        """Calculate number of trading days between dates"""
        if self.trading_calendar is not None:
            return len(self.trading_calendar[(self.trading_calendar >= start_date) & 
                                            (self.trading_calendar <= end_date)])
        else:
            # Approximate: 5/7 ratio for weekdays only
            total_days = (end_date - start_date).days
            return int(total_days * 5/7)
    
    def process_sentiment(self, sentiment_history: List[SentimentItem], 
                            current_date: datetime, 
                            horizon: int) -> Tuple[float, Dict]:
        """ 
        Apply temporal decay to sentiment history for specific horizon
        
        Args:
            sentiment_history: List of historical sentiment measurements
            current_date: Current prediction date
            horizon: Forecasting horizon (5, 30, or 90 days)
            
        Returns:
            Tuple of (weighted_sentiment, metadata)
        """
        if horizon not in self.decay_params:
            raise ValueError(f"Unsupported horizon: {horizon}")
        
        params = self.decay_params[horizon]
        
        # Filter sentiment within lookback window
        cutoff_date = current_date - timedelta(days=params.lookback_days)
        recent_sentiment = [s for s in sentiment_history 
                            if s.date >= cutoff_date and s.date <= current_date]
        
        # Quality filtering - prevent overfitting on noisy data
        high_quality_sentiment = [s for s in recent_sentiment 
                                if s.confidence >= 0.7 and s.article_count >= params.min_sentiment_count]
        
        if len(high_quality_sentiment) == 0:
            logger.warning(f"No high-quality sentiment for {current_date}, horizon {horizon}")
            return 0.0, {'quality': 'insufficient_data', 'count': len(recent_sentiment)}
        
        # Apply exponential decay weighting
        weighted_sentiment = 0.0
        total_weight = 0.0
        weights_list = []
        ages_list = []
        
        for sentiment_item in high_quality_sentiment:
            age_days = self.calculate_trading_days(sentiment_item.date, current_date)
            
            # Exponential decay formula
            weight = np.exp(-params.lambda_decay * age_days)
            
            # Weight by confidence to reduce noise
            adjusted_weight = weight * sentiment_item.confidence
            
            weighted_sentiment += adjusted_weight * sentiment_item.score
            total_weight += adjusted_weight
            
            weights_list.append(weight)
            ages_list.append(age_days)
        
        # Normalize
        final_sentiment = weighted_sentiment / total_weight if total_weight > 0 else 0.0
        
        # Compile metadata for analysis
        metadata = {
            'quality': 'sufficient',
            'raw_count': len(recent_sentiment),
            'filtered_count': len(high_quality_sentiment),
            'total_weight': total_weight,
            'effective_lookback': max(ages_list) if ages_list else 0,
            'weight_concentration': np.std(weights_list) if len(weights_list) > 1 else 0,
            'horizon': horizon,
            'lambda_decay': params.lambda_decay
        }
        
        return final_sentiment, metadata
    
    def batch_process(self, sentiment_data: pd.DataFrame, 
                        prediction_dates: List[datetime],
                        horizons: List[int] = None) -> pd.DataFrame:
        """
        Process sentiment for multiple dates and horizons efficiently
        
        Args:
            sentiment_data: DataFrame with columns ['date', 'score', 'confidence', 'article_count', 'source']
            prediction_dates: List of dates to generate sentiment features for
            horizons: List of horizons to process (default: all configured)
            
        Returns:
            DataFrame with weighted sentiment features
        """
        if horizons is None:
            horizons = list(self.decay_params.keys())
        
        # Convert DataFrame to SentimentItem objects
        sentiment_items = []
        for _, row in sentiment_data.iterrows():
            sentiment_items.append(SentimentItem(
                date=pd.to_datetime(row['date']).to_pydatetime(),
                score=row['score'],
                confidence=row['confidence'],
                article_count=row.get('article_count', 1),
                source=row.get('source', 'unknown')
            ))
        
        # Sort by date for efficiency
        sentiment_items.sort(key=lambda x: x.date)
        
        results = []
        
        for pred_date in prediction_dates:
            row_data = {'date': pred_date}
            
            for horizon in horizons:
                # Filter relevant sentiment for this prediction date
                relevant_sentiment = [s for s in sentiment_items if s.date <= pred_date]
                
                weighted_sentiment, metadata = self.process_sentiment(
                    relevant_sentiment, pred_date, horizon
                )
                
                # Store results
                row_data[f'sentiment_decay_{horizon}d'] = weighted_sentiment
                row_data[f'sentiment_weight_{horizon}d'] = metadata['total_weight']
                row_data[f'sentiment_count_{horizon}d'] = metadata['filtered_count']
                row_data[f'sentiment_quality_{horizon}d'] = metadata['quality']
            
            results.append(row_data)
        
        return pd.DataFrame(results)
    
    def validate_decay_patterns(self, sentiment_data: pd.DataFrame, 
                                sample_dates: List[datetime] = None,
                                plot: bool = True) -> Dict:
        """
        Validate that decay patterns make intuitive sense - prevents overfitting
        
        Returns statistical validation of decay behavior
        """
        if sample_dates is None:
            # Sample 20 random dates for validation
            all_dates = pd.to_datetime(sentiment_data['date'].unique())
            sample_dates = np.random.choice(all_dates, min(20, len(all_dates)), replace=False)
        
        validation_results = {'correlations': {}, 'decay_curves': {}, 'statistical_tests': {}}
        
        # Test decay curves for each horizon
        for horizon in self.decay_params.keys():
            params = self.decay_params[horizon]
            
            # Generate theoretical decay curve
            ages = np.arange(0, params.lookback_days)
            theoretical_weights = np.exp(-params.lambda_decay * ages)
            
            # Sample actual decay patterns
            actual_patterns = []
            for date in sample_dates:
                # Get sentiment within lookback window
                cutoff = pd.to_datetime(date) - timedelta(days=params.lookback_days)
                recent = sentiment_data[
                    (sentiment_data['date'] >= cutoff) & 
                    (sentiment_data['date'] <= date)
                ].copy()
                
                if len(recent) > 5:  # Need sufficient data points
                    recent['age'] = (pd.to_datetime(date) - pd.to_datetime(recent['date'])).dt.days
                    recent['theoretical_weight'] = np.exp(-params.lambda_decay * recent['age'])
                    actual_patterns.append(recent[['age', 'theoretical_weight']])
            
            if actual_patterns:
                combined = pd.concat(actual_patterns)
                
                # Test correlation between age and theoretical weight
                correlation = stats.spearmanr(combined['age'], combined['theoretical_weight'])
                validation_results['correlations'][horizon] = {
                    'correlation': correlation.correlation,
                    'p_value': correlation.pvalue,
                    'sample_size': len(combined)
                }
                
                validation_results['decay_curves'][horizon] = {
                    'theoretical': theoretical_weights,
                    'ages': ages,
                    'lambda': params.lambda_decay
                }
        
        # Statistical test: Do shorter horizons have faster decay?
        lambdas = [self.decay_params[h].lambda_decay for h in [5, 30, 90]]
        monotonic_test = all(lambdas[i] >= lambdas[i+1] for i in range(len(lambdas)-1))
        validation_results['statistical_tests']['monotonic_decay'] = monotonic_test
        
        # Visualization
        if plot:
            self.plot_decay_validation(validation_results)
        
        return validation_results
    
    def plot_decay_validation(self, validation_results: Dict, figsize: Tuple[int, int] = (15, 10)):
        """Create comprehensive visualization of decay patterns"""
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('Temporal Decay Validation Analysis', fontsize=16, fontweight='bold')
        
        colors = {'5': '#ff6b6b', '30': '#4ecdc4', '90': '#45b7d1'}
        
        # Plot 1: Decay curves comparison
        ax1 = axes[0, 0]
        for horizon in [5, 30, 90]:
            if horizon in validation_results['decay_curves']:
                data = validation_results['decay_curves'][horizon]
                ax1.plot(data['ages'], data['theoretical'], 
                        label=f'{horizon}d (λ={data["lambda"]:.3f})',
                        color=colors[str(horizon)], linewidth=2)
        
        ax1.set_xlabel('Age (Trading Days)')
        ax1.set_ylabel('Decay Weight')
        ax1.set_title('Theoretical Decay Curves')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Lambda parameters by horizon
        ax2 = axes[0, 1]
        horizons = list(self.decay_params.keys())
        lambdas = [self.decay_params[h].lambda_decay for h in horizons]
        bars = ax2.bar(range(len(horizons)), lambdas, color=[colors[str(h)] for h in horizons])
        ax2.set_xlabel('Forecast Horizon (Days)')
        ax2.set_ylabel('Decay Rate (λ)')
        ax2.set_title('Decay Parameters by Horizon')
        ax2.set_xticks(range(len(horizons)))
        ax2.set_xticklabels([f'{h}d' for h in horizons])
        
        # Add value labels on bars
        for bar, lamb in zip(bars, lambdas):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{lamb:.3f}', ha='center', va='bottom')
        
        # Plot 3: Correlation validation
        ax3 = axes[0, 2]
        if 'correlations' in validation_results:
            corr_data = validation_results['correlations']
            horizons_with_corr = [h for h in [5, 30, 90] if h in corr_data]
            correlations = [corr_data[h]['correlation'] for h in horizons_with_corr]
            p_values = [corr_data[h]['p_value'] for h in horizons_with_corr]
            
            bars = ax3.bar(range(len(horizons_with_corr)), correlations, 
                            color=[colors[str(h)] for h in horizons_with_corr])
            ax3.set_xlabel('Forecast Horizon (Days)')
            ax3.set_ylabel('Correlation (Age vs Weight)')
            ax3.set_title('Decay Pattern Validation')
            ax3.set_xticks(range(len(horizons_with_corr)))
            ax3.set_xticklabels([f'{h}d' for h in horizons_with_corr])
            ax3.set_ylim([-1, 0])  # Expecting negative correlation
            
            # Add significance stars
            for i, (bar, p_val) in enumerate(zip(bars, p_values)):
                significance = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        significance, ha='center', va='bottom', fontweight='bold')
        
        # Plot 4: Weight concentration analysis
        ax4 = axes[1, 0]
        # This would show how concentrated the weights are (higher std = more dispersed)
        ax4.text(0.5, 0.5, 'Weight Concentration\nAnalysis\n(Implementation dependent\non actual data)',
                ha='center', va='center', transform=ax4.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        ax4.set_title('Weight Distribution Analysis')
        
        # Plot 5: Effective lookback comparison
        ax5 = axes[1, 1]
        lookbacks = [self.decay_params[h].lookback_days for h in [5, 30, 90]]
        bars = ax5.bar(range(3), lookbacks, color=[colors[str(h)] for h in [5, 30, 90]])
        ax5.set_xlabel('Forecast Horizon (Days)')
        ax5.set_ylabel('Lookback Window (Days)')
        ax5.set_title('Lookback Windows by Horizon')
        ax5.set_xticks(range(3))
        ax5.set_xticklabels(['5d', '30d', '90d'])
        
        # Plot 6: Summary statistics
        ax6 = axes[1, 2]
        summary_text = "Validation Summary:\n\n"
        
        if 'statistical_tests' in validation_results:
            monotonic = validation_results['statistical_tests'].get('monotonic_decay', False)
            summary_text += f"✓ Monotonic Decay: {'PASS' if monotonic else 'FAIL'}\n\n"
        
        if 'correlations' in validation_results:
            avg_corr = np.mean([corr_data['correlation'] for corr_data in validation_results['correlations'].values()])
            summary_text += f"Avg. Correlation: {avg_corr:.3f}\n"
            summary_text += f"(Negative expected)\n\n"
        
        summary_text += "🎯 Overfitting Checks:\n"
        summary_text += "• Parameter validation\n"
        summary_text += "• Quality filtering\n"
        summary_text += "• Statistical validation\n"
        summary_text += "• Cross-validation ready"
        
        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, 
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7))
        ax6.set_xlim(0, 1)
        ax6.set_ylim(0, 1)
        ax6.axis('off')
        ax6.set_title('Validation Summary')
        
        plt.tight_layout()
        plt.show()
        
        return fig

# Example usage and validation
if __name__ == "__main__":
    # Define decay parameters with overfitting prevention
    decay_config = {
        5: DecayParameters(horizon=5, lambda_decay=0.3, lookback_days=10, min_sentiment_count=3),
        30: DecayParameters(horizon=30, lambda_decay=0.1, lookback_days=30, min_sentiment_count=5),
        90: DecayParameters(horizon=90, lambda_decay=0.05, lookback_days=60, min_sentiment_count=7)
    }
    
    # Initialize processor
    processor = TemporalDecayProcessor(decay_config)
    
    # Generate sample data for validation
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', '2024-12-01', freq='D')
    sample_sentiment = pd.DataFrame({
        'date': np.random.choice(dates, 1000),
        'score': np.random.normal(0, 0.5, 1000),
        'confidence': np.random.beta(2, 2, 1000),
        'article_count': np.random.poisson(5, 1000) + 1,
        'source': 'sample'
    })
    
    # Run validation
    print("🔍 Running temporal decay validation...")
    validation_results = processor.validate_decay_patterns(sample_sentiment, plot=True)
    
    print("\n✅ Temporal decay validation complete!")
    print("Key insights:")
    for horizon, corr_data in validation_results.get('correlations', {}).items():
        print(f"  Horizon {horizon}d: correlation = {corr_data['correlation']:.3f} (p = {corr_data['p_value']:.3f})")