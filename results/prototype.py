#!/usr/bin/env python3
"""
ETH Price Prediction Prototype
Runs the complete pipeline and generates visualization outputs.
"""

import sys
import os
from pathlib import Path

# Add parent directory to Python path to import modules
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import warnings
from datetime import datetime
import logging

# Now import our custom modules
from preprocess import DataPreprocessor, get_data
from label import create_labels, create_training_labels
from model import create_model, get_model_info
from ensemble import EnsemblePredictor
from train import hierarchical_training_pipeline, compute_metrics, PurgedTimeSeriesSplit

warnings.filterwarnings('ignore')

# Setup logging for prototype
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_prototype_pipeline():
    """Run the complete pipeline and generate visualizations."""
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    logger = logging.getLogger("prototype")
    logger.info("🚀 Starting ETH Price Prediction Prototype Pipeline")
    
    # 1. Load and prepare data
    logger.info("\n📊 Loading and preprocessing data...")
    preprocessor = DataPreprocessor("data")
    data = preprocessor.load_data()
    features_df, targets_df = preprocessor.get_base_dataset()
    
    logger.info(f"Features shape: {features_df.shape}")
    logger.info(f"Date range: {features_df.index.min()} to {features_df.index.max()}")
    logger.info(f"Available features: {len(preprocessor.get_feature_cols())}")
    
    # 2. Create data overview visualization
    create_data_overview(data, results_dir)
    
    # 3. Feature analysis
    create_feature_analysis(features_df, results_dir)
    
    # 4. Label analysis and preparation
    logger.info("\n🎯 Creating and analyzing labels...")
    labeled_data = create_label_analysis(features_df, targets_df, results_dir)
    
    # 5. Model architecture analysis
    logger.info("\n🏗️ Analyzing model architectures...")
    create_model_architecture_analysis(features_df, results_dir)
    
    # 6. Train hierarchical models
    logger.info("\n🤖 Training hierarchical models...")
    model_results = train_hierarchical_models(features_df, targets_df, labeled_data, results_dir)
    
    # 7. Train ensemble system
    logger.info("\n🎼 Training ensemble system...")
    ensemble_results = train_ensemble_system(features_df, targets_df, results_dir)
    
    # 8. Baseline model for comparison
    logger.info("\n📊 Creating baseline model analysis...")
    baseline_results = create_baseline_model_analysis(features_df, targets_df, results_dir)
    
    # 9. Performance comparison
    logger.info("\n📊 Creating performance comparison...")
    create_performance_comparison(model_results, ensemble_results, results_dir)
    
    logger.info("\n✅ Prototype pipeline completed! Check 'results/' folder for outputs.")
    
    # Summary of generated files
    generated_files = list(results_dir.glob("*.png")) + list(results_dir.glob("*.csv"))
    logger.info(f"\n📁 Generated {len(generated_files)} output files:")
    for file in sorted(generated_files):
        logger.info(f"  - {file.name}")
    
    # Final summary
    logger.info(f"\n🎯 Pipeline Summary:")
    logger.info(f"  - Data sources analyzed: {len(data)}")
    logger.info(f"  - Features engineered: {len(features_df.columns)}")
    logger.info(f"  - Training samples: {len(features_df)}")
    logger.info(f"  - Models trained: Hierarchical + Ensemble + Baseline")
    logger.info(f"  - Visualizations created: {len(list(results_dir.glob('*.png')))}")
    logger.info(f"  - Analysis files: {len(list(results_dir.glob('*.csv')))}")

def create_data_overview(data, results_dir):
    """Create overview visualizations of the raw data."""
    logger = logging.getLogger("prototype")
    logger.info("  📈 Creating data overview plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('ETH Data Sources Overview', fontsize=16, fontweight='bold')
    
    # Price data
    if 'price' in data:
        price_df = data['price'].set_index('timestamp')
        axes[0,0].plot(price_df.index, price_df['close'], linewidth=1)
        axes[0,0].set_title('ETH Price (USDT)')
        axes[0,0].set_ylabel('Price')
        axes[0,0].grid(True, alpha=0.3)
        logger.info(f"    Price data: {len(price_df)} records")
    
    # TVL data
    if 'chain_tvl' in data:
        tvl_df = data['chain_tvl'].set_index('timestamp')
        axes[0,1].plot(tvl_df.index, tvl_df['tvl_usd']/1e9, linewidth=1, color='green')
        axes[0,1].set_title('Ethereum Chain TVL')
        axes[0,1].set_ylabel('TVL (Billions USD)')
        axes[0,1].grid(True, alpha=0.3)
        logger.info(f"    TVL data: {len(tvl_df)} records")
    
    # Volume analysis
    if 'price' in data:
        price_df = data['price'].set_index('timestamp')
        axes[1,0].plot(price_df.index, price_df['volume']/1e6, linewidth=1, color='orange')
        axes[1,0].set_title('Trading Volume')
        axes[1,0].set_ylabel('Volume (Millions)')
        axes[1,0].grid(True, alpha=0.3)
    
    # Santiment metrics
    if 'santiment' in data and not data['santiment'].empty:
        sant_df = data['santiment'].set_index('timestamp')
        if 'daily_active_addresses_value' in sant_df.columns:
            axes[1,1].plot(sant_df.index, sant_df['daily_active_addresses_value'], linewidth=1, color='purple')
            axes[1,1].set_title('Daily Active Addresses')
            axes[1,1].set_ylabel('Addresses')
            axes[1,1].grid(True, alpha=0.3)
            logger.info(f"    Santiment data: {len(sant_df)} records")
        else:
            axes[1,1].text(0.5, 0.5, 'Daily Active Addresses\nColumn Not Found', 
                           ha='center', va='center', transform=axes[1,1].transAxes)
            axes[1,1].set_title('Daily Active Addresses')
    else:
        axes[1,1].text(0.5, 0.5, 'Santiment Data\nNot Available', 
                       ha='center', va='center', transform=axes[1,1].transAxes)
        axes[1,1].set_title('Daily Active Addresses')
        logger.info("    Santiment data: Not available")
    
    plt.tight_layout()
    plt.savefig(results_dir / 'data_overview.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("    ✓ Saved data_overview.png")

def create_feature_analysis(features_df, results_dir):
    """Analyze and visualize feature characteristics."""
    logger = logging.getLogger("prototype")
    logger.info("  🔍 Creating feature analysis plots...")
    
    # Feature correlation heatmap
    plt.figure(figsize=(20, 16))
    correlation_matrix = features_df.corr()
    
    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    
    sns.heatmap(correlation_matrix, mask=mask, annot=False, cmap='coolwarm', center=0,
                square=True, linewidths=0.1, cbar_kws={"shrink": .8})
    plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / 'feature_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("    ✓ Saved feature_correlation.png")
    
    # Feature distributions
    num_features = min(24, len(features_df.columns))
    rows = int(np.ceil(num_features / 4))
    fig, axes = plt.subplots(rows, 4, figsize=(20, 4*rows))
    axes = axes.ravel() if rows > 1 else [axes] if rows == 1 else []
    
    for i, col in enumerate(features_df.columns[:num_features]):
        if i < len(axes):
            features_df[col].hist(bins=50, ax=axes[i], alpha=0.7)
            axes[i].set_title(f'{col}', fontsize=10)
            axes[i].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(num_features, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Feature Distributions', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / 'feature_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("    ✓ Saved feature_distributions.png")
    
    # Time series of key features
    key_features = ['close', 'price_return', 'return_vol_24h', 'tvl_usd']
    available_features = [f for f in key_features if f in features_df.columns]
    
    if available_features:
        fig, axes = plt.subplots(len(available_features), 1, figsize=(15, 3*len(available_features)))
        if len(available_features) == 1:
            axes = [axes]
        
        for i, feature in enumerate(available_features):
            axes[i].plot(features_df.index, features_df[feature], linewidth=1)
            axes[i].set_title(f'{feature} over time')
            axes[i].grid(True, alpha=0.3)
    
    plt.suptitle('Key Features Time Series', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / 'key_features_timeseries.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("    ✓ Saved key_features_timeseries.png")

    
    # Feature statistics summary
    feature_stats = features_df.describe()
    feature_stats.to_csv(results_dir / 'feature_statistics.csv')
    logger.info("    ✓ Saved feature_statistics.csv")
    logger.info(f"    📊 Analyzed {len(features_df.columns)} features")

def create_label_analysis(features_df, targets_df, results_dir):
    """Analyze target variables and create labels using triple-barrier method."""
    logger = logging.getLogger("prototype")
    logger.info("  🎯 Creating label analysis plots...")
    
    # Create proper labels using the triple-barrier method
    try:
        labeled_df = create_labels(targets_df, price_col='close')
        logger.info(f"    ✓ Created triple-barrier labels: {len(labeled_df)} samples")
        
        # Extract labels and other info - with fallback for missing columns
        y_dir = labeled_df['y_dir'] if 'y_dir' in labeled_df.columns else labeled_df.get('y', pd.Series(0, index=labeled_df.index))
        hit_times = labeled_df.get('hit_times', pd.Series(np.nan, index=labeled_df.index))
        barriers_hit = labeled_df.get('barriers_hit', pd.Series('timeout', index=labeled_df.index))
        
        # If y_dir is not properly set, create fallback
        if y_dir.nunique() <= 1:
            logger.warning(f"    ⚠️ Triple-barrier labeling produced uniform labels")
            raise ValueError("Uniform labels detected")
        
    except Exception as e:
        logger.warning(f"    ⚠️ Triple-barrier labeling failed: {e}")
        logger.info("    Using simple threshold-based labels as fallback")
        
        # Fallback to simple labeling
        price_returns = features_df['price_return'].copy() if 'price_return' in features_df.columns else targets_df['close'].pct_change()
        volatility = price_returns.rolling(720).std()  # 30-day volatility
        threshold = 0.25 * volatility
        
        y_dir = pd.Series(0, index=price_returns.index)
        y_dir[price_returns > threshold] = 1   # Up
        y_dir[price_returns < -threshold] = -1  # Down
        
        labeled_df = pd.DataFrame({
            'y_dir': y_dir,
            'y_ret': price_returns,
            'hit_times': np.nan,
            'barriers_hit': 'timeout',
            'close': targets_df['close'][:len(y_dir)]  # Add close prices
        })
        hit_times = labeled_df['hit_times']
        barriers_hit = labeled_df['barriers_hit']
    
    # Label distribution analysis
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Label counts
    label_counts = y_dir.value_counts().sort_index()
    colors = ['red' if x == -1 else 'gray' if x == 0 else 'green' for x in label_counts.index]
    axes[0,0].bar(label_counts.index, label_counts.values, color=colors)
    axes[0,0].set_title('Directional Label Distribution')
    axes[0,0].set_xlabel('Direction (-1: Down, 0: Neutral, 1: Up)')
    axes[0,0].set_ylabel('Count')
    
    # Returns distribution
    if 'y_ret' in labeled_df.columns:
        returns = labeled_df['y_ret'].dropna()
        axes[0,1].hist(returns, bins=100, alpha=0.7, density=True)
        axes[0,1].axvline(returns.mean(), color='blue', linestyle='--', label=f'Mean: {returns.mean():.4f}')
        axes[0,1].axvline(returns.std(), color='orange', linestyle='--', label=f'Std: {returns.std():.4f}')
        axes[0,1].set_title('Returns Distribution')
        axes[0,1].set_xlabel('Returns')
    axes[0,1].legend()
    
    # Hit times distribution (if available)
    if not hit_times.isna().all():
        axes[0,2].hist(hit_times.dropna(), bins=30, alpha=0.7)
        axes[0,2].set_title('Hit Times Distribution')
        axes[0,2].set_xlabel('Hours to Barrier Hit')
        axes[0,2].set_ylabel('Frequency')
    else:
        axes[0,2].text(0.5, 0.5, 'Hit Times\nNot Available', 
                       ha='center', va='center', transform=axes[0,2].transAxes)
        axes[0,2].set_title('Hit Times Distribution')
    
    # Volatility regime
    if 'price_return' in features_df.columns:
        vol_regime = features_df['price_return'].rolling(24).std()
    else:
        vol_regime = targets_df['close'].pct_change().rolling(24).std()
    
    axes[1,0].plot(vol_regime.index, vol_regime, linewidth=1, alpha=0.8)
    axes[1,0].set_title('Rolling 24h Volatility')
    axes[1,0].set_ylabel('Volatility')
    
    # Label time series
    axes[1,1].plot(y_dir.index, y_dir, linewidth=1, alpha=0.7)
    axes[1,1].set_title('Directional Labels over Time')
    axes[1,1].set_ylabel('Direction')
    axes[1,1].set_ylim(-1.5, 1.5)
    
    # Barrier hit analysis (if available)
    if not barriers_hit.isna().all() and barriers_hit.nunique() > 1:
        barrier_counts = barriers_hit.value_counts()
        axes[1,2].pie(barrier_counts.values, labels=barrier_counts.index, autopct='%1.1f%%')
        axes[1,2].set_title('Barriers Hit Distribution')
    else:
        axes[1,2].text(0.5, 0.5, 'Barrier Analysis\nNot Available', 
                       ha='center', va='center', transform=axes[1,2].transAxes)
        axes[1,2].set_title('Barriers Hit Distribution')
    
    plt.suptitle('Triple-Barrier Label Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / 'label_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("    ✓ Saved label_analysis.png")
    
    # Save label statistics
    label_stats = {
        'total_samples': len(y_dir),
        'up_labels': (y_dir == 1).sum(),
        'down_labels': (y_dir == -1).sum(),
        'neutral_labels': (y_dir == 0).sum(),
        'up_percentage': (y_dir == 1).mean() * 100,
        'down_percentage': (y_dir == -1).mean() * 100,
        'neutral_percentage': (y_dir == 0).mean() * 100
    }
    
    pd.Series(label_stats).to_csv(results_dir / 'label_statistics.csv', header=['Value'])
    logger.info("    ✓ Saved label_statistics.csv")
    logger.info(f"    📊 Label distribution: Up={label_stats['up_percentage']:.1f}%, Down={label_stats['down_percentage']:.1f}%, Neutral={label_stats['neutral_percentage']:.1f}%")
    
    return labeled_df

def create_model_architecture_analysis(features_df, results_dir):
    """Analyze and visualize model architectures."""
    logger = logging.getLogger("prototype")
    logger.info("  🏗️ Creating model architecture analysis...")
    
    input_size = len(features_df.columns)
    hidden_size = 64  # Standard size for analysis
    
    # Test different model architectures
    models_to_analyze = [
        ("lstm", "PriceLSTM"),
        ("meta_mlp", "MetaMLP"), 
        ("confidence_gru", "ConfidenceGRU"),
        ("hierarchical", "HierarchicalPredictor")
    ]
    
    model_info = []
    
    for model_type, model_name in models_to_analyze:
        try:
            if model_type == "meta_mlp":
                model = create_model(input_size + 1, hidden_size, model_type=model_type)
            else:
                model = create_model(input_size, hidden_size, model_type=model_type)
            
            info = get_model_info(model)
            model_info.append({
                'model_type': model_type,
                'model_name': model_name,
                'total_parameters': info['total_parameters'],
                'trainable_parameters': info['trainable_parameters'],
                'model_size_mb': info.get('model_size_mb', info['total_parameters'] * 4 / (1024 * 1024))  # Estimate if not available
            })
            
            logger.info(f"    ✓ {model_name}: {info['total_parameters']} parameters")
            
        except Exception as e:
            logger.warning(f"    ⚠️ {model_name} analysis failed: {e}")
    
    # Create visualization
    if model_info:
        df = pd.DataFrame(model_info)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Parameter counts
        axes[0].bar(df['model_name'], df['total_parameters'])
        axes[0].set_title('Model Parameter Counts')
        axes[0].set_ylabel('Parameters')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Model sizes
        axes[1].bar(df['model_name'], df['model_size_mb'])
        axes[1].set_title('Model Size (MB)')
        axes[1].set_ylabel('Size (MB)')
        axes[1].tick_params(axis='x', rotation=45)
        
        # Parameter efficiency
        if len(df) > 1:
            baseline_params = df.iloc[0]['total_parameters']
            efficiency = df['total_parameters'] / baseline_params
            axes[2].bar(df['model_name'], efficiency)
            axes[2].set_title('Parameter Efficiency (vs LSTM)')
            axes[2].set_ylabel('Relative Parameters')
            axes[2].tick_params(axis='x', rotation=45)
        
        plt.suptitle('Model Architecture Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(results_dir / 'model_architecture_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save model comparison table
        df.to_csv(results_dir / 'model_comparison.csv', index=False)
        logger.info("    ✓ Saved model_architecture_analysis.png")
        logger.info("    ✓ Saved model_comparison.csv")
    
    return model_info


def train_hierarchical_models(features_df, targets_df, labeled_data, results_dir):
    """Train hierarchical models and analyze performance."""
    logger = logging.getLogger("prototype")
    logger.info("  🤖 Training hierarchical models...")
    
    try:
        # Prepare training data using the sequence creation function
        X, _ = get_data(sequence_length=24)
        
        # Limit data size for prototype
        max_samples = min(1000, len(X))
        X = torch.FloatTensor(X[:max_samples])
        
        # Create training labels from labeled data
        # Ensure labeled_data has the required structure
        if 'close' not in labeled_data.columns:
            # Add close prices from targets_df if missing
            labeled_data = labeled_data.copy()
            labeled_data['close'] = targets_df['close'].iloc[:len(labeled_data)]
        
        y_ret, y_dir, sample_weights = create_training_labels(labeled_data.iloc[:max_samples+24], sequence_length=24)
        
        # Align sizes
        min_len = min(len(X), len(y_ret))
        X = X[:min_len]
        y_ret = y_ret[:min_len]
        y_dir = y_dir[:min_len]
        sample_weights = sample_weights[:min_len]
        
        logger.info(f"    Training data prepared: X={X.shape}, y_ret={y_ret.shape}")
        
        # Train hierarchical model with reduced parameters for speed
        try:
            results = hierarchical_training_pipeline(
                X, y_ret, y_dir, sample_weights,
                input_size=X.shape[2],
                hidden_size=32,  # Smaller for faster training
                num_layers=1,
                batch_size=16,
                device=torch.device("cpu"),  # Use CPU for prototype
                num_epochs=3  # Few epochs for demonstration
            )
        except TypeError as e:
            # If the function signature doesn't match, try a simpler call
            logger.warning(f"    ⚠️ Full hierarchical training failed: {e}")
            logger.info("    Using simplified hierarchical training...")
            results = hierarchical_training_pipeline(
                X, y_ret, y_dir, sample_weights,
                input_size=X.shape[2],
                hidden_size=32,
                device=torch.device("cpu")
            )
        
        logger.info("    ✓ Hierarchical training completed")
        
        # Create performance visualization
        create_hierarchical_performance_plots(results, results_dir)
        
        return results
        
    except Exception as e:
        logger.error(f"    ✗ Hierarchical training failed: {e}")
        return None


def train_ensemble_system(features_df, targets_df, results_dir):
    """Train ensemble system and analyze performance."""
    logger = logging.getLogger("prototype")
    logger.info("  🎼 Training ensemble system...")
    
    try:
        # Initialize ensemble with small parameters for speed
        input_size = len(features_df.columns)
        ensemble = EnsemblePredictor(
            input_size=input_size,
            hidden_size=32,  # Small for prototype
            num_layers=1,
            sequence_length=24
        )
        
        # Limit data for prototype training
        max_samples = min(500, len(features_df))
        features_subset = features_df.iloc[:max_samples]
        targets_subset = targets_df.iloc[:max_samples]
        
        logger.info(f"    Training ensemble on {len(features_subset)} samples")
        
        # Train ensemble
        try:
            results = ensemble.train(
                features_df=features_subset,
                targets_df=targets_subset,
                num_epochs=3,  # Few epochs for demonstration
                run_cv=False   # Skip CV for speed
            )
        except Exception as train_error:
            logger.warning(f"    ⚠️ Ensemble training error: {train_error}")
            # Try with different parameters
            results = ensemble.train(
                features_df=features_subset,
                targets_df=targets_subset,
                num_epochs=2,
                run_cv=False
            )
        
        logger.info("    ✓ Ensemble training completed")
        
        # Test prediction
        X, _ = ensemble._prepare_data(features_subset, targets_subset)
        if len(X) > 0:
            pred_result = ensemble.predict(X[:10], return_confidence=True)
            if isinstance(pred_result, tuple) and len(pred_result) == 2:
                pred, conf = pred_result
            else:
                pred = pred_result
                conf = torch.ones_like(pred) * 0.5  # Default confidence
        
            logger.info(f"    ✓ Prediction test: pred={pred.shape}, conf={conf.shape}")
        else:
            logger.warning("    ⚠️ No data available for prediction test")
        
        # Save ensemble models
        ensemble.save_models(results_dir / "ensemble_models")
        logger.info("    ✓ Ensemble models saved")
        
        # Get performance summary
        try:
            performance = ensemble.get_performance_summary()
            # Save performance metrics
            pd.Series(performance).to_csv(results_dir / 'ensemble_performance.csv', header=['Value'])
            logger.info("    ✓ Saved ensemble_performance.csv")
        except Exception as perf_error:
            logger.warning(f"    ⚠️ Performance summary failed: {perf_error}")
            performance = {'error': 'Performance summary not available'}
        
        return {
            'ensemble': ensemble,
            'performance': performance,
            'results': results
        }
        
    except Exception as e:
        logger.error(f"    ✗ Ensemble training failed: {e}")
        return None


def create_hierarchical_performance_plots(results, results_dir):
    """Create performance plots for hierarchical models."""
    logger = logging.getLogger("prototype")
    
    if not results:
        logger.warning("    ⚠️ No hierarchical results to plot")
        return
    
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Training losses for each level
        for i, level in enumerate(["level_0", "level_1", "level_2"]):
            if level in results and "train_losses" in results[level]:
                losses = results[level]["train_losses"]
                row, col = i // 2, i % 2
                if row < 2 and col < 2:  # Ensure we don't exceed subplot grid
                    axes[row, col].plot(losses, label=f'{level} Training Loss')
                    axes[row, col].set_title(f'{level.replace("_", " ").title()} Training Loss')
                    axes[row, col].set_xlabel('Epoch')
                    axes[row, col].set_ylabel('Loss')
                    axes[row, col].grid(True, alpha=0.3)
        
        # Overall performance comparison
        if len([level for level in ["level_0", "level_1", "level_2"] if level in results]) > 1:
            levels = []
            metrics = []
            
            for level in ["level_0", "level_1", "level_2"]:
                if level in results and "metrics" in results[level]:
                    levels.append(level)
                    # Use a default metric or the first available metric
                    level_metrics = results[level]["metrics"]
                    if "mse" in level_metrics:
                        metrics.append(level_metrics["mse"])
                    elif level_metrics:
                        metrics.append(list(level_metrics.values())[0])
                    else:
                        metrics.append(0)
            
            if len(levels) > 0:
                axes[1, 1].bar(levels, metrics)
                axes[1, 1].set_title('Performance Comparison')
                axes[1, 1].set_ylabel('MSE')
                axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.suptitle('Hierarchical Model Performance', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(results_dir / 'hierarchical_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("    ✓ Saved hierarchical_performance.png")
        
    except Exception as e:
        logger.warning(f"    ⚠️ Failed to create hierarchical performance plots: {e}")


def create_performance_comparison(model_results, ensemble_results, results_dir):
    """Create comparison plots between different modeling approaches."""
    logger = logging.getLogger("prototype")
    logger.info("  📊 Creating performance comparison...")
    
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Model complexity comparison
        complexity_data = []
        if model_results:
            for level in ["level_0", "level_1", "level_2"]:
                if level in model_results and "metrics" in model_results[level]:
                    complexity_data.append({
                        'model': level,
                        'performance': model_results[level]["metrics"].get("mse", 0),
                        'type': 'Hierarchical'
                    })
        
        if ensemble_results and 'performance' in ensemble_results:
            performance = ensemble_results['performance']
            complexity_data.append({
                'model': 'Ensemble',
                'performance': performance.get('mse', 0),
                'type': 'Ensemble'
            })
        
        if complexity_data:
            df_complexity = pd.DataFrame(complexity_data)
            
            # Performance comparison
            hierarchical_models = df_complexity[df_complexity['type'] == 'Hierarchical']
            if len(hierarchical_models) > 0:
                axes[0,0].bar(hierarchical_models['model'], hierarchical_models['performance'], 
                             alpha=0.7, label='Hierarchical')
            
            ensemble_models = df_complexity[df_complexity['type'] == 'Ensemble']
            if len(ensemble_models) > 0:
                axes[0,0].bar(ensemble_models['model'], ensemble_models['performance'], 
                             alpha=0.7, label='Ensemble')
            
            axes[0,0].set_title('Model Performance Comparison')
            axes[0,0].set_ylabel('MSE')
            axes[0,0].legend()
            
        # Placeholder plots for other comparisons
        axes[0,1].text(0.5, 0.5, 'Training Time\nComparison\n(Not Available)', 
                       ha='center', va='center', transform=axes[0,1].transAxes)
        axes[0,1].set_title('Training Time Comparison')
        
        axes[1,0].text(0.5, 0.5, 'Memory Usage\nComparison\n(Not Available)', 
                       ha='center', va='center', transform=axes[1,0].transAxes)
        axes[1,0].set_title('Memory Usage Comparison')
        
        axes[1,1].text(0.5, 0.5, 'Prediction\nAccuracy\n(Not Available)', 
                       ha='center', va='center', transform=axes[1,1].transAxes)
        axes[1,1].set_title('Prediction Accuracy')
        
        plt.suptitle('Model Comparison Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(results_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("    ✓ Saved performance_comparison.png")
        
    except Exception as e:
        logger.warning(f"    ⚠️ Failed to create performance comparison: {e}")


def create_baseline_model_analysis(features_df, targets_df, results_dir):
    """Train a lightweight baseline model for comparison."""
    logger = logging.getLogger("prototype")
    logger.info("  📊 Creating baseline model analysis...")
    
    try:
        # Prepare simple dataset
        X = features_df.fillna(0).values[-1000:]  # Last 1000 samples
        y_price = targets_df['close'].fillna(0).values[-1000:]
        
        # Create simple directional labels
        returns = np.log(y_price[1:] / y_price[:-1])
        X = X[:-1]  # Align with returns
        
        vol = pd.Series(returns).rolling(100).std().fillna(pd.Series(returns).std())
        threshold = 0.25 * vol
        
        y_dir = np.zeros(len(returns))
        y_dir[returns > threshold] = 1
        y_dir[returns < -threshold] = -1
        
        # Split data
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y_dir[:split_idx], y_dir[split_idx:]
        
        # Simple logistic regression baseline
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import classification_report, confusion_matrix
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)
        
        # Model performance visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[0,0], cmap='Blues')
        axes[0,0].set_title('Baseline Model Confusion Matrix')
        axes[0,0].set_xlabel('Predicted')
        axes[0,0].set_ylabel('Actual')

    except Exception as e:
        logger.error(f"    ✗ Baseline model analysis failed: {e}")
        return None
        
    # Feature importance (top 10)
    if hasattr(model, 'coef_'):
        feature_names = features_df.columns
        importance = np.abs(model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_)
        
        # Get top 10 features
        top_indices = np.argsort(importance)[-10:]
        top_features = [feature_names[i] for i in top_indices]
        top_importance = importance[top_indices]
        
        axes[0,1].barh(range(len(top_features)), top_importance)
        axes[0,1].set_yticks(range(len(top_features)))
        axes[0,1].set_yticklabels(top_features)
        axes[0,1].set_title('Top 10 Feature Importance (Baseline)')
    
    # Prediction distribution
    axes[1,0].hist(y_pred_proba.max(axis=1), bins=20, alpha=0.7)
    axes[1,0].set_title('Baseline Prediction Confidence')
    axes[1,0].set_xlabel('Max Probability')
    
    # Performance over time
    test_dates = features_df.index[-len(y_test):]
    correct_preds = (y_test == y_pred).astype(int)
    rolling_accuracy = pd.Series(correct_preds, index=test_dates).rolling(50).mean()
    
    axes[1,1].plot(rolling_accuracy.index, rolling_accuracy, linewidth=2)
    axes[1,1].axhline(y=0.33, color='red', linestyle='--', label='Random baseline')
    axes[1,1].set_title('Baseline Rolling Accuracy (50 periods)')
    axes[1,1].set_ylabel('Accuracy')
    axes[1,1].legend()
    
    plt.suptitle('Baseline Model Performance Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / 'baseline_model_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
        # Save performance summary
    accuracy = (y_test == y_pred).mean()
    baseline_performance = {
        'accuracy': accuracy,
        'precision_macro': classification_report(y_test, y_pred, output_dict=True)['macro avg']['precision'],
        'recall_macro': classification_report(y_test, y_pred, output_dict=True)['macro avg']['recall'],
        'f1_macro': classification_report(y_test, y_pred, output_dict=True)['macro avg']['f1-score']
    }
    
    pd.Series(baseline_performance).to_csv(results_dir / 'baseline_performance.csv', header=['Value'])
    
    logger.info("    ✓ Saved baseline_model_analysis.png")
    logger.info("    ✓ Saved baseline_performance.csv")
    logger.info(f"    📊 Baseline Accuracy: {accuracy:.3f}")
    
    return baseline_performance
        



if __name__ == "__main__":
    run_prototype_pipeline()
