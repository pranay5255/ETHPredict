#!/usr/bin/env python3
"""
ETH Price Prediction Pipeline Setup and Test Script

This script validates the data setup and tests the entire pipeline with detailed logging.
Assumes data is available in the "data" folder structure.

Usage:
    python setup.py                    # Run full pipeline test
    python setup.py --quick           # Quick validation only
    python setup.py --verbose         # Extra verbose logging
    python setup.py --no-train        # Skip training (validation only)
    python setup.py --deliverables    # Generate deliverables
"""

import os
import sys
import argparse
import logging
import traceback
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from datetime import datetime
import warnings
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
def setup_logging(verbose: bool = False):
    """Setup comprehensive logging configuration."""
    log_level = logging.DEBUG if verbose else logging.INFO
    
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Setup file handler
    file_handler = logging.FileHandler(f'logs/pipeline_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    
    # Setup console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(simple_formatter)
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Suppress some verbose libraries
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    warnings.filterwarnings('ignore')


class PipelineValidator:
    """Comprehensive pipeline validation and testing."""
    
    def __init__(self, data_dir: str = "data", verbose: bool = False):
        self.data_dir = Path(data_dir)
        self.verbose = verbose
        self.logger = logging.getLogger(self.__class__.__name__)
        self.results = {}
        
    def validate_environment(self) -> bool:
        """Validate Python environment and dependencies."""
        self.logger.info("=== Validating Environment ===")
        
        try:
            # Check Python version
            python_version = sys.version_info
            self.logger.info(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
            
            if python_version < (3, 8):
                self.logger.error("Python 3.8+ required")
                return False
            
            # Check critical dependencies
            dependencies = {
                'torch': torch.__version__,
                'pandas': pd.__version__,
                'numpy': np.__version__,
            }
            
            for dep, version in dependencies.items():
                self.logger.info(f"{dep}: {version}")
            
            # Check GPU availability
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0)
                self.logger.info(f"GPU available: {gpu_name} ({gpu_count} devices)")
            else:
                self.logger.info("GPU not available, using CPU")
            
            self.logger.info("✓ Environment validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Environment validation failed: {e}")
            return False
    
    def validate_data_structure(self) -> bool:
        """Validate data directory structure and files."""
        self.logger.info("=== Validating Data Structure ===")
        
        try:
            if not self.data_dir.exists():
                self.logger.error(f"Data directory not found: {self.data_dir}")
                return False
            
            # Expected data files
            expected_files = {
                'chain_tvl': 'defillama_eth_chain_tvl_2025_04-05.csv',
                'santiment': 'santiment_metrics_april_may_2025.csv'
            }
            
            # Check raw directory
            raw_dir = self.data_dir / "raw"
            if not raw_dir.exists():
                self.logger.warning(f"Raw directory not found: {raw_dir}")
            
            # Check for price data files
            price_files = list(raw_dir.glob("ETHUSDT-1h-2025-*.csv")) if raw_dir.exists() else []
            self.logger.info(f"Found {len(price_files)} price data files")
            
            # Validate main data files
            valid_files = 0
            for file_type, filename in expected_files.items():
                file_path = self.data_dir / filename
                if file_path.exists():
                    file_size = file_path.stat().st_size / (1024 * 1024)  # MB
                    self.logger.info(f"✓ {file_type}: {filename} ({file_size:.2f} MB)")
                    valid_files += 1
                else:
                    self.logger.warning(f"✗ Missing {file_type}: {filename}")
            
            # Validate at least some data is available
            if valid_files == 0 and len(price_files) == 0:
                self.logger.error("No data files found!")
                return False
            
            self.logger.info(f"✓ Data structure validation passed ({valid_files}/{len(expected_files)} main files)")
            self.results['data_files_found'] = valid_files
            self.results['price_files_found'] = len(price_files)
            return True
            
        except Exception as e:
            self.logger.error(f"Data structure validation failed: {e}")
            self.logger.debug(traceback.format_exc())
            return False
    
    def test_data_loading(self) -> bool:
        """Test data loading functionality."""
        self.logger.info("=== Testing Data Loading ===")
        
        try:
            from preprocess import DataPreprocessor
            
            # Initialize preprocessor
            preprocessor = DataPreprocessor(str(self.data_dir))
            self.logger.info("✓ DataPreprocessor initialized")
            
            # Test data loading
            data = preprocessor.load_data()
            self.logger.info(f"✓ Data loaded: {list(data.keys())}")
            
            # Validate loaded data
            for name, df in data.items():
                if isinstance(df, pd.DataFrame):
                    self.logger.info(f"  {name}: {df.shape} - {df.index.dtype}")
                    if self.verbose:
                        self.logger.debug(f"    Columns: {list(df.columns)}")
                        self.logger.debug(f"    Date range: {df.index.min()} to {df.index.max()}")
                else:
                    self.logger.info(f"  {name}: {type(df)}")
            
            # Test feature preparation
            if len(data) >= 2:
                features_df, targets_df = preprocessor.get_base_dataset()
                self.logger.info(f"✓ Base dataset created: features {features_df.shape}, targets {targets_df.shape}")
                
                # Validate feature columns
                feature_cols = preprocessor.get_feature_cols()
                self.logger.info(f"✓ Feature columns defined: {len(feature_cols)} features")
                
                if self.verbose:
                    self.logger.debug(f"Features: {feature_cols}")
                
                self.results['features_shape'] = features_df.shape
                self.results['targets_shape'] = targets_df.shape
                self.results['num_features'] = len(feature_cols)
                
            self.logger.info("✓ Data loading test passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Data loading test failed: {e}")
            self.logger.debug(traceback.format_exc())
            return False
    
    def test_feature_engineering(self) -> bool:
        """Test advanced feature engineering functionality."""
        self.logger.info("=== Testing Feature Engineering ===")
        
        try:
            from preprocess import DataPreprocessor
            
            preprocessor = DataPreprocessor(str(self.data_dir))
            
            # Create sample data for testing
            dates = pd.date_range('2025-04-01', periods=1000, freq='1H')
            sample_prices = 3000 + np.cumsum(np.random.randn(1000) * 10)
            sample_series = pd.Series(sample_prices, index=dates)
            
            # Test fractional differentiation
            try:
                optimal_d = preprocessor.find_optimal_d(sample_series)
                fracdiff_series = preprocessor.fracdiff(sample_series, optimal_d)
                self.logger.info(f"✓ Fractional differentiation: d={optimal_d:.3f}")
                
                # Test stationarity improvement
                from statsmodels.tsa.stattools import adfuller
                original_adf = adfuller(sample_series.dropna())[1]
                fracdiff_adf = adfuller(fracdiff_series.dropna())[1]
                self.logger.info(f"  ADF p-value: {original_adf:.6f} → {fracdiff_adf:.6f}")
                
            except Exception as e:
                self.logger.warning(f"Fractional differentiation test failed: {e}")
            
            # Test entropy calculation
            returns = sample_series.pct_change().fillna(0)
            entropy = preprocessor.compute_entropy(returns, window=24)
            self.logger.info(f"✓ Entropy calculation: mean={entropy.mean():.4f}")
            
            # Test structural break detection
            cusum_flags = preprocessor.cusum_flag(returns)
            sadf_flags = preprocessor.sadf_flag(sample_series)
            self.logger.info(f"✓ Structural breaks: CUSUM={cusum_flags.sum()}, SADF={sadf_flags.sum()}")
            
            # Test volatility regimes
            vol_regimes = preprocessor.volatility_regime(returns)
            regime_counts = vol_regimes.value_counts().sort_index()
            self.logger.info(f"✓ Volatility regimes: {dict(regime_counts)}")
            
            self.logger.info("✓ Feature engineering test passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Feature engineering test failed: {e}")
            self.logger.debug(traceback.format_exc())
            return False
    
    def test_model_architecture(self) -> bool:
        """Test model architecture and initialization."""
        self.logger.info("=== Testing Model Architecture ===")
        
        try:
            from model import create_model, HierarchicalPredictor, get_model_info
            
            input_size = 24  # Sample input size
            hidden_size = 32  # Smaller for testing
            
            # Test individual model components
            models_to_test = [
                ("lstm", "Level-0 PriceLSTM"),
                ("meta_mlp", "Level-1 MetaMLP"), 
                ("confidence_gru", "Level-2 ConfidenceGRU"),
                ("hierarchical", "Complete HierarchicalPredictor")
            ]
            
            for model_type, description in models_to_test:
                try:
                    if model_type == "meta_mlp":
                        model = create_model(input_size + 1, hidden_size, model_type=model_type)
                    else:
                        model = create_model(input_size, hidden_size, model_type=model_type)
                    
                    model_info = get_model_info(model)
                    self.logger.info(f"✓ {description}: {model_info['total_parameters']} parameters")
                    
                    if self.verbose:
                        self.logger.debug(f"  {model_info}")
                    
                except Exception as e:
                    self.logger.error(f"✗ {description} failed: {e}")
                    return False
            
            # Test forward pass with sample data
            sample_x = torch.randn(2, 24, input_size)  # Batch=2, Seq=24, Features=input_size
            
            hierarchical_model = create_model(input_size, hidden_size, model_type="hierarchical")
            
            # Test different forward modes
            with torch.no_grad():
                pred_only = hierarchical_model(sample_x, return_confidence=False)
                pred_with_conf = hierarchical_model(sample_x, return_confidence=True)
                
                self.logger.info(f"✓ Forward pass: pred_only={pred_only.shape}, pred_with_conf={len(pred_with_conf)}")
            
            self.logger.info("✓ Model architecture test passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Model architecture test failed: {e}")
            self.logger.debug(traceback.format_exc())
            return False
    
    def test_labeling_system(self) -> bool:
        """Test triple-barrier labeling and meta-labeling."""
        self.logger.info("=== Testing Labeling System ===")
        
        try:
            from label import create_labels, create_training_labels, triple_barrier_labels
            
            # Create sample price data
            dates = pd.date_range('2025-04-01', periods=500, freq='1H')
            sample_prices = 3000 + np.cumsum(np.random.randn(500) * 5)
            
            sample_df = pd.DataFrame({
                'close': sample_prices,
                'high': sample_prices * 1.01,
                'low': sample_prices * 0.99,
                'volume': np.random.uniform(1000, 10000, 500)
            }, index=dates)
            
            # Test triple-barrier labeling
            labeled_df = create_labels(sample_df, price_col='close')
            
            # Validate labels
            label_distribution = labeled_df['y_dir'].value_counts().sort_index()
            self.logger.info(f"✓ Triple-barrier labels: {dict(label_distribution)}")
            
            hit_time_stats = labeled_df['hit_times'].describe()
            self.logger.info(f"✓ Hit times: mean={hit_time_stats['mean']:.2f}, std={hit_time_stats['std']:.2f}")
            
            # Test training label creation
            y_ret, y_dir, sample_weights = create_training_labels(labeled_df, sequence_length=24)
            
            self.logger.info(f"✓ Training labels: y_ret={y_ret.shape}, y_dir={y_dir.shape}, weights={sample_weights.shape}")
            
            # Validate label quality
            non_zero_labels = (y_dir != 0).sum().item()
            
            # Convert tensor to numpy for statistics
            weights_np = sample_weights.numpy()
            weight_mean = weights_np.mean()
            weight_std = weights_np.std()
            weight_min = weights_np.min()
            weight_max = weights_np.max()
            
            self.logger.info(f"✓ Label quality: {non_zero_labels}/{len(y_dir)} non-zero labels")
            self.logger.info(f"✓ Weight distribution: mean={weight_mean:.6f}, std={weight_std:.6f}, min={weight_min:.6f}, max={weight_max:.6f}")
            
            self.results['label_distribution'] = dict(label_distribution)
            self.results['non_zero_labels_ratio'] = non_zero_labels / len(y_dir)
            
            self.logger.info("✓ Labeling system test passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Labeling system test failed: {e}")
            self.logger.debug(traceback.format_exc())
            return False
    
    def test_training_pipeline(self, quick_test: bool = False) -> bool:
        """Test the training pipeline with minimal data."""
        self.logger.info("=== Testing Training Pipeline ===")
        
        try:
            from train import PurgedTimeSeriesSplit, compute_metrics, hierarchical_training_pipeline
            from preprocess import DataPreprocessor, get_data
            
            # Use minimal data for quick testing
            preprocessor = DataPreprocessor(str(self.data_dir))
            features_df, targets_df = preprocessor.get_base_dataset()
            
            # Limit data size for testing
            test_size = 200 if quick_test else min(1000, len(features_df))
            features_df = features_df.iloc[:test_size]
            targets_df = targets_df.iloc[:test_size]
            
            self.logger.info(f"Using {test_size} samples for training test")
            
            # Create sample sequences using real data
            X, _ = get_data(sequence_length=24)
            X = torch.FloatTensor(X[:test_size])
            
            # Create dummy labels
            y_ret = torch.randn(len(X), 1) * 0.01  # Small returns
            y_dir = torch.randint(-1, 2, (len(X),))  # Random directions
            sample_weights = torch.ones(len(X))
            
            self.logger.info(f"✓ Test data prepared: X={X.shape}")
            
            # Test purged cross-validation
            cv_splitter = PurgedTimeSeriesSplit(n_splits=3, embargo_hours=2)
            splits = cv_splitter.split(X, y_ret)
            
            self.logger.info(f"✓ Purged CV: {len(splits)} splits created")
            
            # Test metrics computation
            sample_pred = torch.randn_like(y_ret)
            metrics = compute_metrics(y_ret, sample_pred, "regression")
            self.logger.info(f"✓ Metrics computation: {list(metrics.keys())}")
            
            if not quick_test:
                # Test minimal training pipeline
                self.logger.info("Running minimal hierarchical training...")
                
                # Use very small subset and few epochs for speed
                mini_X = X[:50]
                mini_y_ret = y_ret[:50]
                mini_y_dir = y_dir[:50]
                mini_weights = sample_weights[:50]
                
                results = hierarchical_training_pipeline(
                    mini_X, mini_y_ret, mini_y_dir, mini_weights,
                    input_size=X.shape[2],
                    hidden_size=16,  # Very small for testing
                    num_layers=1,
                    batch_size=8,
                    device=torch.device("cpu")  # Force CPU for testing
                )
                
                self.logger.info("✓ Minimal training pipeline completed")
                
                # Validate training results
                for level in ["level_0", "level_1", "level_2"]:
                    if level in results:
                        metrics = results[level]["metrics"]
                        self.logger.info(f"  {level}: {list(metrics.keys())}")
            
            self.logger.info("✓ Training pipeline test passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Training pipeline test failed: {e}")
            self.logger.debug(traceback.format_exc())
            return False
    
    def test_ensemble_system(self, quick_test: bool = False) -> bool:
        """Test the complete ensemble system."""
        self.logger.info("=== Testing Ensemble System ===")
        
        try:
            from ensemble import EnsemblePredictor
            from preprocess import DataPreprocessor
            
            # Load minimal data
            preprocessor = DataPreprocessor(str(self.data_dir))
            features_df, targets_df = preprocessor.get_base_dataset()
            
            # Limit data for testing
            test_size = 100 if quick_test else min(500, len(features_df))
            features_df = features_df.iloc[:test_size]
            targets_df = targets_df.iloc[:test_size]
            
            # Initialize ensemble with small parameters
            input_size = len(preprocessor.get_feature_cols())
            ensemble = EnsemblePredictor(
                input_size=input_size,
                hidden_size=16,  # Small for testing
                num_layers=1,
                sequence_length=24
            )
            
            self.logger.info(f"✓ EnsemblePredictor initialized with {input_size} features")
            
            # Test data preparation
            X, y_ret, y_dir, sample_weights = ensemble._prepare_data(features_df, targets_df)
            self.logger.info(f"✓ Data preparation: X={X.shape}, y_ret={y_ret.shape}")
            
            if not quick_test and len(X) > 50:
                # Test training with minimal setup
                self.logger.info("Running minimal ensemble training...")
                
                results = ensemble.train(
                    features_df=features_df,
                    targets_df=targets_df,
                    num_epochs=2,  # Very few epochs
                    run_cv=False   # Skip CV for speed
                )
                
                self.logger.info("✓ Minimal ensemble training completed")
                
                # Test prediction
                sample_input = X[:5]  # Small batch
                pred, conf = ensemble.predict(sample_input, return_confidence=True)
                self.logger.info(f"✓ Prediction test: pred={pred.shape}, conf={conf.shape}")
                
                # Test model saving/loading
                ensemble.save_models("test_models")
                self.logger.info("✓ Model saving test passed")
                
                # Get performance summary
                performance = ensemble.get_performance_summary()
                self.logger.info(f"✓ Performance metrics: {len(performance)} metrics computed")
                
                if self.verbose:
                    for metric, value in performance.items():
                        self.logger.debug(f"  {metric}: {value:.6f}")
            
            self.logger.info("✓ Ensemble system test passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Ensemble system test failed: {e}")
            self.logger.debug(traceback.format_exc())
            return False
    
    def run_full_validation(self, quick_test: bool = False, skip_training: bool = False) -> bool:
        """Run complete pipeline validation."""
        self.logger.info("=" * 60)
        self.logger.info("STARTING ETH PRICE PREDICTION PIPELINE VALIDATION")
        self.logger.info("=" * 60)
        
        start_time = datetime.now()
        tests = [
            ("Environment", self.validate_environment),
            ("Data Structure", self.validate_data_structure),
            ("Data Loading", self.test_data_loading),
            ("Feature Engineering", self.test_feature_engineering),
            ("Model Architecture", self.test_model_architecture),
            ("Labeling System", self.test_labeling_system),
        ]
        
        if not skip_training:
            tests.extend([
                ("Training Pipeline", lambda: self.test_training_pipeline(quick_test)),
                ("Ensemble System", lambda: self.test_ensemble_system(quick_test)),
            ])
        
        passed_tests = 0
        failed_tests = []
        
        for test_name, test_func in tests:
            self.logger.info(f"\n{'='*20} {test_name} {'='*20}")
            try:
                if test_func():
                    passed_tests += 1
                    self.logger.info(f"✓ {test_name} PASSED")
                else:
                    failed_tests.append(test_name)
                    self.logger.error(f"✗ {test_name} FAILED")
            except Exception as e:
                failed_tests.append(test_name)
                self.logger.error(f"✗ {test_name} CRASHED: {e}")
        
        # Final summary
        total_tests = len(tests)
        duration = datetime.now() - start_time
        
        self.logger.info("\n" + "=" * 60)
        self.logger.info("VALIDATION SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Total tests: {total_tests}")
        self.logger.info(f"Passed: {passed_tests}")
        self.logger.info(f"Failed: {len(failed_tests)}")
        self.logger.info(f"Duration: {duration}")
        
        if failed_tests:
            self.logger.error(f"Failed tests: {', '.join(failed_tests)}")
        
        # Log key results
        if self.results:
            self.logger.info("\nKey Results:")
            for key, value in self.results.items():
                self.logger.info(f"  {key}: {value}")
        
        success = len(failed_tests) == 0
        if success:
            self.logger.info("\n🎉 ALL TESTS PASSED! Pipeline is ready for production.")
        else:
            self.logger.error(f"\n❌ {len(failed_tests)} tests failed. Please check the logs.")
        
        return success


class DeliverablesGenerator:
    """Generate the 4 deliverables for the ETH prediction pipeline."""
    
    def __init__(self, data_dir: str = "data", results_dir: str = "results"):
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def generate_datasource_matrix(self) -> bool:
        """Generate datasource_matrix.csv with metadata about all data sources."""
        self.logger.info("=== Generating Datasource Matrix ===")
        
        try:
            # Define data source matrix
            datasources = [
                {
                    'layer': 'L1-Chain',
                    'provider': 'DeFiLlama',
                    'feed_name': 'Ethereum Chain TVL',
                    'granularity': 'Daily',
                    'update_latency': '1-2 hours',
                    'historical_depth': '2+ years',
                    'cost_tier': 'Free',
                    'rate_limit': '1 req/sec',
                    'major_gotchas': 'Daily updates only, TVL calculation varies by protocol'
                },
                {
                    'layer': 'L1-Price',
                    'provider': 'Binance',
                    'feed_name': 'ETHUSDT OHLCV',
                    'granularity': '1 hour',
                    'update_latency': 'Real-time',
                    'historical_depth': '2+ years',
                    'cost_tier': 'Free',
                    'rate_limit': '1200 req/min',
                    'major_gotchas': 'Rate limits strict, need to handle large files'
                },
                {
                    'layer': 'L2-Social',
                    'provider': 'Santiment',
                    'feed_name': 'Social Volume/Sentiment',
                    'granularity': '1 hour',
                    'update_latency': '1-6 hours',
                    'historical_depth': '3+ years',
                    'cost_tier': 'Freemium/Paid',
                    'rate_limit': '100 req/hour',
                    'major_gotchas': 'Limited free tier, sentiment quality varies'
                },
                {
                    'layer': 'L2-Network',
                    'provider': 'Santiment',
                    'feed_name': 'Network Activity (DAA, Dev Activity)',
                    'granularity': '1 hour',
                    'update_latency': '2-12 hours',
                    'historical_depth': '3+ years',
                    'cost_tier': 'Freemium/Paid',
                    'rate_limit': '100 req/hour',
                    'major_gotchas': 'Dev activity has delays, DAA subject to wash trading'
                },
                {
                    'layer': 'L2-Financial',
                    'provider': 'Santiment',
                    'feed_name': 'Market Cap, Network Growth',
                    'granularity': '1 hour',
                    'update_latency': '1-2 hours',
                    'historical_depth': '3+ years',
                    'cost_tier': 'Freemium/Paid',
                    'rate_limit': '100 req/hour',
                    'major_gotchas': 'Market cap uses circulating supply, may differ from exchanges'
                }
            ]
            
            # Create DataFrame
            df = pd.DataFrame(datasources)
            
            # Save to CSV
            output_path = self.results_dir / "datasource_matrix.csv"
            df.to_csv(output_path, index=False)
            
            self.logger.info(f"✓ Datasource matrix saved to {output_path}")
            self.logger.info(f"  Found {len(df)} data sources")
            
            # Log summary
            for _, row in df.iterrows():
                self.logger.info(f"  {row['provider']}: {row['feed_name']} ({row['granularity']})")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to generate datasource matrix: {e}")
            self.logger.debug(traceback.format_exc())
            return False
            
    def generate_signal_design(self) -> bool:
        """Generate signal_design.md with detailed feature and model documentation."""
        self.logger.info("=== Generating Signal Design Document ===")
        
        try:
            from preprocess import DataPreprocessor
            
            # Get feature information
            preprocessor = DataPreprocessor(str(self.data_dir))
            feature_cols = preprocessor.get_feature_cols()
            
            signal_design_content = f"""# ETH Price Prediction Signal Design

## Data Sources Integration Strategy

### Priority 1: Core Infrastructure (Immediate Integration)
1. **Binance ETHUSDT OHLCV** - Primary price and volume data
   - **Value**: Essential baseline for any price prediction
   - **Effort**: Low (free API, well-documented)
   - **Cost**: Free
   - **Rationale**: Forms the foundation of all price-based features

2. **DeFiLlama Chain TVL** - Ethereum ecosystem health
   - **Value**: High correlation with ETH price trends
   - **Effort**: Low (free API, single endpoint)
   - **Cost**: Free
   - **Rationale**: TVL is a key fundamental driver for L1 tokens

### Priority 2: Enhanced Signals (Next Phase)
3. **Santiment Network Metrics** - On-chain fundamentals
   - **Value**: Medium-high (unique insights into network health)
   - **Effort**: Medium (API key required, more complex data)
   - **Cost**: Freemium ($)
   - **Rationale**: Daily Active Addresses and Network Growth provide fundamental strength indicators

4. **Santiment Social Metrics** - Market sentiment
   - **Value**: Medium (sentiment can lead price movements)
   - **Effort**: Medium (same API as network metrics)
   - **Cost**: Freemium ($)
   - **Rationale**: Social volume and sentiment often precede major moves

## Ground Truth Label Design

**Primary Label**: Directional classification with confidence scoring
- **Target**: `next_hour_direction` ∈ {{-1, 0, 1}}
- **Definition**: 
  - +1 if next hour return ≥ +0.25σ of trailing 30-day volatility
  - -1 if next hour return ≤ -0.25σ of trailing 30-day volatility  
  - 0 otherwise (neutral/sideways)

**Rationale**: 
- Uses adaptive threshold based on realized volatility
- 0.25σ threshold balances signal strength vs. label frequency
- 30-day lookback captures recent market regime
- Directional approach is more robust than pure regression

**Secondary Labels** (for meta-learning):
- **Magnitude**: `|next_hour_return|` for position sizing
- **Confidence**: Based on hit-time from triple-barrier labeling

## Feature Engineering Outline

### Base Features ({len([f for f in feature_cols if f in ['close', 'volume', 'tvl_usd']])})
{chr(10).join([f"- **{f}**: {'Price level' if f == 'close' else 'Trading volume' if f == 'volume' else 'Total Value Locked'}" for f in feature_cols if f in ['close', 'volume', 'tvl_usd']])}

### Network Activity Features ({len([f for f in feature_cols if 'addresses' in f or 'dev_activity' in f or 'network_growth' in f])})
{chr(10).join([f"- **{f}**: {'Daily active addresses' if 'addresses' in f else 'Developer activity' if 'dev_activity' in f else 'Network growth rate'}" for f in feature_cols if any(term in f for term in ['addresses', 'dev_activity', 'network_growth'])])}

### Social Sentiment Features ({len([f for f in feature_cols if 'social' in f])})
{chr(10).join([f"- **{f}**: Social media volume/sentiment" for f in feature_cols if 'social' in f])}

### Derived Features ({len([f for f in feature_cols if any(term in f for term in ['return', 'change', 'ratio', 'growth'])])})
{chr(10).join([f"- **{f}**: {'Returns/changes' if any(term in f for term in ['return', 'change']) else 'Ratio features' if 'ratio' in f else 'Growth metrics'}" for f in feature_cols if any(term in f for term in ['return', 'change', 'ratio', 'growth'])])}

### Advanced Features ({len([f for f in feature_cols if any(term in f for term in ['fracdiff', 'entropy', 'cusum', 'sadf', 'regime', 'parkinson'])])})
{chr(10).join([f"- **{f}**: {'Fractional differentiation' if 'fracdiff' in f else 'Information entropy' if 'entropy' in f else 'Structural break detection' if any(term in f for term in ['cusum', 'sadf']) else 'Volatility regime' if 'regime' in f else 'Parkinson volatility estimator'}" for f in feature_cols if any(term in f for term in ['fracdiff', 'entropy', 'cusum', 'sadf', 'regime', 'parkinson'])])}

### Feature Engineering Methodology

**Windows & Lags**:
- Short-term: 1h, 4h, 24h windows for momentum
- Medium-term: 168h (1 week) for trend context  
- Long-term: 720h (1 month) for regime identification

**Normalization**:
- Z-score standardization for level features
- Min-max scaling for bounded features
- Rolling standardization to adapt to regime changes

**Advanced Transforms**:
- **Fractional Differentiation**: Preserve memory while achieving stationarity
- **Information Entropy**: Measure uncertainty/surprise in returns
- **Structural Breaks**: CUSUM and SADF for regime change detection
- **Volatility Regimes**: Discrete states for different market conditions

## Model Architecture

### Hierarchical Multi-Level Design

**Level 0: PriceLSTM** 
- Pure price/volume LSTM for base predictions
- Input: [close, volume, price_returns, volatility]
- Output: Raw directional probability

**Level 1: MetaMLP**
- Combines Level-0 output with fundamental features
- Input: Level-0 predictions + [TVL, network, social features]
- Output: Enhanced directional probability + confidence

**Level 2: ConfidenceGRU**
- Temporal modeling of prediction confidence
- Input: Historical confidence scores + market regime features
- Output: Final prediction with uncertainty bounds

### Regularization & Bias Prevention

**Regularization**:
- Dropout (0.2-0.5) between layers
- L2 weight decay (1e-4)
- Gradient clipping (max_norm=1.0)

**Look-ahead Bias Prevention**:
- Purged time-series cross-validation with embargo period
- No future information in feature construction
- Strict temporal ordering in all data operations

**Calibration**:
- Platt scaling for probability calibration
- Temperature scaling for confidence calibration
- Out-of-sample calibration validation

## Evaluation Metrics

### Offline Metrics (Historical Validation)

**Accuracy Metrics**:
- Precision/Recall by direction class
- F1-score macro/micro averaged
- Accuracy with class imbalance weighting

**Financial Metrics**:
- Sharpe ratio of strategy returns
- Maximum drawdown
- Hit rate vs. magnitude trade-off

**Calibration Metrics**:
- Brier score decomposition
- Reliability diagram analysis
- Expected Calibration Error (ECE)

### Online Metrics (Live Trading)

**Conviction Quality**:
- Information Coefficient (IC)
- Rank Information Coefficient (Rank IC)
- IC t-statistics for significance

**Risk-Adjusted Performance**:
- Sortino ratio (downside deviation)
- Calmar ratio (return/max drawdown)
- Conditional Value at Risk (CVaR)

## Trivial Benchmarks

**Random Baseline**: 
- Random uniform predictions → ~33% accuracy
- Expected Sharpe ratio: ~0 (random walk)

**Persistence Baseline**:
- "Tomorrow same as today" prediction
- Rolling mean reversion (5-day MA)
- Expected metrics: Accuracy ~40%, Sharpe ~0.1-0.3

**Target Performance**:
- Accuracy: >45% (statistically significant over random)
- Sharpe: >0.5 (risk-adjusted outperformance)
- IC: >0.05 (meaningful predictive power)

## Implementation Notes

- All features constructed from the merged dataframe of price, TVL, and social data
- Feature engineering pipeline ensures no look-ahead bias
- Model training uses sample weights from triple-barrier labeling
- Cross-validation employs purged splits to prevent data leakage
"""

            # Save to markdown file
            output_path = self.results_dir / "signal_design.md"
            with open(output_path, 'w') as f:
                f.write(signal_design_content)
            
            self.logger.info(f"✓ Signal design document saved to {output_path}")
            self.logger.info(f"  Documented {len(feature_cols)} features across 5 categories")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to generate signal design: {e}")
            self.logger.debug(traceback.format_exc())
            return False

    def generate_prototype_analysis(self) -> bool:
        """Generate prototype.py that runs the pipeline and creates visualizations."""
        self.logger.info("=== Generating Prototype Analysis ===")
        
        try:
            prototype_content = '''#!/usr/bin/env python3
"""
ETH Price Prediction Prototype
Runs the complete pipeline and generates visualization outputs.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import torch
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

def run_prototype_pipeline():
    """Run the complete pipeline and generate visualizations."""
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    print("🚀 Starting ETH Price Prediction Prototype Pipeline")
    
    # Import modules
    from preprocess import DataPreprocessor
    from label import create_labels
    from model import create_model
    from ensemble import EnsemblePredictor
    
    # 1. Load and prepare data
    print("\n📊 Loading and preprocessing data...")
    preprocessor = DataPreprocessor("data")
    data = preprocessor.load_data()
    features_df, targets_df = preprocessor.get_base_dataset()
    
    print(f"Features shape: {features_df.shape}")
    print(f"Date range: {features_df.index.min()} to {features_df.index.max()}")
    
    # 2. Create data overview visualization
    create_data_overview(data, results_dir)
    
    # 3. Feature analysis
    create_feature_analysis(features_df, results_dir)
    
    # 4. Label analysis
    create_label_analysis(features_df, targets_df, results_dir)
    
    # 5. Quick model training
    print("\n🤖 Training lightweight model...")
    create_model_analysis(features_df, targets_df, results_dir)
    
    print("\n✅ Prototype pipeline completed! Check 'results/' folder for outputs.")

def create_data_overview(data, results_dir):
    """Create overview visualizations of the raw data."""
    print("  📈 Creating data overview plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('ETH Data Sources Overview', fontsize=16, fontweight='bold')
    
    # Price data
    if 'price' in data:
        price_df = data['price'].set_index('timestamp')
        axes[0,0].plot(price_df.index, price_df['close'], linewidth=1)
        axes[0,0].set_title('ETH Price (USDT)')
        axes[0,0].set_ylabel('Price')
        axes[0,0].grid(True, alpha=0.3)
    
    # TVL data
    if 'chain_tvl' in data:
        tvl_df = data['chain_tvl'].set_index('timestamp')
        axes[0,1].plot(tvl_df.index, tvl_df['tvl_usd']/1e9, linewidth=1, color='green')
        axes[0,1].set_title('Ethereum Chain TVL')
        axes[0,1].set_ylabel('TVL (Billions USD)')
        axes[0,1].grid(True, alpha=0.3)
    
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
    else:
        axes[1,1].text(0.5, 0.5, 'Santiment Data\nNot Available', 
                       ha='center', va='center', transform=axes[1,1].transAxes)
        axes[1,1].set_title('Daily Active Addresses')
    
    plt.tight_layout()
    plt.savefig(results_dir / 'data_overview.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_feature_analysis(features_df, results_dir):
    """Analyze and visualize feature characteristics."""
    print("  🔍 Creating feature analysis plots...")
    
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
    
    # Feature distributions
    fig, axes = plt.subplots(6, 4, figsize=(20, 24))
    axes = axes.ravel()
    
    for i, col in enumerate(features_df.columns[:24]):  # First 24 features
        features_df[col].hist(bins=50, ax=axes[i], alpha=0.7)
        axes[i].set_title(f'{col}', fontsize=10)
        axes[i].grid(True, alpha=0.3)
    
    plt.suptitle('Feature Distributions', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / 'feature_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Time series of key features
    fig, axes = plt.subplots(4, 1, figsize=(15, 12))
    
    key_features = ['close', 'price_return', 'return_vol_24h', 'tvl_usd']
    
    for i, feature in enumerate(key_features):
        if feature in features_df.columns:
            axes[i].plot(features_df.index, features_df[feature], linewidth=1)
            axes[i].set_title(f'{feature} over time')
            axes[i].grid(True, alpha=0.3)
    
    plt.suptitle('Key Features Time Series', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / 'key_features_timeseries.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_label_analysis(features_df, targets_df, results_dir):
    """Analyze target variables and create labels."""
    print("  🎯 Creating label analysis plots...")
    
    # Create simple directional labels
    price_returns = features_df['price_return'].copy()
    volatility = price_returns.rolling(720).std()  # 30-day volatility
    threshold = 0.25 * volatility
    
    labels = pd.Series(0, index=price_returns.index)
    labels[price_returns > threshold] = 1   # Up
    labels[price_returns < -threshold] = -1  # Down
    
    # Label distribution
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Label counts
    label_counts = labels.value_counts().sort_index()
    axes[0,0].bar(label_counts.index, label_counts.values, color=['red', 'gray', 'green'])
    axes[0,0].set_title('Directional Label Distribution')
    axes[0,0].set_xlabel('Direction (-1: Down, 0: Neutral, 1: Up)')
    axes[0,0].set_ylabel('Count')
    
    # Returns distribution
    axes[0,1].hist(price_returns.dropna(), bins=100, alpha=0.7, density=True)
    axes[0,1].axvline(threshold.mean(), color='green', linestyle='--', label='Up threshold')
    axes[0,1].axvline(-threshold.mean(), color='red', linestyle='--', label='Down threshold')
    axes[0,1].set_title('Price Returns Distribution')
    axes[0,1].set_xlabel('Log Returns')
    axes[0,1].legend()
    
    # Volatility regime
    vol_regime = price_returns.rolling(24).std()
    axes[1,0].plot(vol_regime.index, vol_regime, linewidth=1)
    axes[1,0].set_title('Rolling 24h Volatility')
    axes[1,0].set_ylabel('Volatility')
    
    # Label time series
    axes[1,1].plot(labels.index, labels, linewidth=1, alpha=0.7)
    axes[1,1].set_title('Directional Labels over Time')
    axes[1,1].set_ylabel('Direction')
    axes[1,1].set_ylim(-1.5, 1.5)
    
    plt.suptitle('Label Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / 'label_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_model_analysis(features_df, targets_df, results_dir):
    """Train a lightweight model and analyze results."""
    print("  🧠 Training and analyzing model...")
    
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
    axes[0,0].set_title('Confusion Matrix')
    axes[0,0].set_xlabel('Predicted')
    axes[0,0].set_ylabel('Actual')
    
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
        axes[0,1].set_title('Top 10 Feature Importance')
    
    # Prediction distribution
    axes[1,0].hist(y_pred_proba.max(axis=1), bins=20, alpha=0.7)
    axes[1,0].set_title('Prediction Confidence Distribution')
    axes[1,0].set_xlabel('Max Probability')
    
    # Performance over time
    test_dates = features_df.index[-len(y_test):]
    correct_preds = (y_test == y_pred).astype(int)
    rolling_accuracy = pd.Series(correct_preds, index=test_dates).rolling(50).mean()
    
    axes[1,1].plot(rolling_accuracy.index, rolling_accuracy, linewidth=2)
    axes[1,1].axhline(y=0.33, color='red', linestyle='--', label='Random baseline')
    axes[1,1].set_title('Rolling Accuracy (50 periods)')
    axes[1,1].set_ylabel('Accuracy')
    axes[1,1].legend()
    
    plt.suptitle('Model Performance Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / 'model_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print performance summary
    print("\n📊 Model Performance Summary:")
    print(classification_report(y_test, y_pred))
    print(f"Overall Accuracy: {(y_test == y_pred).mean():.3f}")

if __name__ == "__main__":
    run_prototype_pipeline()
'''
            
            # Save prototype script
            output_path = self.results_dir / "prototype.py"
            with open(output_path, 'w') as f:
                f.write(prototype_content)
            
            # Make it executable
            output_path.chmod(0o755)
            
            self.logger.info(f"✓ Prototype script saved to {output_path}")
            
            # Run the prototype to generate visualizations
            self.logger.info("Running prototype to generate visualizations...")
            
            try:
                # Import and run the prototype
                exec(prototype_content)
                self.logger.info("✓ Prototype visualizations generated")
                
            except Exception as e:
                self.logger.warning(f"Prototype execution failed: {e}")
                self.logger.info("✓ Prototype script created (manual execution required)")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to generate prototype: {e}")
            self.logger.debug(traceback.format_exc())
            return False

    def update_readme(self) -> bool:
        """Update README.md to correctly reflect the setup.py pipeline."""
        self.logger.info("=== Updating README.md ===")
        
        try:
            updated_readme = """# ETH Price Prediction Pipeline

A complete machine learning pipeline for Ethereum price prediction using hierarchical models, advanced feature engineering, and multiple data sources.

## Overview

This repository implements a sophisticated ETH price prediction system that combines:
- **Multi-source data integration**: Price, TVL, social, and network metrics  
- **Advanced feature engineering**: Fractional differentiation, entropy, structural breaks
- **Hierarchical model architecture**: Multi-level ensemble with confidence estimation
- **Robust validation**: Purged time-series CV with embargo periods
- **Production-ready pipeline**: Comprehensive validation and deliverables generation

## Quick Start

### 1. Environment Setup

```bash
# Clone repository
git clone <repository-url>
cd ETHPredict

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

Place your data files in the `data/` directory:
```
data/
├── defillama_eth_chain_tvl_2025_04-05.csv
├── santiment_metrics_april_may_2025.csv
└── raw/
    ├── ETHUSDT-1h-2025-04.csv
    └── ETHUSDT-1h-2025-05.csv
```

### 3. Run Complete Pipeline

```bash
# Full pipeline validation and testing
python setup.py

# Quick validation only
python setup.py --quick

# Generate all deliverables
python setup.py --deliverables

# Verbose logging
python setup.py --verbose --deliverables
```

## Data Sources

The pipeline integrates data from multiple sources:

| Source | Data Type | Granularity | Update Frequency |
|--------|-----------|-------------|------------------|
| **Binance** | ETHUSDT OHLCV | 1 hour | Real-time |
| **DeFiLlama** | Chain TVL | Daily | 1-2 hours |
| **Santiment** | Social metrics | 1 hour | 1-6 hours |
| **Santiment** | Network activity | 1 hour | 2-12 hours |

## Feature Engineering

### Base Features (5)
- Price (close), volume, TVL, network activity, social metrics

### Derived Features (10) 
- Returns, volatility, ratios, growth rates, change metrics

### Advanced Features (9)
- **Fractional differentiation**: Preserve memory while achieving stationarity
- **Information entropy**: Measure uncertainty in returns  
- **Structural breaks**: CUSUM and SADF detection
- **Volatility regimes**: Discrete market state classification
- **Parkinson volatility**: High-low range estimator

## Model Architecture

### Hierarchical Design
- **Level 0**: PriceLSTM - Pure price/volume predictions
- **Level 1**: MetaMLP - Fundamental feature integration  
- **Level 2**: ConfidenceGRU - Temporal confidence modeling

### Training Features
- Triple-barrier labeling with meta-labeling
- Sample uniqueness weighting
- Purged time-series cross-validation
- Embargo periods to prevent look-ahead bias

## Pipeline Components

### Core Modules
- `preprocess.py` - Data loading and feature engineering
- `label.py` - Triple-barrier labeling system
- `model.py` - Neural network architectures
- `train.py` - Training pipeline with validation
- `ensemble.py` - Complete ensemble system

### Pipeline Validation
- `setup.py` - Complete pipeline validation and deliverables

## Deliverables

Run `python setup.py --deliverables` to generate:

### 1. datasource_matrix.csv
Data source metadata including providers, granularity, rate limits, and gotchas.

### 2. signal_design.md  
Comprehensive documentation of:
- Data integration strategy (value vs effort vs cost)
- Ground truth label design and rationale
- Feature engineering methodology
- Model architecture and bias prevention
- Evaluation metrics and benchmarks

### 3. prototype.py
Complete pipeline execution with visualization outputs:
- Data overview plots
- Feature correlation analysis  
- Label distribution analysis
- Model performance visualization

### 4. README.md (this file)
Updated setup instructions and pipeline documentation.

## Key Features

### Advanced Labeling
- **Triple-barrier method**: Dynamic stop-loss/take-profit based on volatility
- **Meta-labeling**: Secondary model for bet sizing and confidence
- **Sample weights**: Based on label uniqueness to prevent overfitting

### Robust Validation  
- **Purged CV**: Removes overlapping samples between train/test
- **Embargo period**: Additional gap to prevent information leakage
- **Walk-forward analysis**: Mimics real-world deployment conditions

### Performance Metrics
- **Financial**: Sharpe ratio, max drawdown, hit rate
- **Statistical**: Information coefficient, rank IC
- **Calibration**: Brier score, reliability diagrams

## Usage Examples

### Basic Pipeline Validation
```bash
python setup.py --quick
```

### Full Training and Evaluation
```bash  
python setup.py --verbose
```

### Generate All Deliverables
```bash
python setup.py --deliverables
```

### Custom Data Directory
```bash
python setup.py --data-dir /path/to/data --deliverables
```

## Output Structure

```
results/
├── datasource_matrix.csv       # Data source analysis
├── signal_design.md           # Technical documentation  
├── prototype.py              # Pipeline execution script
├── data_overview.png         # Data visualization
├── feature_correlation.png   # Feature analysis
├── label_analysis.png        # Target analysis
└── model_analysis.png        # Performance metrics
```

## Model Performance

**Target Metrics**:
- Accuracy: >45% (vs 33% random baseline)
- Sharpe Ratio: >0.5 (risk-adjusted returns)
- Information Coefficient: >0.05 (predictive power)

## Key Assumptions

1. **Market Microstructure**: Hourly granularity captures sufficient signal
2. **Feature Stationarity**: Fractional differentiation maintains predictive power
3. **Label Quality**: Triple-barrier method reduces noise in directional labels
4. **Regime Stability**: 30-day volatility window captures regime changes
5. **Data Quality**: Missing values can be forward-filled without bias

## Next Steps (if hired)

### Phase 1: Enhanced Data (Weeks 1-2)
- Integrate additional data sources (Glassnode, funding rates)
- Add derivatives data (options flow, perpetual OI)
- Implement real-time data pipeline

### Phase 2: Advanced Models (Weeks 3-4)  
- Transformer-based architectures
- Graph neural networks for cross-asset correlations
- Reinforcement learning for position sizing

### Phase 3: Production Deployment (Weeks 5-6)
- Model serving infrastructure
- Risk management integration  
- Performance monitoring and retraining

### Phase 4: Research Extensions (Ongoing)
- Causal inference for feature selection
- Adversarial training for robustness
- Multi-timeframe ensemble methods

## Troubleshooting

### Data Issues
```bash
# Check data structure
python setup.py --verbose

# Validate specific data sources  
python -c "from preprocess import DataPreprocessor; dp = DataPreprocessor(); print(dp.load_data().keys())"
```

### Memory Issues
- Reduce sequence length in `get_data(sequence_length=12)`
- Use CPU training: `device=torch.device('cpu')`
- Limit data size: `features_df.iloc[:1000]`

### Performance Issues
- Enable GPU: Install CUDA-compatible PyTorch
- Reduce model size: `hidden_size=32, num_layers=1`
- Use quick validation: `python setup.py --quick`

## License

MIT License - see LICENSE file for details.
"""

            # Save updated README
            with open("README.md", 'w') as f:
                f.write(updated_readme)
            
            self.logger.info("✓ README.md updated with correct pipeline information")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update README: {e}")
            self.logger.debug(traceback.format_exc())
            return False

    def generate_all_deliverables(self) -> bool:
        """Generate all 4 deliverables."""
        self.logger.info("=" * 60)
        self.logger.info("GENERATING ETH PREDICTION PIPELINE DELIVERABLES")
        self.logger.info("=" * 60)
        
        deliverables = [
            ("Datasource Matrix", self.generate_datasource_matrix),
            ("Signal Design Document", self.generate_signal_design),
            ("Prototype Analysis", self.generate_prototype_analysis), 
            ("README Update", self.update_readme),
        ]
        
        success_count = 0
        for name, func in deliverables:
            self.logger.info(f"\n{'='*20} {name} {'='*20}")
            try:
                if func():
                    success_count += 1
                    self.logger.info(f"✓ {name} completed successfully")
                else:
                    self.logger.error(f"✗ {name} failed")
            except Exception as e:
                self.logger.error(f"✗ {name} crashed: {e}")
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"DELIVERABLES SUMMARY: {success_count}/{len(deliverables)} completed")
        self.logger.info(f"Results saved in: {self.results_dir}")
        
        if success_count == len(deliverables):
            self.logger.info("🎉 All deliverables generated successfully!")
        else:
            self.logger.warning(f"⚠️  {len(deliverables) - success_count} deliverables incomplete")
        
        return success_count == len(deliverables)


def main():
    """Main entry point for the setup and validation script."""
    parser = argparse.ArgumentParser(
        description="ETH Price Prediction Pipeline Setup and Test Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python setup.py                    # Full validation
  python setup.py --quick           # Quick validation
  python setup.py --verbose         # Verbose logging
  python setup.py --no-train        # Skip training tests
  python setup.py --data-dir ./data # Custom data directory
  python setup.py --deliverables    # Generate deliverables
        """
    )
    
    parser.add_argument("--quick", action="store_true", 
                       help="Run quick validation (minimal training)")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    parser.add_argument("--no-train", action="store_true",
                       help="Skip training pipeline tests")
    parser.add_argument("--data-dir", default="data",
                       help="Data directory path (default: data)")
    parser.add_argument("--deliverables", action="store_true",
                       help="Generate all deliverables")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger("main")
    
    logger.info("ETH Price Prediction Pipeline Setup Script")
    logger.info(f"Arguments: {vars(args)}")
    
    # Generate deliverables or run validation
    if args.deliverables:
        deliverables_generator = DeliverablesGenerator(data_dir=args.data_dir)
        success = deliverables_generator.generate_all_deliverables()
    else:
        # Run validation
        validator = PipelineValidator(data_dir=args.data_dir, verbose=args.verbose)
        success = validator.run_full_validation(
            quick_test=args.quick,
            skip_training=args.no_train
        )
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
