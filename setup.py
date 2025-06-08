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
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger("main")
    
    logger.info("ETH Price Prediction Pipeline Setup Script")
    logger.info(f"Arguments: {vars(args)}")
    
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
