# ETH Price Prediction Pipeline

A complete machine learning pipeline for Ethereum price prediction using hierarchical models, advanced feature engineering, and multiple data sources.

## Overview

This repository implements a sophisticated ETH price prediction system that combines:
- **Multi-source data integration**: Price, TVL, social, and network metrics  
- **Advanced feature engineering**: Fractional differentiation, entropy, structural breaks
- **Hierarchical model architecture**: Multi-level ensemble with confidence estimation
- **Robust validation**: Purged time-series CV with embargo periods
- **Production-ready pipeline**: Comprehensive validation and deliverables generation

## Project Structure

```
src/
├── data/              # Data loading and preprocessing
│   ├── loader.py      # Data source integration
│   └── validator.py   # Data validation
├── features/          # Feature engineering
│   ├── base.py        # Base feature computation
│   └── advanced.py    # Advanced feature computation
├── models/            # Model architectures
│   ├── hierarchical.py # Hierarchical model components
│   └── ensemble.py    # Ensemble methods
├── training/          # Training pipeline
│   ├── trainer.py     # Training orchestration
│   └── validator.py   # Model validation
├── utils/             # Shared utilities
│   ├── logging.py     # Logging utilities
│   └── metrics.py     # Performance metrics
└── config/            # Configuration management
    ├── loader.py      # Config loading
    └── validator.py   # Config validation
```

## Quick Start

### 1. Environment Setup

```bash
# Clone repository
git clone <repository-url>
cd ETHPredict

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

The system uses a single comprehensive configuration file:

- **`configs/config.yml`** - Complete system configuration including ML training, market making, and backtesting
- **`configs/schema.yaml`** - JSON schema for configuration validation

### 3. Running Experiments

#### Single Experiment Run
```bash
# Run with default config
python runner.py run

# Run with specific config file
python runner.py run --config configs/config.yml

# Run with custom experiment ID and GPU
python runner.py run --id my_experiment --gpu 0
```

#### Parameter Sweep (Hyperparameter Optimization)
```bash
# Run Bayesian optimization (default)
python runner.py grid --trials 100

# Run grid search
python runner.py grid --mode grid --trials 50

# Run random search with multiple GPUs
python runner.py grid --mode random --trials 200 --gpus "0,1" --workers 8
```

#### Generate Reports
```bash
# Generate HTML report
python runner.py report results/

# Generate JSON report
python runner.py report results/ --format json --output analysis.json
```

#### System Information
```bash
# Check system status and configuration
python runner.py info
```

## Data Sources

| Source | Data Type | Granularity | Update Frequency |
|--------|-----------|-------------|------------------|
| **Binance** | ETHUSDT OHLCV | 1 hour | Real-time |
| **DeFiLlama** | Chain TVL | Daily | 1-2 hours |
| **Santiment** | Social metrics | 1 hour | 1-6 hours |
| **Santiment** | Network activity | 1 hour | 2-12 hours |

## Feature Engineering

### Base Features
- Price (close), volume, TVL, network activity, social metrics
- Returns, volatility, ratios, growth rates, change metrics

### Advanced Features
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

## Performance Metrics

### Financial
- Sharpe ratio: >0.5 (risk-adjusted returns)
- Max drawdown: <20%
- Hit rate: >45% (vs 33% random baseline)

### Statistical
- Information coefficient: >0.05
- Rank IC: >0.03
- Brier score: <0.4

## Configuration System

The system uses a single comprehensive configuration file (`configs/config.yml`) that contains all parameters:

### Complete Configuration Structure

```yaml
# Experiment setup
experiment:
  id: exp_{{timestamp}}
  seed: 42
  trials: 1000

# Data sources and preprocessing
data:
  sources: [binance, defillama, santiment]
  start_date: "2022-01-01"
  end_date: "2025-01-01"

# Feature engineering
features:
  frac_diff_order: 0.5
  include: [vol_adj_flow, rsi, macd, bollinger, volume_profile]

# Model architecture (hierarchical ML)
model:
  type: hierarchical
  level0:
    algo: xgboost
    params:
      max_depth: 8
      eta: 0.1
      tree_method: gpu_hist
  level1:
    enabled: true
    algo: mlp
  level2:
    enabled: true
    algo: gru

# Training parameters
training:
  epochs: 100
  learning_rate: 0.001
  batch_size: 32

# Market making strategy
market_maker:
  strategy: glft
  gamma: 0.5              # Risk aversion
  inventory_limit: 10000  # Max position size
  quote_spread: 0.001     # Base spread

# Inventory management
inventory:
  max_long_position: 5000
  var_limit: 1000
  max_drawdown_pct: 0.15

# Bribe optimization
bribe:
  mode: percentile
  percentile: 95
  mev_protection: true

# Backtesting
backtest:
  start: 2024-01-01
  end: 2025-01-01
  initial_capital: 100000

# DEX simulation
sim:
  mode: amm
  amm:
    fee_bps: 30
    inventory: 10000
    mev_enabled: true

# Parameter optimization
optimization:
  parameter_ranges:
    gamma: [0.1, 2.0]
    inventory_limit: [1000, 50000]
    learning_rate: [0.0001, 0.1]
    max_depth: [4, 12]
  
  method: bayesian
  n_trials: 1000
```

### Key Configuration Sections

- **experiment**: Metadata and experiment settings
- **data**: Data sources and preprocessing
- **bars**: Bar sampling configuration
- **features**: Feature engineering parameters
- **model**: Hierarchical ML model architecture
- **training**: Training and validation parameters
- **market_maker**: Market making strategy
- **inventory**: Risk and inventory management
- **bribe**: MEV and bribe optimization
- **backtest**: Backtesting configuration
- **sim**: DEX simulation parameters
- **performance**: Performance evaluation
- **optimization**: Parameter optimization ranges
- **hardware**: GPU and hardware settings

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_models.py

# Run with coverage
pytest --cov=src
```

### Code Style

```bash
# Format code
black src/

# Lint code
ruff check src/

# Type check
mypy src/
```

## License

MIT License - see LICENSE file for details.
