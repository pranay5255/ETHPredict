# ETH Price Prediction Pipeline

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
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Collection (Automated)

Use the automated data setup script to collect all required data:

```bash
# Setup all data sources automatically
python scripts/data_setup.py

# Setup specific data source only
python scripts/data_setup.py --source binance
python scripts/data_setup.py --source defillama
python scripts/data_setup.py --source santiment

# Validate existing data without downloading
python scripts/data_setup.py --validate

# Force re-download all data
python scripts/data_setup.py --force
```

### 3. Data Preparation (Manual Alternative)

If automated setup fails, manually place data files in the `data/` directory:
```
data/
├── defillama_eth_chain_tvl_2025_04-05.csv
├── santiment_metrics_april_may_2025.csv
└── raw/
    ├── ETHUSDT-1h-2025-04.csv
    └── ETHUSDT-1h-2025-05.csv
```

### 4. Pipeline Execution

Choose your execution method based on needs:

```bash
# OPTION A: Full pipeline validation and testing
python setup.py

# OPTION B: Quick validation only (faster)
python setup.py --quick

# OPTION C: Generate deliverables and documentation
python setup.py --deliverables

# OPTION D: Run prototype analysis with visualizations
python results/prototype.py

# Advanced options
python setup.py --verbose --deliverables  # Verbose logging
python setup.py --no-train               # Skip training tests
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

## Key Scripts

### 1. `scripts/data_setup.py` - Automated Data Collection
**Purpose**: Orchestrates collection of all required data sources for the ETH prediction pipeline.

**Key Features**:
- Automated download from Binance, DeFiLlama, and Santiment APIs
- Data validation and integrity checks
- Proper directory structure creation
- Error handling and retry mechanisms

**Usage Examples**:
```bash
python scripts/data_setup.py                    # Setup all data sources
python scripts/data_setup.py --source binance   # Setup specific source
python scripts/data_setup.py --validate         # Validate existing data
python scripts/data_setup.py --force            # Force re-download
```

**Output**: Populates `data/` directory with all required CSV files organized by source.

### 2. `setup.py` - Pipeline Validation & Testing
**Purpose**: Comprehensive validation of the entire pipeline with detailed testing and deliverable generation.

**Key Features**:
- Environment validation (Python version, dependencies, GPU)
- Data structure and integrity validation
- Feature engineering and model architecture testing
- Training pipeline validation with minimal data
- Deliverable generation (datasource matrix, signal design, prototype)

**Usage Examples**:
```bash
python setup.py                    # Full validation (recommended)
python setup.py --quick           # Quick validation (faster)
python setup.py --deliverables    # Generate all deliverables
python setup.py --verbose         # Detailed logging
python setup.py --no-train        # Skip training tests
```

**Output**: 
- Validation logs in `logs/` directory
- Generated deliverables in `results/` directory
- Comprehensive test reports

### 3. `results/prototype.py` - Pipeline Execution & Visualization
**Purpose**: Runs the complete pipeline end-to-end and generates comprehensive visualizations and analysis.

**Key Features**:
- Full data preprocessing and feature engineering
- Triple-barrier label creation and analysis
- Model training (hierarchical + ensemble + baseline)
- Performance visualization and comparison
- Statistical analysis and metrics computation

**Usage**:
```bash
cd results
python prototype.py
```

**Output**: Generates 15+ visualization files and analysis reports in `results/` directory.

## Pipeline Components

### Core Modules
- `preprocess.py` - Data loading and feature engineering
- `label.py` - Triple-barrier labeling system
- `model.py` - Neural network architectures
- `train.py` - Training pipeline with validation
- `ensemble.py` - Complete ensemble system

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

### Complete Workflow (Recommended)
```bash
# 1. Setup data sources automatically
python scripts/data_setup.py

# 2. Validate pipeline and generate deliverables  
python setup.py --deliverables

# 3. Run full pipeline with visualizations
cd results && python prototype.py
```

### Data Collection Scenarios
```bash
# Setup all data sources
python scripts/data_setup.py

# Setup only specific sources
python scripts/data_setup.py --source binance
python scripts/data_setup.py --source defillama

# Validate existing data without downloading
python scripts/data_setup.py --validate

# Force re-download (if data appears corrupted)
python scripts/data_setup.py --force
```

### Pipeline Validation Scenarios
```bash
# Quick validation (recommended for testing)
python setup.py --quick

# Full validation with training tests
python setup.py --verbose

# Generate documentation only
python setup.py --deliverables

# Skip training (faster validation)
python setup.py --no-train
```

### Analysis and Visualization
```bash
# Run complete pipeline analysis (generates 15+ files)
cd results && python prototype.py

# Alternative: Use setup.py to generate core deliverables
python setup.py --deliverables
```

### Custom Configurations
```bash
# Custom data directory
python scripts/data_setup.py --data-dir /path/to/data
python setup.py --data-dir /path/to/data --deliverables
```

## Results Folder Deep Dive

The `results/` folder contains all outputs from pipeline execution, analysis, and deliverables. Files are generated by running `setup.py --deliverables` or `results/prototype.py`.

### Generated Files Overview

**Core Deliverables**:
- `datasource_matrix.csv` - Metadata matrix of all data sources with providers, granularity, rate limits, and gotchas
- `signal_design.md` - Comprehensive technical documentation of feature engineering, labeling methodology, and model architecture
- `prototype.py` - Executable script that runs the complete pipeline and generates all visualizations
- `README.md` - Updated project documentation with setup instructions (this file)

**Data Analysis & Visualization**:
- `data_overview.png` - Multi-panel overview of all raw data sources (price, TVL, volume, network activity)
- `feature_correlation.png` - Heatmap showing correlations between all engineered features
- `feature_distributions.png` - Histograms showing the distribution of the first 24 features
- `key_features_timeseries.png` - Time series plots of critical features (price, returns, volatility, TVL)
- `feature_statistics.csv` - Statistical summary (mean, std, min, max, quartiles) for all features

**Label Analysis**:
- `label_analysis.png` - Multi-panel analysis of triple-barrier labels including distribution, hit times, and temporal patterns
- `label_statistics.csv` - Summary statistics of label distribution (up/down/neutral percentages and counts)

**Model Analysis**:
- `model_architecture_analysis.png` - Comparison of model architectures showing parameter counts, model sizes, and efficiency
- `model_comparison.csv` - Tabular comparison of different model architectures with parameter counts and sizes
- `hierarchical_performance.png` - Training loss curves and performance metrics for each hierarchical level
- `performance_comparison.png` - Comparative analysis between hierarchical and ensemble modeling approaches

**Baseline Analysis**:
- `baseline_model_analysis.png` - Performance analysis of simple logistic regression baseline including confusion matrix and feature importance
- `baseline_performance.csv` - Baseline model metrics (accuracy, precision, recall, F1-score)
- `model_analysis.png` - General model performance visualization with predictions, confidence, and rolling accuracy

### Results Folder Structure

```
results/
├── Core Deliverables
│   ├── datasource_matrix.csv           # Data source metadata and specifications
│   ├── signal_design.md                # Complete technical documentation
│   ├── prototype.py                    # Pipeline execution script
│   └── README.md                       # Project documentation (updated)
├── Data Analysis
│   ├── data_overview.png               # Raw data visualization across all sources
│   ├── feature_correlation.png         # Feature correlation heatmap
│   ├── feature_distributions.png       # Feature distribution histograms
│   ├── key_features_timeseries.png     # Time series of critical features
│   └── feature_statistics.csv          # Feature statistical summaries
├── Label Analysis  
│   ├── label_analysis.png              # Triple-barrier label analysis
│   └── label_statistics.csv            # Label distribution statistics
└── Model Analysis
    ├── model_architecture_analysis.png # Model comparison analysis
    ├── model_comparison.csv            # Model architecture specifications
    ├── hierarchical_performance.png    # Hierarchical model training results
    ├── performance_comparison.png      # Cross-model performance comparison
    ├── baseline_model_analysis.png     # Baseline logistic regression analysis
    ├── baseline_performance.csv        # Baseline model metrics
    └── model_analysis.png              # General model performance visualization
```

### How to Generate Results

**Option 1: Generate Specific Deliverables**
```bash
python setup.py --deliverables  # Creates core deliverables only
```

**Option 2: Full Pipeline with Visualizations**
```bash
cd results && python prototype.py  # Generates all 15+ files
```

**Option 3: Validation + Deliverables**
```bash
python setup.py --verbose --deliverables  # Complete pipeline + outputs
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

## Quick Reference

### Three-Step Execution
1. **Data Collection**: `python scripts/data_setup.py`
2. **Pipeline Validation**: `python setup.py --deliverables` 
3. **Full Analysis**: `cd results && python prototype.py`

### Key Output Locations
- **Logs**: `logs/` directory (pipeline execution logs)
- **Data**: `data/` directory (raw and processed data files)
- **Results**: `results/` directory (15+ analysis files and visualizations)
- **Models**: `results/ensemble_models/` (trained model weights)

### Troubleshooting Quick Fixes
```bash
# Data issues
python scripts/data_setup.py --validate

# Environment issues  
python setup.py --verbose

# Memory issues
python setup.py --quick

# Missing visualizations
cd results && python prototype.py
```

## License

MIT License - see LICENSE file for details.
