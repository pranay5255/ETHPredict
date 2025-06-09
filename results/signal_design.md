# ETH Price Prediction Signal Design

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
- **Target**: `next_hour_direction` ∈ {-1, 0, 1}
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

### Base Features (3)
- **close**: Price level
- **volume**: Trading volume
- **tvl_usd**: Total Value Locked

### Network Activity Features (3)
- **daily_active_addresses_value**: Daily active addresses
- **dev_activity_value**: Developer activity
- **network_growth_value**: Network growth rate

### Social Sentiment Features (2)
- **social_volume_total_value**: Social media volume/sentiment
- **social_dominance_change**: Social media volume/sentiment

### Derived Features (13)
- **network_growth_value**: Growth metrics
- **price_return**: Returns/changes
- **tvl_change**: Returns/changes
- **price_tvl_ratio**: Ratio features
- **return_vol_24h**: Returns/changes
- **volume_tvl_ratio**: Ratio features
- **address_growth**: Growth metrics
- **social_dominance_change**: Returns/changes
- **mcap_tvl_ratio**: Ratio features
- **return_entropy_24h**: Returns/changes
- **return_entropy_168h**: Returns/changes
- **return_vol_1h**: Returns/changes
- **return_vol_168h**: Returns/changes

### Advanced Features (7)
- **fracdiff_close**: Fractional differentiation
- **return_entropy_24h**: Information entropy
- **return_entropy_168h**: Information entropy
- **cusum_flag**: Structural break detection
- **sadf_flag**: Structural break detection
- **vol_regime**: Volatility regime
- **parkinson_vol**: Parkinson volatility estimator

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
