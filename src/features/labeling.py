import pandas as pd
import numpy as np
from typing import Tuple, Optional
import torch


def triple_barrier_labels(
    price_series: pd.Series,
    returns: pd.Series,
    upper_barrier: float = 0.02,  # κ × σ_h
    lower_barrier: float = -0.02,  # -κ × σ_h  
    timeout: int = 288,
    volatility_window: int = 288
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Implement triple-barrier labeling method from López de Prado.
    
    Args:
        price_series: Price time series
        returns: Log returns series
        upper_barrier: Upper barrier threshold (positive)
        lower_barrier: Lower barrier threshold (negative)
        timeout: Maximum holding period in bars
        volatility_window: Bar window for volatility calculation
        
    Returns:
        Tuple of (y_dir, y_ret, hit_times)
        - y_dir: Direction labels {-1, 0, +1}
        - y_ret: Actual returns achieved
        - hit_times: Time to hit barrier or timeout
    """
    n = len(price_series)
    y_dir = pd.Series(0, index=price_series.index)  # Default to 0 (timeout)
    y_ret = pd.Series(0.0, index=price_series.index)
    hit_times = pd.Series(timeout, index=price_series.index)
    
    # Compute rolling volatility for dynamic barriers
    vol = returns.rolling(volatility_window).std()
    
    for i in range(n - timeout):
        current_price = price_series.iloc[i]
        current_vol = vol.iloc[i] if not pd.isna(vol.iloc[i]) else 0.02  # Default vol
        
        # Dynamic barriers based on volatility
        upper_thresh = upper_barrier * current_vol
        lower_thresh = lower_barrier * current_vol
        
        # Look ahead for barrier hits
        for j in range(1, min(timeout + 1, n - i)):
            future_price = price_series.iloc[i + j]
            cumulative_return = np.log(future_price / current_price)
            
            # Check upper barrier
            if cumulative_return >= upper_thresh:
                y_dir.iloc[i] = 1  # Up
                y_ret.iloc[i] = cumulative_return
                hit_times.iloc[i] = j
                break
                
            # Check lower barrier
            elif cumulative_return <= lower_thresh:
                y_dir.iloc[i] = -1  # Down
                y_ret.iloc[i] = cumulative_return
                hit_times.iloc[i] = j
                break
        else:
            # Timeout - use actual return at timeout
            if i + timeout < n:
                timeout_price = price_series.iloc[i + timeout]
                y_ret.iloc[i] = np.log(timeout_price / current_price)
                # Direction based on sign of timeout return
                y_dir.iloc[i] = np.sign(y_ret.iloc[i])
    
    return y_dir, y_ret, hit_times


def adaptive_triple_barrier(
    price_series: pd.Series,
    returns: pd.Series,
    kappa: float = 2.0,  # Multiplier for volatility-based barriers
    timeout: int = 288,
    min_return_threshold: float = 0.005,  # Minimum return threshold
    volatility_window: int = 288,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Adaptive triple-barrier with volatility-scaled barriers.
    
    Args:
        price_series: Price time series
        returns: Log returns
        kappa: Volatility multiplier for barriers
        timeout: Maximum holding period in bars
        min_return_threshold: Minimum return threshold to avoid noise
        volatility_window: Bar window for volatility-scaled barriers
        
    Returns:
        Tuple of (y_dir, y_ret, hit_times)
    """
    # Compute rolling volatility over a wall-clock-preserving bar window.
    rolling_vol = returns.rolling(volatility_window).std()
    
    # Dynamic barriers
    upper_barriers = kappa * rolling_vol
    lower_barriers = -kappa * rolling_vol
    
    # Ensure minimum threshold
    upper_barriers = np.maximum(upper_barriers, min_return_threshold)
    lower_barriers = np.minimum(lower_barriers, -min_return_threshold)
    
    n = len(price_series)
    y_dir = pd.Series(0, index=price_series.index)
    y_ret = pd.Series(0.0, index=price_series.index)
    hit_times = pd.Series(timeout, index=price_series.index)
    
    for i in range(n - timeout):
        current_price = price_series.iloc[i]
        upper_thresh = upper_barriers.iloc[i] if not pd.isna(upper_barriers.iloc[i]) else min_return_threshold
        lower_thresh = lower_barriers.iloc[i] if not pd.isna(lower_barriers.iloc[i]) else -min_return_threshold
        
        for j in range(1, min(timeout + 1, n - i)):
            future_price = price_series.iloc[i + j]
            cum_return = np.log(future_price / current_price)
            
            if cum_return >= upper_thresh:
                y_dir.iloc[i] = 1
                y_ret.iloc[i] = cum_return
                hit_times.iloc[i] = j
                break
            elif cum_return <= lower_thresh:
                y_dir.iloc[i] = -1
                y_ret.iloc[i] = cum_return
                hit_times.iloc[i] = j
                break
        else:
            # Timeout
            if i + timeout < n:
                timeout_price = price_series.iloc[i + timeout]
                y_ret.iloc[i] = np.log(timeout_price / current_price)
                y_dir.iloc[i] = np.sign(y_ret.iloc[i])
    
    return y_dir, y_ret, hit_times


def meta_labeling(
    base_predictions: pd.Series,
    actual_returns: pd.Series,
    prediction_threshold: float = 0.5,
    return_threshold: float = 0.01
) -> pd.Series:
    """
    Meta-labeling: predict whether the base model's prediction is trustworthy.
    
    Args:
        base_predictions: Base model predictions (probabilities or binary)
        actual_returns: Actual returns achieved
        prediction_threshold: Threshold for considering prediction significant
        return_threshold: Minimum return threshold for "success"
        
    Returns:
        Meta-labels: Binary series indicating trustworthy predictions
    """
    # Binarize base predictions if needed
    if base_predictions.dtype == float:
        base_binary = (base_predictions.abs() > prediction_threshold).astype(int)
    else:
        base_binary = base_predictions.abs()
    
    # Define success: prediction direction matches actual return direction
    # and actual return exceeds threshold
    actual_binary = (actual_returns.abs() > return_threshold).astype(int)
    direction_match = (np.sign(base_predictions) == np.sign(actual_returns)).astype(int)
    
    # Meta-label: 1 if base prediction was significant AND direction was correct AND return was meaningful
    meta_labels = base_binary & direction_match & actual_binary
    
    return meta_labels.astype(int)


def create_labels(
    df: pd.DataFrame,
    price_col: str = "close",
    kappa: float = 2.0,
    timeout: int = 288,
    method: str = "adaptive",
    volatility_window: int = 288,
) -> pd.DataFrame:
    """
    Create labels for training using triple-barrier method.
    
    Args:
        df: DataFrame with price data
        price_col: Column name for price
        kappa: Volatility multiplier
        timeout: Maximum holding period in bars
        method: "adaptive" or "fixed"
        volatility_window: Bar window for volatility estimates
        
    Returns:
        DataFrame with additional label columns
    """
    # Compute returns
    returns = np.log(df[price_col]).diff().fillna(0)
    
    if method == "adaptive":
        y_dir, y_ret, hit_times = adaptive_triple_barrier(
            df[price_col], returns, kappa, timeout, volatility_window=volatility_window
        )
    else:
        y_dir, y_ret, hit_times = triple_barrier_labels(
            df[price_col], returns, timeout=timeout, volatility_window=volatility_window
        )
    
    # Add labels to dataframe
    result_df = df.copy()
    result_df["y_dir"] = y_dir
    result_df["y_ret"] = y_ret
    result_df["hit_times"] = hit_times
    result_df["returns"] = returns
    
    # Additional derived labels
    result_df["y_binary"] = (y_dir != 0).astype(int)  # Binary: hit barrier or not
    result_df["y_magnitude"] = y_ret.abs()  # Magnitude of return
    
    return result_df


def sample_weights_from_labels(
    y_dir: pd.Series,
    hit_times: pd.Series,
    returns: pd.Series,
    volatility_window: int = 288
) -> pd.Series:
    """
    Compute sample weights based on label uniqueness and volatility.
    
    Args:
        y_dir: Direction labels
        hit_times: Time to hit barriers
        returns: Return series
        volatility_window: Bar window for volatility calculation
        
    Returns:
        Sample weights series
    """
    # Sample uniqueness weights (1 / number of overlapping samples)
    uniqueness_weights = pd.Series(1.0, index=y_dir.index)
    
    for i in range(len(hit_times)):
        if pd.notna(hit_times.iloc[i]) and hit_times.iloc[i] > 0:
            # Count overlapping samples
            overlap_start = max(0, i - int(hit_times.iloc[i]))
            overlap_end = min(len(hit_times), i + int(hit_times.iloc[i]))
            overlap_count = overlap_end - overlap_start
            uniqueness_weights.iloc[i] = 1.0 / max(overlap_count, 1)
    
    # Volatility-based weights (down-weight high volatility periods)
    vol = returns.rolling(volatility_window).std()
    fill_value = vol.mean()
    if pd.isna(fill_value):
        fill_value = 0.0
    vol_weights = 1.0 / (1.0 + vol.fillna(fill_value))
    
    # Combine weights
    combined_weights = uniqueness_weights * vol_weights
    
    # Normalize to sum to 1
    combined_weights = combined_weights / combined_weights.sum()
    
    return combined_weights


def kelly_position_size(
    probability: pd.Series,
    expected_return: pd.Series,
    volatility: pd.Series,
    risk_free_rate: float = 0.02,
    max_leverage: float = 1.0
) -> pd.Series:
    """
    Kelly criterion for position sizing.
    
    Args:
        probability: Probability of success
        expected_return: Expected return
        volatility: Expected volatility
        risk_free_rate: Risk-free rate
        max_leverage: Maximum leverage allowed
        
    Returns:
        Position sizes
    """
    # Kelly formula: f = (bp - q) / b
    # where b = odds, p = probability of win, q = probability of loss
    
    # Simplified Kelly for continuous case
    # f = (μ - r) / σ²
    excess_return = expected_return - risk_free_rate
    kelly_fraction = excess_return / (volatility ** 2)
    
    # Adjust by probability
    kelly_fraction = kelly_fraction * probability
    
    # Cap at maximum leverage
    kelly_fraction = np.clip(kelly_fraction, -max_leverage, max_leverage)
    
    return kelly_fraction


def create_training_labels(
    df: pd.DataFrame,
    sequence_length: int = 288,
    timeout: int = 288,
    volatility_window: int = 288,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create training labels for the hierarchical model.
    
    Args:
        df: DataFrame with price and feature data
        sequence_length: Sequence length for time series
        timeout: Label timeout in bars
        volatility_window: Bar window for volatility estimates
        
    Returns:
        Tuple of (y_ret, y_dir, sample_weights) as tensors
    """
    # Create triple-barrier labels
    labeled_df = create_labels(df, price_col="close", timeout=timeout, volatility_window=volatility_window)
    
    # Create sample weights
    weights = sample_weights_from_labels(
        labeled_df["y_dir"],
        labeled_df["hit_times"],
        labeled_df["returns"],
        volatility_window=volatility_window,
    )
    
    # Convert to sequences
    y_ret_seq = []
    y_dir_seq = []
    weights_seq = []
    
    for i in range(len(labeled_df) - sequence_length):
        y_ret_seq.append(labeled_df["y_ret"].iloc[i + sequence_length])
        y_dir_seq.append(labeled_df["y_dir"].iloc[i + sequence_length])
        weights_seq.append(weights.iloc[i + sequence_length])
    
    # Convert to tensors
    y_ret_tensor = torch.FloatTensor(y_ret_seq).unsqueeze(1)  # [N, 1]
    y_dir_tensor = torch.LongTensor(y_dir_seq)  # [N] - for classification
    weights_tensor = torch.FloatTensor(weights_seq)  # [N]
    
    return y_ret_tensor, y_dir_tensor, weights_tensor 


def _meta_label_total_cost_bps(costs: dict, horizon_bars: int, granularity: str) -> float:
    from src.data.features_all import bars_for_duration

    bars_per_hour = bars_for_duration(granularity, hours=1)
    horizon_hours = horizon_bars / max(bars_per_hour, 1)
    return (
        2.0 * float(costs.get("fee_bps", 0.0))
        + float(costs.get("spread_bps", 0.0))
        + float(costs.get("slippage_bps", 0.0))
        + float(costs.get("funding_bps_per_hour", 0.0)) * horizon_hours
    )


def meta_triple_barrier_labels(
    signals: pd.DataFrame,
    price_path: pd.DataFrame,
    barrier_config: dict,
    costs: dict,
    *,
    granularity: str = "5m",
) -> pd.DataFrame:
    """Label proposed long/short signals with side-adjusted triple barriers.

    ``signals`` must contain ``sample_index``, ``side``, ``horizon_bars``,
    ``realized_vol`` and ``is_candidate``. Rows without a candidate signal are
    preserved with ``meta_label`` missing so coverage can be reported separately
    from binary meta-label training.
    """

    if signals.empty:
        out = signals.copy()
        out["meta_label"] = pd.Series(dtype="float64")
        return out

    path_df = price_path.reset_index(drop=True).copy()
    for column in ["open", "high", "low", "close"]:
        if column not in path_df.columns:
            raise ValueError(f"price_path missing required column: {column}")
        path_df[column] = pd.to_numeric(path_df[column], errors="coerce").replace([np.inf, -np.inf], np.nan).ffill().bfill()

    profit_kappa = float(barrier_config.get("profit_kappa", 1.5))
    stop_kappa = float(barrier_config.get("stop_kappa", 1.0))
    min_edge_ret = float(barrier_config.get("min_edge_bps", 0.0)) / 10_000.0

    labeled_rows = []
    for _, row in signals.iterrows():
        out = row.to_dict()
        is_candidate = bool(row.get("is_candidate", False)) and int(row.get("side", 0)) != 0
        if not is_candidate:
            out.update(
                {
                    "meta_label": np.nan,
                    "exit_reason": "no_signal",
                    "label_gross_return": 0.0,
                    "label_net_return": 0.0,
                    "label_holding_bars": 0,
                    "profit_barrier": np.nan,
                    "stop_barrier": np.nan,
                }
            )
            labeled_rows.append(out)
            continue

        sample_index = int(row["sample_index"])
        if sample_index < 0 or sample_index >= len(path_df) - 1:
            out.update(
                {
                    "meta_label": 0.0,
                    "exit_reason": "incomplete_path",
                    "label_gross_return": 0.0,
                    "label_net_return": -float(row.get("total_cost_bps", 0.0)) / 10_000.0,
                    "label_holding_bars": 0,
                    "profit_barrier": np.nan,
                    "stop_barrier": np.nan,
                }
            )
            labeled_rows.append(out)
            continue

        side = int(row["side"])
        horizon_bars = int(row.get("horizon_bars", 1))
        vertical = min(horizon_bars, len(path_df) - sample_index - 1)
        total_cost_bps = float(row.get("total_cost_bps", _meta_label_total_cost_bps(costs, horizon_bars, granularity)))
        cost_ret = total_cost_bps / 10_000.0
        realized_vol = float(row.get("realized_vol", 0.0))
        if not np.isfinite(realized_vol) or realized_vol <= 0:
            realized_vol = 1e-8
        profit_barrier = max(profit_kappa * realized_vol, min_edge_ret + cost_ret)
        stop_barrier = max(stop_kappa * realized_vol, cost_ret)
        entry = float(path_df.loc[sample_index, "close"])

        exit_reason = "vertical"
        gross_return = 0.0
        holding_bars = vertical
        for step in range(1, vertical + 1):
            future = path_df.loc[sample_index + step]
            if side > 0:
                favorable = np.log(float(future["high"]) / entry)
                adverse = np.log(float(future["low"]) / entry)
            else:
                favorable = np.log(entry / float(future["low"]))
                adverse = np.log(entry / float(future["high"]))

            if adverse <= -stop_barrier:
                exit_reason = "stop_loss"
                gross_return = -stop_barrier
                holding_bars = step
                break
            if favorable >= profit_barrier:
                exit_reason = "profit_take"
                gross_return = profit_barrier
                holding_bars = step
                break
            if step == vertical:
                exit_close = float(future["close"])
                gross_return = side * np.log(exit_close / entry)

        net_return = gross_return - cost_ret
        meta_label = 1.0 if (exit_reason == "profit_take" or net_return > 0.0) else 0.0
        out.update(
            {
                "meta_label": meta_label,
                "exit_reason": exit_reason,
                "label_gross_return": float(gross_return),
                "label_net_return": float(net_return),
                "label_holding_bars": int(holding_bars),
                "profit_barrier": float(profit_barrier),
                "stop_barrier": float(stop_barrier),
            }
        )
        labeled_rows.append(out)

    return pd.DataFrame(labeled_rows, columns=list(labeled_rows[0].keys()))
