import pandas as pd
import numpy as np
from typing import Any, Tuple, Optional
import torch

from src.features.sample_weights import build_label_spans


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
    """Reconstructable side-adjusted labels; unavailable outcomes stay unlabeled.

    OHLC cannot reveal the order of two touches in one candle. The configured
    ambiguity assumption is recorded, never represented as observed ordering.
    Missing prices are never filled from another bar.
    """
    import hashlib
    import json

    geometry = str(barrier_config.get("geometry", "asymmetric"))
    success = str(barrier_config.get("success_mode", "positive_net_at_vertical"))
    ordering = str(barrier_config.get("intrabar_ordering", "stop_first"))
    if geometry not in {"symmetric", "asymmetric"}:
        raise ValueError(f"Unknown label geometry: {geometry}")
    if success not in {"positive_net_at_vertical", "profit_take_only"}:
        raise ValueError(f"Unknown label success_mode: {success}")
    if ordering not in {"stop_first", "profit_first", "exclude"}:
        raise ValueError(f"Unknown intrabar_ordering: {ordering}")
    profit_kappa = float(barrier_config.get("profit_kappa", 1.5))
    stop_kappa = float(barrier_config.get("stop_kappa", 1.0))
    min_edge = float(barrier_config.get("min_edge_bps", 0.0)) / 10_000.0
    if not all(np.isfinite(x) and x > 0 for x in (profit_kappa, stop_kappa)):
        raise ValueError("profit_kappa and stop_kappa must be finite and positive")
    if not np.isfinite(min_edge) or min_edge < 0:
        raise ValueError("min_edge_bps must be finite and nonnegative")
    identity = hashlib.sha256(json.dumps({"barriers": barrier_config, "costs": costs}, sort_keys=True).encode()).hexdigest()
    path = price_path.reset_index(drop=True).copy()
    for column in ["open", "high", "low", "close"]:
        if column not in path:
            raise ValueError(f"price_path missing required column: {column}")
        path[column] = pd.to_numeric(path[column], errors="coerce")
    values = path[["open", "high", "low", "close"]]
    valid = np.isfinite(values).all(axis=1) & (values > 0).all(axis=1)
    valid &= (path["high"] >= path[["open", "close"]].max(axis=1))
    valid &= (path["low"] <= path[["open", "close"]].min(axis=1)) & (path["high"] >= path["low"])
    timestamps = path["timestamp"].tolist() if "timestamp" in path else list(path.index)

    def integer(value, default=-1):
        try:
            number = float(value)
            return int(number) if np.isfinite(number) and number.is_integer() else default
        except (TypeError, ValueError):
            return default

    def timestamp(index):
        return timestamps[index] if 0 <= index < len(timestamps) else pd.NaT

    rows = []
    for _, signal in signals.iterrows():
        row = signal.to_dict()
        start = integer(row.get("sample_index"))
        horizon = integer(row.get("horizon_bars"))
        expiry = start + horizon
        end = min(max(start, 0), max(len(path) - 1, 0))
        side = integer(row.get("side", 0), 0)
        candidate = bool(row.get("is_candidate", False)) and side in {-1, 1}
        row.update({
            "label_schema_version": 2, "label_config_hash": identity,
            "label_geometry": geometry, "label_success_mode": success,
            "intrabar_ordering": ordering, "ambiguity_flag": False,
            "label_status": "unavailable", "label_exclusion_reason": None,
            "meta_label": np.nan, "exit_reason": "incomplete_path",
            "label_gross_return": np.nan, "label_net_return": np.nan,
            "label_holding_bars": 0, "profit_barrier": np.nan, "stop_barrier": np.nan,
            "profit_barrier_price": np.nan, "stop_barrier_price": np.nan,
            "entry_price": np.nan, "entry_timestamp": timestamp(start),
            "label_volatility": row.get("realized_vol", np.nan),
            "vertical_barrier_idx": expiry, "vertical_barrier_timestamp": timestamp(expiry),
        })
        if not candidate:
            row.update(exit_reason="no_signal", label_exclusion_reason="no_signal")
            end = min(expiry, len(path) - 1)
        elif start < 0 or start >= len(path) or horizon < 1:
            row["label_exclusion_reason"] = "incomplete_path"
        elif not valid.iloc[start]:
            row.update(exit_reason="invalid_ohlc", label_exclusion_reason="invalid_ohlc")
        else:
            entry = float(path.loc[start, "close"])
            row["entry_price"] = entry
            vol = float(row.get("realized_vol", np.nan))
            cost = float(row.get("total_cost_bps", _meta_label_total_cost_bps(costs, horizon, granularity))) / 10_000.0
            if not np.isfinite(vol) or vol <= 0 or not np.isfinite(cost) or cost < 0:
                row.update(exit_reason="invalid_label_inputs", label_exclusion_reason="invalid_label_inputs")
            else:
                profit = max(profit_kappa * vol, min_edge + cost)
                stop = max(stop_kappa * vol, cost)
                if geometry == "symmetric":
                    profit = stop = max(profit, stop)
                row.update(entry_price=entry, profit_barrier=profit, stop_barrier=stop,
                           profit_barrier_price=entry * np.exp(side * profit),
                           stop_barrier_price=entry * np.exp(-side * stop),
                           total_cost_bps=cost * 10_000.0)
                gross = np.nan
                for index in range(start + 1, min(expiry, len(path) - 1) + 1):
                    end = index
                    if not valid.iloc[index]:
                        row.update(exit_reason="invalid_ohlc", label_exclusion_reason="invalid_ohlc")
                        break
                    bar = path.iloc[index]
                    favorable = np.log(bar["high"] / entry) if side > 0 else np.log(entry / bar["low"])
                    adverse = np.log(bar["low"] / entry) if side > 0 else np.log(entry / bar["high"])
                    hit_profit, hit_stop = favorable >= profit, adverse <= -stop
                    if hit_profit and hit_stop:
                        row["ambiguity_flag"] = True
                        if ordering == "exclude":
                            row.update(exit_reason="ambiguous", label_exclusion_reason="ambiguous")
                            break
                        hit_stop = ordering == "stop_first"
                        hit_profit = not hit_stop
                    if hit_stop or hit_profit:
                        row["exit_reason"] = "stop_loss" if hit_stop else "profit_take"
                        gross = -stop if hit_stop else profit
                        break
                    if index == expiry:
                        row["exit_reason"] = "vertical"
                        gross = side * np.log(bar["close"] / entry)
                if np.isfinite(gross):
                    net = gross - cost
                    label = row["exit_reason"] == "profit_take" or (
                        success == "positive_net_at_vertical" and row["exit_reason"] == "vertical" and net > 0
                    )
                    row.update(meta_label=float(label), label_gross_return=float(gross),
                               label_net_return=float(net), label_status="valid")
                elif row["label_exclusion_reason"] is None:
                    row["label_exclusion_reason"] = "incomplete_path"
        row.update({
            "label_holding_bars": max(0, end - start),
            "t1_idx": end, "t1_timestamp": timestamp(end),
            "first_touch_type": row["exit_reason"] if row["label_status"] == "valid" or row["ambiguity_flag"] else None,
            "first_touch_timestamp": timestamp(end) if row["label_status"] == "valid" or row["ambiguity_flag"] else pd.NaT,
            "label_span_start_idx": start, "label_span_end_idx": end,
            "label_span_start_timestamp": timestamp(start), "label_span_end_timestamp": timestamp(end),
        })
        rows.append(row)
    if rows:
        return pd.DataFrame(rows)
    # Keep a stable schema even when candidate generation is empty.
    empty = signals.copy()
    for column in ["meta_label", "label_gross_return", "label_net_return", "t1_idx", "t1_timestamp",
                   "vertical_barrier_idx", "vertical_barrier_timestamp", "label_span_start_idx",
                   "label_span_end_idx", "label_span_start_timestamp", "label_span_end_timestamp",
                   "label_status", "label_exclusion_reason", "ambiguity_flag", "label_config_hash"]:
        empty[column] = pd.Series(dtype="object")
    return empty
