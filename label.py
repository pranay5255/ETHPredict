import pandas as pd
import numpy as np
from typing import Tuple, Optional
import torch


def triple_barrier_labels(
    price_series: pd.Series,
    returns: pd.Series,
    upper_barrier: float = 0.02,  # κ × σ_h
    lower_barrier: float = -0.02,  # -κ × σ_h  
    timeout: int = 24,  # τ hours
    volatility_window: int = 24
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Implement triple-barrier labeling method from López de Prado.
    
    Args:
        price_series: Price time series
        returns: Log returns series
        upper_barrier: Upper barrier threshold (positive)
        lower_barrier: Lower barrier threshold (negative)
        timeout: Maximum holding period in hours
        volatility_window: Window for volatility calculation
        
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
    timeout: int = 24,
    min_return_threshold: float = 0.005  # Minimum return threshold
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Adaptive triple-barrier with volatility-scaled barriers.
    
    Args:
        price_series: Price time series
        returns: Log returns
        kappa: Volatility multiplier for barriers
        timeout: Maximum holding period
        min_return_threshold: Minimum return threshold to avoid noise
        
    Returns:
        Tuple of (y_dir, y_ret, hit_times)
    """
    # Compute rolling volatility
    vol_24h = returns.rolling(24).std()
    
    # Dynamic barriers
    upper_barriers = kappa * vol_24h
    lower_barriers = -kappa * vol_24h
    
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
    timeout: int = 24,
    method: str = "adaptive"
) -> pd.DataFrame:
    """
    Create labels for training using triple-barrier method.
    
    Args:
        df: DataFrame with price data
        price_col: Column name for price
        kappa: Volatility multiplier
        timeout: Maximum holding period
        method: "adaptive" or "fixed"
        
    Returns:
        DataFrame with additional label columns
    """
    # Compute returns
    returns = np.log(df[price_col]).diff().fillna(0)
    
    if method == "adaptive":
        y_dir, y_ret, hit_times = adaptive_triple_barrier(
            df[price_col], returns, kappa, timeout
        )
    else:
        y_dir, y_ret, hit_times = triple_barrier_labels(
            df[price_col], returns, timeout=timeout
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
    volatility_window: int = 24
) -> pd.Series:
    """
    Compute sample weights based on label uniqueness and volatility.
    
    Args:
        y_dir: Direction labels
        hit_times: Time to hit barriers
        returns: Return series
        volatility_window: Window for volatility calculation
        
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
    vol_weights = 1.0 / (1.0 + vol.fillna(vol.mean()))
    
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
    sequence_length: int = 24
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create training labels for the hierarchical model.
    
    Args:
        df: DataFrame with price and feature data
        sequence_length: Sequence length for time series
        
    Returns:
        Tuple of (y_ret, y_dir, sample_weights) as tensors
    """
    # Create triple-barrier labels
    labeled_df = create_labels(df, price_col="close")
    
    # Create sample weights
    weights = sample_weights_from_labels(
        labeled_df["y_dir"],
        labeled_df["hit_times"],
        labeled_df["returns"]
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