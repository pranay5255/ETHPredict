"""Performance metrics for the ETH prediction pipeline."""

import numpy as np
from typing import Dict, Tuple

def calculate_returns(predictions: np.ndarray, actual: np.ndarray) -> float:
    """Calculate returns based on predictions and actual values.
    
    Args:
        predictions: Model predictions
        actual: Actual values
        
    Returns:
        Total returns
    """
    return np.sum(predictions * actual)

def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """Calculate Sharpe ratio.
    
    Args:
        returns: Array of returns
        risk_free_rate: Risk-free rate
        
    Returns:
        Sharpe ratio
    """
    excess_returns = returns - risk_free_rate
    if len(excess_returns) == 0:
        return 0.0
    return np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) != 0 else 0.0

def calculate_max_drawdown(returns: np.ndarray) -> float:
    """Calculate maximum drawdown.
    
    Args:
        returns: Array of returns
        
    Returns:
        Maximum drawdown as a percentage
    """
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (running_max - cumulative) / running_max
    return np.max(drawdown)

def calculate_hit_rate(predictions: np.ndarray, actual: np.ndarray) -> float:
    """Calculate prediction hit rate.
    
    Args:
        predictions: Model predictions
        actual: Actual values
        
    Returns:
        Hit rate as a percentage
    """
    correct = np.sum(np.sign(predictions) == np.sign(actual))
    return correct / len(predictions) if len(predictions) > 0 else 0.0

def calculate_information_coefficient(predictions: np.ndarray, actual: np.ndarray) -> float:
    """Calculate information coefficient (rank correlation).
    
    Args:
        predictions: Model predictions
        actual: Actual values
        
    Returns:
        Information coefficient
    """
    pred_ranks = np.argsort(np.argsort(predictions))
    actual_ranks = np.argsort(np.argsort(actual))
    return np.corrcoef(pred_ranks, actual_ranks)[0, 1]

def calculate_brier_score(predictions: np.ndarray, actual: np.ndarray) -> float:
    """Calculate Brier score (mean squared error of probabilities).
    
    Args:
        predictions: Model predictions (probabilities)
        actual: Actual values (binary)
        
    Returns:
        Brier score
    """
    return np.mean((predictions - actual) ** 2)

def calculate_all_metrics(
    predictions: np.ndarray,
    actual: np.ndarray,
    returns: np.ndarray,
    risk_free_rate: float = 0.0
) -> Dict[str, float]:
    """Calculate all performance metrics.
    
    Args:
        predictions: Model predictions
        actual: Actual values
        returns: Array of returns
        risk_free_rate: Risk-free rate
        
    Returns:
        Dictionary of metric names and values
    """
    return {
        "returns": calculate_returns(predictions, actual),
        "sharpe_ratio": calculate_sharpe_ratio(returns, risk_free_rate),
        "max_drawdown": calculate_max_drawdown(returns),
        "hit_rate": calculate_hit_rate(predictions, actual),
        "information_coefficient": calculate_information_coefficient(predictions, actual),
        "brier_score": calculate_brier_score(predictions, actual)
    } 