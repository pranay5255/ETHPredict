"""Backtesting framework for GLFT market making strategies."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import numpy as np
import pandas as pd
from ..market_maker.glft import GLFTQuoteCalculator, GLFTParams
from ..market_maker.inventory import InventoryBook

@dataclass
class BacktestParams:
    """Parameters for backtesting."""
    start_date: datetime
    end_date: datetime
    initial_capital: float
    seed: int
    gamma: float
    inventory_limit: float
    quote_spread: float

class BacktestResult:
    """Container for backtest results."""
    
    def __init__(self):
        self.trades: List[Dict[str, Any]] = []
        self.pnl_history: List[float] = []
        self.inventory_history: List[float] = []
        self.spread_history: List[float] = []
        self.metrics: Dict[str, float] = {}
        
    def add_trade(
        self,
        timestamp: datetime,
        side: str,
        price: float,
        size: float,
        fee: float,
        pnl: float
    ):
        """Add a trade to the history."""
        self.trades.append({
            "timestamp": timestamp,
            "side": side,
            "price": price,
            "size": size,
            "fee": fee,
            "pnl": pnl
        })
        
    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics."""
        if not self.trades:
            return {}
            
        # Convert to DataFrame
        df = pd.DataFrame(self.trades)
        
        # Calculate metrics
        total_pnl = df["pnl"].sum()
        total_fees = df["fee"].sum()
        num_trades = len(df)
        
        # Calculate returns
        returns = pd.Series(self.pnl_history).pct_change().dropna()
        
        # Calculate metrics
        self.metrics = {
            "total_pnl": total_pnl,
            "total_fees": total_fees,
            "net_pnl": total_pnl - total_fees,
            "num_trades": num_trades,
            "win_rate": (df["pnl"] > 0).mean(),
            "avg_trade_pnl": df["pnl"].mean(),
            "sharpe_ratio": returns.mean() / returns.std() * np.sqrt(252),
            "max_drawdown": self._calculate_max_drawdown(),
            "avg_spread": np.mean(self.spread_history),
            "avg_inventory": np.mean(self.inventory_history)
        }
        
        return self.metrics
        
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        cumulative = pd.Series(self.pnl_history).cumsum()
        rolling_max = cumulative.expanding().max()
        drawdowns = (cumulative - rolling_max) / rolling_max
        return abs(drawdowns.min())

class GLFTBacktester:
    """Backtesting framework for GLFT market making strategies."""
    
    def __init__(
        self,
        params: BacktestParams,
        market_maker: GLFTQuoteCalculator,
        inventory: InventoryBook
    ):
        self.params = params
        self.market_maker = market_maker
        self.inventory = inventory
        
        # Initialize results
        self.results = BacktestResult()
        
        # Set random seed
        np.random.seed(params.seed)
        
    def run(
        self,
        market_data: pd.DataFrame,
        predicted_prices: pd.Series,
        volatility: Optional[pd.Series] = None
    ) -> BacktestResult:
        """
        Run backtest simulation.
        
        Args:
            market_data: DataFrame with market data
            predicted_prices: Series with model's price predictions
            volatility: Optional volatility series
            
        Returns:
            BacktestResult: Backtest results
        """
        for timestamp, row in market_data.iterrows():
            # Get current market state
            mid_price = row["price"]
            predicted_price = predicted_prices[timestamp]
            current_vol = volatility[timestamp] if volatility is not None else None
            
            # Calculate quotes using predicted price
            bid, ask = self.market_maker.calculate_optimal_quotes(
                mid_price=predicted_price,  # Use predicted price instead of current
                inventory=self.inventory.get_net_position("base"),
                volatility=current_vol
            )
            
            # Calculate spread
            spread = (ask - bid) / mid_price
            self.results.spread_history.append(spread)
            
            # Simulate trades based on price difference
            price_diff = predicted_price - mid_price
            
            # More likely to buy if predicted price is higher
            buy_prob = 0.5 + 0.3 * np.tanh(price_diff)
            if np.random.random() < buy_prob:
                size = np.random.uniform(0.1, 1.0)
                fee = size * bid * 0.001  # 0.1% fee
                
                self.inventory.add_position(
                    symbol="base",
                    size=size,
                    price=bid,
                    timestamp=timestamp
                )
                
                self.results.add_trade(
                    timestamp=timestamp,
                    side="buy",
                    price=bid,
                    size=size,
                    fee=fee,
                    pnl=size * (predicted_price - bid) - fee
                )
            
            # More likely to sell if predicted price is lower
            sell_prob = 0.5 - 0.3 * np.tanh(price_diff)
            if np.random.random() < sell_prob:
                size = np.random.uniform(0.1, 1.0)
                fee = size * ask * 0.001  # 0.1% fee
                
                self.inventory.add_position(
                    symbol="base",
                    size=-size,
                    price=ask,
                    timestamp=timestamp
                )
                
                self.results.add_trade(
                    timestamp=timestamp,
                    side="sell",
                    price=ask,
                    size=size,
                    fee=fee,
                    pnl=size * (ask - predicted_price) - fee
                )
            
            # Update inventory history
            self.results.inventory_history.append(
                self.inventory.get_net_position("base")
            )
            
        # Calculate final metrics
        self.results.calculate_metrics()
        return self.results