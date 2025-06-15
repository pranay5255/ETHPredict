"""Backtesting framework for market making strategies."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import numpy as np
import pandas as pd
from .amm import AMMSimulator, AMMParams, AMMType
from .lob import LOBMatchingEngine
from ..market_maker.glft import GLFTQuoteCalculator, GLFTParams
from ..market_maker.inventory import InventoryBook
from ..market_maker.bribe import BribeOptimizer, BribeParams

@dataclass
class BacktestParams:
    """Parameters for backtesting."""
    start_date: datetime
    end_date: datetime
    initial_capital: float
    sim_mode: str  # "amm" or "lob"
    seed: int
    amm_params: Optional[AMMParams] = None
    lob_params: Optional[Dict[str, Any]] = None

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

class Backtester:
    """Backtesting framework for market making strategies."""
    
    def __init__(
        self,
        params: BacktestParams,
        market_maker: GLFTQuoteCalculator,
        inventory: InventoryBook,
        bribe_optimizer: Optional[BribeOptimizer] = None
    ):
        self.params = params
        self.market_maker = market_maker
        self.inventory = inventory
        self.bribe_optimizer = bribe_optimizer
        
        # Initialize simulator
        if params.sim_mode == "amm":
            if not params.amm_params:
                raise ValueError("AMM params required for AMM simulation")
            self.simulator = AMMSimulator(params.amm_params)
        else:
            self.simulator = LOBMatchingEngine(
                tick_size=params.lob_params.get("tick_size", 0.01)
            )
            
        # Initialize results
        self.results = BacktestResult()
        
        # Set random seed
        np.random.seed(params.seed)
        
    def run(
        self,
        market_data: pd.DataFrame,
        volatility: Optional[pd.Series] = None
    ) -> BacktestResult:
        """
        Run backtest simulation.
        
        Args:
            market_data: DataFrame with market data
            volatility: Optional volatility series
            
        Returns:
            BacktestResult: Backtest results
        """
        for timestamp, row in market_data.iterrows():
            # Get current market state
            mid_price = row["price"]
            current_vol = volatility[timestamp] if volatility is not None else None
            
            # Calculate quotes
            bid, ask = self.market_maker.calculate_optimal_quotes(
                mid_price=mid_price,
                inventory=self.inventory.get_net_position("base"),
                volatility=current_vol
            )
            
            # Calculate spread
            spread = (ask - bid) / mid_price
            self.results.spread_history.append(spread)
            
            # Simulate trades
            if self.params.sim_mode == "amm":
                self._simulate_amm_trades(timestamp, bid, ask)
            else:
                self._simulate_lob_trades(timestamp, bid, ask)
                
            # Update inventory history
            self.results.inventory_history.append(
                self.inventory.get_net_position("base")
            )
            
        # Calculate final metrics
        self.results.calculate_metrics()
        return self.results
        
    def _simulate_amm_trades(
        self,
        timestamp: datetime,
        bid: float,
        ask: float
    ):
        """Simulate trades in AMM mode."""
        # Simulate buy order
        if np.random.random() < 0.5:  # 50% chance of buy
            size = np.random.uniform(0.1, 1.0)
            amount_out, price_impact, fee = self.simulator.execute_swap(
                amount_in=size,
                is_x_to_y=True
            )
            
            # Update inventory
            self.inventory.add_position(
                symbol="base",
                size=amount_out,
                price=bid,
                timestamp=timestamp
            )
            
            # Record trade
            self.results.add_trade(
                timestamp=timestamp,
                side="buy",
                price=bid,
                size=amount_out,
                fee=fee,
                pnl=-fee
            )
            
        # Simulate sell order
        if np.random.random() < 0.5:  # 50% chance of sell
            size = np.random.uniform(0.1, 1.0)
            amount_out, price_impact, fee = self.simulator.execute_swap(
                amount_in=size,
                is_x_to_y=False
            )
            
            # Update inventory
            self.inventory.add_position(
                symbol="base",
                size=-amount_out,
                price=ask,
                timestamp=timestamp
            )
            
            # Record trade
            self.results.add_trade(
                timestamp=timestamp,
                side="sell",
                price=ask,
                size=amount_out,
                fee=fee,
                pnl=-fee
            )
            
    def _simulate_lob_trades(
        self,
        timestamp: datetime,
        bid: float,
        ask: float
    ):
        """Simulate trades in LOB mode."""
        # Add market maker orders
        mm_bid_id = f"mm_bid_{timestamp}"
        mm_ask_id = f"mm_ask_{timestamp}"
        
        self.simulator.add_order(
            order_id=mm_bid_id,
            side="buy",
            price=bid,
            size=1.0,
            timestamp=timestamp
        )
        
        self.simulator.add_order(
            order_id=mm_ask_id,
            side="sell",
            price=ask,
            size=1.0,
            timestamp=timestamp
        )
        
        # Simulate market orders
        if np.random.random() < 0.5:  # 50% chance of market order
            side = "buy" if np.random.random() < 0.5 else "sell"
            size = np.random.uniform(0.1, 1.0)
            
            # Add market order
            market_order_id = f"market_{timestamp}"
            self.simulator.add_order(
                order_id=market_order_id,
                side=side,
                price=bid if side == "buy" else ask,
                size=size,
                timestamp=timestamp
            )
            
            # Match orders
            matches = self.simulator.match_orders()
            
            # Process matches
            for buy, sell, price, match_size in matches:
                # Calculate fees
                fee = match_size * 0.001  # 0.1% fee
                
                # Update inventory
                if buy.order_id == mm_bid_id:
                    self.inventory.add_position(
                        symbol="base",
                        size=match_size,
                        price=price,
                        timestamp=timestamp
                    )
                elif sell.order_id == mm_ask_id:
                    self.inventory.add_position(
                        symbol="base",
                        size=-match_size,
                        price=price,
                        timestamp=timestamp
                    )
                    
                # Record trade
                self.results.add_trade(
                    timestamp=timestamp,
                    side="buy" if buy.order_id == mm_bid_id else "sell",
                    price=price,
                    size=match_size,
                    fee=fee,
                    pnl=-fee
                ) 