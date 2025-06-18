# --- Standard Library Imports ---
import os
import sys
import json
import time
import logging
import threading
import argparse
import pickle
from pathlib import Path
from typing import List, Dict, Any, Callable, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import multiprocessing
from contextlib import contextmanager

# --- Third-Party Imports ---
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import yaml
from pydantic import BaseModel as PydanticBaseModel
import optuna
from tqdm import tqdm
from joblib import Parallel, delayed
from statsmodels.tsa.stattools import adfuller
from loguru import logger

# --- Logging Context ---
@contextmanager
def init_logging(cfg):
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    logger.add(log_dir / "pipeline.jsonl", level="INFO", rotation="100 MB", serialize=True)
    try:
        yield logger
    finally:
        logger.info("run complete")

# --- CSV Loader and Validation ---
class DataValidationError(Exception):
    """Raised when a CSV file does not conform to the schema."""

def validate_csvs(raw_dir: Path, schema_path: Path, rejects_dir: Path) -> List[Path]:
    schema = json.load(open(schema_path))
    required_cols = list(schema.keys())
    dtype_map = {k: np.dtype(v) for k, v in schema.items()}
    valid_files: List[Path] = []
    rejects_dir.mkdir(parents=True, exist_ok=True)
    for csv_file in sorted(raw_dir.glob("*.csv")):
        try:
            chunks = pd.read_csv(
                csv_file,
                header=None,
                names=required_cols,
                usecols=range(len(required_cols)),
                chunksize=10000,
            )
            for chunk in chunks:
                for col in required_cols:
                    if chunk[col].dtype != dtype_map[col]:
                        raise DataValidationError(
                            f"Column {col} has {chunk[col].dtype}, expected {dtype_map[col]}"
                        )
            valid_files.append(csv_file)
        except Exception as exc:
            logging.error("CSV validation failed for %s: %s", csv_file, exc)
            (rejects_dir / csv_file.name).write_text("invalid")
    return valid_files

# --- Feature Engineering and Data Preprocessing ---
def sample_bars(df: pd.DataFrame, bar_type: str) -> pd.DataFrame:
    if bar_type == "tick":
        threshold = 1000
        metric = pd.Series(range(len(df))) + 1
    elif bar_type == "volume":
        threshold = 10000
        metric = df["volume"].cumsum()
    else:  # dollar
        threshold = 10_000_000
        metric = (df["close"] * df["volume"]).cumsum()
    groups = metric // threshold
    bars = df.groupby(groups).agg(
        timestamp=("timestamp", "last"),
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    )
    return bars.reset_index(drop=True)

def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("timestamp")
    df["return"] = df["close"].pct_change().fillna(0)
    df["rolling_vol"] = df["return"].rolling(24).std().fillna(0)
    return df

def save_features(asset: str, bar_type: str, df: pd.DataFrame) -> Path:
    date = pd.to_datetime(df["timestamp"].iloc[0], unit="ms").strftime("%Y%m%d")
    out_dir = Path("build/features") / asset / bar_type
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{date}.parquet"
    df.to_parquet(out_path)
    return out_path

class DataPreprocessor:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.granularities = ["30s", "1m", "5m", "1h"]
    def load_price_data(self) -> Dict[str, pd.DataFrame]:
        data = {}
        for gran in self.granularities:
            candidates = list(self.data_dir.glob(f"ETHUSDT-{gran}-*.csv")) + \
                         list((self.data_dir / "raw").glob(f"ETHUSDT-{gran}-*.csv"))
            frames = []
            binance_columns = [
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ]
            for csv_path in candidates:
                try:
                    df = pd.read_csv(csv_path, header=None, names=binance_columns)
                    df["timestamp"] = pd.to_datetime(df["open_time"], unit='ms', errors='coerce')
                    if df["timestamp"].dt.tz is not None:
                        df["timestamp"] = df["timestamp"].dt.tz_localize(None)
                    frames.append(df)
                except Exception as e:
                    print(f"Warning: Could not load {csv_path}: {e}")
            if frames:
                price = pd.concat(frames, ignore_index=True)
                price = price.drop_duplicates(subset="timestamp").sort_values("timestamp")
                data[gran] = price
        return data
    def load_chain_tvl(self) -> Optional[pd.DataFrame]:
        chain_tvl_files = list(self.data_dir.glob("defillama_eth_chain_tvl*.csv"))
        for f in chain_tvl_files:
            df = pd.read_csv(f)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            if df["timestamp"].dt.tz is not None:
                df["timestamp"] = df["timestamp"].dt.tz_localize(None)
            return df
        return None
    def load_santiment(self) -> Optional[pd.DataFrame]:
        santiment_files = list(self.data_dir.glob("santiment_metrics*.csv"))
        for f in santiment_files:
            df = pd.read_csv(f)
            if "datetime" in df.columns:
                df["timestamp"] = pd.to_datetime(df["datetime"])
            else:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            if df["timestamp"].dt.tz is not None:
                df["timestamp"] = df["timestamp"].dt.tz_localize(None)
            return df
        return None
    def fracdiff(self, series: pd.Series, d: float, thres: float = 0.01) -> pd.Series:
        w = [1.0]
        for k in range(1, len(series)):
            w_ = -w[-1] * (d - k + 1) / k
            if abs(w_) < thres:
                break
            w.append(w_)
        w = np.array(w[::-1])
        output = series.copy() * np.nan
        for idx in range(len(w)-1, len(series)):
            output.iloc[idx] = np.dot(w, series.iloc[idx-len(w)+1:idx+1])
        return output
    def find_optimal_d(self, series: pd.Series, max_d: float = 1.0, corr_threshold: float = 0.97) -> float:
        best_d = 0.0
        for d in np.arange(0.1, max_d + 0.01, 0.1):
            diffed = self.fracdiff(series, d).dropna()
            if len(diffed) < 10:
                continue
            try:
                adf_stat, pvalue, *_ = adfuller(diffed)
                corr = np.corrcoef(series.dropna()[-len(diffed):], diffed)[0, 1]
                if pvalue < 0.05 and corr > corr_threshold:
                    best_d = d
                    break
            except Exception:
                continue
        return best_d
    def compute_entropy(self, returns: pd.Series, window: int = 24) -> pd.Series:
        def shannon_entropy(x):
            hist, _ = np.histogram(x, bins=10, density=True)
            hist = hist[hist > 0]
            return -np.sum(hist * np.log(hist))
        return returns.rolling(window, min_periods=window).apply(shannon_entropy, raw=True)
    def cusum_flag(self, series: pd.Series, threshold: float = 2.0) -> pd.Series:
        mean_val = series.mean()
        cusum = (series - mean_val).cumsum()
        flags = (np.abs(cusum) > threshold * series.std()).astype(int)
        return flags
    def sadf_flag(self, series: pd.Series, min_window: int = 24, max_window: int = 100) -> pd.Series:
        flags = pd.Series(0, index=series.index)
        for i in range(max_window, len(series)):
            stats = []
            for w in range(min_window, max_window+1, 10):
                window_data = series.iloc[i-w+1:i+1]
                if window_data.isnull().any() or len(window_data) < w:
                    continue
                try:
                    adf_stat, *_ = adfuller(window_data)
                    stats.append(adf_stat)
                except Exception:
                    continue
            if stats and max(stats) > -1.0:
                flags.iloc[i] = 1
        return flags
    def volatility_regime(self, returns: pd.Series, window: int = 24) -> pd.Series:
        vol = returns.rolling(window).std()
        low, high = vol.quantile(0.33), vol.quantile(0.67)
        regime = pd.Series(1, index=vol.index)
        regime[vol <= low] = 0
        regime[vol >= high] = 2
        return regime
    def parkinson_vol(self, high: pd.Series, low: pd.Series, window: int = 24) -> pd.Series:
        pv = np.sqrt(0.25 * np.log(2) * (np.log(high / low) ** 2))
        return pv.rolling(window).mean()
    def prepare_features(self, granularity: str, sequence_length: int = 24) -> Tuple[np.ndarray, np.ndarray]:
        price_data = self.load_price_data().get(granularity)
        chain_tvl = self.load_chain_tvl()
        santiment = self.load_santiment()
        if price_data is None or chain_tvl is None:
            raise ValueError(f"Price and chain TVL data required for {granularity}")
        price_data = price_data.set_index("timestamp").sort_index()
        freq = pd.infer_freq(price_data.index[:100]) or granularity
        chain_hourly = chain_tvl.set_index("timestamp").resample(freq).ffill()
        if santiment is not None:
            santiment_hourly = santiment.set_index("timestamp").resample(freq).ffill()
        else:
            santiment_hourly = pd.DataFrame(index=chain_hourly.index)
        merged = (
            price_data
            .join(chain_hourly[["tvl_usd"]], how="left")
            .join(santiment_hourly, how="left")
            .sort_index()
        )
        merged.fillna(method="ffill", inplace=True)
        merged["price_return"] = np.log(merged["close"]).diff().fillna(0)
        merged["tvl_change"] = merged["tvl_usd"].pct_change().fillna(0)
        merged["price_tvl_ratio"] = merged["close"] / merged["tvl_usd"]
        merged["return_vol_24"] = merged["price_return"].rolling(24).std().fillna(0)
        merged["volume_tvl_ratio"] = merged["volume"] / merged["tvl_usd"]
        merged["address_growth"] = merged.get("daily_active_addresses_value", pd.Series(0, index=merged.index)).pct_change().fillna(0)
        merged["social_dominance_change"] = merged.get("social_dominance_total_value", pd.Series(0, index=merged.index)).pct_change().fillna(0)
        merged["mcap_tvl_ratio"] = merged.get("marketcap_usd_value", pd.Series(0, index=merged.index)) / merged["tvl_usd"]
        close_d = self.find_optimal_d(merged["close"])
        merged["fracdiff_close"] = self.fracdiff(merged["close"], close_d)
        merged["return_entropy_24"] = self.compute_entropy(merged["price_return"], 24)
        merged["return_entropy_168"] = self.compute_entropy(merged["price_return"], 168)
        merged["cusum_flag"] = self.cusum_flag(merged["price_return"])
        merged["sadf_flag"] = self.sadf_flag(merged["close"])
        merged["vol_regime"] = self.volatility_regime(merged["price_return"])
        merged["return_vol_1"] = merged["price_return"].rolling(1).std().fillna(0)
        merged["return_vol_168"] = merged["price_return"].rolling(168).std().fillna(0)
        merged["parkinson_vol"] = self.parkinson_vol(merged["high"], merged["low"], 24).fillna(0)
        feature_cols = [
            "close", "volume", "tvl_usd", "price_return", "tvl_change", "price_tvl_ratio",
            "return_vol_24", "volume_tvl_ratio", "address_growth", "social_dominance_change",
            "mcap_tvl_ratio", "fracdiff_close", "return_entropy_24", "return_entropy_168",
            "cusum_flag", "sadf_flag", "vol_regime", "return_vol_1", "return_vol_168", "parkinson_vol"
        ]
        features = merged[feature_cols].fillna(0).values
        targets = merged[["tvl_usd"]].values
        features_tensor = torch.FloatTensor(features)
        targets_tensor = torch.FloatTensor(targets)
        features_mean = features_tensor.mean(dim=0)
        features_std = features_tensor.std(dim=0) + 1e-8
        scaled_features = (features_tensor - features_mean) / features_std
        targets_min, targets_max = targets_tensor.min(), targets_tensor.max()
        scaled_targets = (targets_tensor - targets_min) / (targets_max - targets_min + 1e-8)
        X, y = [], []
        for i in range(len(merged) - sequence_length):
            X.append(scaled_features[i : i + sequence_length])
            y.append(scaled_targets[i + sequence_length])
        return np.array(X), np.array(y)
    def get_all_granularity_features(self, sequence_length: int = 24) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        output = {}
        for gran in self.granularities:
            try:
                X, y = self.prepare_features(granularity=gran, sequence_length=sequence_length)
                output[gran] = (X, y)
            except Exception as e:
                print(f"Could not prepare features for {gran}: {e}")
        return output

def save_numpy_arrays(X, y, out_dir: Path, prefix: str):
    np.save(out_dir / f"{prefix}_X.npy", X)
    np.save(out_dir / f"{prefix}_y.npy", y)

def save_pickle(obj, out_dir: Path, filename: str):
    with open(out_dir / filename, "wb") as f:
        pickle.dump(obj, f)

# --- Model Definitions ---
class LSTMModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])

class MetaMLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        layers = []
        prev_size = input_size
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
        layers.append(nn.Linear(hidden_size, 1))
        self.mlp = nn.Sequential(*layers)
    def forward(self, base_pred: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        x = torch.cat([base_pred, features], dim=1)
        return self.mlp(x)

class ConfidenceGRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gru_out, _ = self.gru(x)
        return self.fc(gru_out[:, -1, :])

# --- Trainer Logic ---
def _train_epoch(model: nn.Module, loader: DataLoader, criterion, optimizer):
    model.train()
    loss_total = 0.0
    for x, y in loader:
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out.squeeze(), y.squeeze())
        loss.backward()
        optimizer.step()
        loss_total += loss.item()
    return loss_total / len(loader)

def train_pytorch(df: pd.DataFrame, out_dir: Path, model_cls=LSTMModel):
    features = torch.tensor(df[["return", "rolling_vol"]].values, dtype=torch.float32)
    targets = torch.tensor(df["return"].shift(-1).fillna(0).values, dtype=torch.float32)
    dataset = TensorDataset(features.unsqueeze(1), targets)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = model_cls(features.shape[1])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(3):
        _train_epoch(model, loader, criterion, optimizer)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    path = out_dir / f"model_{ts}.pt"
    torch.save(model.state_dict(), path)
    return path

def train_ray(df: pd.DataFrame, out_dir: Path):
    import ray
    @ray.remote
    def _remote_train(data: pd.DataFrame) -> str:
        return str(train_pytorch(data, out_dir))
    ray.init(ignore_reinit_error=True)
    ref = _remote_train.remote(df)
    result = ray.get(ref)
    ray.shutdown()
    return Path(result)

FRAMEWORKS = {
    "pytorch": train_pytorch,
    "ray": train_ray,
}
def train(df: pd.DataFrame, framework: str, out_dir: Path) -> Path:
    if framework not in FRAMEWORKS:
        raise NotImplementedError(framework)
    trainer_fn = FRAMEWORKS[framework]
    return trainer_fn(df, out_dir)

# --- Backtest Logic ---
from dataclasses import dataclass as dc_dataclass
class BacktestParams:
    def __init__(self, start_date, end_date, initial_capital, seed, gamma, inventory_limit, quote_spread):
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.seed = seed
        self.gamma = gamma
        self.inventory_limit = inventory_limit
        self.quote_spread = quote_spread
class BacktestResult:
    def __init__(self):
        self.trades = []
        self.pnl_history = []
        self.inventory_history = []
        self.spread_history = []
        self.metrics = {}
    def add_trade(self, timestamp, side, price, size, fee, pnl):
        self.trades.append({"timestamp": timestamp, "side": side, "price": price, "size": size, "fee": fee, "pnl": pnl})
    def calculate_metrics(self):
        if not self.trades:
            return {}
        df = pd.DataFrame(self.trades)
        total_pnl = df["pnl"].sum()
        total_fees = df["fee"].sum()
        num_trades = len(df)
        returns = pd.Series(self.pnl_history).pct_change().dropna()
        self.metrics = {
            "total_pnl": total_pnl,
            "total_fees": total_fees,
            "net_pnl": total_pnl - total_fees,
            "num_trades": num_trades,
            "win_rate": (df["pnl"] > 0).mean(),
            "avg_trade_pnl": df["pnl"].mean(),
            "sharpe_ratio": returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else 0,
            "max_drawdown": self._calculate_max_drawdown(),
            "avg_spread": np.mean(self.spread_history) if self.spread_history else 0,
            "avg_inventory": np.mean(self.inventory_history) if self.inventory_history else 0
        }
        return self.metrics
    def _calculate_max_drawdown(self):
        cumulative = pd.Series(self.pnl_history).cumsum()
        rolling_max = cumulative.expanding().max()
        drawdowns = (cumulative - rolling_max) / rolling_max
        return abs(drawdowns.min())
class GLFTParams:
    def __init__(self, gamma, kappa, sigma, dt, max_inventory, min_spread):
        self.gamma = gamma
        self.kappa = kappa
        self.sigma = sigma
        self.dt = dt
        self.max_inventory = max_inventory
        self.min_spread = min_spread
class GLFTQuoteCalculator:
    def __init__(self, params):
        self.params = params
    def calculate_quotes(self, mid_price, inventory, volatility=None):
        sigma = volatility or self.params.sigma
        skew = self.params.kappa * inventory / self.params.max_inventory
        base_spread = np.sqrt(self.params.gamma * sigma**2 * self.params.dt)
        spread = base_spread * (1 + skew)
        spread = max(spread, self.params.min_spread / 10000)
        bid = mid_price * (1 - spread / 2)
        ask = mid_price * (1 + spread / 2)
        return bid, ask
    def calculate_optimal_quotes(self, mid_price, inventory, volatility=None, target_inventory=0.0):
        bid, ask = self.calculate_quotes(mid_price, inventory, volatility)
        diff = inventory - target_inventory
        if diff:
            adj = self.params.kappa * diff / self.params.max_inventory
            bid *= 1 - adj
            ask *= 1 + adj
        return bid, ask
class Position:
    def __init__(self, size, entry_price, timestamp, pnl=0.0):
        self.size = size
        self.entry_price = entry_price
        self.timestamp = timestamp
        self.pnl = pnl
class InventoryBook:
    def __init__(self, max_position, max_drawdown, target_inventory=0.0):
        self.max_position = max_position
        self.max_drawdown = max_drawdown
        self.target_inventory = target_inventory
        self.positions = {}
        self.total_pnl = 0.0
        self.peak_pnl = 0.0
    def add_position(self, symbol, size, price, timestamp):
        current_size = self.get_net_position(symbol)
        if abs(current_size + size) > self.max_position:
            return False
        position = Position(size, price, timestamp)
        if symbol in self.positions:
            current = self.positions[symbol]
            new_size = current.size + size
            if abs(new_size) < 1e-8:
                del self.positions[symbol]
            else:
                current.entry_price = (current.entry_price * current.size + price * size) / new_size
                current.size = new_size
        else:
            self.positions[symbol] = position
        return True
    def update_pnl(self, symbol, current_price, timestamp):
        if symbol not in self.positions:
            return 0.0
        position = self.positions[symbol]
        position.pnl = position.size * (current_price - position.entry_price)
        self.total_pnl = sum(p.pnl for p in self.positions.values())
        self.peak_pnl = max(self.peak_pnl, self.total_pnl)
        return position.pnl
    def get_net_position(self, symbol):
        return self.positions.get(symbol, Position(0, 0, datetime.now())).size
    def get_total_position(self):
        return sum(abs(p.size) for p in self.positions.values())
    def get_drawdown(self):
        if self.peak_pnl == 0:
            return 0.0
        return (self.peak_pnl - self.total_pnl) / abs(self.peak_pnl)
    def check_risk_limits(self):
        total_pos = self.get_total_position()
        if total_pos > self.max_position:
            return False, f"Total position {total_pos} exceeds limit {self.max_position}"
        drawdown = self.get_drawdown()
        if drawdown > self.max_drawdown:
            return False, f"Drawdown {drawdown:.2%} exceeds limit {self.max_drawdown:.2%}"
        return True, "OK"
class GLFTBacktester:
    def __init__(self, params, market_maker, inventory):
        self.params = params
        self.market_maker = market_maker
        self.inventory = inventory
        self.results = BacktestResult()
        np.random.seed(params.seed)
    def run(self, market_data, predicted_prices, volatility=None):
        for timestamp, row in market_data.iterrows():
            mid_price = row["price"]
            predicted_price = predicted_prices[timestamp]
            current_vol = volatility[timestamp] if volatility is not None else None
            bid, ask = self.market_maker.calculate_optimal_quotes(
                mid_price=predicted_price,
                inventory=self.inventory.get_net_position("base"),
                volatility=current_vol
            )
            spread = (ask - bid) / mid_price
            self.results.spread_history.append(spread)
            price_diff = predicted_price - mid_price
            buy_prob = 0.5 + 0.3 * np.tanh(price_diff)
            if np.random.random() < buy_prob:
                size = np.random.uniform(0.1, 1.0)
                fee = size * bid * 0.001
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
            sell_prob = 0.5 - 0.3 * np.tanh(price_diff)
            if np.random.random() < sell_prob:
                size = np.random.uniform(0.1, 1.0)
                fee = size * ask * 0.001
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
            self.results.inventory_history.append(
                self.inventory.get_net_position("base")
            )
        self.results.calculate_metrics()
        return self.results

# --- Experiment Orchestrator ---
@dataclass
class TrialResult:
    trial_id: int
    params: Dict[str, Any]
    metrics: Dict[str, float]
    status: str
    error: Optional[str] = None
    start_time: float = 0.0
    end_time: float = 0.0
    gpu_id: Optional[int] = None
class ExperimentOrchestrator:
    def __init__(self, search_space, search_type="grid", n_trials=1000, n_workers=None, gpu_ids=None, checkpoint_dir="experiment_checkpoints", metric_name="loss", maximize=False):
        self.search_space = search_space
        self.search_type = search_type.lower()
        self.n_trials = n_trials
        self.n_workers = n_workers or multiprocessing.cpu_count()
        self.gpu_ids = gpu_ids or []
        self.checkpoint_dir = Path(checkpoint_dir)
        self.metric_name = metric_name
        self.maximize = maximize
        self.trial_results = {}
        self.current_trial_id = 0
        self.gpu_semaphore = threading.Semaphore(len(self.gpu_ids)) if self.gpu_ids else None
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.checkpoint_dir / 'experiment.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self._load_checkpoint()
    def _generate_grid_search_params(self):
        param_values = []
        for param_name, param_config in self.search_space.items():
            if param_config["type"] == "continuous":
                values = np.linspace(
                    param_config["range"][0],
                    param_config["range"][1],
                    param_config.get("n_points", 10)
                )
            elif param_config["type"] == "discrete":
                values = param_config["values"]
            elif param_config["type"] == "categorical":
                values = param_config["values"]
            else:
                raise ValueError(f"Unknown parameter type: {param_config['type']}")
            param_values.append((param_name, values))
        from itertools import product
        param_names = [p[0] for p in param_values]
        param_combinations = list(product(*[p[1] for p in param_values]))
        return [dict(zip(param_names, combo)) for combo in param_combinations]
    def _generate_random_search_params(self):
        params_list = []
        for _ in range(self.n_trials):
            params = {}
            for param_name, param_config in self.search_space.items():
                if param_config["type"] == "continuous":
                    params[param_name] = np.random.uniform(
                        param_config["range"][0],
                        param_config["range"][1]
                    )
                elif param_config["type"] == "discrete":
                    params[param_name] = np.random.choice(param_config["values"])
                elif param_config["type"] == "categorical":
                    params[param_name] = np.random.choice(param_config["values"])
            params_list.append(params)
        return params_list
    def _bayesian_search_objective(self, trial):
        params = {}
        for param_name, param_config in self.search_space.items():
            if param_config["type"] == "continuous":
                params[param_name] = trial.suggest_float(
                    param_name,
                    param_config["range"][0],
                    param_config["range"][1]
                )
            elif param_config["type"] == "discrete":
                params[param_name] = trial.suggest_int(
                    param_name,
                    min(param_config["values"]),
                    max(param_config["values"])
                )
            elif param_config["type"] == "categorical":
                params[param_name] = trial.suggest_categorical(
                    param_name,
                    param_config["values"]
                )
        result = self._run_single_trial(params)
        return result.metrics[self.metric_name]
    def _allocate_gpu(self):
        if not self.gpu_ids or not self.gpu_semaphore:
            return None
        self.gpu_semaphore.acquire()
        available_gpu = next(
            (gpu_id for gpu_id in self.gpu_ids 
             if not torch.cuda.is_initialized() or 
             torch.cuda.memory_allocated(gpu_id) == 0),
            self.gpu_ids[0]
        )
        return available_gpu
    def _release_gpu(self, gpu_id):
        if gpu_id is not None and self.gpu_semaphore:
            self.gpu_semaphore.release()
    def _run_single_trial(self, params):
        trial_id = self.current_trial_id
        self.current_trial_id += 1
        result = TrialResult(
            trial_id=trial_id,
            params=params,
            metrics={},
            status="in_progress",
            start_time=time.time()
        )
        try:
            gpu_id = self._allocate_gpu()
            result.gpu_id = gpu_id
            metrics = self.experiment_fn(params, gpu_id)
            result.metrics = metrics
            result.status = "success"
        except Exception as e:
            result.status = "failed"
            result.error = str(e)
            self.logger.error(f"Trial {trial_id} failed: {str(e)}")
        finally:
            result.end_time = time.time()
            self._release_gpu(gpu_id)
            self.trial_results[trial_id] = result
            self._save_checkpoint()
        return result
    def _save_checkpoint(self):
        checkpoint = {
            "trial_results": {str(k): asdict(v) for k, v in self.trial_results.items()},
            "current_trial_id": self.current_trial_id,
            "search_space": self.search_space,
            "search_type": self.search_type,
            "n_trials": self.n_trials
        }
        with open(self.checkpoint_dir / "checkpoint.json", "w") as f:
            json.dump(checkpoint, f, indent=2)
    def _load_checkpoint(self):
        checkpoint_file = self.checkpoint_dir / "checkpoint.json"
        if checkpoint_file.exists():
            with open(checkpoint_file, "r") as f:
                checkpoint = json.load(f)
            self.trial_results = {
                int(k): TrialResult(**v) 
                for k, v in checkpoint["trial_results"].items()
            }
            self.current_trial_id = checkpoint["current_trial_id"]
            self.logger.info(f"Loaded checkpoint with {len(self.trial_results)} completed trials")
    def run_experiment(self, experiment_fn):
        self.experiment_fn = experiment_fn
        if self.search_type == "grid":
            param_combinations = self._generate_grid_search_params()
            self.n_trials = len(param_combinations)
        elif self.search_type == "random":
            param_combinations = self._generate_random_search_params()
        elif self.search_type == "bayesian":
            study = optuna.create_study(
                direction="maximize" if self.maximize else "minimize"
            )
            study.optimize(
                self._bayesian_search_objective,
                n_trials=self.n_trials,
                n_jobs=self.n_workers
            )
            return study
        else:
            raise ValueError(f"Unknown search type: {self.search_type}")
        with tqdm(total=self.n_trials, desc="Running trials") as pbar:
            results = Parallel(n_jobs=self.n_workers)(
                delayed(self._run_single_trial)(params)
                for params in param_combinations
            )
            pbar.update(len(results))
        return self.trial_results
    def get_best_trial(self):
        if not self.trial_results:
            return None
        successful_trials = [
            t for t in self.trial_results.values()
            if t.status == "success"
        ]
        if not successful_trials:
            return None
        return max(
            successful_trials,
            key=lambda x: x.metrics[self.metric_name] if self.maximize 
            else -x.metrics[self.metric_name]
        )
    def get_summary(self):
        successful_trials = [t for t in self.trial_results.values() if t.status == "success"]
        failed_trials = [t for t in self.trial_results.values() if t.status == "failed"]
        return {
            "total_trials": len(self.trial_results),
            "successful_trials": len(successful_trials),
            "failed_trials": len(failed_trials),
            "best_trial": asdict(self.get_best_trial()) if self.get_best_trial() else None,
            "average_metrics": {
                metric: np.mean([t.metrics[metric] for t in successful_trials])
                for metric in successful_trials[0].metrics.keys()
            } if successful_trials else {}
        }

# --- Pipeline Runner and Main ---
class Config(PydanticBaseModel):
    dataset: str
    trainer: dict
    market_maker: dict
    foundry: dict
    flags: dict

def load_config(path: str) -> Config:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return Config(**cfg)

def ingest_raw_csvs(cfg: Config):
    raw_dir = Path("data/raw")
    validate_csvs(raw_dir, Path("data/schema.json"), Path("rejects"))

def build_bars_and_features(cfg: Config, feature_mode: str = "full", sequence_length: int = 24) -> pd.DataFrame:
    csv_file = next(Path("data/raw").glob("*.csv"))
    df = pd.read_csv(csv_file, header=None, names=["timestamp", "open", "high", "low", "close", "volume"], usecols=range(6))
    bars = sample_bars(df, "tick")
    if feature_mode == "simple":
        features = generate_features(bars)
        save_features(cfg.dataset, "tick", features)
        return features
    else:
        preprocessor = DataPreprocessor(data_dir="data")
        all_granularity_data = preprocessor.get_all_granularity_features(sequence_length=sequence_length)
        out_dir = Path("data/processed_features")
        out_dir.mkdir(exist_ok=True)
        for granularity, (X, y) in all_granularity_data.items():
            prefix = f"{granularity}_seq{sequence_length}"
            print(f"Saving features for {granularity} to {out_dir} as {prefix}_X.npy and {prefix}_y.npy")
            save_numpy_arrays(X, y, out_dir, prefix)
            meta = {
                "X_shape": X.shape,
                "y_shape": y.shape,
                "sequence_length": sequence_length,
                "granularity": granularity,
            }
            save_pickle(meta, out_dir, f"{prefix}_meta.pkl")
        if "1h" in all_granularity_data:
            X, y = all_granularity_data["1h"]
            features_df = pd.DataFrame(X.reshape(X.shape[0], -1))
            return features_df
        else:
            gran, (X, y) = next(iter(all_granularity_data.items()))
            features_df = pd.DataFrame(X.reshape(X.shape[0], -1))
            return features_df

def maybe_train_models(cfg: Config, features: pd.DataFrame) -> Path:
    return train(features, cfg.trainer.get("framework", "pytorch"), Path("artifacts/models"))

def run_backtest_and_sim(cfg: Config, features: pd.DataFrame, model_path: Path):
    if cfg.flags.get("enable_dex_sim"):
        raise NotImplementedError("DEX simulation")
    preds = pd.Series(0, index=features.index)
    # You may want to load your model and generate predictions here
    # For now, we use zero predictions as a placeholder
    # To use GLFTBacktester, you would need to construct the required objects and call .run()
    # Example (pseudo):
    # params = BacktestParams(...)
    # market_maker = GLFTQuoteCalculator(...)
    # inventory = InventoryBook(...)
    # backtester = GLFTBacktester(params, market_maker, inventory)
    # result = backtester.run(features, preds)
    # print(result.metrics)
    # For now, just simulate a basic backtest
    # run_backtest(features, preds, Path("results") / f"run-{int(datetime.now().timestamp())}")
    pass

def summarise_results(cfg: Config):
    summ = Path("results/summary.csv")
    summ.touch(exist_ok=True)
    with summ.open("a") as f:
        f.write(f"{datetime.now().isoformat()},ok\n")

def main():
    cfg = load_config(sys.argv[1])
    with init_logging(cfg):
        ingest_raw_csvs(cfg)
        features = build_bars_and_features(cfg, feature_mode="full", sequence_length=24)
        model_path = maybe_train_models(cfg, features)
        run_backtest_and_sim(cfg, features, model_path)
        summarise_results(cfg)

if __name__ == "__main__":
    main()
