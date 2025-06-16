# ETHPredict: Function, Dependency, and Library Requirements Table

This table lists all functions needed for full repo operation, their locations, detailed descriptions, internal dependencies, and **NOTES ON POSSIBLE EXTERNAL LIBRARIES OR MODULES** that may not yet be implemented or imported. At the end, I include **key questions to clarify missing or ambiguous parts of the architecture**.

---

| Function Name            | File Location                  | Description & Internal Dependencies                                                                                                                                                                     | Possible External/Unimplemented Libraries or Modules         |
|--------------------------|-------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------|
| setup_experiment         | src/experiment/setup.py        | Sets experiment id/seed, creates folders, initializes logging/checkpointing.                                                                                                                           | `os`, `random`, `logging`, custom checkpoint module         |
| setup_hardware           | src/utils/hardware.py          | Configures GPU, memory, workers, precision if needed.                                                                                                           | `torch`, `cuda`, `tensorflow`, memory management utils      |
| load_and_validate_data   | src/data/loader.py             | Loads data from sources (Binance, Defillama, Santiment), filters and cleans per config.                                                                         | `pandas`, API/data connectors for Binance/Defillama/Santiment|
| construct_bars           | src/data/bar_sampling.py       | Samples bars (time/tick/volume/dollar), applies EWMA, thresholds, etc.                                                                                          | `numpy`, `pandas`, bar sampling utilities                   |
| compute_features         | src/features/engineer.py       | Computes features (RSI, MACD, Bollinger, fractional diff, etc.), applies window/periods, filters by importance/correlation.                                     | `ta-lib`, `numpy`, `scipy`, custom feature functions        |
| train_model              | src/model/trainer.py           | Builds/trains models (XGBoost, LightGBM, CatBoost, LSTM, MLP, GRU). Handles meta-labeling/confidence models per config.                                        | `xgboost`, `lightgbm`, `catboost`, `torch`, `tensorflow`    |
| run_market_maker         | src/strategy/market_maker.py   | Runs market making logic (GLFT, Avellaneda-Stoikov, etc.), dynamic spread/inventory.                                                                           | Custom strategy module, math/statistics libraries           |
| manage_inventory         | src/strategy/inventory.py      | Enforces inventory/risk/timeout/var/drawdown constraints, rebalances as needed.                                                                                 | `numpy`, risk/portfolio libraries                           |
| optimize_bribe           | src/strategy/bribe.py          | Computes bribe based on mode, applies MEV/front-run/sandwich protections.                                                                                       | Custom or external MEV/simulation modules                   |
| run_backtest             | src/backtest/engine.py         | Runs historical simulation with trading costs, slippage, capital constraints. Calls market maker, bribe, inventory.                                            | `pandas`, custom backtest framework, slippage/cost models   |
| evaluate_performance     | src/performance/evaluator.py   | Calculates metrics (Sharpe, drawdown, win rate, info coefficient, etc.), checks thresholds.                                                                    | `numpy`, `scipy`, custom metrics functions                  |
| generate_report          | src/report/generator.py        | Produces HTML/JSON reports, exports trades/positions/P&L with compression if needed.                                                                           | `jinja2`, `markdown`, `json`, `gzip`, reporting templates   |
| save_model               | src/model/io.py                | Serializes/saves model to disk for reuse/deployment.                                                                                                           | `pickle`, `joblib`, model-specific save routines            |
| load_model               | src/model/io.py                | Loads model from disk for evaluation/inference.                                                                                                                | `pickle`, `joblib`                                          |
| run_pipeline             | runner.py (orchestrator)       | Main orchestrator: executes all above functions in order, passing config/data/models as needed, handles errors/logs.                                           | All above modules; CLI/argparse/typer for entrypoint        |

---

## **Possible Missing/Unimplemented Libraries or Components**

- **Data Connectors/APIs** for Binance, Defillama, Santiment (for `load_and_validate_data`)
- **Bar Sampling Utilities** (e.g. dollar bars, tick bars -- often not present in standard libraries)
- **Feature Engineering**: If using indicators like RSI/MACD/Bollinger, need either `ta-lib`, `pandas_ta`, or custom implementations.
- **Deep Learning Frameworks**: `torch` (PyTorch) or `tensorflow` for MLP/LSTM/GRU models.
- **Tree Models**: `xgboost`, `lightgbm`, `catboost` for non-sequential ML.
- **MEV/Bribe Simulation**: Custom simulation logic or specialized libraries for Ethereum MEV modeling.
- **Backtest Engine**: No standard library, may need custom simulation for inventory/market-making/backtest loop.
- **Reporting**: `jinja2` for HTML, `markdown` for markdown, `json` for exports, `gzip` for compression.
- **Risk/Portfolio**: Custom or third-party portfolio/risk metrics.
- **Checkpointing/Experiment Tracking**: Could use `mlflow`, `wandb`, or homegrown logic.
- **CLI Framework**: `typer`, `argparse`, or `click` for command-line entry.

---
