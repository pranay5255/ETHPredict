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

## **Questions to Clarify/Find Missing Pieces**

1. **Data Connectors**:  
   - Are connectors for Binance, Defillama, and Santiment data sources implemented? If not, do you want them as live API, CSV/Parquet, or both?
2. **Bar Sampling**:  
   - Do you already have utilities for advanced bar sampling (dollar, tick, EWMA), or should these be built from scratch?
3. **Feature Library**:  
   - Should we use `ta-lib`/`pandas_ta` for indicators, or implement all technical indicators manually?
4. **Model Frameworks**:  
   - Are you committed to PyTorch, TensorFlow, or a mix for deep learning? (Needed for MLP/LSTM/GRU.)
5. **Market Making Logic**:  
   - Is there existing code for GLFT, Avellaneda-Stoikov, etc., or is this all to be implemented?
6. **MEV/Bribe Simulation**:  
   - Is there any external or in-house library for MEV protection, or do we need to model it from scratch?
7. **Backtesting Engine**:  
   - Is there a homegrown backtest engine, or should we adapt from `backtrader`, `zipline`, or build a custom one?
8. **Reporting**:  
   - Should HTML/JSON reports follow a specific template or style? Do you need any analytics dashboard integration?
9. **Experiment Tracking**:  
   - Should we integrate with something like MLflow/W&B, or keep all tracking local (CSV/JSON/logs)?
10. **Deployment/Serving**:  
    - Are there requirements for model deployment (API server, batch, etc.)? Should we add this to the scope?
11. **Testing/Validation**:  
    - Will you provide sample datasets and expected outputs for module/unit testing, or should we generate synthetic data for tests?
12. **Configuration Validation**:  
    - Is the current schema validation sufficient, or do you expect more complex runtime config checks?

---

Answers:

code and repopulate markdown table after these answers to questions in "Questions to clarify" section
1. Uses CSVs already present in the data folder in csv format. Will be changed into smaller granularity later manually. It should show errors for non-compatibility.
2. in @pranay5255/ETHPredict "src/data/featureGen.py" and other file to persist features.
3. most are implemented don't add any other fetaures
4. committed to only pytorch, ray, vllm and sglang. Whichever is easiest to orchestrate.
5. in @pranay5255/ETHPredict "src/market-maker/glft.py".
6. in @pranay5255/ETHPredict "src/market-maker/bribe.py". Only use foundry to call and simulate functions for eth like swap from main contracts which will be provided in the "configs/config.yml" file. Where its currently not present. Modify and research it to place it there.
7. Backtesting is implemnented but not used in simulations beacause some of it has not been implemented. Only backtest for price data and ignore DEX simulations.
8. mot required explicitly only a detailed log files of everything that happens with tests so i can debug them. 
9. all tracking in logs,csvs only in local
10. no requirement for model deployment but only on llocal and to be used with other services. Write docker services orchestrating this backend where foundry and on-chain state is separate and modelling,backtesting and training are in one instance. 
11. some sample datasets are provided in @pranay5255/ETHPredict "data/" folder. Some other datasets will be given but testing, logging and reporting must all be combined in one single file.
12. simple check and reporting of data schema and types is required. 

 

**Please answer or clarify the above points so I can provide a more precise implementation plan and highlight any further missing dependencies or design gaps!**
