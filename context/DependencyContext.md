# ETHPredict: Function, Dependency, and Library Requirements Table

This table lists all functions needed for full repo operation, their locations, detailed descriptions, internal dependencies, and **NOTES ON POSSIBLE EXTERNAL LIBRARIES OR MODULES** that may not yet be implemented or imported. At the end, I include **key questions to clarify missing or ambiguous parts of the architecture**.

---

| Function Name            | File Location                  | Description & Internal Dependencies                                                                                                                                                                     | Possible External/Unimplemented Libraries or Modules         |
|--------------------------|-------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------|
| setup_experiment         | src/main.py                    | Sets experiment id/seed, creates folders, initializes logging/checkpointing.                                                                                                                           | `os`, `random`, `logging`                                   |
| setup_hardware           | src/utils/logging.py           | Configures logging, hardware info if needed.                                                                                                                    | `logging`, `os`                                             |
| load_and_validate_data   | src/data/features_all.py       | Loads data from CSVs, filters and cleans per config. Errors if incompatible.                                                                                    | `pandas`, CSV readers                                       |
| compute_features         | src/data/features_all.py, src/features/labeling.py | Computes features (most indicators implemented), applies window/periods, labeling (triple-barrier, meta-labeling, etc.).                                        | `numpy`, `pandas`, custom feature functions                 |
| train_model              | src/models/model.py, src/training/trainer.py       | Builds/trains models (LSTM, MLP, GRU, ensemble). Handles meta-labeling/confidence models per config. Only PyTorch, Ray, vllm, sglang used.                      | `torch`, `ray`, `vllm`, `sglang`                           |
| run_market_maker         | src/market_maker/glft.py       | Runs market making logic (GLFT, inventory skew, etc.).                                                                                                          | Custom strategy module, math/statistics libraries           |
| manage_inventory         | src/market_maker/inventory.py  | Enforces inventory/risk/timeout/var/drawdown constraints, rebalances as needed.                                                                                 | `numpy`, risk/portfolio libraries                           |
| run_backtest             | src/simulator/backtest.py      | Runs historical simulation with trading costs, slippage, capital constraints. Calls market maker, inventory. Only for price data, not DEX.                      | `pandas`, custom backtest framework                         |
| evaluate_performance     | src/utils/metrics.py           | Calculates metrics (Sharpe, drawdown, win rate, info coefficient, etc.), checks thresholds.                                                                    | `numpy`, `scipy`, custom metrics functions                  |
| generate_report          | src/utils/logger.py            | Produces log files and CSVs of all results, no HTML/JSON reporting required.                                                                                   | `logging`, `csv`                                            |
| save_model               | src/models/model.py            | Serializes/saves model to disk for reuse/deployment.                                                                                                           | `pickle`, model-specific save routines                      |
| load_model               | src/models/model.py            | Loads model from disk for evaluation/inference.                                                                                                                | `pickle`                                                    |
| run_pipeline             | src/orchestrator/main.py       | Main orchestrator: executes all above functions in order, passing config/data/models as needed, handles errors/logs.                                           | All above modules; CLI/argparse/typer for entrypoint        |
| config_loader            | src/config/loader.py           | Loads and validates YAML/JSON config files.                                                                                                                    | `yaml`, `jsonschema`                                        |
| integration_layer        | src/adaptor/adapter.py         | Handles integration and adaptation between modules.                                                                                                            | None                                                        |

---

## **Possible Missing/Unimplemented Libraries or Components**

- **Data Connectors/APIs**: Not needed, all data is loaded from CSVs in the data folder. Errors if incompatible.
- **Feature Engineering**: Most indicators are implemented; do not add new features.
- **Deep Learning Frameworks**: Only PyTorch, Ray, vllm, and sglang are used for modeling and orchestration.
- **Tree Models**: Not used.
- **Backtest Engine**: Implemented for price data only, not for DEX simulation.
- **Reporting**: Only detailed log files and CSVs are required, no HTML/JSON reports.
- **Risk/Portfolio**: Custom or third-party portfolio/risk metrics as needed.
- **Checkpointing/Experiment Tracking**: All tracking is local (logs, CSVs).
- **CLI Framework**: Typer/argparse for command-line entry.
- **Docker Orchestration**: Backend (modeling, backtesting, training) and foundry/on-chain state are separate services, orchestrated via Docker Compose.

---

## **Questions to Clarify/Find Missing Pieces**

1. **Data Connectors**:  
   - All data is loaded from CSVs in the data folder. Errors are shown for non-compatibility.
2. **Feature Library**:  
   - Most indicators are implemented; do not add new features.
3. **Model Frameworks**:  
   - Only PyTorch, Ray, vllm, and sglang are used for modeling and orchestration.
4. **Market Making Logic**:  
   - Implemented in `src/market_maker/glft.py`.
5. **Backtesting Engine**:  
   - Implemented for price data only. Ignore DEX simulations for now.
6. **Reporting**:  
   - Only detailed log files and CSVs are required for debugging and analysis.
7. **Experiment Tracking**:  
   - All tracking is local (logs, CSVs only).
8. **Deployment/Serving**:  
    - No deployment required; only local use. Docker Compose orchestrates backend (modeling, backtesting, training) and foundry/on-chain state as separate services.
9. **Testing/Validation**:  
    - Some sample datasets are provided in the `data/` folder. All testing, logging, and reporting must be combined in one file.
10. **Configuration Validation**:  
    - Simple check and reporting of data schema and types is required.

---

**Please answer or clarify the above points so I can provide a more precise implementation plan and highlight any further missing dependencies or design gaps!**
