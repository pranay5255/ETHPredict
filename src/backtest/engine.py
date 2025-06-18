from pathlib import Path
import pandas as pd


def run_backtest(features: pd.DataFrame, predictions: pd.Series, out_dir: Path) -> None:
    pnl = (predictions.shift(1).fillna(0) * features["return"]).cumsum()
    equity = 1 + pnl
    out_dir.mkdir(parents=True, exist_ok=True)
    equity_curve = pd.DataFrame({"equity": equity})
    equity_curve.to_csv(out_dir / "equity_curve.csv", index=False)
    metrics = {"final_equity": equity.iloc[-1]}
    (out_dir / "metrics.json").write_text(pd.Series(metrics).to_json())
