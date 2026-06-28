"""Export `.env` Lighter credentials to the upstream examples config format."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.trading.lighter_config import LighterConfigError, load_lighter_trading_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/lighter_trading.yml")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument(
        "--out",
        default="api_key_config.json",
        help="Output path read by upstream examples when run from this repo root.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        cfg = load_lighter_trading_config(args.config, args.env_file)
        cfg.require_trading_credentials()
        assert cfg.account_index is not None
        assert cfg.api_key_index is not None
        assert cfg.api_private_key is not None

        out = Path(args.out)
        out.write_text(
            json.dumps(
                {
                    "baseUrl": cfg.base_url,
                    "accountIndex": cfg.account_index,
                    "privateKeys": {str(cfg.api_key_index): cfg.api_private_key},
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"Wrote upstream example config to {out}")
    except LighterConfigError as exc:
        print(f"error: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
