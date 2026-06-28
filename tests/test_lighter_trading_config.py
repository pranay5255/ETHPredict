from pathlib import Path

import pytest

from src.trading.lighter_config import (
    LighterConfigError,
    decimal_to_scaled_int,
    load_lighter_trading_config,
)


def _write_config(path: Path):
    path.write_text(
        """
network: testnet
base_url: "https://testnet.zklighter.elliot.ai"
websocket_path: "/stream"
nonce_management: optimistic
allow_mainnet: false
default_market: eth_perp
markets:
  eth_perp:
    symbol: "ETH"
    market_type: "perp"
    market_index: 0
    size_decimals: 4
    price_decimals: 2
    min_base_amount: "0.0050"
    min_quote_amount: "10.000000"
""",
        encoding="utf-8",
    )


def test_load_lighter_trading_config_defaults(tmp_path, monkeypatch):
    config_path = tmp_path / "lighter_trading.yml"
    env_path = tmp_path / ".env"
    _write_config(config_path)
    env_path.write_text(
        """
LIGHTER_ACCOUNT_INDEX=65
LIGHTER_API_KEY_INDEX=3
LIGHTER_API_PRIVATE_KEY=abc123
""",
        encoding="utf-8",
    )
    monkeypatch.delenv("LIGHTER_BASE_URL", raising=False)
    monkeypatch.delenv("LIGHTER_ACCOUNT_INDEX", raising=False)
    monkeypatch.delenv("LIGHTER_API_KEY_INDEX", raising=False)
    monkeypatch.delenv("LIGHTER_API_PRIVATE_KEY", raising=False)

    cfg = load_lighter_trading_config(config_path, env_path)

    assert cfg.base_url == "https://testnet.zklighter.elliot.ai"
    assert cfg.account_index == 65
    assert cfg.api_key_index == 3
    assert cfg.market.market_index == 0
    assert cfg.websocket_url == "wss://testnet.zklighter.elliot.ai/stream"


def test_mainnet_refused_without_guard(tmp_path, monkeypatch):
    config_path = tmp_path / "lighter_trading.yml"
    env_path = tmp_path / ".env"
    _write_config(config_path)
    env_path.write_text("LIGHTER_BASE_URL=https://mainnet.zklighter.elliot.ai\n", encoding="utf-8")
    monkeypatch.delenv("LIGHTER_BASE_URL", raising=False)
    monkeypatch.delenv("LIGHTER_ALLOW_MAINNET", raising=False)

    with pytest.raises(LighterConfigError, match="Refusing"):
        load_lighter_trading_config(config_path, env_path)


def test_decimal_to_scaled_int_rejects_unsupported_precision():
    assert decimal_to_scaled_int("0.0050", 4, "base_amount") == 50
    assert decimal_to_scaled_int("4050.00", 2, "price") == 405000
    with pytest.raises(LighterConfigError, match="more than 4"):
        decimal_to_scaled_int("0.00001", 4, "base_amount")
