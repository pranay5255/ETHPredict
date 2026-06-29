from pathlib import Path
import os
from decimal import Decimal
from types import SimpleNamespace

import pytest

from scripts import lighter_perp_functionality_check as perp_check

from src.trading.lighter_config import (
    LighterConfigError,
    LighterMarketConfig,
    LighterTradingConfig,
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


def _perp_runner_cfg(network="testnet", base_url="https://testnet.zklighter.elliot.ai", allow_mainnet=False):
    return LighterTradingConfig(
        network=network,
        base_url=base_url,
        websocket_path="/stream",
        nonce_management="optimistic",
        allow_mainnet=allow_mainnet,
        account_index=65,
        api_key_index=3,
        api_private_key="not-a-real-key",
        eth_private_key=None,
        market_profile="eth_perp",
        market=LighterMarketConfig(
            symbol="ETH",
            market_type="perp",
            market_index=0,
            size_decimals=4,
            price_decimals=2,
            min_base_amount=Decimal("0.0050"),
            min_quote_amount=Decimal("10.000000"),
        ),
    )


def test_lighter_perp_env_file_is_loaded_exactly(tmp_path, monkeypatch):
    env_path = tmp_path / ".env.testnet"
    env_path.write_text(
        """LIGHTER_NETWORK=testnet
LIGHTER_BASE_URL=https://testnet.zklighter.elliot.ai
LIGHTER_API_KEY_INDEX=4
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("LIGHTER_ACCOUNT_INDEX", "999")
    monkeypatch.setenv("LIGHTER_API_PRIVATE_KEY", "old-secret")

    set_keys = perp_check.apply_lighter_env_file(env_path)

    assert "LIGHTER_API_KEY_INDEX" in set_keys
    assert os.environ["LIGHTER_API_KEY_INDEX"] == "4"
    assert "LIGHTER_ACCOUNT_INDEX" not in os.environ
    assert "LIGHTER_API_PRIVATE_KEY" not in os.environ


def test_lighter_perp_redacts_secret_shaped_fields():
    payload = {
        "api_private_key": "secret",
        "nested": {"auth": "token", "safe": "value"},
        "tx_info": {"L1Sig": "signature"},
    }

    assert perp_check.redact_payload(payload) == {
        "api_private_key": "<redacted>",
        "nested": {"auth": "<redacted>", "safe": "value"},
        "tx_info": {"L1Sig": "<redacted>"},
    }


def test_lighter_perp_position_sign_and_price_helpers():
    cfg = _perp_runner_cfg()

    assert perp_check.effective_position({"position": "0.1000", "sign": 1}) == Decimal("0.1000")
    assert perp_check.effective_position({"position": "0.1000", "sign": -1}) == Decimal("-0.1000")
    assert perp_check.price_decimal_to_int(cfg, Decimal("123.451"), perp_check.ROUND_CEILING) == 12346
    assert perp_check.price_decimal_to_int(cfg, Decimal("123.459"), perp_check.ROUND_FLOOR) == 12345
    runner = perp_check.PerpFunctionalityRunner.__new__(perp_check.PerpFunctionalityRunner)
    assert 0 <= runner.next_client_order_index() < perp_check.MAX_CLIENT_ORDER_INDEX


def test_lighter_perp_mutating_modes_require_testnet_ack():
    cfg = _perp_runner_cfg()
    args = SimpleNamespace(i_understand_this_mutates_testnet=False)

    with pytest.raises(perp_check.RunnerError, match="mutates testnet state"):
        perp_check.validate_mode_safety("limit-http", cfg, args)

    args.i_understand_this_mutates_testnet = True
    perp_check.validate_mode_safety("limit-http", cfg, args)


def test_lighter_perp_refuses_mainnet_mutations():
    cfg = _perp_runner_cfg(
        network="mainnet",
        base_url="https://mainnet.zklighter.elliot.ai",
        allow_mainnet=True,
    )
    args = SimpleNamespace(i_understand_this_mutates_testnet=True)

    with pytest.raises(perp_check.RunnerError, match="not allowed against mainnet"):
        perp_check.validate_mode_safety("limit-http", cfg, args)

    perp_check.validate_mode_safety("mainnet-readonly", cfg, args)
