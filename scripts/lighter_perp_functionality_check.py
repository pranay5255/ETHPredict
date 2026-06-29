"""Controlled Lighter ETH perp functionality checks.

This runner executes the testnet-to-mainnet parity plan without using the SDK
example ``api_key_config.json`` file. It loads the requested ``.env`` file
directly, writes an audit trail under ``/tmp/lighter-perp-runs``, and refuses
all mainnet mutations.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

import websockets
from dotenv import dotenv_values

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import lighter
from lighter.signer_client import CreateOrderTxReq

from src.trading.lighter_config import (
    API_KEY_SETUP_ENV_SPECS,
    OPTIONAL_ENV_SPECS,
    REQUIRED_TRADING_ENV_SPECS,
    LighterConfigError,
    LighterTradingConfig,
    load_lighter_trading_config,
)


MODES = (
    "preflight",
    "public-data",
    "paper",
    "limit-http",
    "limit-ws",
    "batch-http",
    "batch-ws",
    "market-open-close",
    "margin",
    "sl-tp",
    "cleanup",
    "full-testnet",
    "mainnet-readonly",
)
MUTATING_MODES = {
    "limit-http",
    "limit-ws",
    "batch-http",
    "batch-ws",
    "market-open-close",
    "margin",
    "sl-tp",
    "cleanup",
    "full-testnet",
}
LIGHTER_ENV_KEYS = {
    spec.name
    for spec in (
        REQUIRED_TRADING_ENV_SPECS
        + API_KEY_SETUP_ENV_SPECS
        + OPTIONAL_ENV_SPECS
    )
}
LIGHTER_ENV_KEYS.update(
    {
        "LIGHTER_API_PUBLIC_KEY",
        "LIGHTER_MARKET_INDEX",
        "LIGHTER_MARKET_SYMBOL",
        "LIGHTER_MARKET_TYPE",
        "LIGHTER_MIN_BASE_AMOUNT",
        "LIGHTER_MIN_QUOTE_AMOUNT",
        "LIGHTER_PRICE_DECIMALS",
        "LIGHTER_SIZE_DECIMALS",
    }
)
SECRET_KEY_FRAGMENTS = ("private", "secret", "token", "auth", "signature", "l1sig")
CODE_OK = 200
MAX_CLIENT_ORDER_INDEX = 281_474_976_710_655
TRANSIENT_ERROR_MARKERS = (
    "502",
    "503",
    "504",
    "Bad Gateway",
    "Gateway Time-out",
    "Gateway Timeout",
    "timed out",
    "timeout",
    "Server disconnected",
)


class RunnerError(RuntimeError):
    """Raised when a functionality check cannot continue safely."""


@dataclass(frozen=True)
class BookTop:
    bid: Decimal
    ask: Decimal


@dataclass(frozen=True)
class SignedTx:
    label: str
    tx_type: int
    tx_info: str
    tx_hash: str
    api_key_index: int
    nonce: int


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def to_plain(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [to_plain(item) for item in value]
    if isinstance(value, tuple):
        return [to_plain(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_plain(item) for key, item in value.items()}
    if hasattr(value, "to_dict"):
        return to_plain(value.to_dict())
    if hasattr(value, "model_dump"):
        return to_plain(value.model_dump())
    return repr(value)


def redact_payload(value: Any) -> Any:
    value = to_plain(value)
    if isinstance(value, list):
        return [redact_payload(item) for item in value]
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            normalized = str(key).lower()
            if any(fragment in normalized for fragment in SECRET_KEY_FRAGMENTS):
                redacted[str(key)] = "<redacted>"
            else:
                redacted[str(key)] = redact_payload(item)
        return redacted
    return value


def decimal_value(value: Any, default: str = "0") -> Decimal:
    if value is None or value == "":
        value = default
    return Decimal(str(value))


def mode_default_env_file(mode: str) -> str:
    return ".env.mainnet" if mode == "mainnet-readonly" else ".env.testnet"


def apply_lighter_env_file(env_path: Path) -> tuple[str, ...]:
    if not env_path.exists():
        raise RunnerError(f"Missing env file: {env_path}")

    values = dotenv_values(env_path)
    managed_keys = set(LIGHTER_ENV_KEYS)
    managed_keys.update(key for key in os.environ if key.startswith("LIGHTER_"))
    managed_keys.update(key for key in values if key.startswith("LIGHTER_"))

    for key in managed_keys:
        os.environ.pop(key, None)

    set_keys: list[str] = []
    for key, value in values.items():
        if not key.startswith("LIGHTER_") or value in (None, ""):
            continue
        os.environ[key] = str(value)
        set_keys.append(key)
    return tuple(sorted(set_keys))


def load_config_from_env_file(config_path: Path, env_path: Path) -> LighterTradingConfig:
    apply_lighter_env_file(env_path)
    return load_lighter_trading_config(config_path, env_path, load_env_file=False)


def is_mainnet_config(cfg: LighterTradingConfig) -> bool:
    return "mainnet" in cfg.base_url.lower() or cfg.network.lower() == "mainnet"


def config_summary(cfg: LighterTradingConfig, env_file: Path) -> dict[str, Any]:
    return {
        "env_file": str(env_file),
        "network": cfg.network,
        "base_url": cfg.base_url,
        "websocket_url": cfg.websocket_url,
        "account_index": cfg.account_index,
        "api_key_index": cfg.api_key_index,
        "market_profile": cfg.market_profile,
        "market": {
            "symbol": cfg.market.symbol,
            "market_type": cfg.market.market_type,
            "market_index": cfg.market.market_index,
            "size_decimals": cfg.market.size_decimals,
            "price_decimals": cfg.market.price_decimals,
            "min_base_amount": str(cfg.market.min_base_amount),
            "min_quote_amount": str(cfg.market.min_quote_amount),
        },
    }


def quantize_decimal(value: Decimal, decimals: int, rounding: str) -> Decimal:
    quantum = Decimal(1).scaleb(-decimals)
    return value.quantize(quantum, rounding=rounding)


def decimal_to_int(value: Decimal, decimals: int) -> int:
    return int(value * (Decimal(10) ** decimals))


def price_decimal_to_int(
    cfg: LighterTradingConfig,
    value: Decimal,
    rounding: str = ROUND_HALF_UP,
) -> int:
    quantized = quantize_decimal(value, cfg.market.price_decimals, rounding)
    return decimal_to_int(quantized, cfg.market.price_decimals)


def amount_decimal_to_int(cfg: LighterTradingConfig, value: Decimal) -> int:
    quantized = quantize_decimal(value, cfg.market.size_decimals, ROUND_HALF_UP)
    return decimal_to_int(quantized, cfg.market.size_decimals)


def int_price_to_decimal(cfg: LighterTradingConfig, value: int) -> Decimal:
    return Decimal(value) / (Decimal(10) ** cfg.market.price_decimals)


def response_code(response: Any) -> Optional[int]:
    if response is None:
        return None
    if hasattr(response, "code"):
        return int(response.code)
    if isinstance(response, dict) and "code" in response:
        return int(response["code"])
    return None


def effective_position(position: Any) -> Decimal:
    if position is None:
        return Decimal("0")
    if isinstance(position, dict):
        amount = decimal_value(position.get("position"))
        sign = int(position.get("sign") or 0)
    else:
        amount = decimal_value(getattr(position, "position", "0"))
        sign = int(getattr(position, "sign", 0) or 0)
    if amount == 0 or sign == 0:
        return Decimal("0")
    return amount if sign > 0 else -amount


def find_market_position(account: Any, market_id: int) -> Any:
    for position in getattr(account, "positions", None) or []:
        if int(getattr(position, "market_id", -1)) == market_id:
            return position
    return None


def cancel_order_index(order: Any) -> int:
    client_order_index = getattr(order, "client_order_index", None)
    if client_order_index is not None:
        return int(client_order_index)
    return int(getattr(order, "order_index"))


class AuditRun:
    def __init__(
        self,
        root: Path,
        network: str,
        mode: str,
        cfg: LighterTradingConfig,
        env_file: Path,
    ) -> None:
        self.path = root / network / utc_timestamp()
        self.path.mkdir(parents=True, exist_ok=True)
        self.summary: dict[str, Any] = {
            "mode": mode,
            "network": network,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "status": "running",
            "steps": [],
        }
        self.write_json("config.json", config_summary(cfg, env_file))
        self.flush()

    def write_json(self, name: str, payload: Any) -> None:
        target = self.path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(redact_payload(payload), f, indent=2, sort_keys=True)
            f.write("\n")

    def add_step(self, name: str, status: str, payload: Optional[dict[str, Any]] = None) -> None:
        step = {
            "name": name,
            "status": status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if payload:
            step.update(redact_payload(payload))
        self.summary["steps"].append(step)
        self.flush()

    def flush(self) -> None:
        with open(self.path / "summary.json", "w", encoding="utf-8") as f:
            json.dump(redact_payload(self.summary), f, indent=2, sort_keys=True)
            f.write("\n")

    def finish(self, status: str) -> None:
        self.summary["status"] = status
        self.summary["finished_at"] = datetime.now(timezone.utc).isoformat()
        self.flush()


class PerpFunctionalityRunner:
    def __init__(self, cfg: LighterTradingConfig, audit: AuditRun, args: argparse.Namespace) -> None:
        self.cfg = cfg
        self.audit = audit
        self.args = args
        self.market_id = cfg.market.market_index
        self.size_base = decimal_value(args.size_base)
        self.size_int = amount_decimal_to_int(cfg, self.size_base)
        self.slippage = Decimal(args.max_slippage_bps) / Decimal("10000")

    async def run_step(
        self,
        name: str,
        func: Callable[[], Awaitable[Any]],
    ) -> Any:
        print(f"[{name}] starting")
        started = time.monotonic()
        try:
            result = await func()
        except Exception as exc:
            self.audit.add_step(
                name,
                "fail",
                {"elapsed_seconds": round(time.monotonic() - started, 3), "error": str(exc)},
            )
            print(f"[{name}] failed: {exc}")
            raise
        self.audit.add_step(
            name,
            "pass",
            {"elapsed_seconds": round(time.monotonic() - started, 3)},
        )
        print(f"[{name}] passed")
        return result

    def is_transient_error(self, exc: Exception) -> bool:
        text = str(exc).lower()
        return any(marker.lower() in text for marker in TRANSIENT_ERROR_MARKERS)

    async def retry_read(
        self,
        label: str,
        factory: Callable[[], Awaitable[Any]],
    ) -> Any:
        attempts = max(1, int(self.args.api_retries))
        for attempt in range(1, attempts + 1):
            try:
                return await factory()
            except Exception as exc:
                if attempt >= attempts or not self.is_transient_error(exc):
                    raise
                print(f"[{label}] transient error on attempt {attempt}/{attempts}: {exc}")
                await asyncio.sleep(self.args.retry_sleep_seconds * attempt)
        raise RunnerError(f"{label} retry loop exhausted")

    async def create_signer(self) -> lighter.SignerClient:
        attempts = max(1, int(self.args.api_retries))
        for attempt in range(1, attempts + 1):
            try:
                return self.cfg.create_signer_client()
            except Exception as exc:
                if attempt >= attempts or not self.is_transient_error(exc):
                    raise
                print(f"[create-signer] transient error on attempt {attempt}/{attempts}: {exc}")
                await asyncio.sleep(self.args.retry_sleep_seconds * attempt)
        raise RunnerError("create_signer retry loop exhausted")

    async def preflight(self, *, strict_neutral: bool) -> dict[str, Any]:
        cfg = self.cfg
        api_client = cfg.create_api_client()
        signer = None
        try:
            root_api = lighter.RootApi(api_client)
            order_api = lighter.OrderApi(api_client)
            account_api = lighter.AccountApi(api_client)
            tx_api = lighter.TransactionApi(api_client)

            status = await self.retry_read("root.status", root_api.status)
            details = await self.retry_read(
                "order_book_details",
                lambda: order_api.order_book_details(
                    market_id=self.market_id,
                    filter=cfg.market.market_type,
                ),
            )
            account = await self.fetch_account(account_api)
            next_nonce = None
            if cfg.account_index is not None and cfg.api_key_index is not None:
                next_nonce = await self.retry_read(
                    "next_nonce",
                    lambda: tx_api.next_nonce(
                        account_index=cfg.account_index,
                        api_key_index=cfg.api_key_index,
                    ),
                )

            if not cfg.has_trading_credentials:
                raise RunnerError("Signed checks require account index, API key index, and API private key")
            signer = await self.create_signer()
            signer_err = signer.check_client()
            if signer_err is not None:
                raise RunnerError(f"Signer check failed: {signer_err}")
            auth, auth_err = signer.create_auth_token_with_expiry(api_key_index=cfg.api_key_index)
            if auth_err is not None:
                raise RunnerError(f"Auth token check failed: {auth_err}")
            active_orders = await self.fetch_active_orders(signer, auth=auth)

            position = find_market_position(account, self.market_id)
            signed_position = effective_position(position)
            summary = {
                "status": to_plain(status),
                "market_detail": to_plain(details),
                "account": to_plain(account),
                "next_nonce": to_plain(next_nonce),
                "active_orders": to_plain(active_orders),
                "signed_position": str(signed_position),
            }
            self.audit.write_json("preflight.json", summary)

            active_count = len(active_orders.orders or [])
            if strict_neutral and active_count:
                raise RunnerError(f"Preflight abort: {active_count} active ETH perp orders exist")
            if strict_neutral and signed_position != 0:
                raise RunnerError(f"Preflight abort: ETH perp position is {signed_position}")

            return {
                "active_order_count": active_count,
                "signed_position": str(signed_position),
                "available_balance": getattr(account, "available_balance", None),
                "collateral": getattr(account, "collateral", None),
            }
        finally:
            if signer is not None:
                await signer.close()
            await api_client.close()

    async def public_data(self) -> dict[str, Any]:
        api_client = self.cfg.create_api_client()
        try:
            root_api = lighter.RootApi(api_client)
            order_api = lighter.OrderApi(api_client)
            candle_api = lighter.CandlestickApi(api_client)
            funding_api = lighter.FundingApi(api_client)

            now = int(time.time())
            start = now - 6 * 60 * 60
            payload = {
                "root_info": await root_api.info(),
                "status": await root_api.status(),
                "order_books": await order_api.order_books(),
                "order_book_details": await order_api.order_book_details(
                    market_id=self.market_id,
                    filter=self.cfg.market.market_type,
                ),
                "order_book_orders": await order_api.order_book_orders(
                    market_id=self.market_id,
                    limit=5,
                ),
                "asset_details": await order_api.asset_details(asset_id=1),
                "candles": await candle_api.candles(
                    market_id=self.market_id,
                    resolution="1h",
                    start_timestamp=start,
                    end_timestamp=now,
                    count_back=6,
                ),
                "fundings": await candle_api.fundings(
                    market_id=self.market_id,
                    resolution="1h",
                    start_timestamp=start,
                    end_timestamp=now,
                    count_back=6,
                ),
                "funding_rates": await funding_api.funding_rates(),
                "exchange_stats": await order_api.exchange_stats(),
                "exchange_metrics_volume_all": await order_api.exchange_metrics(
                    period="all",
                    kind="volume",
                    filter="byMarket",
                    value=self.cfg.market.symbol,
                ),
                "recent_trades": await order_api.recent_trades(
                    market_id=self.market_id,
                    limit=10,
                ),
            }
            self.audit.write_json("public_data.json", payload)
        finally:
            await api_client.close()

        ws_payload = await self.ws_checks(include_account=True)
        self.audit.write_json("public_and_account_ws.json", ws_payload)
        return {"public_sections": len(payload), "ws_sections": len(ws_payload)}

    async def ws_checks(self, *, include_account: bool) -> dict[str, Any]:
        cfg = self.cfg
        messages: dict[str, Any] = {}
        signer = None
        try:
            async with websockets.connect(cfg.websocket_url) as ws:
                messages["connected"] = await self.recv_ws_json(ws)
                await ws.send(json.dumps({"type": "subscribe", "channel": f"order_book/{self.market_id}"}))
                messages["order_book"] = await self.recv_ws_until(
                    ws,
                    {"subscribed/order_book", "update/order_book"},
                )

                if include_account:
                    if not cfg.has_trading_credentials:
                        raise RunnerError("Account-assets WS check requires signing credentials")
                    signer = await self.create_signer()
                    auth, err = signer.create_auth_token_with_expiry(api_key_index=cfg.api_key_index)
                    if err is not None:
                        raise RunnerError(f"Auth token failed for account-assets WS: {err}")
                    await ws.send(
                        json.dumps(
                            {
                                "type": "subscribe",
                                "channel": f"account_all_assets/{cfg.account_index}",
                                "auth": auth,
                            }
                        )
                    )
                    messages["account_all_assets"] = await self.recv_ws_until(
                        ws,
                        {"subscribed/account_all_assets", "update/account_all_assets"},
                    )
        finally:
            if signer is not None:
                await signer.close()
        return messages

    async def paper_trading(self) -> dict[str, Any]:
        api_client = self.cfg.create_api_client()
        results: dict[str, Any] = {}
        try:
            paper = lighter.PaperClient(
                api_client,
                initial_collateral_usdc=10_000,
                ws_url=self.cfg.websocket_url,
            )
            await paper.track_market_snapshot(self.market_id)
            buy = await paper.create_paper_order(
                lighter.PaperOrderRequest(
                    market_id=self.market_id,
                    side=lighter.PaperOrderSide.BUY,
                    base_amount=float(self.size_base),
                )
            )
            sell = await paper.create_paper_order(
                lighter.PaperOrderRequest(
                    market_id=self.market_id,
                    side=lighter.PaperOrderSide.SELL,
                    base_amount=float(self.size_base),
                )
            )
            results["snapshot"] = {
                "buy": buy,
                "sell": sell,
                "account": paper.get_account(),
                "portfolio_value": paper.get_portfolio_value(),
            }

            live = lighter.PaperClient(
                api_client,
                initial_collateral_usdc=10_000,
                ws_url=self.cfg.websocket_url,
            )
            try:
                await live.track_market(self.market_id)
                await asyncio.sleep(self.args.paper_live_seconds)
                live_buy = await live.create_paper_order(
                    lighter.PaperOrderRequest(
                        market_id=self.market_id,
                        side=lighter.PaperOrderSide.BUY,
                        base_amount=float(self.size_base),
                    )
                )
                await asyncio.sleep(self.args.paper_live_seconds)
                live_sell = await live.create_paper_order(
                    lighter.PaperOrderRequest(
                        market_id=self.market_id,
                        side=lighter.PaperOrderSide.SELL,
                        base_amount=float(self.size_base),
                    )
                )
                results["live"] = {
                    "buy": live_buy,
                    "sell": live_sell,
                    "account": live.get_account(),
                    "portfolio_value": live.get_portfolio_value(),
                }
            finally:
                await live.close()

            health = lighter.PaperClient(
                api_client,
                initial_collateral_usdc=1_500,
                ws_url=self.cfg.websocket_url,
            )
            await health.track_market_snapshot(self.market_id)
            health_sell = await health.create_paper_order(
                lighter.PaperOrderRequest(
                    market_id=self.market_id,
                    side=lighter.PaperOrderSide.SELL,
                    base_amount=float(self.size_base),
                )
            )
            results["health"] = {
                "sell": health_sell,
                "health": health.get_health(),
                "liquidation_price": health.get_liquidation_price(self.market_id),
                "position": health.get_position(self.market_id),
            }
        finally:
            await api_client.close()

        self.audit.write_json("paper_trading.json", results)
        return {"sections": tuple(results.keys())}

    async def limit_lifecycle(self, *, transport: str) -> None:
        await self.assert_neutral()
        signer = await self.create_signer()
        try:
            top = await self.fetch_book_top()
            client_index = self.next_client_order_index()
            far_ask = price_decimal_to_int(
                self.cfg,
                top.ask * Decimal("1.35"),
                ROUND_CEILING,
            )
            farther_ask = price_decimal_to_int(
                self.cfg,
                top.ask * Decimal("1.45"),
                ROUND_CEILING,
            )
            submissions: list[dict[str, Any]] = []
            if transport == "ws":
                async with websockets.connect(self.cfg.websocket_url) as ws:
                    await self.recv_ws_json(ws)
                    submissions.append(
                        await self.sign_and_send_order(
                            signer,
                            "limit_create_ws",
                            transport,
                            ws=ws,
                            client_order_index=client_index,
                            base_amount=self.size_int,
                            price=far_ask,
                            is_ask=True,
                            order_type=signer.ORDER_TYPE_LIMIT,
                            time_in_force=signer.ORDER_TIME_IN_FORCE_GOOD_TILL_TIME,
                            reduce_only=False,
                        )
                    )
                    await self.settle()
                    submissions.append(
                        await self.sign_and_send_modify(
                            signer,
                            "limit_modify_ws",
                            transport,
                            ws=ws,
                            order_index=client_index,
                            base_amount=self.size_int,
                            price=farther_ask,
                        )
                    )
                    await self.settle()
                    submissions.append(
                        await self.sign_and_send_cancel(
                            signer,
                            "limit_cancel_ws",
                            transport,
                            ws=ws,
                            order_index=client_index,
                        )
                    )
            else:
                submissions.append(
                    await self.sign_and_send_order(
                        signer,
                        "limit_create_http",
                        transport,
                        client_order_index=client_index,
                        base_amount=self.size_int,
                        price=far_ask,
                        is_ask=True,
                        order_type=signer.ORDER_TYPE_LIMIT,
                        time_in_force=signer.ORDER_TIME_IN_FORCE_GOOD_TILL_TIME,
                        reduce_only=False,
                    )
                )
                await self.settle()
                submissions.append(
                    await self.sign_and_send_modify(
                        signer,
                        "limit_modify_http",
                        transport,
                        order_index=client_index,
                        base_amount=self.size_int,
                        price=farther_ask,
                    )
                )
                await self.settle()
                submissions.append(
                    await self.sign_and_send_cancel(
                        signer,
                        "limit_cancel_http",
                        transport,
                        order_index=client_index,
                    )
                )
            await self.settle()
            self.audit.write_json(f"limit_{transport}.json", submissions)
        finally:
            await signer.close()
        await self.assert_neutral()

    async def batch_lifecycle(self, *, transport: str) -> None:
        await self.assert_neutral()
        signer = await self.create_signer()
        try:
            top = await self.fetch_book_top()
            ask_index = self.next_client_order_index()
            bid_index = ask_index + 1
            far_ask = price_decimal_to_int(self.cfg, top.ask * Decimal("1.35"), ROUND_CEILING)
            far_bid = price_decimal_to_int(self.cfg, top.bid * Decimal("0.65"), ROUND_FLOOR)
            farther_ask = price_decimal_to_int(self.cfg, top.ask * Decimal("1.45"), ROUND_CEILING)
            farther_bid = price_decimal_to_int(self.cfg, top.bid * Decimal("0.55"), ROUND_FLOOR)

            submissions: list[dict[str, Any]] = []
            create_batch = [
                self.sign_order(
                    signer,
                    "batch_create_ask",
                    client_order_index=ask_index,
                    base_amount=self.size_int,
                    price=far_ask,
                    is_ask=True,
                    order_type=signer.ORDER_TYPE_LIMIT,
                    time_in_force=signer.ORDER_TIME_IN_FORCE_GOOD_TILL_TIME,
                    reduce_only=False,
                ),
                self.sign_order(
                    signer,
                    "batch_create_bid",
                    client_order_index=bid_index,
                    base_amount=self.size_int,
                    price=far_bid,
                    is_ask=False,
                    order_type=signer.ORDER_TYPE_LIMIT,
                    time_in_force=signer.ORDER_TIME_IN_FORCE_GOOD_TILL_TIME,
                    reduce_only=False,
                ),
            ]
            modify_batch = [
                self.sign_modify(
                    signer,
                    "batch_modify_ask",
                    order_index=ask_index,
                    base_amount=self.size_int,
                    price=farther_ask,
                    prior_api_key_index=create_batch[-1].api_key_index,
                ),
                self.sign_modify(
                    signer,
                    "batch_modify_bid",
                    order_index=bid_index,
                    base_amount=self.size_int,
                    price=farther_bid,
                    prior_api_key_index=create_batch[-1].api_key_index,
                ),
            ]
            cancel_batch = [
                self.sign_cancel(
                    signer,
                    "batch_cancel_ask",
                    order_index=ask_index,
                    prior_api_key_index=modify_batch[-1].api_key_index,
                ),
                self.sign_cancel(
                    signer,
                    "batch_cancel_bid",
                    order_index=bid_index,
                    prior_api_key_index=modify_batch[-1].api_key_index,
                ),
            ]

            if transport == "ws":
                async with websockets.connect(self.cfg.websocket_url) as ws:
                    await self.recv_ws_json(ws)
                    submissions.append(await self.send_batch_ws(ws, "create_batch_ws", create_batch))
                    await self.settle()
                    submissions.append(await self.send_batch_ws(ws, "modify_batch_ws", modify_batch))
                    await self.settle()
                    submissions.append(await self.send_batch_ws(ws, "cancel_batch_ws", cancel_batch))
            else:
                submissions.append(await self.send_batch_http(signer, "create_batch_http", create_batch))
                await self.settle()
                submissions.append(await self.send_batch_http(signer, "modify_batch_http", modify_batch))
                await self.settle()
                submissions.append(await self.send_batch_http(signer, "cancel_batch_http", cancel_batch))

            await self.settle()
            self.audit.write_json(f"batch_{transport}.json", submissions)
        finally:
            await signer.close()
        await self.assert_neutral()

    async def market_open_close(self) -> None:
        await self.assert_neutral()
        signer = await self.create_signer()
        try:
            submissions = [
                await self.market_order(signer, "market_buy_open", is_ask=False, base_amount=self.size_int, reduce_only=False)
            ]
            await self.settle()
            position = await self.current_signed_position()
            if position <= 0:
                raise RunnerError(f"Expected long position after buy, got {position}")
            close_amount = amount_decimal_to_int(self.cfg, abs(position))
            submissions.append(
                await self.market_order(
                    signer,
                    "market_sell_close",
                    is_ask=True,
                    base_amount=close_amount,
                    reduce_only=True,
                )
            )
            await self.settle()
            self.audit.write_json("market_open_close.json", submissions)
        finally:
            await signer.close()
        await self.assert_neutral()

    async def margin_flow(self) -> None:
        await self.assert_neutral()
        signer = await self.create_signer()
        try:
            baseline_account = await self.fetch_account(lighter.AccountApi(signer.api_client))
            baseline_position = find_market_position(baseline_account, self.market_id)
            market_detail = await self.fetch_market_detail()
            default_fraction = int(getattr(market_detail, "default_initial_margin_fraction"))
            min_fraction = int(getattr(market_detail, "min_initial_margin_fraction") or 0)
            baseline_fraction = (
                int(decimal_value(getattr(baseline_position, "initial_margin_fraction", None)))
                if baseline_position is not None
                else default_fraction
            )
            baseline_mode = (
                int(getattr(baseline_position, "margin_mode"))
                if baseline_position is not None
                else signer.CROSS_MARGIN_MODE
            )
            if baseline_fraction <= 0 or baseline_fraction < min_fraction or baseline_fraction > 10_000:
                restore_fraction = default_fraction
                restore_mode = signer.CROSS_MARGIN_MODE
            else:
                restore_fraction = baseline_fraction
                restore_mode = baseline_mode

            submissions: list[dict[str, Any]] = []
            submissions.append(
                await self.update_leverage(
                    signer,
                    "margin_cross_20x_http",
                    margin_mode=signer.CROSS_MARGIN_MODE,
                    leverage=Decimal("20"),
                    transport="http",
                )
            )
            await self.settle()
            async with websockets.connect(self.cfg.websocket_url) as ws:
                await self.recv_ws_json(ws)
                submissions.append(
                    await self.update_leverage(
                        signer,
                        "margin_isolated_50x_ws",
                        margin_mode=signer.ISOLATED_MARGIN_MODE,
                        leverage=Decimal("50"),
                        transport="ws",
                        ws=ws,
                    )
                )
            await self.settle()

            submissions.append(
                await self.market_order(
                    signer,
                    "margin_temp_buy_open",
                    is_ask=False,
                    base_amount=self.size_int,
                    reduce_only=False,
                )
            )
            await self.settle()
            submissions.append(
                await self.update_margin(
                    signer,
                    "margin_add_10_5_http",
                    usdc_amount=Decimal("10.5"),
                    direction=signer.ISOLATED_MARGIN_ADD_COLLATERAL,
                    transport="http",
                )
            )
            await self.settle()
            submissions.append(
                await self.update_margin(
                    signer,
                    "margin_remove_5_http",
                    usdc_amount=Decimal("5"),
                    direction=signer.ISOLATED_MARGIN_REMOVE_COLLATERAL,
                    transport="http",
                )
            )
            await self.settle()
            close_record = await self.close_any_position(signer, label_prefix="margin")
            if close_record is not None:
                submissions.append(close_record)
            await self.settle()

            self.audit.write_json(
                "margin_progress.json",
                {
                    "baseline_position": baseline_position,
                    "default_fraction": default_fraction,
                    "min_fraction": min_fraction,
                    "restore_fraction": restore_fraction,
                    "restore_mode": restore_mode,
                    "submissions": submissions,
                },
            )

            submissions.append(
                await self.update_leverage_fraction(
                    signer,
                    "margin_restore_baseline_http",
                    margin_mode=restore_mode,
                    fraction=restore_fraction,
                    transport="http",
                )
            )
            await self.settle()
            self.audit.write_json(
                "margin.json",
                {
                    "baseline_position": baseline_position,
                    "restore_fraction": restore_fraction,
                    "restore_mode": restore_mode,
                    "submissions": submissions,
                },
            )
        finally:
            await signer.close()
        await self.assert_neutral()

    async def sl_tp_flow(self) -> None:
        await self.assert_neutral()
        signer = await self.create_signer()
        submissions: list[dict[str, Any]] = []
        try:
            top = await self.fetch_book_top()
            ioc_price = price_decimal_to_int(self.cfg, top.bid * Decimal("0.99"), ROUND_FLOOR)
            tp_trigger = price_decimal_to_int(self.cfg, top.bid * Decimal("0.97"), ROUND_FLOOR)
            tp_price = price_decimal_to_int(self.cfg, top.bid * Decimal("0.975"), ROUND_CEILING)
            sl_trigger = price_decimal_to_int(self.cfg, top.ask * Decimal("1.03"), ROUND_CEILING)
            sl_price = price_decimal_to_int(self.cfg, top.ask * Decimal("1.035"), ROUND_CEILING)

            attached_orders = [
                CreateOrderTxReq(
                    MarketIndex=self.market_id,
                    ClientOrderIndex=self.next_client_order_index(),
                    BaseAmount=self.size_int,
                    Price=ioc_price,
                    IsAsk=1,
                    Type=signer.ORDER_TYPE_LIMIT,
                    TimeInForce=signer.ORDER_TIME_IN_FORCE_IMMEDIATE_OR_CANCEL,
                    ReduceOnly=0,
                    TriggerPrice=signer.NIL_TRIGGER_PRICE,
                    OrderExpiry=signer.DEFAULT_IOC_EXPIRY,
                ),
                CreateOrderTxReq(
                    MarketIndex=self.market_id,
                    ClientOrderIndex=self.next_client_order_index(),
                    BaseAmount=0,
                    Price=tp_price,
                    IsAsk=0,
                    Type=signer.ORDER_TYPE_TAKE_PROFIT_LIMIT,
                    TimeInForce=signer.ORDER_TIME_IN_FORCE_GOOD_TILL_TIME,
                    ReduceOnly=1,
                    TriggerPrice=tp_trigger,
                    OrderExpiry=signer.DEFAULT_28_DAY_ORDER_EXPIRY,
                ),
                CreateOrderTxReq(
                    MarketIndex=self.market_id,
                    ClientOrderIndex=self.next_client_order_index(),
                    BaseAmount=0,
                    Price=sl_price,
                    IsAsk=0,
                    Type=signer.ORDER_TYPE_STOP_LOSS_LIMIT,
                    TimeInForce=signer.ORDER_TIME_IN_FORCE_GOOD_TILL_TIME,
                    ReduceOnly=1,
                    TriggerPrice=sl_trigger,
                    OrderExpiry=signer.DEFAULT_28_DAY_ORDER_EXPIRY,
                ),
            ]
            submissions.append(
                await self.sign_and_send_grouped(
                    signer,
                    "attached_ioc_sl_tp_http",
                    grouping_type=signer.GROUPING_TYPE_ONE_TRIGGERS_A_ONE_CANCELS_THE_OTHER,
                    orders=attached_orders,
                )
            )
            await self.settle()
            active = await self.fetch_active_orders(signer)
            if len(active.orders or []) < 2:
                raise RunnerError("Expected attached TP/SL reduce-only orders after IOC fill")
            submissions.extend(
                await self.cancel_active_orders(signer, label_prefix="attached_sl_tp")
            )
            close_record = await self.close_any_position(
                signer,
                label_prefix="attached_sl_tp",
            )
            if close_record is not None:
                submissions.append(close_record)
            await self.settle()
            await self.assert_neutral()

            submissions.append(
                await self.market_order(
                    signer,
                    "position_tied_short_open",
                    is_ask=True,
                    base_amount=self.size_int,
                    reduce_only=False,
                )
            )
            await self.settle()
            top = await self.fetch_book_top()
            tp_trigger = price_decimal_to_int(self.cfg, top.bid * Decimal("0.97"), ROUND_FLOOR)
            tp_price = price_decimal_to_int(self.cfg, top.bid * Decimal("0.975"), ROUND_CEILING)
            sl_trigger = price_decimal_to_int(self.cfg, top.ask * Decimal("1.03"), ROUND_CEILING)
            sl_price = price_decimal_to_int(self.cfg, top.ask * Decimal("1.035"), ROUND_CEILING)
            position_tied_orders = [
                CreateOrderTxReq(
                    MarketIndex=self.market_id,
                    ClientOrderIndex=self.next_client_order_index(),
                    BaseAmount=0,
                    Price=tp_price,
                    IsAsk=0,
                    Type=signer.ORDER_TYPE_TAKE_PROFIT_LIMIT,
                    TimeInForce=signer.ORDER_TIME_IN_FORCE_GOOD_TILL_TIME,
                    ReduceOnly=1,
                    TriggerPrice=tp_trigger,
                    OrderExpiry=signer.DEFAULT_28_DAY_ORDER_EXPIRY,
                ),
                CreateOrderTxReq(
                    MarketIndex=self.market_id,
                    ClientOrderIndex=self.next_client_order_index(),
                    BaseAmount=0,
                    Price=sl_price,
                    IsAsk=0,
                    Type=signer.ORDER_TYPE_STOP_LOSS_LIMIT,
                    TimeInForce=signer.ORDER_TIME_IN_FORCE_GOOD_TILL_TIME,
                    ReduceOnly=1,
                    TriggerPrice=sl_trigger,
                    OrderExpiry=signer.DEFAULT_28_DAY_ORDER_EXPIRY,
                ),
            ]
            submissions.append(
                await self.sign_and_send_grouped(
                    signer,
                    "position_tied_sl_tp_http",
                    grouping_type=signer.GROUPING_TYPE_ONE_CANCELS_THE_OTHER,
                    orders=position_tied_orders,
                )
            )
            await self.settle()
            active = await self.fetch_active_orders(signer)
            if len(active.orders or []) < 2:
                raise RunnerError("Expected position-tied OCO reduce-only orders")
            submissions.extend(
                await self.cancel_active_orders(signer, label_prefix="position_tied_sl_tp")
            )
            close_record = await self.close_any_position(
                signer,
                label_prefix="position_tied_sl_tp",
            )
            if close_record is not None:
                submissions.append(close_record)
            await self.settle()
            self.audit.write_json("sl_tp.json", submissions)
        finally:
            await signer.close()
        await self.assert_neutral()

    async def cleanup(self) -> None:
        signer = await self.create_signer()
        cleanup_records: list[dict[str, Any]] = []
        try:
            active = await self.fetch_active_orders(signer)
            for order in active.orders or []:
                cleanup_records.append(
                    await self.sign_and_send_cancel(
                        signer,
                        f"cleanup_cancel_{cancel_order_index(order)}",
                        "http",
                        order_index=cancel_order_index(order),
                    )
                )
                await self.settle()

            remaining = await self.fetch_active_orders(signer)
            if remaining.orders:
                signed = self.sign_cancel_all(signer, "cleanup_cancel_all_market")
                cleanup_records.append(await self.send_signed_http(signer, signed))
                await self.settle()

            position = await self.current_signed_position()
            if position != 0:
                close_record = await self.close_any_position(signer, label_prefix="cleanup")
                if close_record is not None:
                    cleanup_records.append(close_record)
                await self.settle()

            self.audit.write_json("cleanup.json", cleanup_records)
        finally:
            await signer.close()
        await self.assert_neutral()

    async def assert_neutral(self) -> None:
        signer = await self.create_signer()
        try:
            active = await self.fetch_active_orders(signer)
            position = await self.current_signed_position()
            if active.orders:
                raise RunnerError(f"Expected zero active ETH orders, found {len(active.orders)}")
            if position != 0:
                raise RunnerError(f"Expected zero ETH position, found {position}")
        finally:
            await signer.close()

    async def current_signed_position(self) -> Decimal:
        api_client = self.cfg.create_api_client()
        try:
            account = await self.fetch_account(lighter.AccountApi(api_client))
            return effective_position(find_market_position(account, self.market_id))
        finally:
            await api_client.close()

    async def fetch_account(self, account_api: lighter.AccountApi) -> Any:
        if self.cfg.account_index is None:
            raise RunnerError("LIGHTER_ACCOUNT_INDEX is required")
        accounts = await self.retry_read(
            "account",
            lambda: account_api.account(by="index", value=str(self.cfg.account_index)),
        )
        if not accounts.accounts:
            raise RunnerError(f"No Lighter account found for index {self.cfg.account_index}")
        return accounts.accounts[0]

    async def fetch_active_orders(
        self,
        signer: lighter.SignerClient,
        *,
        auth: Optional[str] = None,
    ) -> Any:
        if auth is None:
            auth, err = signer.create_auth_token_with_expiry(api_key_index=self.cfg.api_key_index)
            if err is not None:
                raise RunnerError(f"Auth token failed: {err}")
        return await self.retry_read(
            "account_active_orders",
            lambda: signer.order_api.account_active_orders(
                authorization=auth,
                account_index=self.cfg.account_index,
                market_id=self.market_id,
                market_type=self.cfg.market.market_type,
            ),
        )

    async def fetch_book_top(self) -> BookTop:
        api_client = self.cfg.create_api_client()
        try:
            orders = await self.retry_read(
                "order_book_orders_top",
                lambda: lighter.OrderApi(api_client).order_book_orders(
                    market_id=self.market_id,
                    limit=1,
                ),
            )
            if not orders.bids or not orders.asks:
                raise RunnerError("Order book top is empty")
            return BookTop(
                bid=decimal_value(orders.bids[0].price),
                ask=decimal_value(orders.asks[0].price),
            )
        finally:
            await api_client.close()

    async def fetch_market_detail(self) -> Any:
        api_client = self.cfg.create_api_client()
        try:
            details = await self.retry_read(
                "market_detail",
                lambda: lighter.OrderApi(api_client).order_book_details(
                    market_id=self.market_id,
                    filter=self.cfg.market.market_type,
                ),
            )
            for detail in details.order_book_details or []:
                if detail.market_id == self.market_id:
                    return detail
            raise RunnerError(f"Market detail not found for market {self.market_id}")
        finally:
            await api_client.close()

    async def settle(self) -> None:
        await asyncio.sleep(self.args.settle_seconds)

    def next_client_order_index(self) -> int:
        return (int(time.time() * 1000) * 100 + (time.monotonic_ns() % 100)) % MAX_CLIENT_ORDER_INDEX

    def next_nonce(self, signer: lighter.SignerClient, prior_api_key_index: Optional[int] = None) -> tuple[int, int]:
        if prior_api_key_index is None:
            return signer.nonce_manager.next_nonce(self.cfg.api_key_index)
        return signer.nonce_manager.next_nonce(prior_api_key_index)

    def sign_order(
        self,
        signer: lighter.SignerClient,
        label: str,
        *,
        client_order_index: int,
        base_amount: int,
        price: int,
        is_ask: bool,
        order_type: int,
        time_in_force: int,
        reduce_only: bool,
        trigger_price: int = 0,
        order_expiry: Optional[int] = None,
        prior_api_key_index: Optional[int] = None,
    ) -> SignedTx:
        api_key_index, nonce = self.next_nonce(signer, prior_api_key_index)
        tx_type, tx_info, tx_hash, err = signer.sign_create_order(
            market_index=self.market_id,
            client_order_index=client_order_index,
            base_amount=base_amount,
            price=price,
            is_ask=is_ask,
            order_type=order_type,
            time_in_force=time_in_force,
            reduce_only=reduce_only,
            trigger_price=trigger_price,
            order_expiry=(
                signer.DEFAULT_28_DAY_ORDER_EXPIRY if order_expiry is None else order_expiry
            ),
            nonce=nonce,
            api_key_index=api_key_index,
        )
        if err is not None:
            signer.nonce_manager.acknowledge_failure(api_key_index)
            raise RunnerError(f"{label} sign failed: {err}")
        return SignedTx(label, tx_type, tx_info, tx_hash, api_key_index, nonce)

    def sign_modify(
        self,
        signer: lighter.SignerClient,
        label: str,
        *,
        order_index: int,
        base_amount: int,
        price: int,
        trigger_price: int = 0,
        prior_api_key_index: Optional[int] = None,
    ) -> SignedTx:
        api_key_index, nonce = self.next_nonce(signer, prior_api_key_index)
        tx_type, tx_info, tx_hash, err = signer.sign_modify_order(
            market_index=self.market_id,
            order_index=order_index,
            base_amount=base_amount,
            price=price,
            trigger_price=trigger_price,
            nonce=nonce,
            api_key_index=api_key_index,
        )
        if err is not None:
            signer.nonce_manager.acknowledge_failure(api_key_index)
            raise RunnerError(f"{label} sign failed: {err}")
        return SignedTx(label, tx_type, tx_info, tx_hash, api_key_index, nonce)

    def sign_cancel(
        self,
        signer: lighter.SignerClient,
        label: str,
        *,
        order_index: int,
        prior_api_key_index: Optional[int] = None,
    ) -> SignedTx:
        api_key_index, nonce = self.next_nonce(signer, prior_api_key_index)
        tx_type, tx_info, tx_hash, err = signer.sign_cancel_order(
            market_index=self.market_id,
            order_index=order_index,
            nonce=nonce,
            api_key_index=api_key_index,
        )
        if err is not None:
            signer.nonce_manager.acknowledge_failure(api_key_index)
            raise RunnerError(f"{label} sign failed: {err}")
        return SignedTx(label, tx_type, tx_info, tx_hash, api_key_index, nonce)

    def sign_cancel_all(self, signer: lighter.SignerClient, label: str) -> SignedTx:
        api_key_index, nonce = self.next_nonce(signer)
        tx_type, tx_info, tx_hash, err = signer.sign_cancel_all_orders(
            time_in_force=signer.CANCEL_ALL_TIF_IMMEDIATE,
            timestamp_ms=0,
            cancel_all_market_index=self.market_id,
            nonce=nonce,
            api_key_index=api_key_index,
        )
        if err is not None:
            signer.nonce_manager.acknowledge_failure(api_key_index)
            raise RunnerError(f"{label} sign failed: {err}")
        return SignedTx(label, tx_type, tx_info, tx_hash, api_key_index, nonce)

    def sign_grouped(
        self,
        signer: lighter.SignerClient,
        label: str,
        *,
        grouping_type: int,
        orders: list[CreateOrderTxReq],
    ) -> SignedTx:
        api_key_index, nonce = self.next_nonce(signer)
        tx_type, tx_info, tx_hash, err = signer.sign_create_grouped_orders(
            grouping_type=grouping_type,
            orders=orders,
            nonce=nonce,
            api_key_index=api_key_index,
        )
        if err is not None:
            signer.nonce_manager.acknowledge_failure(api_key_index)
            raise RunnerError(f"{label} sign failed: {err}")
        return SignedTx(label, tx_type, tx_info, tx_hash, api_key_index, nonce)

    def sign_update_leverage_fraction_tx(
        self,
        signer: lighter.SignerClient,
        label: str,
        *,
        margin_mode: int,
        fraction: int,
    ) -> SignedTx:
        api_key_index, nonce = self.next_nonce(signer)
        tx_type, tx_info, tx_hash, err = signer.sign_update_leverage(
            market_index=self.market_id,
            fraction=fraction,
            margin_mode=margin_mode,
            nonce=nonce,
            api_key_index=api_key_index,
        )
        if err is not None:
            signer.nonce_manager.acknowledge_failure(api_key_index)
            raise RunnerError(f"{label} sign failed: {err}")
        return SignedTx(label, tx_type, tx_info, tx_hash, api_key_index, nonce)

    def sign_update_margin_tx(
        self,
        signer: lighter.SignerClient,
        label: str,
        *,
        usdc_amount: Decimal,
        direction: int,
    ) -> SignedTx:
        api_key_index, nonce = self.next_nonce(signer)
        usdc_int = int(usdc_amount * Decimal("1000000"))
        tx_type, tx_info, tx_hash, err = signer.sign_update_margin(
            market_index=self.market_id,
            usdc_amount=usdc_int,
            direction=direction,
            nonce=nonce,
            api_key_index=api_key_index,
        )
        if err is not None:
            signer.nonce_manager.acknowledge_failure(api_key_index)
            raise RunnerError(f"{label} sign failed: {err}")
        return SignedTx(label, tx_type, tx_info, tx_hash, api_key_index, nonce)

    async def sign_and_send_order(
        self,
        signer: lighter.SignerClient,
        label: str,
        transport: str,
        *,
        client_order_index: int,
        base_amount: int,
        price: int,
        is_ask: bool,
        order_type: int,
        time_in_force: int,
        reduce_only: bool,
        trigger_price: int = 0,
        order_expiry: Optional[int] = None,
        ws: Any = None,
    ) -> dict[str, Any]:
        signed = self.sign_order(
            signer,
            label,
            client_order_index=client_order_index,
            base_amount=base_amount,
            price=price,
            is_ask=is_ask,
            order_type=order_type,
            time_in_force=time_in_force,
            reduce_only=reduce_only,
            trigger_price=trigger_price,
            order_expiry=order_expiry,
        )
        return await self.send_signed(signer, signed, transport, ws=ws)

    async def sign_and_send_modify(
        self,
        signer: lighter.SignerClient,
        label: str,
        transport: str,
        *,
        order_index: int,
        base_amount: int,
        price: int,
        ws: Any = None,
    ) -> dict[str, Any]:
        signed = self.sign_modify(
            signer,
            label,
            order_index=order_index,
            base_amount=base_amount,
            price=price,
        )
        return await self.send_signed(signer, signed, transport, ws=ws)

    async def sign_and_send_cancel(
        self,
        signer: lighter.SignerClient,
        label: str,
        transport: str,
        *,
        order_index: int,
        ws: Any = None,
    ) -> dict[str, Any]:
        signed = self.sign_cancel(signer, label, order_index=order_index)
        return await self.send_signed(signer, signed, transport, ws=ws)

    async def sign_and_send_grouped(
        self,
        signer: lighter.SignerClient,
        label: str,
        *,
        grouping_type: int,
        orders: list[CreateOrderTxReq],
    ) -> dict[str, Any]:
        signed = self.sign_grouped(signer, label, grouping_type=grouping_type, orders=orders)
        return await self.send_signed_http(signer, signed)

    async def send_signed(
        self,
        signer: lighter.SignerClient,
        signed: SignedTx,
        transport: str,
        *,
        ws: Any = None,
    ) -> dict[str, Any]:
        if transport == "ws":
            if ws is None:
                raise RunnerError("WS transport requires an open websocket")
            return await self.send_signed_ws(ws, signed)
        return await self.send_signed_http(signer, signed)

    async def send_signed_http(
        self,
        signer: lighter.SignerClient,
        signed: SignedTx,
    ) -> dict[str, Any]:
        try:
            response = await signer.send_tx(signed.tx_type, signed.tx_info)
        except Exception:
            signer.nonce_manager.acknowledge_failure(signed.api_key_index)
            raise
        code = response_code(response)
        record = {
            "label": signed.label,
            "transport": "http",
            "tx_type": signed.tx_type,
            "tx_hash": signed.tx_hash,
            "api_key_index": signed.api_key_index,
            "nonce": signed.nonce,
            "response": response,
        }
        if code is not None and code != CODE_OK:
            signer.nonce_manager.acknowledge_failure(signed.api_key_index)
            raise RunnerError(f"{signed.label} HTTP send failed with code {code}")
        return record

    async def send_signed_ws(self, ws: Any, signed: SignedTx) -> dict[str, Any]:
        request_id = f"{signed.label}_{signed.nonce}"
        await ws.send(
            json.dumps(
                {
                    "type": "jsonapi/sendtx",
                    "data": {
                        "id": request_id,
                        "tx_type": signed.tx_type,
                        "tx_info": json.loads(signed.tx_info),
                    },
                }
            )
        )
        response = await self.recv_ws_json(ws)
        record = {
            "label": signed.label,
            "transport": "ws",
            "tx_type": signed.tx_type,
            "tx_hash": signed.tx_hash,
            "api_key_index": signed.api_key_index,
            "nonce": signed.nonce,
            "response": response,
        }
        self.assert_ws_response_ok(signed.label, response)
        return record

    async def send_batch_http(
        self,
        signer: lighter.SignerClient,
        label: str,
        signed_txs: list[SignedTx],
    ) -> dict[str, Any]:
        response = await signer.send_tx_batch(
            [tx.tx_type for tx in signed_txs],
            [tx.tx_info for tx in signed_txs],
        )
        code = response_code(response)
        if code is not None and code != CODE_OK:
            for signed in signed_txs:
                signer.nonce_manager.acknowledge_failure(signed.api_key_index)
            raise RunnerError(f"{label} HTTP batch failed with code {code}")
        return {
            "label": label,
            "transport": "http",
            "tx_hashes": [tx.tx_hash for tx in signed_txs],
            "nonces": [tx.nonce for tx in signed_txs],
            "response": response,
        }

    async def send_batch_ws(
        self,
        ws: Any,
        label: str,
        signed_txs: list[SignedTx],
    ) -> dict[str, Any]:
        await ws.send(
            json.dumps(
                {
                    "type": "jsonapi/sendtxbatch",
                    "data": {
                        "id": label,
                        "tx_types": json.dumps([tx.tx_type for tx in signed_txs]),
                        "tx_infos": json.dumps([tx.tx_info for tx in signed_txs]),
                    },
                }
            )
        )
        response = await self.recv_ws_json(ws)
        self.assert_ws_response_ok(label, response)
        return {
            "label": label,
            "transport": "ws",
            "tx_hashes": [tx.tx_hash for tx in signed_txs],
            "nonces": [tx.nonce for tx in signed_txs],
            "response": response,
        }

    async def market_order(
        self,
        signer: lighter.SignerClient,
        label: str,
        *,
        is_ask: bool,
        base_amount: int,
        reduce_only: bool,
    ) -> dict[str, Any]:
        top = await self.fetch_book_top()
        if is_ask:
            limit_price = price_decimal_to_int(
                self.cfg,
                top.bid * (Decimal("1") - self.slippage),
                ROUND_FLOOR,
            )
        else:
            limit_price = price_decimal_to_int(
                self.cfg,
                top.ask * (Decimal("1") + self.slippage),
                ROUND_CEILING,
            )
        return await self.sign_and_send_order(
            signer,
            label,
            "http",
            client_order_index=self.next_client_order_index(),
            base_amount=base_amount,
            price=limit_price,
            is_ask=is_ask,
            order_type=signer.ORDER_TYPE_MARKET,
            time_in_force=signer.ORDER_TIME_IN_FORCE_IMMEDIATE_OR_CANCEL,
            reduce_only=reduce_only,
            order_expiry=signer.DEFAULT_IOC_EXPIRY,
        )

    async def update_leverage(
        self,
        signer: lighter.SignerClient,
        label: str,
        *,
        margin_mode: int,
        leverage: Decimal,
        transport: str,
        ws: Any = None,
    ) -> dict[str, Any]:
        fraction = int(Decimal("10000") / leverage)
        return await self.update_leverage_fraction(
            signer,
            label,
            margin_mode=margin_mode,
            fraction=fraction,
            transport=transport,
            ws=ws,
        )

    async def update_leverage_fraction(
        self,
        signer: lighter.SignerClient,
        label: str,
        *,
        margin_mode: int,
        fraction: int,
        transport: str,
        ws: Any = None,
    ) -> dict[str, Any]:
        signed = self.sign_update_leverage_fraction_tx(
            signer,
            label,
            margin_mode=margin_mode,
            fraction=fraction,
        )
        return await self.send_signed(signer, signed, transport, ws=ws)

    async def update_margin(
        self,
        signer: lighter.SignerClient,
        label: str,
        *,
        usdc_amount: Decimal,
        direction: int,
        transport: str,
        ws: Any = None,
    ) -> dict[str, Any]:
        signed = self.sign_update_margin_tx(
            signer,
            label,
            usdc_amount=usdc_amount,
            direction=direction,
        )
        return await self.send_signed(signer, signed, transport, ws=ws)

    async def close_any_position(self, signer: lighter.SignerClient, *, label_prefix: str) -> Optional[dict[str, Any]]:
        position = await self.current_signed_position()
        if position == 0:
            return None
        base_amount = amount_decimal_to_int(self.cfg, abs(position))
        return await self.market_order(
            signer,
            f"{label_prefix}_close_position",
            is_ask=position > 0,
            base_amount=base_amount,
            reduce_only=True,
        )

    async def cancel_active_orders(self, signer: lighter.SignerClient, *, label_prefix: str) -> list[dict[str, Any]]:
        active = await self.fetch_active_orders(signer)
        cancellations = []
        for order in active.orders or []:
            cancellations.append(
                await self.sign_and_send_cancel(
                    signer,
                    f"{label_prefix}_cancel_{cancel_order_index(order)}",
                    "http",
                    order_index=cancel_order_index(order),
                )
            )
            await self.settle()
        return cancellations

    async def recv_ws_json(self, ws: Any) -> dict[str, Any]:
        deadline = time.monotonic() + self.args.ws_timeout
        while True:
            timeout = deadline - time.monotonic()
            if timeout <= 0:
                raise RunnerError("Timed out waiting for websocket message")
            raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
            if not isinstance(raw, str):
                raise RunnerError(f"Unexpected websocket binary payload: {raw!r}")
            message = json.loads(raw)
            if message.get("type") == "ping":
                await ws.send(json.dumps({"type": "pong"}))
                continue
            return message

    async def recv_ws_until(self, ws: Any, wanted_types: set[str]) -> dict[str, Any]:
        observed: list[str] = []
        deadline = time.monotonic() + self.args.ws_timeout
        while True:
            timeout = deadline - time.monotonic()
            if timeout <= 0:
                raise RunnerError(f"Timed out waiting for {sorted(wanted_types)}; saw {observed}")
            raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
            if not isinstance(raw, str):
                raise RunnerError(f"Unexpected websocket binary payload: {raw!r}")
            message = json.loads(raw)
            msg_type = message.get("type")
            observed.append(str(msg_type))
            if msg_type == "ping":
                await ws.send(json.dumps({"type": "pong"}))
                continue
            if msg_type in wanted_types:
                return message

    def assert_ws_response_ok(self, label: str, response: dict[str, Any]) -> None:
        if "error" in response:
            raise RunnerError(f"{label} WS response error: {response['error']}")
        data = response.get("data")
        if isinstance(data, dict) and "code" in data and int(data["code"]) != CODE_OK:
            raise RunnerError(f"{label} WS response code {data['code']}: {response}")
        if "code" in response and int(response["code"]) != CODE_OK:
            raise RunnerError(f"{label} WS response code {response['code']}: {response}")


async def execute_mode(runner: PerpFunctionalityRunner, mode: str) -> None:
    if mode == "preflight":
        await runner.run_step("preflight", lambda: runner.preflight(strict_neutral=True))
    elif mode == "public-data":
        await runner.run_step("public-data", runner.public_data)
    elif mode == "paper":
        await runner.run_step("paper", runner.paper_trading)
    elif mode == "limit-http":
        await runner.run_step("limit-http", lambda: runner.limit_lifecycle(transport="http"))
    elif mode == "limit-ws":
        await runner.run_step("limit-ws", lambda: runner.limit_lifecycle(transport="ws"))
    elif mode == "batch-http":
        await runner.run_step("batch-http", lambda: runner.batch_lifecycle(transport="http"))
    elif mode == "batch-ws":
        await runner.run_step("batch-ws", lambda: runner.batch_lifecycle(transport="ws"))
    elif mode == "market-open-close":
        await runner.run_step("market-open-close", runner.market_open_close)
    elif mode == "margin":
        await runner.run_step("margin", runner.margin_flow)
    elif mode == "sl-tp":
        await runner.run_step("sl-tp", runner.sl_tp_flow)
    elif mode == "cleanup":
        await runner.run_step("cleanup", runner.cleanup)
    elif mode == "mainnet-readonly":
        await runner.run_step("mainnet-preflight-readonly", lambda: runner.preflight(strict_neutral=False))
        await runner.run_step("mainnet-public-data", runner.public_data)
    elif mode == "full-testnet":
        await runner.run_step("preflight", lambda: runner.preflight(strict_neutral=True))
        await runner.run_step("public-data", runner.public_data)
        await runner.run_step("paper", runner.paper_trading)
        await runner.run_step("limit-http", lambda: runner.limit_lifecycle(transport="http"))
        await runner.run_step("limit-ws", lambda: runner.limit_lifecycle(transport="ws"))
        await runner.run_step("batch-http", lambda: runner.batch_lifecycle(transport="http"))
        await runner.run_step("batch-ws", lambda: runner.batch_lifecycle(transport="ws"))
        await runner.run_step("market-open-close", runner.market_open_close)
        await runner.run_step("margin", runner.margin_flow)
        await runner.run_step("sl-tp", runner.sl_tp_flow)
        await runner.run_step("cleanup", runner.cleanup)
    else:
        raise RunnerError(f"Unsupported mode: {mode}")


def validate_mode_safety(mode: str, cfg: LighterTradingConfig, args: argparse.Namespace) -> None:
    if mode == "mainnet-readonly":
        if not is_mainnet_config(cfg):
            raise RunnerError("mainnet-readonly requires LIGHTER_NETWORK=mainnet and a mainnet URL")
        return

    if is_mainnet_config(cfg):
        raise RunnerError(f"Mode {mode} is not allowed against mainnet")

    if mode in MUTATING_MODES and not args.i_understand_this_mutates_testnet:
        raise RunnerError(
            f"Mode {mode} mutates testnet state. Re-run with "
            "--i-understand-this-mutates-testnet to execute it."
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=MODES)
    parser.add_argument("--config", default="configs/lighter_trading.yml")
    parser.add_argument(
        "--env-file",
        default=None,
        help="Defaults to .env.testnet, or .env.mainnet for mainnet-readonly.",
    )
    parser.add_argument("--run-root", default="/tmp/lighter-perp-runs")
    parser.add_argument("--size-base", default="0.1")
    parser.add_argument("--settle-seconds", type=float, default=2.0)
    parser.add_argument("--ws-timeout", type=float, default=15.0)
    parser.add_argument("--paper-live-seconds", type=float, default=2.0)
    parser.add_argument("--max-slippage-bps", default="100")
    parser.add_argument("--api-retries", type=int, default=4)
    parser.add_argument("--retry-sleep-seconds", type=float, default=2.0)
    parser.add_argument(
        "--i-understand-this-mutates-testnet",
        action="store_true",
        help="Required for any testnet state-changing mode.",
    )
    parser.add_argument(
        "--no-cleanup-on-failure",
        dest="cleanup_on_failure",
        action="store_false",
        help="Do not attempt testnet cleanup after a mutating mode fails.",
    )
    parser.set_defaults(cleanup_on_failure=True)
    return parser


async def async_main(args: argparse.Namespace) -> int:
    env_file = Path(args.env_file or mode_default_env_file(args.mode))
    cfg = load_config_from_env_file(Path(args.config), env_file)
    validate_mode_safety(args.mode, cfg, args)

    audit = AuditRun(Path(args.run_root), cfg.network.lower(), args.mode, cfg, env_file)
    runner = PerpFunctionalityRunner(cfg, audit, args)
    print(f"Audit directory: {audit.path}")
    try:
        await execute_mode(runner, args.mode)
    except Exception:
        if (
            args.cleanup_on_failure
            and args.mode in MUTATING_MODES
            and args.mode != "cleanup"
            and not is_mainnet_config(cfg)
        ):
            try:
                await runner.run_step("cleanup-after-failure", runner.cleanup)
            except Exception as cleanup_exc:
                print(f"[cleanup-after-failure] failed: {cleanup_exc}")
        audit.finish("fail")
        raise
    audit.finish("pass")
    return 0


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        raise SystemExit(asyncio.run(async_main(args)))
    except (LighterConfigError, RunnerError) as exc:
        print(f"error: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
