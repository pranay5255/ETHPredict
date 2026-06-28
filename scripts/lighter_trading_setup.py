"""Inspect and bootstrap Lighter testnet trading configuration."""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import lighter
from dotenv import set_key
from eth_account import Account

from src.trading.lighter_config import (
    API_KEY_SETUP_ENV_SPECS,
    OPTIONAL_ENV_SPECS,
    REQUIRED_TRADING_ENV_SPECS,
    LighterConfigError,
    env_value_status,
    load_lighter_trading_config,
)


def _to_dict(value: Any) -> dict:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return {}


def _print_env_status() -> None:
    groups = [
        ("Required for normal trading", REQUIRED_TRADING_ENV_SPECS),
        ("Required only to create or rotate API keys", API_KEY_SETUP_ENV_SPECS),
        ("Optional overrides", OPTIONAL_ENV_SPECS),
    ]
    for title, specs in groups:
        print(title)
        for spec, is_set in env_value_status(specs):
            marker = "set" if is_set else "missing"
            print(f"  {spec.name}: {marker} - {spec.description}")


def _first_market_detail(details: Any, market_type: str) -> Optional[dict]:
    data = _to_dict(details)
    key = "spot_order_book_details" if market_type == "spot" else "order_book_details"
    entries = data.get(key) or []
    return entries[0] if entries else None


async def check_setup(args: argparse.Namespace) -> int:
    cfg = load_lighter_trading_config(args.config, args.env_file)
    print(f"Network: {cfg.network}")
    print(f"REST: {cfg.base_url}")
    print(f"WS: {cfg.websocket_url}")
    print(
        f"Market: {cfg.market.symbol} {cfg.market.market_type} "
        f"index={cfg.market.market_index}"
    )

    api_client = cfg.create_api_client()
    try:
        status = await lighter.RootApi(api_client).status()
        print(f"Status endpoint: ok code={getattr(status, 'code', 'unknown')}")

        details = await lighter.OrderApi(api_client).order_book_details(
            market_id=cfg.market.market_index,
            filter=cfg.market.market_type,
        )
        detail = _first_market_detail(details, cfg.market.market_type)
        if not detail:
            print("Order book details: no matching market returned")
            return 1
        print(
            "Order book details: ok "
            f"symbol={detail.get('symbol')} "
            f"status={detail.get('status')} "
            f"size_decimals={detail.get('size_decimals')} "
            f"price_decimals={detail.get('price_decimals')}"
        )

        if getattr(args, "public_only", False):
            return 0

        if cfg.account_index is None:
            print("Account check: skipped, LIGHTER_ACCOUNT_INDEX is missing")
        else:
            await lighter.AccountApi(api_client).account(
                by="index",
                value=str(cfg.account_index),
            )
            print(f"Account check: ok account_index={cfg.account_index}")

        if cfg.account_index is not None and cfg.api_key_index is not None:
            nonce = await lighter.TransactionApi(api_client).next_nonce(
                account_index=cfg.account_index,
                api_key_index=cfg.api_key_index,
            )
            print(f"Next nonce: {nonce.nonce}")
        else:
            print("Next nonce: skipped, account/api key index missing")

    finally:
        await api_client.close()

    if not cfg.has_trading_credentials:
        print("Signer check: skipped, normal trading env vars are incomplete")
        _print_env_status()
        return 1

    signer = cfg.create_signer_client()
    try:
        err = signer.check_client()
        if err is not None:
            print(f"Signer check: failed: {err}")
            return 1
        print("Signer check: ok")

        auth, err = signer.create_auth_token_with_expiry(api_key_index=cfg.api_key_index)
        if err is not None:
            print(f"Auth token check: failed: {err}")
            return 1
        assert cfg.account_index is not None
        active_orders = await signer.order_api.account_active_orders(
            authorization=auth,
            account_index=cfg.account_index,
            market_id=cfg.market.market_index,
            market_type=cfg.market.market_type,
        )
        print(f"Active orders auth check: ok count={len(active_orders.orders or [])}")
    finally:
        await signer.close()

    return 0


async def register_api_key(args: argparse.Namespace) -> int:
    cfg = load_lighter_trading_config(args.config, args.env_file)
    cfg.require_eth_private_key()
    assert cfg.eth_private_key is not None

    api_key_index = args.api_key_index if args.api_key_index is not None else cfg.api_key_index
    if api_key_index is None:
        raise LighterConfigError("Set LIGHTER_API_KEY_INDEX or pass --api-key-index")

    api_client = cfg.create_api_client()
    tx_client = None
    try:
        account_index = cfg.account_index
        if account_index is None:
            eth_address = Account.from_key(cfg.eth_private_key).address
            accounts = await lighter.AccountApi(api_client).accounts_by_l1_address(
                l1_address=eth_address
            )
            account_index = min(int(account.index) for account in accounts.sub_accounts)
            print(f"Resolved account_index={account_index} for {eth_address}")

        private_key, public_key, err = lighter.create_api_key()
        if err is not None:
            raise LighterConfigError(err)

        tx_client = lighter.SignerClient(
            url=cfg.base_url,
            account_index=account_index,
            api_private_keys={api_key_index: private_key},
            nonce_management_type=cfg.nonce_manager_type(),
        )
        response, err = await tx_client.change_api_key(
            eth_private_key=cfg.eth_private_key,
            new_pubkey=public_key,
            api_key_index=api_key_index,
        )
        if err is not None:
            raise LighterConfigError(err)
        print(f"Submitted change_api_key tx for api_key_index={api_key_index}: {response}")

        if args.settle_seconds > 0:
            time.sleep(args.settle_seconds)

        err = tx_client.check_client()
        if err is not None:
            raise LighterConfigError(err)
        print("Registered API key verified")

        env_file = Path(args.env_file)
        env_file.touch(exist_ok=True)
        set_key(str(env_file), "LIGHTER_ACCOUNT_INDEX", str(account_index))
        set_key(str(env_file), "LIGHTER_API_KEY_INDEX", str(api_key_index))
        set_key(str(env_file), "LIGHTER_API_PRIVATE_KEY", private_key)
        set_key(str(env_file), "LIGHTER_BASE_URL", cfg.base_url)
        set_key(str(env_file), "LIGHTER_NETWORK", cfg.network)
        print(f"Wrote trading credentials to {env_file}")
    finally:
        if tx_client is not None:
            await tx_client.close()
        await api_client.close()

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/lighter_trading.yml")
    parser.add_argument("--env-file", default=".env")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("env", help="Show required Lighter environment variables")
    public = subparsers.add_parser("public", help="Verify public testnet endpoint and ETH market metadata")
    public.set_defaults(public_only=True)

    check = subparsers.add_parser("check", help="Verify testnet endpoint, account, API key, nonce, and auth")
    check.set_defaults(public_only=False)

    register = subparsers.add_parser("register-api-key", help="Create and register a Lighter API key")
    register.add_argument("--api-key-index", type=int, default=None)
    register.add_argument("--settle-seconds", type=int, default=10)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.command == "env":
            load_lighter_trading_config(args.config, args.env_file)
            _print_env_status()
            raise SystemExit(0)
        if args.command in {"public", "check"}:
            raise SystemExit(asyncio.run(check_setup(args)))
        if args.command == "register-api-key":
            raise SystemExit(asyncio.run(register_api_key(args)))
    except LighterConfigError as exc:
        print(f"error: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
