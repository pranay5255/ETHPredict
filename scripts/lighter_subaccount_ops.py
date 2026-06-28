"""Dry-run-first Lighter subaccount helpers."""

from __future__ import annotations

import argparse
import asyncio
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import lighter

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.trading.lighter_config import (
    LighterConfigError,
    decimal_to_scaled_int,
    load_lighter_trading_config,
)


ROUTES = {
    "perp": lighter.SignerClient.ROUTE_PERP,
    "spot": lighter.SignerClient.ROUTE_SPOT,
}


def _as_dict(value: Any) -> dict:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return {}


def _decimal(value: str, name: str) -> Decimal:
    try:
        amount = Decimal(value)
    except InvalidOperation as exc:
        raise LighterConfigError(f"{name} must be a decimal number") from exc
    if amount <= 0:
        raise LighterConfigError(f"{name} must be positive")
    return amount


def _is_mainnet_url(base_url: str) -> bool:
    return "mainnet" in base_url.lower()


def _memo(value: str) -> str:
    if not value.startswith("0x") or len(value) != 66:
        raise LighterConfigError("memo must be 0x followed by exactly 32 bytes of hex")
    try:
        bytes.fromhex(value[2:])
    except ValueError as exc:
        raise LighterConfigError("memo must be valid hex") from exc
    return value


def _require_write_confirmation(args: argparse.Namespace, cfg) -> None:
    if not args.submit:
        raise LighterConfigError("Write operation was not submitted because --submit is missing")
    if _is_mainnet_url(cfg.base_url) and not args.confirm_mainnet:
        raise LighterConfigError("Mainnet writes require --confirm-mainnet")


async def list_l1_accounts(args: argparse.Namespace) -> int:
    cfg = load_lighter_trading_config(args.config, args.env_file)
    api_client = cfg.create_api_client()
    try:
        response = await lighter.AccountApi(api_client).accounts_by_l1_address(
            l1_address=args.l1_address
        )
        data = _as_dict(response)
        accounts = data.get("sub_accounts") or []
        if not accounts:
            print(f"No Lighter accounts found for {args.l1_address}")
            return 1
        for account in accounts:
            print(
                "Account: "
                f"index={account.get('index')} "
                f"type={account.get('account_type')} "
                f"status={account.get('status')} "
                f"collateral={account.get('collateral')}"
            )
        return 0
    finally:
        await api_client.close()


async def show_account(args: argparse.Namespace) -> int:
    cfg = load_lighter_trading_config(args.config, args.env_file)
    if cfg.account_index is None:
        raise LighterConfigError("LIGHTER_ACCOUNT_INDEX is required")

    api_client = cfg.create_api_client()
    try:
        account = await lighter.AccountApi(api_client).account(
            by="index",
            value=str(cfg.account_index),
        )
        data = _as_dict(account)
        accounts = data.get("accounts") or []
        if not accounts:
            print(f"No account found for index {cfg.account_index}")
            return 1
        acct = accounts[0]
        print(
            "Account: "
            f"index={acct.get('index')} "
            f"type={acct.get('account_type')} "
            f"status={acct.get('status')} "
            f"collateral={acct.get('collateral')} "
            f"available_balance={acct.get('available_balance')}"
        )
        for asset in acct.get("assets") or []:
            if asset.get("symbol") == "USDC":
                print(
                    "USDC: "
                    f"balance={asset.get('balance')} "
                    f"locked={asset.get('locked_balance')} "
                    f"margin={asset.get('margin_balance')}"
                )
        return 0
    finally:
        await api_client.close()


async def create_subaccount(args: argparse.Namespace) -> int:
    cfg = load_lighter_trading_config(args.config, args.env_file)
    if cfg.account_index is None:
        raise LighterConfigError("LIGHTER_ACCOUNT_INDEX is required")

    if not args.submit:
        print(
            "Dry run: would create a Lighter subaccount from "
            f"account_index={cfg.account_index} on {cfg.base_url}."
        )
        print("Pass --submit to send the create_sub_account transaction.")
        return 0

    _require_write_confirmation(args, cfg)
    cfg.require_trading_credentials()
    signer = cfg.create_signer_client()
    try:
        tx_info, response, err = await signer.create_sub_account()
        print(f"Create subaccount response: tx={tx_info} response={response} err={err}")
        return 1 if err is not None else 0
    finally:
        await signer.close()


async def transfer_usdc(args: argparse.Namespace) -> int:
    cfg = load_lighter_trading_config(args.config, args.env_file)
    if cfg.account_index is None:
        raise LighterConfigError("LIGHTER_ACCOUNT_INDEX is required")
    amount = _decimal(args.amount, "amount")
    route_from = ROUTES[args.route_from]
    route_to = ROUTES[args.route_to]
    memo = _memo(args.memo or ("0x" + "00" * 32))

    print(
        "Prepared USDC transfer: "
        f"from_account={cfg.account_index} to_account={args.to_account_index} "
        f"amount={amount} route_from={args.route_from} route_to={args.route_to} "
        f"network={cfg.network}"
    )

    if not args.submit:
        print("Dry run only. Pass --submit to send this transfer.")
        return 0

    _require_write_confirmation(args, cfg)
    signer = cfg.create_signer_client()
    try:
        auth_token, err = signer.create_auth_token_with_expiry(api_key_index=cfg.api_key_index)
        if err is not None:
            raise LighterConfigError(f"Auth token failed: {err}")
        assert cfg.account_index is not None
        fee_info = await lighter.InfoApi(signer.api_client).transfer_fee_info(
            authorization=auth_token,
            account_index=cfg.account_index,
            to_account_index=args.to_account_index,
        )
        fee = fee_info.transfer_fee_usdc
        usdc_amount = decimal_to_scaled_int(str(amount), 6, "amount")
        api_key_index, nonce = signer.nonce_manager.next_nonce(cfg.api_key_index)
        tx_type, tx_info, tx_hash, err = signer.sign_transfer_same_master_account(
            to_account_index=args.to_account_index,
            asset_id=signer.ASSET_ID_USDC,
            route_from=route_from,
            route_to=route_to,
            usdc_amount=usdc_amount,
            fee=fee,
            memo=memo,
            nonce=nonce,
            api_key_index=api_key_index,
        )
        if err is not None:
            print(f"USDC transfer signing failed: {err}")
            return 1
        response = await signer.send_tx(tx_type=tx_type, tx_info=tx_info)
        print(
            "USDC transfer response: "
            f"tx_hash={tx_hash} tx_info={tx_info} response={response}"
        )
        return 0
    finally:
        await signer.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/lighter_trading.yml")
    parser.add_argument("--env-file", default=".env")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_l1 = subparsers.add_parser("list-l1", help="List Lighter accounts for an L1 address")
    list_l1.add_argument("--l1-address", required=True)

    subparsers.add_parser("account", help="Show account and USDC state for LIGHTER_ACCOUNT_INDEX")

    create = subparsers.add_parser("create-subaccount", help="Create a subaccount from the current account")
    create.add_argument("--submit", action="store_true")
    create.add_argument("--confirm-mainnet", action="store_true")

    transfer = subparsers.add_parser("transfer-usdc", help="Transfer USDC to a same-master account")
    transfer.add_argument("--to-account-index", type=int, required=True)
    transfer.add_argument("--amount", required=True)
    transfer.add_argument("--route-from", choices=sorted(ROUTES), default="perp")
    transfer.add_argument("--route-to", choices=sorted(ROUTES), default="perp")
    transfer.add_argument("--memo", default=None, help="32-byte hex memo; defaults to all zeros")
    transfer.add_argument("--submit", action="store_true")
    transfer.add_argument("--confirm-mainnet", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        if args.command == "list-l1":
            raise SystemExit(asyncio.run(list_l1_accounts(args)))
        if args.command == "account":
            raise SystemExit(asyncio.run(show_account(args)))
        if args.command == "create-subaccount":
            raise SystemExit(asyncio.run(create_subaccount(args)))
        if args.command == "transfer-usdc":
            raise SystemExit(asyncio.run(transfer_usdc(args)))
    except LighterConfigError as exc:
        print(f"error: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
