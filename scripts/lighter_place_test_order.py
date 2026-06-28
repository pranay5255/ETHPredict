"""Place a guarded Lighter testnet limit order."""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.trading.lighter_config import LighterConfigError, load_lighter_trading_config


def _side_to_is_ask(side: str) -> bool:
    normalized = side.lower()
    if normalized == "sell":
        return True
    if normalized == "buy":
        return False
    raise LighterConfigError("--side must be buy or sell")


async def place_order(args: argparse.Namespace) -> int:
    cfg = load_lighter_trading_config(args.config, args.env_file)
    cfg.validate_network_safety()
    cfg.market.assert_min_base_amount(args.size_base)

    base_amount = cfg.market.base_amount_to_int(args.size_base)
    price = cfg.market.price_to_int(args.price)
    client_order_index = args.client_order_index or int(time.time() * 1000)
    is_ask = _side_to_is_ask(args.side)

    print(
        "Prepared order: "
        f"network={cfg.network} market={cfg.market.market_index} "
        f"side={args.side} size={args.size_base} price={args.price} "
        f"base_amount={base_amount} price_int={price} "
        f"client_order_index={client_order_index}"
    )

    if not args.submit:
        print("Dry run only. Pass --submit to send this order to Lighter testnet.")
        return 0

    cfg.require_trading_credentials()
    signer = cfg.create_signer_client()
    try:
        api_key_index, nonce = signer.nonce_manager.next_nonce(cfg.api_key_index)
        tx, response, err = await signer.create_order(
            market_index=cfg.market.market_index,
            client_order_index=client_order_index,
            base_amount=base_amount,
            price=price,
            is_ask=is_ask,
            order_type=signer.ORDER_TYPE_LIMIT,
            time_in_force=signer.ORDER_TIME_IN_FORCE_GOOD_TILL_TIME,
            reduce_only=args.reduce_only,
            trigger_price=signer.NIL_TRIGGER_PRICE,
            nonce=nonce,
            api_key_index=api_key_index,
        )
        print(f"Create order response: tx={tx} response={response} err={err}")
        if err is not None:
            return 1

        if args.cancel_after_submit:
            api_key_index, nonce = signer.nonce_manager.next_nonce(api_key_index)
            cancel_tx, cancel_response, cancel_err = await signer.cancel_order(
                market_index=cfg.market.market_index,
                order_index=client_order_index,
                nonce=nonce,
                api_key_index=api_key_index,
            )
            print(
                "Cancel order response: "
                f"tx={cancel_tx} response={cancel_response} err={cancel_err}"
            )
            if cancel_err is not None:
                return 1
    finally:
        await signer.close()

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/lighter_trading.yml")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--side", choices=["buy", "sell"], default="sell")
    parser.add_argument("--size-base", default="0.0050", help="Human base size, e.g. 0.0050 ETH")
    parser.add_argument("--price", required=True, help="Human limit price, e.g. 4050.00")
    parser.add_argument("--client-order-index", type=int, default=None)
    parser.add_argument("--reduce-only", action="store_true")
    parser.add_argument("--submit", action="store_true", help="Actually send the order")
    parser.add_argument(
        "--no-cancel-after-submit",
        dest="cancel_after_submit",
        action="store_false",
        help="Leave the submitted testnet order open",
    )
    parser.set_defaults(cancel_after_submit=True)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        raise SystemExit(asyncio.run(place_order(args)))
    except LighterConfigError as exc:
        print(f"error: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
