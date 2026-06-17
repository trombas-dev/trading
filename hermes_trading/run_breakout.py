"""
hermes_trading/run_breakout.py — Session Breakout portfolio orchestrator.

Spawns one BreakoutTradingLoop per active instrument, all concurrently.
State lives under state/breakout/{SYMBOL}/ so it is separate from the
Fibonacci strategy state while still being visible to the shared
status server (via the updated status_server.py).

Usage:
  uv run python -m hermes_trading.run_breakout                      # all active
  uv run python -m hermes_trading.run_breakout --asset BTCUSD       # single symbol
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import yaml
from rich.logging import RichHandler

from hermes_trading.loop_breakout import BreakoutTradingLoop
import MetaTrader5 as mt5

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)


def _init_mt5(cfg: dict) -> None:
    ok = mt5.initialize(
        server   = cfg.get("mt5_server",   "ICMarketsEU-Demo"),
        login    = int(cfg.get("mt5_login",    52037890)),
        password = cfg.get("mt5_password", "$8lyQHf3PnjvAx"),
    )
    if not ok:
        raise RuntimeError(f"MT5 init failed: {mt5.last_error()}")
    info = mt5.account_info()
    logger.info(f"MT5 connected: {info.login}  balance {info.balance:.2f} {info.currency}")


def get_active_symbols(strategy_path: Path) -> list[str]:
    try:
        with open(strategy_path) as f:
            cfg = yaml.safe_load(f)
        syms_cfg = cfg.get("symbols", {})
        return [
            sym for sym, params in syms_cfg.items()
            if sym != "default"
            and isinstance(params, dict)
            and params.get("active", True)
        ]
    except Exception as exc:
        logger.warning(f"Could not load active symbols: {exc}")
        return []


async def run_portfolio(
    state_dir: Path,
    strategy_path: Path,
    symbols: list[str],
) -> None:
    logger.info(
        f"Breakout portfolio starting — {len(symbols)} instruments: "
        f"{', '.join(symbols)}"
    )

    loops: list[BreakoutTradingLoop] = []
    for sym in symbols:
        sym_dir = state_dir / sym
        sym_dir.mkdir(parents=True, exist_ok=True)
        loop = BreakoutTradingLoop(
            asset         = sym,
            state_dir     = sym_dir,
            strategy_path = strategy_path,
        )
        loops.append(loop)
        logger.info(f"  + {sym}  state={sym_dir}")

    results = await asyncio.gather(
        *[loop.run() for loop in loops],
        return_exceptions=True,
    )
    for sym, res in zip(symbols, results):
        if isinstance(res, Exception):
            logger.error(f"[{sym}] loop terminated: {res}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Hermes Breakout Worker")
    parser.add_argument("--asset",     type=str,  default="")
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=Path(__file__).parent.parent / "state" / "breakout",
    )
    args = parser.parse_args()

    strategy_path = args.state_dir.parent / "breakout_strategy.yaml"
    if not strategy_path.exists():
        logger.error(f"breakout_strategy.yaml not found at {strategy_path}")
        sys.exit(1)

    with open(strategy_path) as f:
        cfg = yaml.safe_load(f)

    _init_mt5(cfg)

    if args.asset:
        symbols = [args.asset]
    else:
        symbols = get_active_symbols(strategy_path)

    if not symbols:
        logger.error("No active symbols found in breakout_strategy.yaml")
        sys.exit(1)

    try:
        asyncio.run(run_portfolio(args.state_dir, strategy_path, symbols))
    except KeyboardInterrupt:
        logger.info("Shutdown requested.")
    except RuntimeError as e:
        logger.critical(str(e))
        sys.exit(1)
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()
