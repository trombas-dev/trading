"""
hermes_trading/run.py — Multi-symbol paper trading orchestrator.

Spawns one MTFTradingLoop per active symbol (all those without
``active: false`` in strategy.yaml), all running concurrently inside a
single asyncio event loop.  A single shared status HTTP server is started
on $PORT and reads from every per-symbol subdirectory.

Usage:
  uv run python -m hermes_trading.run              # all active symbols
  uv run python -m hermes_trading.run --asset GBPAUD  # single symbol (debug)
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import yaml
from rich.logging import RichHandler

from hermes_trading.loop import MTFTradingLoop

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)

# Canonical fallback order — matches strategy.yaml active section
_FALLBACK_SYMBOLS = ["GBPAUD", "BTCUSD", "GBPUSD", "USDCHF", "XAUUSD", "NZDUSD"]


def load_goal(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_active_symbols(strategy_path: Path) -> list[str]:
    """Return symbols that do *not* have ``active: false`` in strategy.yaml."""
    try:
        with open(strategy_path) as f:
            cfg = yaml.safe_load(f)
        symbols_cfg = cfg.get("symbols", {})
        active = [
            sym
            for sym, params in symbols_cfg.items()
            if sym != "default"
            and isinstance(params, dict)
            and params.get("active", True)
        ]
        return active if active else _FALLBACK_SYMBOLS
    except Exception as exc:
        logger.warning(f"Could not read active symbols from strategy.yaml: {exc}")
        return _FALLBACK_SYMBOLS


async def run_portfolio(
    state_dir: Path,
    strategy_path: Path,
    goal: dict,
    symbols: list[str],
) -> None:
    """Start the status server + one loop per symbol, all concurrently."""
    logger.info(f"Hermes portfolio starting — {len(symbols)} symbols: {', '.join(symbols)}")

    # Single shared status server on the root state dir
    try:
        from hermes_trading.status_server import start as start_status
        asyncio.create_task(start_status(state_dir))
        logger.info("Status server scheduled on $PORT")
    except Exception as exc:
        logger.warning(f"Status server could not start: {exc}")

    # Create per-symbol dirs and loop instances
    loops: list[MTFTradingLoop] = []
    for symbol in symbols:
        sym_dir = state_dir / symbol
        sym_dir.mkdir(parents=True, exist_ok=True)
        loop = MTFTradingLoop(
            asset=symbol,
            goal=goal,
            state_dir=sym_dir,
            strategy_path=strategy_path,
        )
        loops.append(loop)
        logger.info(f"  + {symbol}  state={sym_dir}")

    # Run all loops concurrently — an exception in one does not kill the others
    results = await asyncio.gather(
        *[loop.run(start_status_server=False) for loop in loops],
        return_exceptions=True,
    )
    for sym, res in zip(symbols, results):
        if isinstance(res, Exception):
            logger.error(f"[{sym}] loop terminated: {res}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Hermes Trading Worker")
    parser.add_argument(
        "--asset",
        type=str,
        default="",
        help="Single symbol override (leave blank to run all active symbols)",
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=Path(__file__).parent.parent / "state",
        help="Root state directory (per-symbol subdirs are created automatically)",
    )
    args = parser.parse_args()

    state_dir     = args.state_dir
    strategy_path = state_dir / "strategy.yaml"
    goal_path     = state_dir / "goal.yaml"

    if not goal_path.exists():
        logger.error(f"goal.yaml not found at {goal_path}")
        sys.exit(1)

    goal = load_goal(goal_path)

    if args.asset:
        symbols = [args.asset]
        logger.info(f"Single-symbol mode: {args.asset}")
    else:
        symbols = get_active_symbols(strategy_path)

    try:
        asyncio.run(run_portfolio(state_dir, strategy_path, goal, symbols))
    except KeyboardInterrupt:
        logger.info("Shutdown requested.")
    except RuntimeError as e:
        logger.critical(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
