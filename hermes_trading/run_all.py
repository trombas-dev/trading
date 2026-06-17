"""
hermes_trading/run_all.py — Combined orchestrator.

Runs both strategies concurrently in a single process:
  1. MTF Fibonacci loops  (existing, one per symbol)
  2. Session Breakout loops  (new, one per instrument)

Single process = single filesystem = status server sees everything.

Usage:
  uv run python -m hermes_trading.run_all
"""

import asyncio
import logging
import sys
from pathlib import Path

import yaml
from rich.logging import RichHandler

from hermes_trading.loop         import MTFTradingLoop
from hermes_trading.loop_breakout import BreakoutTradingLoop

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)

_FIB_FALLBACK  = ["GBPAUD", "BTCUSD", "GBPUSD", "USDCHF", "XAUUSD", "NZDUSD"]
_BO_FALLBACK   = ["XNGUSD", "XAUUSD", "BTCUSD", "US500", "US30", "XTIUSD", "JP225", "XBRUSD"]


def _active_fib_symbols(strategy_path: Path) -> list[str]:
    try:
        with open(strategy_path) as f:
            cfg = yaml.safe_load(f)
        syms = cfg.get("symbols", {})
        result = [s for s, p in syms.items()
                  if s != "default" and isinstance(p, dict) and p.get("active", True)]
        return result if result else _FIB_FALLBACK
    except Exception:
        return _FIB_FALLBACK


def _active_bo_symbols(strategy_path: Path) -> list[str]:
    try:
        with open(strategy_path) as f:
            cfg = yaml.safe_load(f)
        syms = cfg.get("symbols", {})
        return [s for s, p in syms.items()
                if s != "default" and isinstance(p, dict) and p.get("active", True)]
    except Exception:
        return _BO_FALLBACK


async def main_async(state_dir: Path) -> None:
    fib_strategy_path = state_dir / "strategy.yaml"
    bo_strategy_path  = state_dir / "breakout_strategy.yaml"
    goal_path         = state_dir / "goal.yaml"

    if not goal_path.exists():
        logger.error(f"goal.yaml not found at {goal_path}")
        sys.exit(1)

    with open(goal_path) as f:
        goal = yaml.safe_load(f)

    fib_symbols = _active_fib_symbols(fib_strategy_path)
    bo_symbols  = _active_bo_symbols(bo_strategy_path)

    logger.info(f"Fibonacci  symbols : {', '.join(fib_symbols)}")
    logger.info(f"Breakout instruments: {', '.join(bo_symbols)}")

    # ── Status server (single shared instance) ──────────────────────────────
    try:
        from hermes_trading.status_server import start as start_status
        asyncio.create_task(start_status(state_dir))
        logger.info("Status server started")
    except Exception as exc:
        logger.warning(f"Status server could not start: {exc}")

    # ── Fibonacci loops ──────────────────────────────────────────────────────
    fib_loops = []
    for sym in fib_symbols:
        sym_dir = state_dir / sym
        sym_dir.mkdir(parents=True, exist_ok=True)
        fib_loops.append(MTFTradingLoop(
            asset         = sym,
            goal          = goal,
            state_dir     = sym_dir,
            strategy_path = fib_strategy_path,
        ))
        logger.info(f"  [Fib] + {sym}")

    # ── Breakout loops ───────────────────────────────────────────────────────
    bo_loops = []
    bo_state_dir = state_dir / "breakout"
    for sym in bo_symbols:
        sym_dir = bo_state_dir / sym
        sym_dir.mkdir(parents=True, exist_ok=True)
        bo_loops.append(BreakoutTradingLoop(
            asset         = sym,
            state_dir     = sym_dir,
            strategy_path = bo_strategy_path,
        ))
        logger.info(f"  [BO]  + {sym}")

    # ── Run everything concurrently ─────────────────────────────────────────
    all_loops = (
        [loop.run(start_status_server=False) for loop in fib_loops] +
        [loop.run() for loop in bo_loops]
    )
    results = await asyncio.gather(*all_loops, return_exceptions=True)

    symbols = fib_symbols + bo_symbols
    for sym, res in zip(symbols, results):
        if isinstance(res, Exception):
            logger.error(f"[{sym}] terminated: {res}")


def main() -> None:
    state_dir = Path(__file__).parent.parent / "state"
    try:
        asyncio.run(main_async(state_dir))
    except KeyboardInterrupt:
        logger.info("Shutdown requested.")
    except RuntimeError as e:
        logger.critical(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
