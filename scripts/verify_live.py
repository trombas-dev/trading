"""
verify_live.py — MT5 paper-trade round-trip verification.

Runs ONE tick of the MTF loop in dry-run mode and prints a full diagnostic
report: bar fetch, structure detection, signal detection, spread fetch.
No position is ever opened.

Usage
-----
  # Verify EURUSD against live MT5 (requires terminal running)
  uv run python scripts/verify_live.py --symbol EURUSD --source mt5

  # Verify against parquet (offline, no MT5 needed)
  uv run python scripts/verify_live.py --symbol BTCUSD --source parquet

  # Verify all symbols
  uv run python scripts/verify_live.py --all --source mt5
"""

from __future__ import annotations

import argparse
import asyncio
import io
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
sys.path.insert(0, str(ROOT))


ALL_SYMBOLS = [
    "BTCUSD", "EURUSD", "GBPUSD", "USDJPY", "XAUUSD",
    "GBPJPY", "EURJPY", "AUDUSD", "USDCAD", "NZDUSD",
]


async def verify_symbol(symbol: str, source: str) -> dict:
    """
    Run one tick of the MTF pipeline for `symbol` and return a status dict.
    """
    result = {
        "symbol":   symbol,
        "source":   source,
        "ok":       False,
        "error":    None,
        "bars":     {},
        "spread":   None,
        "bias":     None,
        "n_choch":  None,
        "n_windows": None,
        "signal":   None,
    }

    try:
        import yaml
        from hermes_trading.adapters.mt5_price import fetch_mtf_bars, fetch_half_spread
        from hermes_trading.strategy.backtest  import BacktestParams
        from hermes_trading.strategy.live      import get_live_signal

        # ── Load strategy + apply per-symbol overrides ────────────────────────
        state_path = ROOT / "state" / "strategy.yaml"
        with open(state_path) as f:
            raw = yaml.safe_load(f)

        # Simple merge (mirrors loop._merge_symbol_config)
        symbols_cfg = raw.get("symbols", {})
        overrides   = symbols_cfg.get(symbol) or symbols_cfg.get("default") or {}
        s = {**raw, **overrides}

        params = BacktestParams(
            n_bias            = int(s.get("n_bias",            9)),
            n_signal          = int(s.get("n_signal",          3)),
            n_entry           = int(s.get("n_entry",           3)),
            zone_lo           = float(s.get("zone_lo",         0.236)),
            zone_hi           = float(s.get("zone_hi",         0.618)),
            timeout_bars      = int(s.get("timeout_bars",      6)),
            max_bars_in_trade = int(s.get("max_bars_in_trade", 100)),
            scenario          = int(s.get("scenario",          1)),
            use_bos_windows   = bool(s.get("use_bos_windows",  False)),
            bos_fib_mode      = str(s.get("bos_fib_mode",      "last_leg")),
        )

        tf_bias, tf_sig, tf_ent = params.tfs()

        n_bars_map = {
            "1D":  int(s.get("n_bars_1d",  250)),
            "4H":  int(s.get("n_bars_4h",  500)),
            "1H":  int(s.get("n_bars_1h",  1000)),
            "15M": int(s.get("n_bars_15m", 500)),
            "5M":  int(s.get("n_bars_5m",  500)),
        }

        # ── Fetch bars ────────────────────────────────────────────────────────
        # For parquet source, load from disk
        if source == "parquet":
            import pandas as pd

            sym_dir = DATA_DIR / symbol
            dfs = {}
            for tf in ("1D", "4H", "1H", "15M", "5M"):
                path = sym_dir / f"{tf}.parquet"
                if path.exists():
                    df = pd.read_parquet(path, engine="pyarrow")
                    df.columns = [c.lower() for c in df.columns]
                    if df.index.tzinfo is None:
                        df.index = df.index.tz_localize("UTC")
                    dfs[tf] = df.iloc[-n_bars_map.get(tf, 500):]
        else:
            dfs = await fetch_mtf_bars(
                symbol, source=source,
                scenario = params.scenario,
                n_bias   = n_bars_map[tf_bias],
                n_signal = n_bars_map[tf_sig],
                n_entry  = n_bars_map[tf_ent],
            )

        result["bars"] = {tf: len(df) for tf, df in dfs.items()}

        # Verify required TFs are present
        for tf in (tf_bias, tf_sig, tf_ent):
            if tf not in dfs or len(dfs[tf]) == 0:
                result["error"] = f"Missing or empty TF: {tf}"
                return result

        # ── Fetch spread ──────────────────────────────────────────────────────
        half_spread = await fetch_half_spread(symbol, source=source, extra_pts=3)
        result["spread"] = round(half_spread * 2, 6)   # show full round-trip

        # ── Run live pipeline ─────────────────────────────────────────────────
        entry_signal, ctx = get_live_signal(dfs, params)

        result["bias"]     = ctx.get("bias", "neutral")
        result["n_choch"]  = ctx.get("n_choch", 0)
        result["n_windows"] = ctx.get("n_windows", 0)
        result["signal"]   = (
            f"{entry_signal.direction} @ {entry_signal.entry_price:.4f}  "
            f"SL={entry_signal.stop_loss:.4f}  "
            f"TP1={entry_signal.tp1:.4f}  "
            f"RR={entry_signal.rr1:.2f}"
        ) if entry_signal else None

        result["ok"] = True

    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["trace"] = traceback.format_exc()

    return result


def _print_report(results: list[dict]) -> None:
    try:
        from rich.table import Table
        from rich import box
        from rich.console import Console

        t = Table(
            title="Hermes Live Verification",
            box=box.SIMPLE_HEAD, show_edge=False,
        )
        for col in ["Symbol", "Status", "Bars (bias/sig/ent)",
                    "Spread (RT)", "Bias", "CHoCH", "Windows", "Entry signal"]:
            t.add_column(col)

        for r in results:
            tf_labels = list(r["bars"].values())[:3]
            bars_str  = "/".join(str(b) for b in tf_labels) if tf_labels else "—"
            status    = "[green]OK[/]" if r["ok"] else "[red]FAIL[/]"
            t.add_row(
                r["symbol"],
                status,
                bars_str,
                f"{r['spread']}" if r["spread"] is not None else "—",
                r["bias"] or "—",
                str(r["n_choch"]) if r["n_choch"] is not None else "—",
                str(r["n_windows"]) if r["n_windows"] is not None else "—",
                r["signal"] or ("—" if r["ok"] else f"[red]{r['error']}[/]"),
            )

        Console().print(t)

        ok     = sum(r["ok"] for r in results)
        failed = len(results) - ok
        print(f"\n  {ok}/{len(results)} OK   {failed} failed")

        for r in results:
            if not r["ok"] and r.get("trace"):
                print(f"\n--- {r['symbol']} traceback ---")
                print(r["trace"])

    except ImportError:
        # Fallback without rich
        for r in results:
            status = "OK" if r["ok"] else f"FAIL: {r['error']}"
            print(f"  {r['symbol']:10s}  {status}")


async def main_async(args: argparse.Namespace) -> None:
    symbols = ALL_SYMBOLS if args.all else [args.symbol]
    print(f"\n[verify_live]  source={args.source}  symbols={symbols}\n")

    tasks = [verify_symbol(sym, args.source) for sym in symbols]
    results = await asyncio.gather(*tasks)
    _print_report(list(results))


def main() -> None:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Verify Hermes live pipeline end-to-end (dry-run, no orders)."
    )
    parser.add_argument("--symbol", default="EURUSD",
                        help="Symbol to verify (default: EURUSD)")
    parser.add_argument("--all",    action="store_true",
                        help="Verify all 10 symbols")
    parser.add_argument("--source", choices=["mt5", "parquet", "yfinance"],
                        default="parquet",
                        help="Data source (default: parquet)")
    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
