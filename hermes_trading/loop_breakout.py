"""
loop_breakout.py — Session Breakout strategy live loop for Hermes.

Runs alongside the existing MTF Fibonacci loop (loop.py).
Uses identical state layout so status_server.py picks it up automatically:

  state/breakout/{SYMBOL}/
    trades.jsonl
    heartbeat.json

Strategy:
  Entry  : Session open breakout + retest on H1 bars
           Sessions: Asia (00-08), London (08-16), NY (13-21)
           Entry window: first 4 h of each session
           Breakout: close > resistance (or < support) with volume > 1.5x avg
           Retest: next 1-8 H1 bars pull back to within 0.35 ATR of broken level
  Exit   : Session close — hold until the end of the breakout session's H1 bars
           (or stop loss if price hits stop before session ends)
  Risk   : 1% of account per trade (configurable via breakout_strategy.yaml)
  Spread : Live bid-ask from MT5 tick; applied to entry and risk
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

from hermes_trading.adapters.mt5_price import fetch_half_spread, _fetch_bars
from hermes_trading.strategy_breakout.signal import (
    BreakoutSignal,
    SESSIONS,
    find_latest_entry,
    session_end_bar,
    _atr,
    _avg_vol,
    build_session_hl,
    attach_sr,
)

logger = logging.getLogger(__name__)

LOOP_INTERVAL_S          = 300      # 5-minute cadence (same as MTFTradingLoop)
MAX_CONSECUTIVE_FAILURES = 5
H1_BARS_TO_FETCH         = 2000     # ~83 days of H1 history


# ── H1 data fetch (MT5 primary, yfinance fallback on Linux/Railway) ───────────

async def _fetch_h1(symbol: str, source: str = "mt5") -> pd.DataFrame:
    """
    Fetch H1 bars — tries MT5 first, falls back to yfinance automatically.
    On Railway (Linux) MT5 is unavailable so yfinance is always used.
    """
    return await _fetch_bars(symbol, "1H", H1_BARS_TO_FETCH, source)


# ── Main loop ─────────────────────────────────────────────────────────────────

class BreakoutTradingLoop:
    """
    Session breakout paper trading loop — one instance per symbol.
    Mirrors the MTFTradingLoop interface so run_breakout.py can orchestrate
    multiple symbols with asyncio.gather().
    """

    STRATEGY_VERSION = "bo-01"

    def __init__(
        self,
        asset: str,
        state_dir: Path,
        strategy_path: Path,
    ):
        self.asset          = asset
        self.state_dir      = state_dir
        self.strategy_path  = strategy_path

        self.trades_path    = state_dir / "trades.jsonl"
        self.heartbeat_path = state_dir / "heartbeat.json"

        self.position:            Optional[dict] = None
        self.last_acted_entry_ts: Optional[pd.Timestamp] = None
        self.bars_in_position:    int  = 0
        self.consecutive_failures: int = 0

    # ── Config ────────────────────────────────────────────────────────────────

    def _load_cfg(self) -> dict:
        with open(self.strategy_path) as f:
            base = yaml.safe_load(f)
        sym_cfg = base.get("symbols", {}).get(self.asset, {})
        return {**base, **sym_cfg}

    # ── I/O ───────────────────────────────────────────────────────────────────

    def write_heartbeat(self, status: str, extra: dict | None = None) -> None:
        payload: dict = {
            "ts":       datetime.now(timezone.utc).isoformat(),
            "status":   status,
            "asset":    self.asset,
            "strategy": "breakout",
        }
        if extra:
            payload.update(extra)
        self.heartbeat_path.write_text(json.dumps(payload))

    def log_trade(self, trade: dict) -> None:
        with open(self.trades_path, "a") as f:
            f.write(json.dumps(trade) + "\n")

    # ── Position helpers ──────────────────────────────────────────────────────

    def _open_position(
        self,
        sig: BreakoutSignal,
        half_spread: float,
        pos_size_r: float,
    ) -> None:
        d        = sig.direction
        adj      = half_spread if d == "long" else -half_spread
        entry_px = sig.entry_px + adj
        risk     = max(abs(entry_px - sig.stop), 1e-8)

        self.position = {
            "direction":    d,
            "entry_price":  entry_px,
            "stop":         sig.stop,
            "risk":         risk,
            "half_spread":  half_spread,
            "session":      sig.session,
            "entry_time":   datetime.now(timezone.utc).isoformat(),
            "pos_size_r":   pos_size_r,
            "entry_bar_ts": sig.bar_time.isoformat(),
        }
        self.last_acted_entry_ts = sig.bar_time
        self.bars_in_position    = 0

        logger.info(
            f"[BO] OPEN {d.upper()} {self.asset}  "
            f"@ {entry_px:.5f}  SL={sig.stop:.5f}  "
            f"sess={sig.session}  spread={half_spread*2:.5f}"
        )

    def _close_position(
        self,
        exit_px: float,
        exit_kind: str,
        ts: str,
    ) -> None:
        pos   = self.position
        d     = pos["direction"]
        ep    = pos["entry_price"]
        risk  = pos["risk"]
        hs    = pos["half_spread"]

        # Widen exit price for spread
        eff_exit = exit_px - hs if d == "long" else exit_px + hs
        pnl_r    = (eff_exit - ep) / risk if d == "long" else (ep - eff_exit) / risk
        pnl_pct  = round(pnl_r * pos["pos_size_r"] * 100, 4)

        trade = {
            "ts_open":          pos["entry_time"],
            "ts_close":         ts,
            "asset":            self.asset,
            "direction":        pos["direction"],
            "entry_price":      pos["entry_price"],
            "exit_price":       round(eff_exit, 5),
            "exit_kind":        exit_kind,
            "pnl_r":            round(pnl_r, 4),
            "pnl_pct":          pnl_pct,
            "bars_in_trade":    self.bars_in_position,
            "strategy_version": self.STRATEGY_VERSION,
            "mode":             "paper",
            "context": {
                "session":  pos["session"],
                "strategy": "session_breakout",
            },
        }
        self.log_trade(trade)
        logger.info(
            f"[BO] CLOSE {exit_kind.upper()} {self.asset}  "
            f"@ {eff_exit:.5f}  pnl={pnl_r:+.3f}R  ({pnl_pct:+.2f}%)"
        )
        self.position       = None
        self.bars_in_position = 0

    # ── Main tick ─────────────────────────────────────────────────────────────

    async def run_once(self) -> bool:
        cfg        = self._load_cfg()
        pos_size_r = float(cfg.get("position_size_r", 0.01))

        source = os.environ.get("DATA_SOURCE") or cfg.get("source", "yfinance")
        logger.info(f"[BO] {self.asset}: tick start source={source}")
        df = await _fetch_h1(self.asset, source)
        logger.info(f"[BO] {self.asset}: fetched {len(df) if df is not None else 0} bars")
        if df is None or len(df) < 50:
            logger.warning(f"[BO] {self.asset}: insufficient H1 bars")
            return False

        df["atr"]     = _atr(df)
        df["avg_vol"] = _avg_vol(df)
        hl = build_session_hl(df)
        df = attach_sr(df, hl)

        half_spread = await fetch_half_spread(self.asset, source=source, extra_pts=3)
        ts          = datetime.now(timezone.utc).isoformat()
        last_bar    = df.iloc[-1]
        current_px  = float(last_bar["close"])

        # ── Check open position ───────────────────────────────────────────────
        if self.position is not None:
            self.bars_in_position += 1
            pos  = self.position
            d    = pos["direction"]
            stop = pos["stop"]
            sess = pos["session"]

            stop_hit = (current_px <= stop) if d == "long" else (current_px >= stop)
            if stop_hit:
                self._close_position(stop, "stop_loss", ts)

            else:
                # Check if the session has ended (session close exit)
                end_ts = session_end_bar(df, sess, pd.Timestamp(pos["entry_bar_ts"]))
                if end_ts is not None and df.index[-1] >= end_ts:
                    self._close_position(current_px, "sess_close", ts)

        # ── Look for new entry ────────────────────────────────────────────────
        if self.position is None:
            sig = find_latest_entry(df, self.asset, self.last_acted_entry_ts)
            if sig is not None:
                self._open_position(sig, half_spread, pos_size_r)

        # ── Heartbeat ─────────────────────────────────────────────────────────
        self.write_heartbeat(
            "ok",
            {
                "price":               current_px,
                "in_position":         self.position is not None,
                "direction":           self.position["direction"] if self.position else None,
                "bars_in_trade":       self.bars_in_position,
                "session":             self.position["session"] if self.position else None,
                "last_acted_entry_ts": self.last_acted_entry_ts.isoformat()
                                       if self.last_acted_entry_ts is not None else None,
            },
        )
        return True

    # ── Event loop ────────────────────────────────────────────────────────────

    def _restore_state(self) -> None:
        """Reload last_acted_entry_ts from heartbeat so restarts don't re-trade old signals.
        If no saved timestamp exists, default to 2 hours ago so only fresh bars are traded."""
        from datetime import timedelta
        fallback = pd.Timestamp.utcnow().tz_localize("UTC") - pd.Timedelta(hours=2)
        try:
            if self.heartbeat_path.exists():
                hb = json.loads(self.heartbeat_path.read_text())
                ts_str = hb.get("last_acted_entry_ts")
                if ts_str:
                    ts = pd.Timestamp(ts_str)
                    if ts.tzinfo is None:
                        ts = ts.tz_localize("UTC")
                    else:
                        ts = ts.tz_convert("UTC")
                    self.last_acted_entry_ts = ts
                    logger.info(f"[BO] {self.asset}: restored last_acted_entry_ts={ts_str}")
                    return
        except Exception:
            pass
        self.last_acted_entry_ts = fallback
        logger.info(f"[BO] {self.asset}: no saved state — ignoring signals before {fallback.isoformat()}")

    async def run(self) -> None:
        logger.info(
            f"[BO] BreakoutLoop starting — asset={self.asset} "
            f"interval={LOOP_INTERVAL_S}s  mode=paper"
        )
        self._restore_state()
        self.write_heartbeat("starting")

        while True:
            # BTCUSD trades 24/7; other instruments skip forex weekend
            now_utc = datetime.now(timezone.utc)
            wd, h   = now_utc.weekday(), now_utc.hour
            market_closed = (
                wd == 5
                or (wd == 6 and h < 22)
                or (wd == 4 and h >= 22)
            )
            if market_closed and self.asset != "BTCUSD":
                logger.info(f"[BO] {self.asset}: market closed — sleeping 1h")
                await asyncio.sleep(3600)
                continue

            try:
                success = await self.run_once()
                if success:
                    self.consecutive_failures = 0
                else:
                    self.consecutive_failures += 1
                    logger.warning(
                        f"[BO] {self.asset} run_once returned False "
                        f"(failures={self.consecutive_failures}/{MAX_CONSECUTIVE_FAILURES})"
                    )
            except Exception as exc:
                logger.exception(f"[BO] {self.asset} unhandled error: {exc}")
                self.consecutive_failures += 1

            if self.consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                msg = (f"[BO] {self.asset} circuit breaker: "
                       f"{self.consecutive_failures} consecutive failures")
                logger.critical(msg)
                self.write_heartbeat(
                    "circuit_breaker_open",
                    {"failures": self.consecutive_failures},
                )
                raise RuntimeError(msg)

            await asyncio.sleep(LOOP_INTERVAL_S)
