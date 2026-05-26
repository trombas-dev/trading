"""
loop.py — MTF Fibonacci strategy live loop.

Replaces the legacy RSI loop (Phase 1-7) with the full multi-timeframe
1H -> 15M -> 5M pipeline built in Phase 8+.

Tick cadence
------------
  Default interval: 300 s (5 minutes) — aligned to 5M bar closes.
  On each tick:
    1. Load strategy.yaml params.
    2. Fetch fresh 1H / 15M / 5M bars from MT5 or yfinance.
    3. If in position  -> check SL / TP against latest 5M close.
    4. If no position  -> run MTF pipeline; open if fresh entry signal found.
    5. Trigger reflection after every `reflection_every` closed trades.

Position lifecycle (two-phase)
-------------------------------
  Phase 1  Full position (100 %)
    Bullish: close <= SL  -> stop_loss  (-1 R)
             close >= TP1 -> tp1        (+0.5 * RR1), SL moves to entry (BE)
  Phase 2  Half position, SL at breakeven
    Bullish: close <= entry -> breakeven  (0 R on remaining half)
             close >= TP2   -> tp2        (+0.5 * RR2)
             bars > max_bars -> timeout   (mark-to-market)

Bearish checks are symmetric.

Trade log fields (compatible with score.py)
--------------------------------------------
  ts_open, ts_close, asset, direction
  entry_price, exit_price, exit_kind
  pnl_r      : PnL in R-multiples
  pnl_pct    : pnl_r * position_size_r * 100  (% of account)
  strategy_version, mode: "paper", context: dict
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from hermes_trading.adapters.mt5_price  import fetch_mtf_bars, fetch_half_spread
from hermes_trading.strategy.backtest   import BacktestParams
from hermes_trading.strategy.live       import get_live_signal
from hermes_trading.score               import score as compute_score

logger = logging.getLogger(__name__)

LOOP_INTERVAL_S           = 300     # 5-minute tick
MAX_CONSECUTIVE_FAILURES  = 5
ADAPTER_RETRIES           = 3
RETRY_BASE_DELAY          = 2.0


# ── Utility ───────────────────────────────────────────────────────────────────

async def fetch_with_retry(fetch_fn, name: str) -> Any:
    """Call an async function up to ADAPTER_RETRIES times with back-off."""
    for attempt in range(ADAPTER_RETRIES):
        try:
            return await fetch_fn()
        except Exception as exc:
            delay = RETRY_BASE_DELAY ** (attempt + 1)
            logger.warning(
                f"{name} fetch failed (attempt {attempt + 1}/{ADAPTER_RETRIES}): {exc}. "
                f"Retrying in {delay:.1f}s"
            )
            if attempt < ADAPTER_RETRIES - 1:
                await asyncio.sleep(delay)
    logger.error(f"{name} failed after {ADAPTER_RETRIES} attempts")
    return None


# ── Position exit helpers ─────────────────────────────────────────────────────

def _check_exit(position: dict, close: float) -> dict | None:  # noqa: C901
    """
    Evaluate one 5M bar close against the open position's SL / TP.

    Returns an exit event dict (kind, price, fraction, pnl_r) or None if
    no exit is triggered.  SL is evaluated on CLOSE only (no wick stops).
    """
    d           = position["direction"]
    phase       = position["phase"]
    sl          = position["sl_live"]
    tp1         = position["tp1"]
    tp2         = position["tp2"]
    ep          = position["entry_price"]
    risk        = position["risk"]
    half_spread = position.get("half_spread", 0.0)

    def _r(exit_px: float, frac: float) -> float:
        # Worsen exit price by half_spread (sell at bid / buy at ask)
        if d == "bullish":
            eff = exit_px - half_spread
            return frac * (eff - ep) / risk
        eff = exit_px + half_spread
        return frac * (ep - eff) / risk

    if phase == 1:
        sl_hit  = close <= sl if d == "bullish" else close >= sl
        tp1_hit = close >= tp1 if d == "bullish" else close <= tp1

        if sl_hit:
            return {"kind": "stop_loss",  "price": sl,   "fraction": 1.0, "pnl_r": _r(sl,   1.0)}
        if tp1_hit:
            return {"kind": "tp1",        "price": tp1,  "fraction": 0.5, "pnl_r": _r(tp1,  0.5)}

    elif phase == 2:
        be_hit  = close <= sl if d == "bullish" else close >= sl  # sl == entry after TP1
        tp2_hit = close >= tp2 if d == "bullish" else close <= tp2

        if be_hit:
            return {"kind": "breakeven",  "price": sl,   "fraction": 0.5, "pnl_r": _r(sl,   0.5)}
        if tp2_hit:
            return {"kind": "tp2",        "price": tp2,  "fraction": 0.5, "pnl_r": _r(tp2,  0.5)}

    return None


def _is_full_close(kind: str) -> bool:
    return kind in ("stop_loss", "breakeven", "tp2", "timeout")


# ── Main trading loop ─────────────────────────────────────────────────────────

class MTFTradingLoop:
    """
    Multi-timeframe Fibonacci strategy live loop.

    Reads parameters from state_dir/strategy.yaml on every tick — Claude can
    mutate the file between ticks to adjust the strategy without restart.
    """

    def __init__(
        self,
        asset: str,
        goal: dict,
        state_dir: Path,
        strategy_path: Path | None = None,
    ):
        self.asset      = asset         # MT5 symbol name, e.g. "BTCUSD"
        self.goal       = goal
        self.state_dir  = state_dir

        self.trades_path    = state_dir / "trades.jsonl"
        self.heartbeat_path = state_dir / "heartbeat.json"
        # Allow a shared strategy.yaml outside the per-symbol state dir
        self.strategy_path  = strategy_path or state_dir / "strategy.yaml"

        self.consecutive_failures          = 0
        self.position:         dict | None = None
        self.last_acted_entry_ts           = None   # pd.Timestamp | None
        self.bars_in_position:  int        = 0
        self.closed_trades_since_reflect:  int = 0
        self.dry_run:           bool       = False  # set True for --dry-run mode

    # ── State I/O ─────────────────────────────────────────────────────────────

    def load_strategy(self) -> dict:
        with open(self.strategy_path) as f:
            return yaml.safe_load(f)

    @staticmethod
    def _merge_symbol_config(base: dict, symbol: str) -> dict:
        """
        Merge per-symbol overrides from the 'symbols' section into the base dict.

        Priority order (highest wins):
          per-symbol override  >  base global params  >  hardcoded defaults

        Falls back to 'symbols.default' if the symbol has no explicit entry.
        """
        symbols_cfg = base.get("symbols", {})
        overrides   = symbols_cfg.get(symbol) or symbols_cfg.get("default") or {}
        return {**base, **overrides}

    @staticmethod
    def _strategy_to_params(s: dict) -> BacktestParams:
        return BacktestParams(
            n_bias            = int(s.get("n_bias",            7)),
            n_signal          = int(s.get("n_signal",          5)),
            n_entry           = int(s.get("n_entry",           3)),
            zone_lo           = float(s.get("zone_lo",         0.382)),
            zone_hi           = float(s.get("zone_hi",         0.786)),
            timeout_bars      = int(s.get("timeout_bars",      4)),
            max_bars_in_trade = int(s.get("max_bars_in_trade", 100)),
            scenario          = int(s.get("scenario",          1)),
            use_bos_windows   = bool(s.get("use_bos_windows",  False)),
            bos_fib_mode      = str(s.get("bos_fib_mode",      "last_leg")),
            # half_spread is set at runtime from MT5 tick data (see run_once)
            # Regime filter params
            regime_filter     = str(s.get("regime_filter",     "none")),
            regime_tf         = str(s.get("regime_tf",         "1D")),
            regime_adx_period = int(s.get("regime_adx_period", 14)),
            regime_adx_min    = float(s.get("regime_adx_min",  20.0)),
            regime_atr_short  = int(s.get("regime_atr_short",  10)),
            regime_atr_long   = int(s.get("regime_atr_long",   50)),
            regime_atr_ratio  = float(s.get("regime_atr_ratio", 0.80)),
        )

    def write_heartbeat(self, status: str, extra: dict | None = None) -> None:
        payload: dict = {
            "ts":     datetime.now(timezone.utc).isoformat(),
            "status": status,
            "asset":  self.asset,
        }
        if extra:
            payload.update(extra)
        self.heartbeat_path.write_text(json.dumps(payload))

    def log_trade(self, trade: dict) -> None:
        with open(self.trades_path, "a") as f:
            f.write(json.dumps(trade) + "\n")

    def load_recent_trades(self, n: int = 100) -> list[dict]:
        if not self.trades_path.exists():
            return []
        lines = self.trades_path.read_text().strip().splitlines()
        return [json.loads(l) for l in lines if l.strip()][-n:]

    # ── Position management ───────────────────────────────────────────────────

    def _open_position(
        self,
        entry_signal,
        position_size_r: float,
        context: dict,
        half_spread: float = 0.0,
    ) -> None:
        """
        Record a new virtual position from an EntrySignal.

        half_spread is applied to the effective entry price and risk so that
        the position tracks the realistic filled price (ask for longs, bid for
        shorts) rather than the bar-close mid price.
        """
        d = entry_signal.direction

        # Adjust entry for live bid-ask spread (buy at ask, sell at bid)
        if half_spread > 0.0:
            adj          = half_spread if d == "bullish" else -half_spread
            entry_price  = entry_signal.entry_price + adj
            # Risk widens because we entered at a worse price than the SL reference
            risk         = max(abs(entry_price - entry_signal.stop_loss), 1e-8)
        else:
            entry_price  = entry_signal.entry_price
            risk         = entry_signal.risk

        self.position = {
            "direction":       d,
            "entry_price":     entry_price,
            "sl_live":         entry_signal.stop_loss,  # moves to BE after TP1
            "tp1":             entry_signal.tp1,
            "tp2":             entry_signal.tp2,
            "risk":            risk,
            "half_spread":     half_spread,
            "phase":           1,
            "tp1_pnl_r":       None,
            "entry_time":      datetime.now(timezone.utc).isoformat(),
            "position_size_r": position_size_r,
            "context":         context,
        }
        self.last_acted_entry_ts = entry_signal.timestamp
        self.bars_in_position    = 0
        logger.info(
            f"OPEN  {d.upper()}  "
            f"@ {entry_price:.4f}  "
            f"(signal close={entry_signal.entry_price:.4f}  "
            f"half_spread={half_spread:.5f})  "
            f"SL={entry_signal.stop_loss:.4f}  "
            f"TP1={entry_signal.tp1:.4f}  "
            f"TP2={entry_signal.tp2:.4f}  "
            f"R:R1={entry_signal.rr1:.2f}"
        )

    def _process_exit(
        self,
        evt: dict,
        current_price: float,
        strategy: dict,
        ts: str,
    ) -> bool:
        """
        Apply an exit event to the open position.

        Returns True if the position is now fully closed.
        """
        pos  = self.position
        kind = evt["kind"]

        if kind == "tp1":
            # Partial close: stay in phase 2
            pos["tp1_pnl_r"] = evt["pnl_r"]
            pos["sl_live"]   = pos["entry_price"]   # SL -> breakeven
            pos["phase"]     = 2
            logger.info(
                f"TP1 HIT  +{evt['pnl_r']:+.3f}R (50%)  SL moved to BE @ {pos['sl_live']:.4f}"
            )
            return False   # still open

        # Full close: stop_loss / breakeven / tp2 / timeout
        total_pnl_r = evt["pnl_r"]
        if pos["tp1_pnl_r"] is not None:
            total_pnl_r += pos["tp1_pnl_r"]

        pnl_pct = round(total_pnl_r * pos["position_size_r"] * 100, 4)

        trade = {
            "ts_open":          pos["entry_time"],
            "ts_close":         ts,
            "asset":            self.asset,
            "direction":        pos["direction"],
            "entry_price":      pos["entry_price"],
            "exit_price":       evt["price"],
            "exit_kind":        kind,
            "pnl_r":            round(total_pnl_r, 4),
            "pnl_pct":          pnl_pct,
            "bars_in_trade":    self.bars_in_position,
            "strategy_version": str(strategy.get("version", "01")),
            "mode":             "paper",
            "context":          pos.get("context", {}),
        }
        self.log_trade(trade)
        logger.info(
            f"CLOSE {kind.upper()}  {pos['direction'].upper()}  "
            f"@ {evt['price']:.4f}  "
            f"pnl={total_pnl_r:+.3f}R  ({pnl_pct:+.2f}%)"
        )

        self.position      = None
        self.bars_in_position = 0
        self.closed_trades_since_reflect += 1
        return True

    def _timeout_exit(self, current_price: float, strategy: dict, ts: str) -> None:
        """Close remaining position at current price on timeout."""
        pos     = self.position
        risk    = pos["risk"]
        ep      = pos["entry_price"]
        d       = pos["direction"]
        frac    = 0.5 if pos["tp1_pnl_r"] is not None else 1.0

        if d == "bullish":
            pnl_r = frac * (current_price - ep) / risk
        else:
            pnl_r = frac * (ep - current_price) / risk

        if pos["tp1_pnl_r"] is not None:
            pnl_r += pos["tp1_pnl_r"]

        evt = {"kind": "timeout", "price": current_price, "fraction": frac, "pnl_r": pnl_r}
        self._process_exit(evt, current_price, strategy, ts)

    # ── Main tick ─────────────────────────────────────────────────────────────

    async def run_once(self) -> bool:
        strategy = self.load_strategy()
        # Merge per-symbol overrides before building params
        strategy = self._merge_symbol_config(strategy, self.asset)

        # Honour active: false — skip suspended symbols without error
        if not strategy.get("active", True):
            logger.info(f"[{self.asset}] active=false in strategy.yaml — skipping tick")
            return True   # return True so the outer loop keeps running (not a fatal error)

        params   = self._strategy_to_params(strategy)
        source   = strategy.get("source", "yfinance")
        pos_size = float(strategy.get("position_size_r", 0.01))

        # ── Bar counts by timeframe (strategy.yaml can tune per-TF) ──────────
        n_bars_map: dict[str, int] = {
            "1D":  int(strategy.get("n_bars_1d",  250)),
            "4H":  int(strategy.get("n_bars_4h",  500)),
            "1H":  int(strategy.get("n_bars_1h",  1000)),
            "15M": int(strategy.get("n_bars_15m", 500)),
            "5M":  int(strategy.get("n_bars_5m",  500)),
        }

        tf_bias, tf_sig, tf_ent = params.tfs()

        # Extra TFs needed by the regime filter (e.g. "1D" for ADX when
        # scenario 1 only uses 1H/15M/5M).  Deduplication handled in fetch_mtf_bars.
        extra_tfs: dict[str, int] = {}
        if (params.regime_filter != "none"
                and params.regime_tf not in (tf_bias, tf_sig, tf_ent)):
            n_regime = n_bars_map.get(params.regime_tf, 250)
            extra_tfs[params.regime_tf] = n_regime

        # ── Fetch fresh bars + live spread ───────────────────────────────────
        try:
            dfs = await fetch_mtf_bars(
                self.asset, source=source,
                scenario  = params.scenario,
                n_bias    = n_bars_map[tf_bias],
                n_signal  = n_bars_map[tf_sig],
                n_entry   = n_bars_map[tf_ent],
                extra_tfs = extra_tfs or None,
            )
        except Exception as exc:
            logger.error(f"Bar fetch failed: {exc}")
            return False

        # Fetch live half-spread from MT5 tick data (0.0 for yfinance / on error)
        half_spread = await fetch_half_spread(self.asset, source=source, extra_pts=3)
        if half_spread > 0:
            logger.debug(
                f"Live half-spread {self.asset}: {half_spread:.5f} price units"
            )

        ts            = datetime.now(timezone.utc).isoformat()
        current_price = float(dfs[tf_ent]["close"].iloc[-1])

        # ── Check open position exits ─────────────────────────────────────────
        if self.position is not None:
            self.bars_in_position += 1
            max_bars = params.max_bars_in_trade

            if self.bars_in_position > max_bars:
                logger.info(f"TIMEOUT after {self.bars_in_position} bars")
                self._timeout_exit(current_price, strategy, ts)
            else:
                evt = _check_exit(self.position, current_price)
                if evt:
                    self._process_exit(evt, current_price, strategy, ts)

        # ── Check for new entry (only when flat) ──────────────────────────────
        if self.position is None:
            entry_signal, context = get_live_signal(dfs, params)

            if entry_signal is not None and self._is_fresh(entry_signal):
                if getattr(self, "dry_run", False):
                    logger.info(
                        f"[DRY-RUN] Signal found — would OPEN "
                        f"{entry_signal.direction.upper()} "
                        f"@ {entry_signal.entry_price:.4f}  "
                        f"SL={entry_signal.stop_loss:.4f}  "
                        f"TP1={entry_signal.tp1:.4f}  "
                        f"half_spread={half_spread:.5f}"
                    )
                    self.last_acted_entry_ts = entry_signal.timestamp
                else:
                    self._open_position(entry_signal, pos_size, context,
                                        half_spread=half_spread)
        else:
            context = self.position.get("context", {})

        # ── Trigger reflection ────────────────────────────────────────────────
        if self.closed_trades_since_reflect >= self.goal.get("reflection_every", 10):
            recent    = self.load_recent_trades(25)
            score_val = compute_score(recent, self.goal)
            try:
                from hermes_trading.reflect import maybe_reflect
                maybe_reflect(
                    mode          = "hermes",
                    trades        = recent,
                    goal          = self.goal,
                    strategy_path = self.strategy_path,
                    state_dir     = self.state_dir,
                    score         = score_val,
                )
            except Exception as exc:
                logger.warning(f"Reflection failed: {exc}")
            self.closed_trades_since_reflect = 0

        # ── Heartbeat ─────────────────────────────────────────────────────────
        self.write_heartbeat(
            "ok",
            {
                "price":       current_price,
                "in_position": self.position is not None,
                "direction":   self.position["direction"] if self.position else None,
                "bars_in_trade": self.bars_in_position,
                **(context if self.position is None else {}),
            },
        )
        return True

    def _is_fresh(self, entry_signal) -> bool:
        """
        True if `entry_signal` has not already been acted upon.

        Prevents re-entering the same signal on consecutive ticks while
        flat (e.g. between tp1 hit and tp2 / BE resolution in the same session).
        """
        if self.last_acted_entry_ts is None:
            return True
        return entry_signal.timestamp > self.last_acted_entry_ts

    # ── Event loop ────────────────────────────────────────────────────────────

    async def run(self, start_status_server: bool = True) -> None:
        logger.info(
            f"MTF Loop starting — asset={self.asset} "
            f"interval={LOOP_INTERVAL_S}s  mode=paper"
        )
        self.write_heartbeat("starting")

        # Start the status HTTP server as a non-fatal background task.
        # In multi-symbol mode the orchestrator (run.py) starts it once with
        # the root state dir and passes start_status_server=False here.
        if start_status_server:
            try:
                from hermes_trading.status_server import start as start_status
                asyncio.create_task(start_status(self.state_dir))
            except Exception as exc:
                logger.warning(f"Status server could not start: {exc}")

        while True:
            try:
                success = await self.run_once()
                self.consecutive_failures = (
                    0 if success else self.consecutive_failures + 1
                )
            except Exception as exc:
                logger.exception(f"Unhandled error in run_once: {exc}")
                self.consecutive_failures += 1

            if self.consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                msg = (
                    f"Circuit breaker: {self.consecutive_failures} "
                    f"consecutive failures"
                )
                logger.critical(msg)
                self.write_heartbeat(
                    "circuit_breaker_open",
                    {"failures": self.consecutive_failures},
                )
                raise RuntimeError(msg)

            await asyncio.sleep(LOOP_INTERVAL_S)


# ── Backward-compat alias ─────────────────────────────────────────────────────
# run.py still instantiates TradingLoop; point it at the new class.
TradingLoop = MTFTradingLoop
