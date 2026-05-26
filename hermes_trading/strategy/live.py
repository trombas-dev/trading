"""
live.py — Live pipeline adapter for the Hermes MTF strategy.

Runs the bias -> signal -> entry pipeline on the CURRENT trading window only
(the window that is active right now, determined by _build_windows), then returns:
  - The most recent actionable entry signal (or None if none exist)
  - A rich context dict used for heartbeat + reflection prompts

All three scenarios are supported via params.tfs():
  Scenario 1 (default) : 1H bias → 15M signal → 5M entry
  Scenario 2           : 4H bias → 1H signal  → 15M entry
  Scenario 3           : 1D bias → 4H signal  → 1H entry

Why _build_windows instead of fibs_from_last_choch?
----------------------------------------------------
The backtest uses _build_windows() which handles:
  - CHoCH-only mode
  - BOS windows with last_leg / full_trend / delayed fib modes
  - Window boundaries (ts_start, ts_end, fib, bias)

Using the same function here ensures backtest ≡ live.  We take the LAST
returned window (the one covering "now") rather than iterating all windows.

Public API
----------
  get_live_signal(dfs, params) -> (EntrySignal | None, dict)

  # Legacy positional form still accepted:
  get_live_signal(df_1h, df_15m, df_5m, params)
"""

from __future__ import annotations

import logging

import pandas as pd

from hermes_trading.strategy.backtest   import BacktestParams, _build_windows, _apply_regime_filter
from hermes_trading.strategy.regime     import compute_regime_ok
from hermes_trading.strategy.swing      import find_swings
from hermes_trading.strategy.structure  import find_structure
from hermes_trading.strategy.signal     import find_signals
from hermes_trading.strategy.entry      import find_entries, EntrySignal

logger = logging.getLogger(__name__)


def get_live_signal(
    dfs_or_df_bias,
    df_signal: pd.DataFrame | None = None,
    df_entry:  pd.DataFrame | None = None,
    params:    BacktestParams | None = None,
) -> tuple[EntrySignal | None, dict]:
    """
    Run the MTF pipeline on fresh bar data and return the current trading state.

    Uses _build_windows() from backtest.py to determine the active window,
    ensuring live behaviour is identical to the backtest simulation.

    Calling conventions
    -------------------
    New (recommended):
        get_live_signal(dfs, params)
        where dfs = {"1H": df, "15M": df, "5M": df}   (or 4H/1H/15M etc.)

    Legacy (still supported):
        get_live_signal(df_1h, df_15m, df_5m, params)

    Parameters
    ----------
    dfs_or_df_bias : dict[str, pd.DataFrame] OR bias-TF DataFrame (legacy).
    df_signal      : Signal-TF bars (legacy positional only).
    df_entry       : Entry-TF bars  (legacy positional only).
    params         : Strategy parameters (BacktestParams).

    Returns
    -------
    (entry_signal, context)
      entry_signal : Latest EntrySignal from find_entries, or None.
      context      : dict with pipeline state for heartbeat + reflection.
    """
    if params is None:
        params = BacktestParams()

    # ── Resolve DataFrames ────────────────────────────────────────────────────
    if isinstance(dfs_or_df_bias, dict):
        dfs = dfs_or_df_bias
    else:
        # Legacy positional form — map to scenario-1 names
        dfs = {"1H": dfs_or_df_bias, "15M": df_signal, "5M": df_entry}

    tf_bias, tf_sig, tf_ent = params.tfs()

    try:
        df_bias = dfs[tf_bias]
        df_sig  = dfs[tf_sig]
        df_ent  = dfs[tf_ent]
    except KeyError as exc:
        logger.error(
            f"get_live_signal: missing TF {exc} in dfs dict. "
            f"Available: {list(dfs.keys())}"
        )
        return None, {"bias": "neutral", "n_choch": 0}

    ctx: dict = {
        "bias":      "neutral",
        "n_choch":   0,
        "scenario":  params.scenario,
        "tf_bias":   tf_bias,
        "tf_signal": tf_sig,
        "tf_entry":  tf_ent,
    }

    # ── Bias TF: swings + structure ───────────────────────────────────────────
    try:
        sw_bias = find_swings(df_bias, n=params.n_bias)
        st_bias = find_structure(df_bias, sw_bias)
    except Exception as exc:
        logger.warning(f"{tf_bias} structure failed: {exc}")
        return None, ctx

    ctx["n_choch"] = len(st_bias.choch)

    # ── Build windows — identical logic to the backtest ───────────────────────
    raw_events = st_bias.events if params.use_bos_windows else st_bias.choch

    if not raw_events:
        return None, ctx

    ts_end_of_data = df_bias.index[-1]
    if ts_end_of_data.tzinfo is None:
        ts_end_of_data = ts_end_of_data.tz_localize("UTC")

    windows = _build_windows(raw_events, sw_bias, df_bias, params, ts_end_of_data)

    # ── Regime filter: remove windows that opened in choppy regimes ───────────
    if params.regime_filter != "none" and isinstance(dfs, dict) and params.regime_tf in dfs:
        regime_ok = compute_regime_ok(
            dfs[params.regime_tf],
            params.regime_filter,
            adx_period    = params.regime_adx_period,
            adx_min       = params.regime_adx_min,
            atr_short     = params.regime_atr_short,
            atr_long      = params.regime_atr_long,
            atr_ratio_min = params.regime_atr_ratio,
        )
        windows = _apply_regime_filter(windows, regime_ok)

        # Report current regime state in context
        ts_now = df_bias.index[-1]
        import pandas as _pd
        if ts_now.tzinfo is None:
            ts_now = ts_now.tz_localize("UTC")
        if not regime_ok.empty:
            _idx = regime_ok.index
            if _idx.tzinfo is None:
                regime_ok.index = _idx.tz_localize("UTC")
            _val = regime_ok.asof(ts_now)
            ctx["regime_ok"] = bool(_val) if _pd.notna(_val) else True
        else:
            ctx["regime_ok"] = True
        ctx["regime_filter"] = params.regime_filter

    if not windows:
        return None, ctx

    # The last window is the currently active one (ts_end == ts_end_of_data)
    cur_win  = windows[-1]
    fib      = cur_win["fib"]
    bias     = cur_win["bias"]
    ts_start = cur_win["ts_start"]

    ctx["bias"]           = bias
    ctx["last_choch_ts"]  = str(ts_start)[:16]
    ctx["last_choch_dir"] = bias
    ctx["fib_high"]       = round(fib.high, 4)
    ctx["fib_low"]        = round(fib.low, 4)
    ctx["fib_range_pct"]  = round(fib.range_pct, 2)
    ctx["fib_tp1"]        = round(fib.tp1.price, 4) if fib.tp1 else None
    ctx["fib_tp2"]        = round(fib.tp2.price, 4) if fib.tp2 else None
    ctx["n_windows"]      = len(windows)

    # ── Slice signal + entry TFs to this window ───────────────────────────────
    w_sig = df_sig[df_sig.index >= ts_start].copy()
    w_ent = df_ent[df_ent.index >= ts_start].copy()

    ctx[f"w_{tf_sig}_bars"] = len(w_sig)
    ctx[f"w_{tf_ent}_bars"] = len(w_ent)

    min_sig = params.n_signal * 4
    min_ent = params.n_entry  * 4

    if len(w_sig) < min_sig:
        logger.debug(
            f"Insufficient {tf_sig} bars in window: {len(w_sig)} < {min_sig}"
        )
        return None, ctx

    # ── Signal TF pipeline ────────────────────────────────────────────────────
    try:
        sw_sig = find_swings(w_sig, n=params.n_signal)
        st_sig = find_structure(w_sig, sw_sig)
        sig    = find_signals(
            w_sig, fib=fib, bias=bias, struct_res=st_sig,
            zone_lo=params.zone_lo, zone_hi=params.zone_hi,
        )
    except Exception as exc:
        logger.warning(f"{tf_sig} pipeline failed: {exc}")
        return None, ctx

    ctx[f"n_{tf_sig}_signals"]      = len(sig.signals)
    ctx[f"n_{tf_sig}_zone_entries"] = len(sig.zone_entries)

    if not sig.signals:
        return None, ctx

    if len(w_ent) < min_ent:
        logger.debug(
            f"Insufficient {tf_ent} bars in window: {len(w_ent)} < {min_ent}"
        )
        return None, ctx

    # ── Entry TF pipeline ─────────────────────────────────────────────────────
    try:
        sw_ent = find_swings(w_ent, n=params.n_entry)
        st_ent = find_structure(w_ent, sw_ent)
        ent    = find_entries(
            w_ent, sig_res=sig, struct_res=st_ent, fib=fib,
            timeout_bars=params.timeout_bars,
        )
    except Exception as exc:
        logger.warning(f"{tf_ent} pipeline failed: {exc}")
        return None, ctx

    ctx["n_entries"] = len(ent.entries)

    if not ent.entries:
        return None, ctx

    latest = ent.entries[-1]
    ctx["latest_entry_ts"]    = str(latest.timestamp)[:16]
    ctx["latest_entry_price"] = round(latest.entry_price, 4)
    ctx["latest_entry_sl"]    = round(latest.stop_loss, 4)
    ctx["latest_entry_tp1"]   = round(latest.tp1, 4)
    ctx["latest_entry_tp2"]   = round(latest.tp2, 4)
    ctx["latest_entry_rr1"]   = round(latest.rr1, 2)

    return latest, ctx
