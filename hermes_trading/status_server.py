"""
Lightweight async HTTP status server — multi-symbol portfolio dashboard.

Runs alongside the trading loop on $PORT (Railway injects this).
Exposes:
  GET /           → HTML portfolio dashboard (auto-refreshes every 60 s)
  GET /api/status → JSON (all symbols aggregated + per-symbol detail)

State layout expected:
  state_dir/
    strategy.yaml        ← shared strategy (with per-symbol overrides)
    goal.yaml            ← portfolio targets
    history/             ← strategy version archive
    GBPAUD/
      heartbeat.json
      trades.jsonl
      hypotheses.jsonl
    BTCUSD/ ...          ← same structure for each active pair
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path


# ── Fibonacci strategy OOS reference (grid optimizer, real MT5 spreads 2024-2026) ─
_OOS_REF: dict[str, dict] = {
    "GBPAUD": {"oos_r": 44.92, "oos_sharpe": 1.52, "oos_trades": None,  "val_r":  -4.10, "note": ""},
    "BTCUSD": {"oos_r": 31.94, "oos_sharpe": 1.14, "oos_trades": 166,   "val_r": -11.89, "note": "VAL caution"},
    "GBPUSD": {"oos_r": 25.33, "oos_sharpe": 1.43, "oos_trades": None,  "val_r":   None, "note": ""},
    "USDCHF": {"oos_r": 22.43, "oos_sharpe": 0.96, "oos_trades": 123,   "val_r":   5.40, "note": ""},
    "XAUUSD": {"oos_r": 19.32, "oos_sharpe": 0.89, "oos_trades": None,  "val_r":   None, "note": ""},
    "NZDUSD": {"oos_r": 15.30, "oos_sharpe": 1.08, "oos_trades":  16,   "val_r":   None, "note": "16 OOS trades"},
}

_ACTIVE_SYMBOLS = list(_OOS_REF.keys())

# ── Breakout strategy OOS reference (exit_optimizer.py, Session Close exit, 2024-2025) ─
_BO_OOS_REF: dict[str, dict] = {
    "XNGUSD": {"oos_r": 29.15, "oos_sharpe": 4.60, "oos_trades": 164, "note": "best"},
    "XAUUSD": {"oos_r": 12.67, "oos_sharpe": 3.03, "oos_trades": 108, "note": ""},
    "BTCUSD": {"oos_r": 11.07, "oos_sharpe": 2.80, "oos_trades":  54, "note": ""},
    "US500":  {"oos_r": 14.19, "oos_sharpe": 2.81, "oos_trades":  90, "note": ""},
    "US30":   {"oos_r":  8.99, "oos_sharpe": 2.60, "oos_trades":  83, "note": ""},
    "XTIUSD": {"oos_r": 10.02, "oos_sharpe": 2.38, "oos_trades": 105, "note": ""},
    "JP225":  {"oos_r":  4.98, "oos_sharpe": 1.92, "oos_trades":  71, "note": ""},
    "XBRUSD": {"oos_r":  6.24, "oos_sharpe": 1.92, "oos_trades":  87, "note": ""},
}

_BO_SYMBOLS = list(_BO_OOS_REF.keys())


# ── Data helpers ───────────────────────────────────────────────────────────────

def _load_json_lines(path: Path, limit: int = 200) -> list[dict]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    return [json.loads(l) for l in lines[-limit:] if l.strip()]


def _compute_stats(trades: list[dict]) -> dict:
    """Aggregate trade statistics.  Works with both pnl_r and pnl_pct fields."""
    if not trades:
        return {
            "total": 0, "wins": 0, "losses": 0,
            "win_rate": 0.0,
            "total_r": 0.0, "avg_r": 0.0,
            "total_pnl_pct": 0.0, "avg_pnl_pct": 0.0,
            "best_pnl_pct": 0.0, "worst_pnl_pct": 0.0,
        }
    rs   = [t.get("pnl_r",   0.0) for t in trades]
    pnls = [t.get("pnl_pct", 0.0) for t in trades]
    wins = [r for r in rs if r > 0]
    return {
        "total":          len(trades),
        "wins":           len(wins),
        "losses":         len(trades) - len(wins),
        "win_rate":       round(len(wins) / len(trades) * 100, 1),
        "total_r":        round(sum(rs),   2),
        "avg_r":          round(sum(rs)   / len(rs),   3),
        "total_pnl_pct":  round(sum(pnls), 4),
        "avg_pnl_pct":    round(sum(pnls) / len(pnls), 4),
        "best_pnl_pct":   round(max(pnls), 4),
        "worst_pnl_pct":  round(min(pnls), 4),
    }


def _symbol_status(sym_dir: Path) -> dict:
    """Load per-symbol heartbeat, trades and hypotheses."""
    heartbeat: dict = {}
    hb_path = sym_dir / "heartbeat.json"
    if hb_path.exists():
        try:
            heartbeat = json.loads(hb_path.read_text())
        except Exception:
            pass

    trades     = _load_json_lines(sym_dir / "trades.jsonl")
    hypotheses = _load_json_lines(sym_dir / "hypotheses.jsonl")
    return {
        "heartbeat":  heartbeat,
        "stats":      _compute_stats(trades),
        "trades":     trades,
        "hypotheses": hypotheses,
    }


def _build_status(state_dir: Path) -> dict:
    """Build the full status dict, aggregating all per-symbol subdirs."""
    # ── shared config ──────────────────────────────────────────────────────
    strategy: dict = {}
    st_path = state_dir / "strategy.yaml"
    if st_path.exists():
        try:
            import yaml
            with open(st_path) as f:
                strategy = yaml.safe_load(f)
        except Exception:
            pass

    goal: dict = {}
    goal_path = state_dir / "goal.yaml"
    if goal_path.exists():
        try:
            import yaml
            with open(goal_path) as f:
                goal = yaml.safe_load(f)
        except Exception:
            pass

    versions = sorted(
        [p.stem for p in (state_dir / "history").glob("v*.yaml")]
    ) if (state_dir / "history").exists() else []

    # ── detect per-symbol dirs ─────────────────────────────────────────────
    sym_dirs = {
        sym: state_dir / sym
        for sym in _ACTIVE_SYMBOLS
        if (state_dir / sym).exists()
    }

    if sym_dirs:
        # ── multi-symbol mode ──────────────────────────────────────────────
        per_symbol: dict[str, dict] = {}
        all_trades: list[dict] = []
        all_hyps:   list[dict] = []

        # Portfolio-level hypotheses (root state dir) come first
        all_hyps = _load_json_lines(state_dir / "hypotheses.jsonl")

        for sym, sym_dir in sym_dirs.items():
            s = _symbol_status(sym_dir)
            per_symbol[sym] = s
            all_trades.extend(s["trades"])
            all_hyps.extend(s["hypotheses"])

        # ── Breakout strategy state ────────────────────────────────────────
        bo_state_dir = state_dir / "breakout"
        bo_per_symbol: dict[str, dict] = {}
        bo_all_trades: list[dict] = []

        if bo_state_dir.exists():
            for sym in _BO_SYMBOLS:
                sym_dir = bo_state_dir / sym
                if sym_dir.exists():
                    s = _symbol_status(sym_dir)
                    bo_per_symbol[sym] = s
                    bo_all_trades.extend(s["trades"])

        bo_all_trades.sort(key=lambda t: t.get("ts_close", ""), reverse=True)
        bo_portfolio_stats = _compute_stats(bo_all_trades)

        # Combine for the global recent trades feed
        all_trades.extend(bo_all_trades)

        # Sort all trades newest-first
        all_trades.sort(key=lambda t: t.get("ts_close", ""), reverse=True)
        all_hyps.sort(  key=lambda h: h.get("ts", ""),       reverse=True)

        portfolio_stats = _compute_stats(all_trades)

        return {
            "generated_at":          datetime.now(timezone.utc).isoformat(),
            "mode":                  "portfolio",
            "strategy":              strategy,
            "goal":                  goal,
            "strategy_versions":     versions,
            "portfolio_stats":       portfolio_stats,
            "per_symbol":            per_symbol,
            "bo_per_symbol":         bo_per_symbol,
            "bo_portfolio_stats":    bo_portfolio_stats,
            "last_20_trades":        all_trades[:20],
            "last_5_hypotheses":     all_hyps[:5],
            # Legacy single-symbol fields (for backward-compat with /api/status consumers)
            "stats":                 portfolio_stats,
            "heartbeat":             {},
            "last_10_trades":        all_trades[:10],
        }

    else:
        # ── legacy single-symbol fallback (pre-multi-symbol) ──────────────
        trades     = _load_json_lines(state_dir / "trades.jsonl")
        hypotheses = _load_json_lines(state_dir / "hypotheses.jsonl")
        heartbeat: dict = {}
        hb_path = state_dir / "heartbeat.json"
        if hb_path.exists():
            try:
                heartbeat = json.loads(hb_path.read_text())
            except Exception:
                pass

        return {
            "generated_at":       datetime.now(timezone.utc).isoformat(),
            "mode":               "single",
            "strategy":           strategy,
            "goal":               goal,
            "strategy_versions":  versions,
            "stats":              _compute_stats(trades),
            "heartbeat":          heartbeat,
            "last_10_trades":     trades[-10:][::-1],
            "last_5_hypotheses":  hypotheses[-5:][::-1],
            "portfolio_stats":    _compute_stats(trades),
            "per_symbol":         {},
            "last_20_trades":     trades[-20:][::-1],
        }


# ── HTML renderer ──────────────────────────────────────────────────────────────

def _pnl_color(v: float) -> str:
    if v > 0:  return "color:#22c55e"
    if v < 0:  return "color:#ef4444"
    return "color:#94a3b8"


def _staleness_badge(hb: dict) -> str:
    ts_str = hb.get("ts", "")
    if not ts_str:
        return '<span class="badge badge-red">NO DATA</span>'
    try:
        ts = datetime.fromisoformat(ts_str)
        age_s = (datetime.now(timezone.utc) - ts).total_seconds()
        if age_s < 600:   # < 10 min → fresh
            return '<span class="badge badge-green">OK</span>'
        if age_s < 1800:  # < 30 min → warn
            return '<span class="badge badge-yellow">STALE</span>'
    except Exception:
        pass
    return '<span class="badge badge-red">STALE</span>'


def _bo_sym_rows(bo_per_symbol: dict) -> str:
    """Build HTML table rows for the breakout strategy symbol table."""
    rows = ""
    for sym in _BO_SYMBOLS:
        ref = _BO_OOS_REF[sym]
        if sym in bo_per_symbol:
            s_data = bo_per_symbol[sym]
            hb     = s_data["heartbeat"]
            s      = s_data["stats"]
            badge  = _staleness_badge(hb)
            last_t = hb.get("ts", "—")
            last_t = last_t[:16].replace("T", " ") if last_t != "—" else "—"
            in_pos = hb.get("in_position", False)
            sess   = hb.get("session", "")
            pos_badge = (f'<span class="badge badge-green">OPEN {sess}</span>' if in_pos
                         else '<span class="badge" style="background:#1e293b;color:#475569">—</span>')
            t_cnt  = s["total"]
            wr     = f"{s['win_rate']:.0f}%" if t_cnt else "—"
            lr_val = s["total_r"]
            lr_str = f"{lr_val:+.2f}R" if t_cnt else "—"
            lr_col = _pnl_color(lr_val) if t_cnt else "color:#475569"
        else:
            badge     = '<span class="badge badge-red">OFFLINE</span>'
            last_t    = "—"
            pos_badge = "—"
            t_cnt     = 0
            wr        = "—"
            lr_str    = "—"
            lr_col    = "color:#475569"

        note_span = (f' <span style="color:#64748b;font-size:11px">({ref["note"]})</span>'
                     if ref.get("note") else "")
        rows += (
            f"<tr>"
            f"<td style='font-weight:600;color:#e2e8f0'>{sym}</td>"
            f"<td>{badge}</td>"
            f"<td style='color:#64748b;font-size:12px'>{last_t}</td>"
            f"<td>{pos_badge}</td>"
            f"<td style='text-align:right'>{t_cnt}</td>"
            f"<td style='text-align:right'>{wr}</td>"
            f"<td style='text-align:right;{lr_col}'>{lr_str}</td>"
            f"<td style='text-align:right;color:#22c55e'>+{ref['oos_r']:.2f}R{note_span}</td>"
            f"<td style='text-align:right;color:#94a3b8'>{ref['oos_sharpe']:.2f}</td>"
            f"<td style='text-align:right;color:#475569'>{ref['oos_trades']}</td>"
            f"</tr>"
        )
    return rows


def _html_portfolio(status: dict) -> str:
    ps    = status["portfolio_stats"]
    goal  = status["goal"]
    st    = status["strategy"]
    now   = status["generated_at"][:19].replace("T", " ")
    per_s = status["per_symbol"]
    oos_total_r = sum(v["oos_r"] for v in _OOS_REF.values())

    bo_per_s        = status.get("bo_per_symbol", {})
    bo_ps           = status.get("bo_portfolio_stats", {})
    bo_oos_total_r  = sum(v["oos_r"] for v in _BO_OOS_REF.values())

    # ── per-symbol table rows ─────────────────────────────────────────────
    sym_rows = ""
    for sym in _ACTIVE_SYMBOLS:
        ref = _OOS_REF[sym]
        if sym in per_s:
            s_data = per_s[sym]
            hb     = s_data["heartbeat"]
            s      = s_data["stats"]
            badge  = _staleness_badge(hb)
            last_t = hb.get("ts", "—")
            last_t = last_t[:16].replace("T", " ") if last_t != "—" else "—"
            in_pos = hb.get("in_position", False)
            pos_badge = ('<span class="badge badge-green">OPEN</span>' if in_pos
                         else '<span class="badge" style="background:#1e293b;color:#475569">—</span>')
            t_cnt   = s["total"]
            wr      = f"{s['win_rate']:.0f}%" if t_cnt else "—"
            lr_val  = s["total_r"]
            lr_str  = f"{lr_val:+.2f}R" if t_cnt else "—"
            lr_col  = _pnl_color(lr_val) if t_cnt else "color:#475569"
        else:
            badge   = '<span class="badge badge-red">OFFLINE</span>'
            last_t  = "—"
            pos_badge = "—"
            t_cnt   = 0
            wr      = "—"
            lr_str  = "—"
            lr_col  = "color:#475569"

        note_span = (f' <span style="color:#64748b;font-size:11px">({ref["note"]})</span>'
                     if ref.get("note") else "")
        oos_t = str(ref["oos_trades"]) if ref["oos_trades"] else "—"

        sym_rows += (
            f"<tr>"
            f"<td style='font-weight:600;color:#e2e8f0'>{sym}</td>"
            f"<td>{badge}</td>"
            f"<td style='color:#64748b;font-size:12px'>{last_t}</td>"
            f"<td>{pos_badge}</td>"
            f"<td style='text-align:right'>{t_cnt}</td>"
            f"<td style='text-align:right'>{wr}</td>"
            f"<td style='text-align:right;{lr_col}'>{lr_str}</td>"
            f"<td style='text-align:right;color:#22c55e'>+{ref['oos_r']:.2f}R{note_span}</td>"
            f"<td style='text-align:right;color:#94a3b8'>{ref['oos_sharpe']:.2f}</td>"
            f"<td style='text-align:right;color:#475569'>{oos_t}</td>"
            f"</tr>"
        )

    # ── all-pair trade history rows ───────────────────────────────────────
    trades_rows = ""
    for t in status["last_20_trades"]:
        p   = t.get("pnl_r",   0.0)
        pct = t.get("pnl_pct", 0.0)
        col = "#22c55e" if p > 0 else "#ef4444"
        kind_badges = {
            "tp2":       "badge-green",
            "tp1":       "badge-green",
            "stop_loss": "badge-red",
            "breakeven": "badge-yellow",
            "timeout":   "badge-yellow",
        }
        kind = t.get("exit_kind", "—")
        kbadge_cls = kind_badges.get(kind, "")
        kbadge = (f'<span class="badge {kbadge_cls}">{kind}</span>'
                  if kbadge_cls else f'<span style="color:#64748b">{kind}</span>')
        trades_rows += (
            f"<tr>"
            f"<td style='color:#64748b;font-size:12px'>{t.get('ts_close','')[:16].replace('T',' ')}</td>"
            f"<td style='font-weight:600'>{t.get('asset','')}</td>"
            f"<td style='color:#94a3b8'>{t.get('direction','')[:4]}</td>"
            f"<td>{kbadge}</td>"
            f"<td style='text-align:right'>{t.get('entry_price',0):,.4f}</td>"
            f"<td style='text-align:right'>{t.get('exit_price',0):,.4f}</td>"
            f"<td style='text-align:right;color:{col};font-weight:600'>{p:+.3f}R</td>"
            f"<td style='text-align:right;color:{col}'>{pct:+.3f}%</td>"
            f"</tr>"
        )
    if not trades_rows:
        trades_rows = '<tr><td colspan=8 style="color:#475569;padding:16px">No closed trades yet — loops are watching for CHoCH + Fib entries</td></tr>'

    # ── reflection rows ───────────────────────────────────────────────────
    hyp_rows = ""
    for h in status["last_5_hypotheses"]:
        hyp_rows += (
            f"<tr>"
            f"<td style='color:#64748b;font-size:12px'>{h.get('ts','')[:16].replace('T',' ')}</td>"
            f"<td>v{h.get('from_version','?')} → v{h.get('to_version','?')}</td>"
            f"<td style='color:#67e8f9'>{h.get('variable','')}</td>"
            f"<td>{h.get('old_value','')} → {h.get('new_value','')}</td>"
            f"<td style='color:#64748b;font-size:12px'>{str(h.get('rationale',''))[:90]}</td>"
            f"</tr>"
        )
    if not hyp_rows:
        hyp_rows = '<tr><td colspan=5 style="color:#475569">No reflections yet</td></tr>'

    # ── P&L cards ─────────────────────────────────────────────────────────
    fib_ps    = status.get("portfolio_stats", ps)   # Fibonacci only stats
    total_r   = fib_ps.get("total_r", 0.0)
    total_pct = fib_ps.get("total_pnl_pct", 0.0)
    wr        = fib_ps.get("win_rate", 0.0)
    total_t   = fib_ps.get("total", 0)
    wr_col    = "green" if wr >= 50 else "red"
    r_col     = "green" if total_r > 0 else ("red" if total_r < 0 else "muted")

    bo_total_r   = bo_ps.get("total_r", 0.0)
    bo_total_pct = bo_ps.get("total_pnl_pct", 0.0)
    bo_wr        = bo_ps.get("win_rate", 0.0)
    bo_total_t   = bo_ps.get("total", 0)
    bo_wr_col    = "green" if bo_wr >= 50 else "red"
    bo_r_col     = "green" if bo_total_r > 0 else ("red" if bo_total_r < 0 else "muted")

    bo_sym_rows_html = _bo_sym_rows(bo_per_s)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="60">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Hermes Portfolio — Status</title>
<style>
  body{{background:#0f172a;color:#e2e8f0;font-family:system-ui,sans-serif;margin:0;padding:24px;max-width:1400px}}
  h1{{color:#f8fafc;font-size:20px;margin:0 0 4px}}
  h2{{color:#94a3b8;font-size:14px;font-weight:600;margin:32px 0 4px;border-top:1px solid #1e293b;padding-top:20px}}
  .sub{{color:#64748b;font-size:13px;margin-bottom:24px}}
  .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin-bottom:24px}}
  .card{{background:#1e293b;border-radius:8px;padding:16px}}
  .card .label{{color:#64748b;font-size:11px;text-transform:uppercase;letter-spacing:.05em}}
  .card .value{{font-size:22px;font-weight:700;margin-top:4px}}
  .green{{color:#22c55e}} .red{{color:#ef4444}} .yellow{{color:#f59e0b}} .muted{{color:#94a3b8}}
  table{{width:100%;border-collapse:collapse;font-size:13px;margin-bottom:28px}}
  th{{text-align:left;color:#475569;padding:6px 8px;border-bottom:1px solid #1e293b;font-size:11px;text-transform:uppercase;letter-spacing:.05em}}
  th.r{{text-align:right}}
  td{{padding:6px 8px;border-bottom:1px solid #0f172a}}
  tr:hover td{{background:#1a2540}}
  .section-title{{color:#64748b;font-size:11px;text-transform:uppercase;letter-spacing:.08em;margin:24px 0 8px}}
  .badge{{display:inline-block;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:600}}
  .badge-green{{background:#14532d;color:#86efac}}
  .badge-red{{background:#450a0a;color:#fca5a5}}
  .badge-yellow{{background:#451a03;color:#fcd34d}}
  .badge-blue{{background:#1e3a5f;color:#93c5fd}}
</style>
</head>
<body>
<h1>Hermes Portfolio</h1>
<div class="sub">
  Two independent strategies running in parallel &nbsp;·&nbsp;
  Paper mode &nbsp;·&nbsp;
  Updated: {now} UTC
</div>

<!-- ── FIBONACCI STRATEGY ───────────────────────────────────────────── -->
<h2>Strategy 1 &mdash; MTF Fibonacci (1H&rarr;15M&rarr;5M)</h2>
<div class="sub">{len(_ACTIVE_SYMBOLS)} pairs active &nbsp;·&nbsp; OOS ref: +{oos_total_r:.0f}R</div>
<div class="grid">
  <div class="card"><div class="label">Active Pairs</div><div class="value muted">{len(per_s)}/{len(_ACTIVE_SYMBOLS)}</div></div>
  <div class="card"><div class="label">Fib Trades</div><div class="value muted">{total_t}</div></div>
  <div class="card"><div class="label">Fib WR</div><div class="value {wr_col}">{wr:.0f}%</div></div>
  <div class="card"><div class="label">Fib Live R</div><div class="value {r_col}">{total_r:+.2f}R</div></div>
  <div class="card"><div class="label">Fib Live %</div><div class="value" style="{_pnl_color(total_pct)}">{total_pct:+.2f}%</div></div>
  <div class="card"><div class="label">OOS Ref</div><div class="value green">+{oos_total_r:.0f}R</div></div>
  <div class="card"><div class="label">Strategy Versions</div><div class="value muted">{len(status['strategy_versions'])}</div></div>
  <div class="card"><div class="label">Target 30d</div><div class="value muted">{goal.get('target_return_30d',0.05)*100:.0f}%</div></div>
</div>

<div class="section-title">Fibonacci Pairs — Live vs Backtest Reference</div>
<table>
  <thead>
    <tr>
      <th>Symbol</th><th>Status</th><th>Last Tick (UTC)</th><th>Position</th>
      <th class="r">Trades</th><th class="r">WR%</th><th class="r">Live R</th>
      <th class="r">OOS R (ref)</th><th class="r">OOS Sharpe</th><th class="r">OOS Trades</th>
    </tr>
  </thead>
  <tbody>{sym_rows}</tbody>
</table>

<!-- ── BREAKOUT STRATEGY ──────────────────────────────────────────────── -->
<h2>Strategy 2 &mdash; Session Breakout (H1, Asia/London/NY)</h2>
<div class="sub">{len(_BO_SYMBOLS)} instruments active &nbsp;·&nbsp; OOS ref: +{bo_oos_total_r:.0f}R (2024-2025, Session Close exit)</div>
<div class="grid">
  <div class="card"><div class="label">Active Instruments</div><div class="value muted">{len(bo_per_s)}/{len(_BO_SYMBOLS)}</div></div>
  <div class="card"><div class="label">Breakout Trades</div><div class="value muted">{bo_total_t}</div></div>
  <div class="card"><div class="label">Breakout WR</div><div class="value {bo_wr_col}">{bo_wr:.0f}%</div></div>
  <div class="card"><div class="label">Breakout Live R</div><div class="value {bo_r_col}">{bo_total_r:+.2f}R</div></div>
  <div class="card"><div class="label">Breakout Live %</div><div class="value" style="{_pnl_color(bo_total_pct)}">{bo_total_pct:+.2f}%</div></div>
  <div class="card"><div class="label">OOS Ref</div><div class="value green">+{bo_oos_total_r:.0f}R</div></div>
</div>

<div class="section-title">Breakout Instruments — Live vs Backtest Reference</div>
<table>
  <thead>
    <tr>
      <th>Symbol</th><th>Status</th><th>Last Tick (UTC)</th><th>Position</th>
      <th class="r">Trades</th><th class="r">WR%</th><th class="r">Live R</th>
      <th class="r">OOS R (ref)</th><th class="r">OOS Sharpe</th><th class="r">OOS Trades</th>
    </tr>
  </thead>
  <tbody>{bo_sym_rows_html if bo_sym_rows_html else '<tr><td colspan=10 style="color:#475569;padding:16px">Session Breakout loop not started yet — run: python -m hermes_trading.run_breakout</td></tr>'}</tbody>
</table>

<!-- ── COMBINED TRADE FEED ────────────────────────────────────────────── -->
<div class="section-title">Recent Trades — All Strategies (newest first)</div>
<table>
  <thead>
    <tr>
      <th>Closed (UTC)</th><th>Symbol</th><th>Dir</th><th>Exit</th>
      <th class="r">Entry</th><th class="r">Exit Px</th>
      <th class="r">P&L (R)</th><th class="r">P&L (%)</th>
    </tr>
  </thead>
  <tbody>{trades_rows}</tbody>
</table>

<div class="section-title">Hermes Reflection Log (Fibonacci strategy)</div>
<table>
  <thead><tr><th>Time (UTC)</th><th>Version</th><th>Variable</th><th>Change</th><th>Rationale</th></tr></thead>
  <tbody>{hyp_rows}</tbody>
</table>

<div style="color:#334155;font-size:11px;margin-top:24px">
  Auto-refreshes every 60 s &nbsp;·&nbsp;
  <a href="/api/status" style="color:#475569">JSON API</a>
</div>
</body></html>"""


def _html(status: dict) -> str:
    """Route to the correct HTML renderer based on mode."""
    if status.get("mode") == "portfolio" and status.get("per_symbol"):
        return _html_portfolio(status)
    # ── legacy single-symbol fallback ─────────────────────────────────────
    s      = status["stats"]
    hb     = status["heartbeat"]
    st     = status["strategy"]
    last_t = hb.get("ts", "—")[:19].replace("T", " ") if hb.get("ts") else "—"
    price  = hb.get("price", "—")
    in_pos = hb.get("in_position", False)

    trades_rows = ""
    for t in status["last_10_trades"]:
        p   = t.get("pnl_pct", 0)
        col = "#22c55e" if p > 0 else "#ef4444"
        trades_rows += (
            f"<tr>"
            f"<td>{t.get('ts_close','')[:19].replace('T',' ')}</td>"
            f"<td>{t.get('asset','')}</td>"
            f"<td>{t.get('entry_price','')}</td>"
            f"<td>{t.get('exit_price','')}</td>"
            f"<td style='color:{col};font-weight:600'>{p:+.4f}%</td>"
            f"<td>v{t.get('strategy_version','?')}</td>"
            f"</tr>"
        )

    hyp_rows = ""
    for h in status.get("last_5_hypotheses", []):
        hyp_rows += (
            f"<tr>"
            f"<td>{h.get('ts','')[:19].replace('T',' ')}</td>"
            f"<td>v{h.get('from_version','?')} → v{h.get('to_version','?')}</td>"
            f"<td>{h.get('variable','')}</td>"
            f"<td>{h.get('old_value','')} → {h.get('new_value','')}</td>"
            f"<td style='color:#94a3b8;font-size:12px'>{h.get('rationale','')[:80]}</td>"
            f"</tr>"
        )

    def pnl_color(v):
        if v > 0: return "color:#22c55e"
        if v < 0: return "color:#ef4444"
        return "color:#94a3b8"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="60">
<title>Hermes Trading</title>
<style>
  body{{background:#0f172a;color:#e2e8f0;font-family:system-ui,sans-serif;margin:0;padding:24px}}
  h1{{color:#f8fafc;font-size:20px;margin:0 0 4px}}
  .sub{{color:#64748b;font-size:13px;margin-bottom:24px}}
  .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin-bottom:24px}}
  .card{{background:#1e293b;border-radius:8px;padding:16px}}
  .card .label{{color:#64748b;font-size:11px;text-transform:uppercase}}
  .card .value{{font-size:24px;font-weight:700;margin-top:4px}}
  .green{{color:#22c55e}} .red{{color:#ef4444}} .muted{{color:#94a3b8}}
  table{{width:100%;border-collapse:collapse;font-size:13px;margin-bottom:24px}}
  th{{text-align:left;color:#64748b;padding:6px 8px;border-bottom:1px solid #1e293b}}
  td{{padding:6px 8px;border-bottom:1px solid #1e293b}}
  .section-title{{color:#94a3b8;font-size:12px;text-transform:uppercase;margin:20px 0 8px}}
  .badge{{display:inline-block;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:600}}
  .badge-green{{background:#14532d;color:#86efac}}
  .badge-yellow{{background:#451a03;color:#fcd34d}}
</style>
</head>
<body>
<h1>Hermes Trading</h1>
<div class="sub">
  {st.get('asset', status.get('goal',{}).get('asset','?'))} &nbsp;·&nbsp;
  {'<span class="badge badge-green">IN POSITION</span>' if in_pos else '<span class="badge badge-yellow">WATCHING</span>'}
  &nbsp;·&nbsp; Last tick: {last_t} UTC
</div>
<div class="grid">
  <div class="card"><div class="label">Price</div><div class="value muted">{price}</div></div>
  <div class="card"><div class="label">Total Trades</div><div class="value muted">{s['total']}</div></div>
  <div class="card"><div class="label">Win Rate</div><div class="value {'green' if s['win_rate']>=50 else 'red'}">{s['win_rate']}%</div></div>
  <div class="card"><div class="label">Total P&L</div><div class="value" style="{pnl_color(s['total_pnl_pct'])}">{s['total_pnl_pct']:+.2f}%</div></div>
</div>
<div class="section-title">Last 10 Trades</div>
<table>
  <thead><tr><th>Closed</th><th>Asset</th><th>Entry</th><th>Exit</th><th>P&L</th><th>Strategy</th></tr></thead>
  <tbody>{trades_rows or '<tr><td colspan=6 style="color:#475569">No closed trades yet</td></tr>'}</tbody>
</table>
<div class="section-title">Reflection Log</div>
<table>
  <thead><tr><th>Time</th><th>Version</th><th>Variable</th><th>Change</th><th>Rationale</th></tr></thead>
  <tbody>{hyp_rows or '<tr><td colspan=5 style="color:#475569">No reflections yet</td></tr>'}</tbody>
</table>
<div style="color:#334155;font-size:11px;margin-top:24px">
  Auto-refreshes every 60s &nbsp;·&nbsp;
  <a href="/api/status" style="color:#475569">JSON API</a>
</div>
</body></html>"""


# ── HTTP server ────────────────────────────────────────────────────────────────

async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, state_dir: Path):
    try:
        request_line = (await reader.readline()).decode(errors="replace").strip()
        while True:
            line = await reader.readline()
            if line in (b"\r\n", b"\n", b""):
                break

        path   = request_line.split(" ")[1] if " " in request_line else "/"
        status = _build_status(state_dir)

        if path.startswith("/api/status"):
            body         = json.dumps(status, default=str).encode()
            content_type = "application/json"
        else:
            body         = _html(status).encode()
            content_type = "text/html; charset=utf-8"

        response = (
            f"HTTP/1.1 200 OK\r\n"
            f"Content-Type: {content_type}\r\n"
            f"Content-Length: {len(body)}\r\n"
            f"Connection: close\r\n\r\n"
        ).encode() + body

        writer.write(response)
        await writer.drain()
    except Exception:
        pass
    finally:
        writer.close()


async def start(state_dir: Path):
    port   = int(os.getenv("PORT", "8080"))
    server = await asyncio.start_server(
        lambda r, w: handle(r, w, state_dir),
        host="0.0.0.0",
        port=port,
    )
    addr = server.sockets[0].getsockname()
    import logging as _log
    _log.getLogger(__name__).info(f"Status server on http://{addr[0]}:{addr[1]}")
    async with server:
        await server.serve_forever()
