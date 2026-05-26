"""
hermes-stats — local / remote dashboard for the Hermes portfolio.

Usage:
  hermes-stats                    # reads local state/
  hermes-stats --url https://...  # fetches from Railway status endpoint
"""

import argparse
import json
import sys
from pathlib import Path


def _load_local(state_dir: Path) -> dict:
    from hermes_trading.status_server import _build_status
    return _build_status(state_dir)


def _load_remote(url: str) -> dict:
    import httpx
    resp = httpx.get(url.rstrip("/") + "/api/status", timeout=10)
    resp.raise_for_status()
    return resp.json()


def _render(status: dict):
    import io
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-16"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    from rich.console import Console
    from rich.table   import Table
    from rich         import box
    from rich.panel   import Panel
    from rich.columns import Columns
    from rich.text    import Text

    console = Console()
    s    = status.get("portfolio_stats") or status["stats"]
    goal = status.get("goal", {})
    st   = status.get("strategy", {})

    is_portfolio = status.get("mode") == "portfolio"

    # ── Header ──────────────────────────────────────────────────────────────
    console.print()
    if is_portfolio:
        per_s = status.get("per_symbol", {})
        console.print(
            f"[bold white]Hermes Portfolio[/]  [dim]·[/]  "
            f"[cyan]{len(per_s)}[/] active pairs  [dim]·[/]  "
            f"Paper mode  [dim]·[/]  "
            f"[dim]{status['generated_at'][:19].replace('T',' ')} UTC[/]"
        )
    else:
        hb     = status.get("heartbeat", {})
        asset  = st.get("asset", goal.get("asset", "?"))
        in_pos = hb.get("in_position", False)
        pos    = "[bold green]IN POSITION[/]" if in_pos else "[bold yellow]WATCHING[/]"
        console.print(
            f"[bold white]Hermes Trading[/]  [dim]·[/]  {asset}  [dim]·[/]  {pos}"
        )
    console.print()

    # ── Portfolio KPI cards ──────────────────────────────────────────────────
    def kpi(label, value, style="white"):
        return Panel(f"[{style}]{value}[/]\n[dim]{label}[/]", expand=True)

    total_r   = s.get("total_r", 0.0)
    total_pct = s.get("total_pnl_pct", 0.0)
    wr        = s["win_rate"]
    r_style   = "green" if total_r > 0 else ("red" if total_r < 0 else "white")
    wr_style  = "green" if wr >= 50      else "red"

    from hermes_trading.status_server import _OOS_REF
    oos_total = sum(v["oos_r"] for v in _OOS_REF.values())

    console.print(Columns([
        kpi("Total Trades",    str(s["total"])),
        kpi("Portfolio WR",    f"{wr:.0f}%",          wr_style),
        kpi("Live P&L (R)",    f"{total_r:+.2f}R",    r_style),
        kpi("Live P&L (%)",    f"{total_pct:+.2f}%",  r_style),
        kpi("OOS Ref (total)", f"+{oos_total:.0f}R",  "green"),
        kpi("Target 30d",      f"{goal.get('target_return_30d',0.05)*100:.0f}%"),
    ], equal=True))
    console.print()

    # ── Per-symbol table ─────────────────────────────────────────────────────
    if is_portfolio:
        per_s = status.get("per_symbol", {})
        sym_table = Table(
            title="Active Pairs — Live vs Backtest Reference",
            box=box.SIMPLE_HEAD,
            show_edge=False,
            title_style="dim",
        )
        sym_table.add_column("Symbol",        style="bold white", no_wrap=True)
        sym_table.add_column("Status",        no_wrap=True)
        sym_table.add_column("Last Tick",     style="dim",   no_wrap=True)
        sym_table.add_column("Pos",           no_wrap=True)
        sym_table.add_column("Trades",        justify="right")
        sym_table.add_column("WR%",           justify="right")
        sym_table.add_column("Live R",        justify="right")
        sym_table.add_column("OOS R (ref)",   justify="right", style="green")
        sym_table.add_column("OOS Sharpe",    justify="right", style="dim")
        sym_table.add_column("OOS Trades",    justify="right", style="dim")

        for sym in _OOS_REF:
            ref = _OOS_REF[sym]
            oos_r_str  = f"+{ref['oos_r']:.2f}R"
            oos_sh_str = f"{ref['oos_sharpe']:.2f}"
            oos_t_str  = str(ref["oos_trades"]) if ref["oos_trades"] else "—"

            if sym in per_s:
                sd   = per_s[sym]
                hb   = sd["heartbeat"]
                ss   = sd["stats"]
                ts   = hb.get("ts", "")
                last_t = ts[:16].replace("T", " ") if ts else "—"

                # staleness
                status_txt = "[dim]—[/]"
                try:
                    from datetime import datetime, timezone
                    age = (datetime.now(timezone.utc) -
                           datetime.fromisoformat(ts)).total_seconds()
                    if age < 600:
                        status_txt = "[green]OK[/]"
                    elif age < 1800:
                        status_txt = "[yellow]STALE[/]"
                    else:
                        status_txt = "[red]STALE[/]"
                except Exception:
                    status_txt = "[dim]?[/]"

                in_pos = hb.get("in_position", False)
                pos_str = "[green]OPEN[/]" if in_pos else "[dim]—[/]"
                t_cnt  = ss["total"]
                wr_sym = f"{ss['win_rate']:.0f}%" if t_cnt else "—"
                lr_val = ss.get("total_r", 0.0)
                lr_str = Text(f"{lr_val:+.2f}R", style="green" if lr_val > 0 else "red") if t_cnt else Text("—", style="dim")

                note = f" [dim]({ref['note']})[/]" if ref.get("note") else ""
                sym_table.add_row(
                    sym, status_txt, last_t, pos_str,
                    str(t_cnt), wr_sym, lr_str,
                    oos_r_str + note, oos_sh_str, oos_t_str,
                )
            else:
                sym_table.add_row(
                    sym,
                    "[red]OFFLINE[/]", "—", "—",
                    "—", "—", Text("—", style="dim"),
                    oos_r_str, oos_sh_str, oos_t_str,
                )

        console.print(sym_table)

    # ── Trade history ────────────────────────────────────────────────────────
    trades = status.get("last_20_trades") or status.get("last_10_trades", [])
    t_table = Table(
        title=f"Recent Trades (newest first)",
        box=box.SIMPLE_HEAD,
        show_edge=False,
        title_style="dim",
    )
    t_table.add_column("Closed (UTC)", style="dim", no_wrap=True)
    t_table.add_column("Symbol",       style="bold")
    t_table.add_column("Dir",          style="dim")
    t_table.add_column("Exit Kind",    no_wrap=True)
    t_table.add_column("Entry",        justify="right")
    t_table.add_column("Exit Px",      justify="right")
    t_table.add_column("P&L (R)",      justify="right")
    t_table.add_column("P&L (%)",      justify="right")

    if trades:
        for t in trades[:20]:
            pnl_r   = t.get("pnl_r", 0.0)
            pnl_pct = t.get("pnl_pct", 0.0)
            r_text  = Text(f"{pnl_r:+.3f}R",  style="green" if pnl_r  > 0 else "red")
            p_text  = Text(f"{pnl_pct:+.3f}%", style="green" if pnl_pct > 0 else "red")
            kind    = t.get("exit_kind", "—")
            kind_styles = {
                "tp2":       "[green]tp2[/]",
                "tp1":       "[green]tp1[/]",
                "stop_loss": "[red]stop_loss[/]",
                "breakeven": "[yellow]breakeven[/]",
                "timeout":   "[yellow]timeout[/]",
            }
            kind_str = kind_styles.get(kind, f"[dim]{kind}[/]")
            t_table.add_row(
                t.get("ts_close", "")[:16].replace("T", " "),
                t.get("asset", ""),
                t.get("direction", "")[:4],
                kind_str,
                f"{t.get('entry_price', 0):,.4f}",
                f"{t.get('exit_price',  0):,.4f}",
                r_text,
                p_text,
            )
    else:
        t_table.add_row("[dim]No closed trades yet[/]", "", "", "", "", "", "", "")

    console.print(t_table)

    # ── Reflection log ───────────────────────────────────────────────────────
    hyps    = status.get("last_5_hypotheses", [])
    h_table = Table(
        title="Hermes Reflection Log (newest first)",
        box=box.SIMPLE_HEAD,
        show_edge=False,
        title_style="dim",
    )
    h_table.add_column("Time (UTC)",  style="dim", no_wrap=True)
    h_table.add_column("Version")
    h_table.add_column("Variable",   style="cyan")
    h_table.add_column("Change")
    h_table.add_column("Rationale",  style="dim", max_width=55)

    if hyps:
        for h in hyps:
            h_table.add_row(
                h.get("ts", "")[:16].replace("T", " "),
                f"v{h.get('from_version','?')} -> v{h.get('to_version','?')}",
                h.get("variable", ""),
                f"{h.get('old_value','')} -> {h.get('new_value','')}",
                h.get("rationale", ""),
            )
    else:
        h_table.add_row("[dim]No reflections yet[/]", "", "", "", "")

    console.print(h_table)

    # ── Strategy snapshot ────────────────────────────────────────────────────
    if not is_portfolio:
        entry = st.get("entry", {})
        console.print(
            f"[dim]Strategy:[/]  "
            f"indicator=[cyan]{entry.get('indicator','?')}[/]  "
            f"threshold=[cyan]{entry.get('threshold','?')}[/]  "
            f"direction=[cyan]{entry.get('direction','?')}[/]"
        )
    console.print()


def main():
    parser = argparse.ArgumentParser(description="Hermes trading stats")
    parser.add_argument("--url",      type=str,  default="", help="Railway status URL")
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=Path(__file__).parent.parent / "state",
        help="Local state directory",
    )
    args = parser.parse_args()

    try:
        status = _load_remote(args.url) if args.url else _load_local(args.state_dir)
        _render(status)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
