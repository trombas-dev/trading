import numpy as np


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def _realised_return(trades: list[dict]) -> float:
    if not trades:
        return 0.0
    return sum(t.get("pnl_pct", 0) for t in trades) / 100


def _max_drawdown(trades: list[dict]) -> float:
    if not trades:
        return 0.0
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for t in trades:
        equity *= 1 + t.get("pnl_pct", 0) / 100
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak
        if dd > max_dd:
            max_dd = dd
    return max_dd


def _sharpe(trades: list[dict]) -> float:
    if len(trades) < 2:
        return 0.0
    returns = [t.get("pnl_pct", 0) / 100 for t in trades]
    mean = float(np.mean(returns))
    std = float(np.std(returns, ddof=1))
    if std == 0:
        return 0.0
    return mean / std * (len(returns) ** 0.5)


def score(trades: list[dict], goal: dict) -> float:
    """Composite score in [-1, +1]: return (50%) + drawdown (30%) + sharpe (20%)."""
    if not trades:
        return 0.0

    target_return = goal.get("target_return_30d", 0.05)
    max_dd = goal.get("max_drawdown", 0.08)
    min_sharpe = goal.get("min_sharpe", 1.0)
    failure_below = goal.get("failure_below", -0.04)

    realised = _realised_return(trades)

    if realised < failure_below:
        return -1.0

    drawdown = _max_drawdown(trades)
    sharpe = _sharpe(trades)

    return_score = _clamp(realised / target_return, -1.0, 1.0) if target_return else 0.0
    drawdown_score = _clamp(1.0 - drawdown / max_dd, -1.0, 1.0) if max_dd else 0.0
    sharpe_score = _clamp(sharpe / min_sharpe, -1.0, 1.0) if min_sharpe else 0.0

    composite = return_score * 0.5 + drawdown_score * 0.3 + sharpe_score * 0.2
    return round(_clamp(composite, -1.0, 1.0), 4)
