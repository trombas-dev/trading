import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


def _bump_version(current: str) -> str:
    try:
        n = int(current.lstrip("v0") or "0")
    except ValueError:
        n = 0
    return f"{n + 1:02d}"


def _save_history(strategy: dict, history_dir: Path):
    version = str(strategy.get("version", "01"))
    dest = history_dir / f"v{version.zfill(4)}.yaml"
    history_dir.mkdir(parents=True, exist_ok=True)
    with open(dest, "w") as f:
        yaml.dump(strategy, f, default_flow_style=False)


def _append_hypothesis(hypotheses_path: Path, hypothesis: dict):
    with open(hypotheses_path, "a") as f:
        f.write(json.dumps(hypothesis) + "\n")


def _fallback_reflect(
    trades: list[dict], goal: dict, strategy: dict, score: float
) -> tuple[dict, str | None, object, object]:
    """
    Change exactly ONE variable based on performance.

    Supports both legacy RSI strategy keys and the new MTF Fibonacci keys.
    Returns (new_strategy, var_name, old_value, new_value).
    """
    realised = sum(t.get("pnl_pct", 0) for t in trades) / 100 if trades else 0.0
    worst    = min((t.get("pnl_pct", 0) for t in trades), default=0.0)
    drawdown = abs(worst) / 100

    target = goal.get("target_return_30d", 0.05)
    max_dd = goal.get("max_drawdown", 0.08)

    new_strategy = dict(strategy)

    # ── MTF strategy keys ─────────────────────────────────────────────────────
    if "zone_lo" in strategy or "n_bias" in strategy:
        win_rate = (
            sum(1 for t in trades if t.get("pnl_r", 0) > 0) / len(trades) * 100
            if trades else 50.0
        )

        if drawdown > max_dd:
            # Tighten entry zone (raise zone_lo) to improve signal quality
            old = float(new_strategy.get("zone_lo", 0.382))
            candidates = [v for v in (0.236, 0.382, 0.500) if v > old]
            new = candidates[0] if candidates else old
            if new != old:
                new_strategy["zone_lo"] = new
                return new_strategy, "zone_lo", old, new

        if realised < target:
            if win_rate < 40:
                # Narrow zone to raise selectivity
                old = float(new_strategy.get("zone_hi", 0.786))
                candidates = [v for v in (0.618, 0.705, 0.786) if v < old]
                new = candidates[-1] if candidates else old
                if new != old:
                    new_strategy["zone_hi"] = new
                    return new_strategy, "zone_hi", old, new
            else:
                # Extend timeout to give entries more room to work
                old = int(new_strategy.get("timeout_bars", 4))
                new = min(8, old + 2)
                if new != old:
                    new_strategy["timeout_bars"] = new
                    return new_strategy, "timeout_bars", old, new

        return strategy, None, None, None

    # ── Legacy RSI strategy keys ──────────────────────────────────────────────
    if drawdown > max_dd:
        old = new_strategy.get("stop_loss_pct", 2.0)
        new = round(max(0.5, old - 0.2), 2)
        new_strategy["stop_loss_pct"] = new
        return new_strategy, "stop_loss_pct", old, new

    if realised < target:
        old_entry = dict(new_strategy.get("entry", {}))
        old = old_entry.get("threshold", 30)
        new = old + 2
        new_strategy["entry"] = {**old_entry, "threshold": new}
        return new_strategy, "entry.threshold", old, new

    return strategy, None, None, None


def _hermes_reflect(
    trades: list[dict], goal: dict, strategy: dict
) -> tuple[dict, str, object, object, str]:
    """Call hermes subprocess for a one-variable proposal."""
    prompt_data = {
        "trades": trades,
        "goal": goal,
        "strategy": strategy,
        "instruction": (
            "Review the last trades and current strategy. "
            "Propose exactly ONE variable change most likely to improve performance. "
            "Respond ONLY with JSON: "
            '{"variable": "<name>", "old_value": <val>, "new_value": <val>, "rationale": "<text>"}'
        ),
    }

    result = subprocess.run(
        ["hermes"],
        input=json.dumps(prompt_data),
        capture_output=True,
        text=True,
        timeout=60,
    )

    if result.returncode != 0:
        raise RuntimeError(f"hermes exited {result.returncode}: {result.stderr.strip()}")

    proposal = json.loads(result.stdout)
    variable = proposal["variable"]
    old_value = proposal.get("old_value")
    new_value = proposal["new_value"]
    rationale = proposal.get("rationale", "")

    new_strategy = dict(strategy)
    if "." in variable:
        top, sub = variable.split(".", 1)
        new_strategy[top] = {**new_strategy.get(top, {}), sub: new_value}
    else:
        new_strategy[variable] = new_value

    return new_strategy, variable, old_value, new_value, rationale


def maybe_reflect(
    mode: str,
    trades: list[dict],
    goal: dict,
    strategy_path: Path,
    state_dir: Path,
    score: float,
):
    with open(strategy_path) as f:
        strategy = yaml.safe_load(f)

    history_dir = state_dir / "history"
    hypotheses_path = state_dir / "hypotheses.jsonl"

    _save_history(strategy, history_dir)

    rationale = ""
    if mode == "fallback":
        new_strategy, variable_changed, old_value, new_value = _fallback_reflect(
            trades, goal, strategy, score
        )
    elif mode == "hermes":
        try:
            new_strategy, variable_changed, old_value, new_value, rationale = _hermes_reflect(
                trades, goal, strategy
            )
        except Exception as e:
            logger.warning(f"Hermes reflect failed, falling back to deterministic: {e}")
            new_strategy, variable_changed, old_value, new_value = _fallback_reflect(
                trades, goal, strategy, score
            )
    else:
        raise ValueError(f"Unknown reflect mode: {mode}")

    if variable_changed is None:
        logger.info("Reflect: performance on target — no change")
        return

    old_version = str(strategy.get("version", "01"))
    new_version = _bump_version(old_version)
    new_strategy["version"] = new_version

    with open(strategy_path, "w") as f:
        yaml.dump(new_strategy, f, default_flow_style=False)

    hypothesis = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "from_version": old_version,
        "to_version": new_version,
        "variable": variable_changed,
        "old_value": old_value,
        "new_value": new_value,
        "score_at_reflect": round(score, 4),
        "mode": mode,
        "rationale": rationale,
    }
    _append_hypothesis(hypotheses_path, hypothesis)
    logger.info(
        f"Reflect [{mode}]: v{old_version}->v{new_version} | "
        f"{variable_changed}: {old_value} -> {new_value}"
    )


if __name__ == "__main__":
    import argparse
    import logging
    import sys

    from rich.logging import RichHandler

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )

    parser = argparse.ArgumentParser(description="Hermes reflect — manual trigger")
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--fallback", action="store_true", help="Deterministic rule-based reflection")
    mode_group.add_argument("--hermes", action="store_true", help="Hermes LLM-based reflection")
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=Path(__file__).parent.parent / "state",
        help="Path to state directory (default: ../state relative to this file)",
    )
    cli_args = parser.parse_args()

    state_dir = cli_args.state_dir
    goal_path = state_dir / "goal.yaml"

    if not goal_path.exists():
        print(f"ERROR: goal.yaml not found at {goal_path}", file=sys.stderr)
        sys.exit(1)

    with open(goal_path) as f:
        goal = yaml.safe_load(f)

    trades: list[dict] = []
    trades_path = state_dir / "trades.jsonl"
    if trades_path.exists():
        lines = trades_path.read_text().strip().splitlines()
        trades = [json.loads(line) for line in lines if line.strip()]

    from hermes_trading.score import score as compute_score
    score_val = compute_score(trades, goal)

    mode = "fallback" if cli_args.fallback else "hermes"
    maybe_reflect(
        mode=mode,
        trades=trades,
        goal=goal,
        strategy_path=state_dir / "strategy.yaml",
        state_dir=state_dir,
        score=score_val,
    )
    print(f"\nReflect [{mode}] complete. Check state/strategy.yaml and state/hypotheses.jsonl")
