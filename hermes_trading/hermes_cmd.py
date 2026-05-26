#!/usr/bin/env python3
"""
hermes — AI trading strategy advisor.

Reads a JSON context from stdin, returns exactly one variable-change
proposal as JSON on stdout.

Expected stdin shape:
  {
    "trades":      [...],     # recent closed trades
    "goal":        {...},     # goal.yaml contents
    "strategy":    {...},     # current strategy.yaml
    "instruction": "..."      # task description
  }

Output shape:
  {
    "variable":  "entry.threshold",
    "old_value": 30,
    "new_value": 28,
    "rationale": "RSI threshold loosened to capture more entries given low win rate."
  }
"""

import json
import os
import sys


SYSTEM_PROMPT = """You are a quantitative trading strategy advisor embedded in an
autonomous paper-trading agent. Your sole job is to review recent trade outcomes
and propose EXACTLY ONE change to the strategy that is most likely to improve
risk-adjusted returns.

Rules you must never break:
1. Propose EXACTLY one variable. Never two.
2. Respond with ONLY valid JSON — no markdown, no explanation outside the JSON.
3. Valid variables: entry.threshold, entry.direction, stop_loss_pct, position_size_r
4. entry.direction must be "long" or "short"
5. entry.threshold must be an integer 1-99
6. stop_loss_pct must be a positive float
7. position_size_r must be a positive float ≤ 2.0
8. If performance is already on target, tighten stop_loss_pct by 0.2 to lock in gains.

Output schema (JSON only, no other text):
{"variable": "<name>", "old_value": <current>, "new_value": <proposed>, "rationale": "<one sentence>"}
"""


def _build_user_message(context: dict) -> str:
    trades = context.get("trades", [])
    goal = context.get("goal", {})
    strategy = context.get("strategy", {})

    trade_summary = "No closed trades yet — strategy has not fired."
    if trades:
        pnls = [t.get("pnl_pct", 0) for t in trades]
        wins = sum(1 for p in pnls if p > 0)
        trade_summary = (
            f"{len(trades)} closed trades | "
            f"win rate {wins}/{len(trades)} | "
            f"total pnl {sum(pnls):+.2f}% | "
            f"worst trade {min(pnls):+.2f}% | "
            f"best trade {max(pnls):+.2f}%\n\n"
            f"Last 5 trades:\n{json.dumps(trades[-5:], indent=2)}"
        )

    return f"""Current strategy (v{strategy.get('version', '?')}):
{json.dumps(strategy, indent=2)}

Goal:
  target_return_30d : {goal.get('target_return_30d', 0.05) * 100:.1f}%
  max_drawdown      : {goal.get('max_drawdown', 0.05) * 100:.1f}%
  min_sharpe        : {goal.get('min_sharpe', 1.0)}

Trade results:
{trade_summary}

Propose exactly one change."""


def _extract_json(text: str) -> str:
    """Strip markdown fences if the model wraps its output."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        inner = [l for l in lines if not l.startswith("```")]
        text = "\n".join(inner).strip()
    return text


def main():
    raw = sys.stdin.read().strip()
    if not raw:
        print(json.dumps({"error": "hermes: empty stdin"}), file=sys.stderr)
        sys.exit(1)

    try:
        context = json.loads(raw)
    except json.JSONDecodeError as e:
        print(json.dumps({"error": f"hermes: invalid JSON on stdin: {e}"}), file=sys.stderr)
        sys.exit(1)

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print(json.dumps({"error": "hermes: ANTHROPIC_API_KEY not set"}), file=sys.stderr)
        sys.exit(1)

    model = os.getenv("HERMES_MODEL", "claude-3-5-haiku-20241022")

    try:
        import anthropic
    except ImportError:
        print(json.dumps({"error": "hermes: anthropic package not installed — run: uv add anthropic"}), file=sys.stderr)
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    try:
        message = client.messages.create(
            model=model,
            max_tokens=300,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": _build_user_message(context)}],
        )
    except anthropic.APIError as e:
        print(json.dumps({"error": f"hermes: Claude API error: {e}"}), file=sys.stderr)
        sys.exit(1)

    response_text = _extract_json(message.content[0].text)

    try:
        proposal = json.loads(response_text)
    except json.JSONDecodeError:
        # Soft fallback: return raw text as rationale, change threshold by +2
        strategy = context.get("strategy", {})
        old_threshold = strategy.get("entry", {}).get("threshold", 30)
        proposal = {
            "variable": "entry.threshold",
            "old_value": old_threshold,
            "new_value": old_threshold + 2,
            "rationale": f"Parse fallback. Raw response: {response_text[:120]}",
        }

    # Always write valid JSON to stdout
    print(json.dumps(proposal))


if __name__ == "__main__":
    main()
