"""
Daily Claude review of the strategy's live performance. The monitor sends a JSON snapshot of
recent stats; Claude returns a short observation ONLY when something is worth flagging, and the
single word "OK" (-> None here) when everything looks healthy. So the daily email stays silent
about strategy health unless there is genuinely something to say.

Requires the official Anthropic SDK and an API key:
    pip install anthropic
    secrets.env:  ANTHROPIC_API_KEY=sk-ant-...

If the key or package is missing the review is simply skipped (returns None) - it never blocks
the trader or the daily summary.
"""
from __future__ import annotations
import os, json

MODEL = "claude-opus-4-8"

SYSTEM = (
    "You are a risk monitor for a LIVE crypto trading strategy: a cross-confirmed BTC momentum "
    "basket (4 strategies that passed a 6-test gauntlet on both BTC and ETH) with "
    "volatility-target position sizing and a causal trailing-MA trend filter that flattens "
    "exposure in downtrends. You receive a JSON snapshot of recent live performance.\n\n"
    "Your job: flag genuine problems the operator should act on - an unusually deep or fast "
    "drawdown, a sharp single-day equity drop, the strategy stuck pinned long or short for a long "
    "time, exposure/leverage outside the expected range, a stalled or stale data feed, or a run of "
    "losing days. Do NOT comment on ordinary fluctuation or normal small P&L swings.\n\n"
    "Respond in 2-5 concise, concrete sentences ONLY if there is something noteworthy. If "
    "everything looks within normal range, reply with exactly the single word: OK"
)


def review_performance(stats: dict, model: str = MODEL) -> str | None:
    """Return observations to email, or None when healthy / unavailable."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    try:
        import anthropic
    except ImportError:
        print("[claude_review] anthropic package not installed (pip install anthropic) - skipping")
        return None
    try:
        client = anthropic.Anthropic(api_key=key)
        resp = client.messages.create(
            model=model,
            max_tokens=4096,
            thinking={"type": "adaptive"},
            system=SYSTEM,
            messages=[{
                "role": "user",
                "content": "Live performance snapshot:\n```json\n"
                           + json.dumps(stats, indent=2, default=str) + "\n```",
            }],
        )
        text = "".join(b.text for b in resp.content if b.type == "text").strip()
        if not text or text.upper().strip().rstrip(".") == "OK":
            return None
        return text
    except Exception as e:
        print(f"[claude_review] review failed: {e}")
        return None
