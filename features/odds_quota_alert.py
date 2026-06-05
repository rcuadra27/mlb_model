#!/usr/bin/env python3
"""Pushover alert when Odds API monthly usage crosses a daily threshold (default 50%)."""

from __future__ import annotations

import os
import sys
import urllib.error
import urllib.parse
import urllib.request

from features.pipeline_alert import notify

SPORTS_URL = "https://api.the-odds-api.com/v4/sports"


def _usage_headers(api_key: str) -> dict[str, str]:
    url = f"{SPORTS_URL}?{urllib.parse.urlencode({'apiKey': api_key})}"
    req = urllib.request.Request(url, headers={"User-Agent": "mlb-odds-quota-check/1"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return {k.lower(): v for k, v in resp.headers.items()}
    except urllib.error.HTTPError as exc:
        return {k.lower(): v for k, v in exc.headers.items()}


def main() -> int:
    api_key = (os.environ.get("ODDS_API_KEY") or "").strip()
    if not api_key:
        print("ODDS_API_KEY not set — skip quota check", file=sys.stderr)
        return 0

    quota = int(os.environ.get("ODDS_MONTHLY_QUOTA", "100000"))
    pct = float(os.environ.get("ODDS_ALERT_PCT", "50"))
    threshold = int(quota * (pct / 100.0))

    hdr = _usage_headers(api_key)
    used_s = hdr.get("x-requests-used", "")
    remaining_s = hdr.get("x-requests-remaining", "")
    try:
        used = int(used_s) if used_s != "" else None
        remaining = int(remaining_s) if remaining_s != "" else None
    except ValueError:
        used = remaining = None

    print(f"Odds API usage: used={used} remaining={remaining} quota={quota} alert_at={threshold}")

    if used is None:
        print("Could not parse usage headers", file=sys.stderr)
        return 1

    if used < threshold:
        return 0

    body = (
        f"Odds API used {used:,} credits ({100.0 * used / quota:.1f}% of {quota:,} monthly quota). "
        f"Remaining: {remaining if remaining is not None else 'unknown'}. "
        "Check the-odds-api.com and client bundles for unexpected polling."
    )
    return notify(
        "odds_quota",
        f"Odds API {pct:.0f}% daily threshold ({used:,} used)",
        body,
        severity="critical",
    )


if __name__ == "__main__":
    sys.exit(main())
