#!/usr/bin/env python3
"""Send immediate pipeline failure alerts (webhook and/or email)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime
from zoneinfo import ZoneInfo

PT = ZoneInfo("America/Los_Angeles")


def _now_pt() -> str:
    return datetime.now(PT).strftime("%Y-%m-%d %H:%M:%S %Z")


def log_alert_line(severity: str, job: str, message: str) -> None:
    """Structured line for Cloud Logging metrics / alert policies."""
    line = (
        f"PIPELINE_ALERT severity={severity} job={job} "
        f"time_pt={_now_pt()} message={message}"
    )
    print(line, file=sys.stderr)


def post_webhook(url: str, payload: dict) -> bool:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return 200 <= resp.status < 300
    except urllib.error.HTTPError as exc:
        print(f"Webhook HTTP {exc.code}: {exc.read()[:300]!r}", file=sys.stderr)
        return False
    except Exception as exc:
        print(f"Webhook error: {exc}", file=sys.stderr)
        return False


def send_sendgrid_email(to_addr: str, subject: str, body: str) -> bool:
    api_key = os.environ.get("SENDGRID_API_KEY", "").strip()
    from_addr = os.environ.get("PIPELINE_ALERT_FROM", "").strip() or to_addr
    if not api_key:
        return False
    payload = {
        "personalizations": [{"to": [{"email": to_addr}]}],
        "from": {"email": from_addr},
        "subject": subject,
        "content": [{"type": "text/plain", "value": body}],
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        "https://api.sendgrid.com/v3/mail/send",
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return 200 <= resp.status < 300
    except urllib.error.HTTPError as exc:
        print(f"SendGrid HTTP {exc.code}: {exc.read()[:300]!r}", file=sys.stderr)
        return False
    except Exception as exc:
        print(f"SendGrid error: {exc}", file=sys.stderr)
        return False


def notify(job: str, subject: str, body: str, severity: str = "critical") -> int:
    log_alert_line(severity, job, subject)

    sent = False
    webhook = os.environ.get("PIPELINE_ALERT_WEBHOOK_URL", "").strip()
    if webhook:
        payload = {
            "text": f"{subject}\n\n{body}",
            "job": job,
            "severity": severity,
            "time_pt": _now_pt(),
        }
        # Pushover-compatible shape when token/user env vars are set
        push_token = os.environ.get("PUSHOVER_APP_TOKEN", "").strip()
        push_user = os.environ.get("PUSHOVER_USER_KEY", "").strip()
        if push_token and push_user and "pushover.net" in webhook:
            payload = {
                "token": push_token,
                "user": push_user,
                "title": subject[:250],
                "message": body[:1024],
                "priority": 1 if severity == "critical" else 0,
            }
        sent = post_webhook(webhook, payload) or sent

    email_to = os.environ.get("PIPELINE_ALERT_EMAIL", "").strip()
    if email_to:
        sent = send_sendgrid_email(email_to, subject, body) or sent

    if not sent:
        print(
            "No alert channel delivered. Set PIPELINE_ALERT_WEBHOOK_URL and/or "
            "PIPELINE_ALERT_EMAIL + SENDGRID_API_KEY on the Cloud Run job.",
            file=sys.stderr,
        )
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Pipeline failure notification")
    ap.add_argument("--job", required=True, help="Job name, e.g. morning_inference")
    ap.add_argument("--date", default="", help="Slate date YYYY-MM-DD")
    ap.add_argument("--message", default="", help="Summary message")
    ap.add_argument(
        "--failures",
        default="",
        help="Comma-separated failed step names",
    )
    ap.add_argument(
        "--counts",
        default="",
        help="JSON object of table row counts from smoke test",
    )
    ap.add_argument("--severity", default="critical", choices=("critical", "warning"))
    args = ap.parse_args()

    failures = [f.strip() for f in args.failures.split(",") if f.strip()]
    subject = f"[MLB Pipeline] {args.job} FAILED"
    if args.date:
        subject += f" ({args.date})"

    lines = [
        f"Job: {args.job}",
        f"Slate date: {args.date or 'n/a'}",
        f"Time (PT): {_now_pt()}",
        "",
        args.message or "Pipeline did not complete successfully.",
    ]
    if failures:
        lines.extend(["", "Failed steps:", *[f"  - {f}" for f in failures]])
    if args.counts:
        try:
            counts = json.loads(args.counts)
            lines.extend(["", "Row counts:"])
            for k, v in sorted(counts.items()):
                flag = " *** EMPTY ***" if v == 0 else ""
                lines.append(f"  {k}: {v}{flag}")
        except json.JSONDecodeError:
            lines.append(f"\nCounts: {args.counts}")

    lines.extend([
        "",
        "Console: Cloud Run → Jobs → mlb-morning-inference → Logs",
        "Project: mlb-model-491223",
    ])
    body = "\n".join(lines)
    return notify(args.job, subject, body, severity=args.severity)


if __name__ == "__main__":
    sys.exit(main())
