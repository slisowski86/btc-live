"""
Email transport for live-trader alerts. SMTP (Gmail or any provider), configured from env /
secrets.env. Every send is BEST-EFFORT: if email isn't configured or a send fails, it logs a
warning and returns False - it NEVER raises, so a notification problem can't take down the trader.

secrets.env keys (see secrets.env.example):
    EMAIL_ENABLED=1                       # set 0 to mute all email
    SMTP_HOST=smtp.gmail.com
    SMTP_PORT=587                          # 587 = STARTTLS, 465 = SSL
    SMTP_USER=you@gmail.com
    SMTP_PASSWORD=your_16_char_app_password   # Gmail: an APP PASSWORD, not your login password
    EMAIL_FROM=you@gmail.com               # optional; defaults to SMTP_USER
    EMAIL_TO=you@gmail.com                  # comma-separate for multiple recipients

Gmail app password: Google Account -> Security -> 2-Step Verification -> App passwords.
"""
from __future__ import annotations
import os, ssl, smtplib
from email.message import EmailMessage


def _int(v, default):
    try:
        return int(str(v).split("#")[0].strip())   # tolerate a stray inline comment
    except (TypeError, ValueError):
        return default


def _cfg():
    return {
        "enabled": os.environ.get("EMAIL_ENABLED", "1").strip().lower() not in ("0", "false", "no", ""),
        "host": os.environ.get("SMTP_HOST", "smtp.gmail.com").strip(),
        "port": _int(os.environ.get("SMTP_PORT", "587"), 587),
        "user": os.environ.get("SMTP_USER", "").strip(),
        "password": os.environ.get("SMTP_PASSWORD", "").strip(),
        "sender": (os.environ.get("EMAIL_FROM") or os.environ.get("SMTP_USER", "")).strip(),
        "to": [x.strip() for x in os.environ.get("EMAIL_TO", "").split(",") if x.strip()],
    }


def email_configured() -> bool:
    c = _cfg()
    return bool(c["enabled"] and c["user"] and c["password"] and c["to"])


def send_email(subject: str, body: str, prefix: str = "[BTC-LIVE]") -> bool:
    """Send a plaintext email. Returns True on success, False otherwise. Never raises."""
    c = _cfg()
    if not c["enabled"]:
        return False
    if not (c["user"] and c["password"] and c["to"]):
        print("[notifier] email not configured (SMTP_USER / SMTP_PASSWORD / EMAIL_TO) - skipping send")
        return False

    msg = EmailMessage()
    msg["Subject"] = f"{prefix} {subject}".strip() if prefix else subject
    msg["From"] = c["sender"] or c["user"]
    msg["To"] = ", ".join(c["to"])
    msg.set_content(body)

    try:
        ctx = ssl.create_default_context()
        if c["port"] == 465:
            with smtplib.SMTP_SSL(c["host"], c["port"], context=ctx, timeout=30) as s:
                s.login(c["user"], c["password"])
                s.send_message(msg)
        else:
            with smtplib.SMTP(c["host"], c["port"], timeout=30) as s:
                s.ehlo()
                s.starttls(context=ctx)
                s.login(c["user"], c["password"])
                s.send_message(msg)
        return True
    except Exception as e:
        print(f"[notifier] send failed: {e}")
        return False


if __name__ == "__main__":
    # quick test:  py notifier.py   (loads secrets.env if present)
    try:
        from paper_trader import load_secrets
        load_secrets()
    except Exception:
        pass
    ok = send_email("test email", "If you can read this, email alerts are configured correctly.")
    print("sent OK" if ok else "NOT sent (check EMAIL_* config in secrets.env)")
