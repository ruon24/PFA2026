"""Mint a LiveKit access token for the local --dev server.

Paste the printed JWT into the LiveKit Agents Playground
(https://agents-playground.livekit.io) along with ws://localhost:7880.

Usage from PFA2026/:
    uv run python -m agent.mint_token
    uv run python -m agent.mint_token --room demo --identity jawher --valid-for 24h

Reads LIVEKIT_API_KEY / LIVEKIT_API_SECRET from agent/.env.local (or .env).
Falls back to the Docker --dev defaults (devkey / secret).
"""

from __future__ import annotations

import argparse
import os
from datetime import timedelta
from pathlib import Path

from dotenv import load_dotenv
from livekit import api

_AGENT_DIR = Path(__file__).resolve().parent
load_dotenv(_AGENT_DIR / ".env.local")
load_dotenv(_AGENT_DIR / ".env")


def _parse_duration(text: str) -> timedelta:
    text = text.strip().lower()
    if text.endswith("h"):
        return timedelta(hours=float(text[:-1]))
    if text.endswith("m"):
        return timedelta(minutes=float(text[:-1]))
    if text.endswith("s"):
        return timedelta(seconds=float(text[:-1]))
    return timedelta(seconds=float(text))


def main() -> int:
    parser = argparse.ArgumentParser(description="Mint a LiveKit room JWT.")
    parser.add_argument("--room", default="derja-test", help="Room name to join")
    parser.add_argument("--identity", default="student", help="Participant identity")
    parser.add_argument("--name", default=None, help="Display name (defaults to identity)")
    parser.add_argument("--valid-for", default="24h", help="Token lifetime, e.g. 24h, 30m")
    args = parser.parse_args()

    key = os.environ.get("LIVEKIT_API_KEY", "devkey")
    secret = os.environ.get("LIVEKIT_API_SECRET", "secret")
    url = os.environ.get("LIVEKIT_URL", "ws://localhost:7880")

    token = (
        api.AccessToken(key, secret)
        .with_identity(args.identity)
        .with_name(args.name or args.identity)
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room=args.room,
                can_publish=True,
                can_subscribe=True,
                can_publish_data=True,
            )
        )
        .with_ttl(_parse_duration(args.valid_for))
        .to_jwt()
    )

    print(f"URL:      {url}")
    print(f"Room:     {args.room}")
    print(f"Identity: {args.identity}")
    print(f"TTL:      {args.valid_for}")
    print()
    print(token)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
