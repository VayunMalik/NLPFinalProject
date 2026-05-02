"""
Fetch Yann LeCun's tweets (excluding retweets) from the Community Archive's
public Supabase/PostgREST API. No X account, no dev key, no scraping.

Community Archive: https://www.community-archive.org

Install:
    pip install requests langdetect

Usage:
    python scrapeTwitter.py
    python scrapeTwitter.py --username ylecun --output tweets.json
    python scrapeTwitter.py --include-retweets
    python scrapeTwitter.py --exclude-langs fr,es   # default: fr
    python scrapeTwitter.py --exclude-langs ""       # keep all languages
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from typing import Any

try:
    import requests
except ImportError:
    sys.exit("ERROR: run `pip install requests` first.")

try:
    from langdetect import DetectorFactory, detect
    DetectorFactory.seed = 0
    _HAS_LANGDETECT = True
except ImportError:
    _HAS_LANGDETECT = False


BASE = "https://fabxmporizzqflnftavs.supabase.co/rest/v1"
ANON_KEY = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
    "eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZhYnhtcG9yaXp6cWZsbmZ0YXZzIiwicm9sZSI6ImFub24i"
    "LCJpYXQiOjE3MjIyNDQ5MTIsImV4cCI6MjAzNzgyMDkxMn0."
    "UIEJiUNkLsW28tBHmG-RQDW-I5JNlJLt62CSk9D_qG8"
)
HEADERS = {"apikey": ANON_KEY, "Authorization": f"Bearer {ANON_KEY}"}
PAGE_SIZE = 1000


def resolve_account(username: str) -> dict[str, Any]:
    for table in ("all_account", "account"):
        r = requests.get(
            f"{BASE}/{table}",
            params={"username": f"eq.{username}", "select": "*"},
            headers=HEADERS,
            timeout=30,
        )
        r.raise_for_status()
        rows = r.json()
        if rows:
            return rows[0]
    sys.exit(f"ERROR: @{username} not found in Community Archive.")


def fetch_tweets(account_id: str, include_retweets: bool) -> list[dict[str, Any]]:
    params: list[tuple[str, str]] = [
        ("account_id", f"eq.{account_id}"),
        ("select", "*"),
        ("order", "created_at.desc"),
    ]
    if not include_retweets:
        params.append(("full_text", "not.like.RT @*"))

    results: list[dict[str, Any]] = []
    offset = 0
    while True:
        headers = {**HEADERS, "Range-Unit": "items", "Range": f"{offset}-{offset + PAGE_SIZE - 1}"}
        r = requests.get(f"{BASE}/tweets", params=params, headers=headers, timeout=60)
        if r.status_code not in (200, 206):
            r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        results.extend(batch)
        print(f"  fetched {len(results)}...", flush=True)
        if len(batch) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
    return results


_URL_RE = re.compile(r"https?://\S+")
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#\w+")


def _clean_for_lang(text: str) -> str:
    text = _URL_RE.sub(" ", text)
    text = _MENTION_RE.sub(" ", text)
    text = _HASHTAG_RE.sub(" ", text)
    return text.strip()


def filter_by_language(tweets: list[dict[str, Any]], exclude: set[str]) -> list[dict[str, Any]]:
    if not exclude:
        return tweets
    if not _HAS_LANGDETECT:
        sys.exit("ERROR: language filtering needs `pip install langdetect`.")

    kept: list[dict[str, Any]] = []
    dropped = 0
    for t in tweets:
        text = _clean_for_lang(t.get("full_text") or "")
        if len(text) < 10:
            kept.append(t)
            continue
        try:
            lang = detect(text)
        except Exception:
            kept.append(t)
            continue
        if lang in exclude:
            dropped += 1
            continue
        t["_detected_lang"] = lang
        kept.append(t)
    print(f"  dropped {dropped} tweets in {sorted(exclude)}", flush=True)
    return kept


def main() -> None:
    ap = argparse.ArgumentParser(description="Pull tweets from the Community Archive API.")
    ap.add_argument("--username", default="ylecun", help="handle without @ (default: ylecun)")
    ap.add_argument("--output", default="yann_lecun_tweets.json", help="output JSON path")
    ap.add_argument("--include-retweets", action="store_true", help="keep RTs (default: filtered out)")
    ap.add_argument("--exclude-langs", default="fr", help="comma-separated lang codes to drop (default: fr)")
    args = ap.parse_args()

    print(f"Resolving @{args.username}...", flush=True)
    account = resolve_account(args.username)
    print(
        f"  account_id={account['account_id']} "
        f"display_name={account.get('account_display_name')!r} "
        f"num_tweets={account.get('num_tweets')}",
        flush=True,
    )

    label = "all tweets" if args.include_retweets else "tweets (retweets excluded)"
    print(f"Fetching {label}...", flush=True)
    tweets = fetch_tweets(account["account_id"], args.include_retweets)

    exclude_langs = {s.strip() for s in args.exclude_langs.split(",") if s.strip()}
    if exclude_langs:
        print(f"Filtering out languages: {sorted(exclude_langs)}...", flush=True)
        tweets = filter_by_language(tweets, exclude_langs)

    payload = {
        "user": account,
        "retweets_excluded": not args.include_retweets,
        "excluded_langs": sorted(exclude_langs),
        "tweet_count": len(tweets),
        "tweets": tweets,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    print(f"Wrote {len(tweets)} tweets to {args.output}", flush=True)


if __name__ == "__main__":
    main()
