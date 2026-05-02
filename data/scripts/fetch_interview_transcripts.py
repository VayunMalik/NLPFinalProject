"""Fetch transcripts/article text for each row in data/raw/lecun_interviews.json.

YouTube URLs use youtube-transcript-api. Web articles must be supplied via the
--articles-jsonl argument (each row: {"url": ..., "text": ...}); this script
does not fetch HTML directly so paywalled/JS-rendered sites are handled out of
band (e.g. via WebFetch). Apple Podcasts URLs are recorded as unsupported.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlparse

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
)


YT_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"}
APPLE_HOSTS = {"podcasts.apple.com"}
ENGLISH_LANGS = ["en", "en-US", "en-GB", "a.en"]


def extract_video_id(url: str) -> Optional[str]:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if host == "youtu.be":
        return parsed.path.lstrip("/") or None
    if host in YT_HOSTS:
        qs = parse_qs(parsed.query)
        v = qs.get("v", [None])[0]
        return v
    return None


def is_youtube(url: str) -> bool:
    return urlparse(url).netloc.lower() in YT_HOSTS


def is_apple_podcast(url: str) -> bool:
    return urlparse(url).netloc.lower() in APPLE_HOSTS


def fetch_youtube(video_id: str) -> tuple[Optional[str], Optional[str]]:
    api = YouTubeTranscriptApi()
    try:
        fetched = api.fetch(video_id, languages=ENGLISH_LANGS)
    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable) as e:
        return None, f"{type(e).__name__}: {e}"
    except Exception as e:
        return None, f"unexpected: {type(e).__name__}: {e}"
    parts = [snip.text for snip in fetched]
    text = re.sub(r"\s+", " ", " ".join(parts)).strip()
    return (text or None), (None if text else "empty transcript")


def load_articles_index(path: Optional[Path]) -> dict[str, str]:
    if not path:
        return {}
    idx: dict[str, str] = {}
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            idx[row["url"]] = row["text"]
    return idx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/raw/lecun_interviews.json")
    ap.add_argument("--output", default="data/raw/lecun_interviews.json")
    ap.add_argument(
        "--articles-jsonl",
        default=None,
        help="Optional JSONL of pre-fetched article text for non-YouTube URLs.",
    )
    ap.add_argument("--backup-suffix", default=".bak")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    data = json.loads(in_path.read_text())

    if in_path == out_path:
        backup = in_path.with_suffix(in_path.suffix + args.backup_suffix)
        backup.write_text(in_path.read_text())
        print(f"[backup] wrote {backup}", file=sys.stderr)

    article_idx = load_articles_index(
        Path(args.articles_jsonl) if args.articles_jsonl else None
    )

    summary = {"youtube_ok": 0, "youtube_fail": 0, "article_ok": 0, "article_missing": 0, "skipped_apple": 0}

    for iv in data["interviews"]:
        url = iv.get("url", "")
        if is_youtube(url):
            vid = extract_video_id(url)
            if not vid:
                iv["transcript"] = None
                iv["transcript_source"] = "youtube"
                iv["transcript_error"] = "could not parse video id"
                summary["youtube_fail"] += 1
                continue
            text, err = fetch_youtube(vid)
            iv["transcript"] = text
            iv["transcript_source"] = "youtube_captions"
            iv["transcript_error"] = err
            iv["video_id"] = vid
            if text:
                summary["youtube_ok"] += 1
            else:
                summary["youtube_fail"] += 1
            print(f"[yt {'OK' if text else 'ERR'}] {vid}: {err or len(text)} chars")
        elif is_apple_podcast(url):
            iv["transcript"] = None
            iv["transcript_source"] = "apple_podcasts"
            iv["transcript_error"] = "apple podcasts has no public transcript api"
            summary["skipped_apple"] += 1
            print(f"[apple SKIP] {url[:80]}")
        else:
            text = article_idx.get(url)
            iv["transcript"] = text
            iv["transcript_source"] = "article_html" if text else None
            iv["transcript_error"] = None if text else "no article text supplied"
            if text:
                summary["article_ok"] += 1
            else:
                summary["article_missing"] += 1
            print(f"[article {'OK' if text else 'MISS'}] {url[:80]}: {len(text) if text else 0} chars")

    data["transcripts_fetched_at"] = datetime.utcnow().isoformat()
    data["transcript_summary"] = summary

    out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"\nSummary: {summary}")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
