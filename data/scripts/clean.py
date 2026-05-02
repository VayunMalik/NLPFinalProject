import argparse
import html
import json
import re
import sys
from collections import Counter
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.seed import seed_everything

# arXiv-style citations like [1], [12], [1, 2], [1,2,3]
CITATION_RE = re.compile(r"\[\d+(?:\s*,\s*\d+)*\]")
WHITESPACE_RE = re.compile(r"\s+")
URL_RE = re.compile(r"https?://\S+")
LEADING_MENTIONS_RE = re.compile(r"^(?:@\w+\s+)+")
RT_BOILERPLATE_RE = re.compile(r"^RT\s+@\w+:?", re.IGNORECASE)


def collapse_whitespace(s: str) -> str:
    return WHITESPACE_RE.sub(" ", s).strip()


def clean_paper_text(title: str, abstract: str) -> str:
    abstract = CITATION_RE.sub("", abstract)
    abstract = collapse_whitespace(abstract)
    title = collapse_whitespace(title)
    return f"{title}\n\n{abstract}"


def clean_tweet_text(raw: str) -> str:
    s = html.unescape(raw)
    s = LEADING_MENTIONS_RE.sub("", s)
    s = URL_RE.sub("", s)
    return collapse_whitespace(s)


def process_papers(papers):
    for p in papers:
        abstract = p.get("abstract")
        if not abstract or len(abstract) < 200:
            continue
        title = p.get("title") or ""
        text = clean_paper_text(title, abstract)
        yield {
            "source": "papers",
            "id": p.get("paper_id") or p.get("arxiv_id") or p.get("doi") or title[:64],
            "text": text,
            "meta": {
                "title": title,
                "year": p.get("year"),
                "venue": p.get("venue"),
                "authors": p.get("authors"),
                "arxiv_id": p.get("arxiv_id"),
                "doi": p.get("doi"),
                "url": p.get("url"),
            },
        }


def process_interviews(interviews):
    # TODO: replace description-as-text placeholder with fetched transcripts
    # before training. Descriptions are low-signal snippets (often truncated
    # YouTube preview blurbs) and do not represent LeCun's voice well.
    for iv in interviews:
        desc = iv.get("description") or ""
        desc = collapse_whitespace(desc)
        if len(desc) < 80:
            continue
        yield {
            "source": "interviews",
            "id": iv.get("url") or iv.get("title") or "",
            "text": desc,
            "meta": {
                "title": iv.get("title"),
                "source_name": iv.get("source"),
                "date": iv.get("date"),
                "url": iv.get("url"),
                "placeholder": True,
            },
        }


def process_tweets(tweets):
    for t in tweets:
        if t.get("_detected_lang") == "fr":
            continue
        raw = t.get("full_text") or ""
        if RT_BOILERPLATE_RE.match(raw):
            continue
        cleaned = clean_tweet_text(raw)
        if len(cleaned) < 40:
            continue
        yield {
            "source": "tweets",
            "id": str(t.get("tweet_id")),
            "text": cleaned,
            "meta": {
                "created_at": t.get("created_at"),
                "reply_to_username": t.get("reply_to_username"),
                "favorite_count": t.get("favorite_count"),
                "retweet_count": t.get("retweet_count"),
                "lang": t.get("_detected_lang"),
            },
        }


def main():
    ap = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    ap.add_argument("--papers", default=str(root / "raw" / "lecun_research.json"))
    ap.add_argument("--interviews", default=str(root / "raw" / "lecun_interviews.json"))
    ap.add_argument("--tweets", default=str(root / "raw" / "yann_lecun_tweets.json"))
    ap.add_argument("--output", default=str(root / "processed" / "clean.jsonl"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    seed_everything(args.seed)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    papers = json.loads(Path(args.papers).read_text()).get("papers", [])
    interviews = json.loads(Path(args.interviews).read_text()).get("interviews", [])
    tweets = json.loads(Path(args.tweets).read_text()).get("tweets", [])

    counts = Counter()
    with out_path.open("w") as f:
        for rec in tqdm(process_papers(papers), total=len(papers), desc="papers"):
            f.write(json.dumps(rec) + "\n")
            counts[rec["source"]] += 1
        for rec in tqdm(process_interviews(interviews), total=len(interviews), desc="interviews"):
            f.write(json.dumps(rec) + "\n")
            counts[rec["source"]] += 1
        for rec in tqdm(process_tweets(tweets), total=len(tweets), desc="tweets"):
            f.write(json.dumps(rec) + "\n")
            counts[rec["source"]] += 1

    print(f"wrote {out_path}")
    for src in ("papers", "interviews", "tweets"):
        print(f"  {src}: {counts[src]}")
    print(f"  total: {sum(counts.values())}")


if __name__ == "__main__":
    main()
