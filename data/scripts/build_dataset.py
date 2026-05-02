import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.seed import seed_everything

SPLIT_RATIOS = (0.8, 0.1, 0.1)  # train / dev / test


def stratified_split(records, seed: int):
    by_src = defaultdict(list)
    for r in records:
        by_src[r["source"]].append(r)

    rng = random.Random(seed)
    train, dev, test = [], [], []
    for src, items in by_src.items():
        rng.shuffle(items)
        n = len(items)
        n_train = int(n * SPLIT_RATIOS[0])
        n_dev = int(n * SPLIT_RATIOS[1])
        # Remainder goes to test so all records land in exactly one split.
        train.extend(items[:n_train])
        dev.extend(items[n_train:n_train + n_dev])
        test.extend(items[n_train + n_dev:])

    # Shuffle within-split so source ordering is not implicit.
    rng.shuffle(train)
    rng.shuffle(dev)
    rng.shuffle(test)
    return train, dev, test


def write_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_baseline(path: Path, source: str, n: int = 1000):
    path.parent.mkdir(parents=True, exist_ok=True)
    if source == "none":
        with path.open("w") as f:
            f.write(
                "# TODO: populate with everyday-speech corpus. "
                "Run build_dataset.py with --baseline-source daily_dialog to fill.\n"
            )
        return 0

    if source == "daily_dialog":
        from datasets import load_dataset

        ds = load_dataset("daily_dialog", split="train")
        utterances = []
        for ex in ds:
            for line in ex["dialog"]:
                line = line.strip()
                if line:
                    utterances.append(line)
                if len(utterances) >= n:
                    break
            if len(utterances) >= n:
                break

        with path.open("w") as f:
            for i, u in enumerate(utterances[:n]):
                f.write(json.dumps({
                    "source": "baseline_daily_dialog",
                    "id": f"dd_{i}",
                    "chunk_idx": 0,
                    "text": u,
                    "n_tokens": None,
                }) + "\n")
        return min(len(utterances), n)

    raise ValueError(f"unknown baseline source: {source}")


def main():
    ap = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    ap.add_argument("--input", default=str(root / "processed" / "chunked.jsonl"))
    ap.add_argument("--splits-dir", default=str(root / "splits"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--baseline-source",
        default="none",
        choices=["none", "daily_dialog"],
    )
    ap.add_argument("--baseline-n", type=int, default=1000)
    ap.add_argument(
        "--force-relock",
        action="store_true",
        help="Override an existing test.sha256 lock and rewrite test.jsonl.",
    )
    args = ap.parse_args()

    seed_everything(args.seed)

    in_path = Path(args.input)
    splits_dir = Path(args.splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)

    records = []
    with in_path.open() as f:
        for line in tqdm(f, desc="loading"):
            records.append(json.loads(line))

    train, dev, test = stratified_split(records, args.seed)

    train_path = splits_dir / "train.jsonl"
    dev_path = splits_dir / "dev.jsonl"
    test_path = splits_dir / "test.jsonl"
    sha_path = splits_dir / "test.sha256"
    baseline_path = splits_dir / "baseline.jsonl"

    # Train and dev are always rewritten.
    write_jsonl(train_path, train)
    write_jsonl(dev_path, dev)

    # Compute prospective test hash by writing to a staging file, hashing it,
    # then comparing before committing.
    staging = splits_dir / "test.jsonl.staging"
    write_jsonl(staging, test)
    new_hash = sha256_file(staging)

    if sha_path.exists() and not args.force_relock:
        existing = sha_path.read_text().strip().split()[0]
        if existing != new_hash:
            staging.unlink()
            raise RuntimeError(
                f"test set lock violated: existing sha256={existing} "
                f"but new run would produce {new_hash}. "
                f"Refusing to overwrite {test_path}. "
                f"Pass --force-relock to override (this invalidates prior eval results)."
            )
        # Hash matches; ensure the file exists with the expected content.
        staging.replace(test_path)
    else:
        if args.force_relock and sha_path.exists():
            sys.stderr.write(
                f"WARNING: --force-relock set; resetting test set lock at {sha_path}\n"
            )
        staging.replace(test_path)
        sha_path.write_text(new_hash + "\n")

    n_baseline = write_baseline(baseline_path, args.baseline_source, args.baseline_n)

    print(f"train: {len(train)} -> {train_path}")
    print(f"dev:   {len(dev)} -> {dev_path}")
    print(f"test:  {len(test)} -> {test_path} (sha256={new_hash[:12]}...)")
    print(f"baseline: {n_baseline} ({args.baseline_source}) -> {baseline_path}")


if __name__ == "__main__":
    main()
