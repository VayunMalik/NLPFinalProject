import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.seed import seed_everything

PAPER_WINDOW = 1024
PAPER_STRIDE = 128


def chunk_paper(token_ids, window: int, stride: int):
    # Sliding window with `stride` token overlap between consecutive windows.
    if len(token_ids) <= window:
        yield token_ids
        return
    step = window - stride
    start = 0
    while start < len(token_ids):
        end = start + window
        chunk = token_ids[start:end]
        yield chunk
        if end >= len(token_ids):
            break
        start += step


def main():
    ap = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    ap.add_argument("--input", default=str(root / "processed" / "clean.jsonl"))
    ap.add_argument("--output", default=str(root / "processed" / "chunked.jsonl"))
    ap.add_argument(
        "--tokenizer",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="HF tokenizer id; override for offline/public testing.",
    )
    ap.add_argument("--window", type=int, default=PAPER_WINDOW)
    ap.add_argument("--stride", type=int, default=PAPER_STRIDE)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    seed_everything(args.seed)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_in = sum(1 for _ in in_path.open())
    n_out = 0
    with in_path.open() as fin, out_path.open("w") as fout:
        for line in tqdm(fin, total=n_in, desc="chunking"):
            rec = json.loads(line)
            ids = tok.encode(rec["text"], add_special_tokens=False)

            if rec["source"] == "papers":
                for i, chunk_ids in enumerate(chunk_paper(ids, args.window, args.stride)):
                    # Decode back to text so downstream re-tokenization is canonical.
                    text = tok.decode(chunk_ids, skip_special_tokens=True)
                    fout.write(json.dumps({
                        "source": rec["source"],
                        "id": rec["id"],
                        "chunk_idx": i,
                        "text": text,
                        "n_tokens": len(chunk_ids),
                    }) + "\n")
                    n_out += 1
            else:
                # Tweets and interview snippets are short; emit whole.
                fout.write(json.dumps({
                    "source": rec["source"],
                    "id": rec["id"],
                    "chunk_idx": 0,
                    "text": rec["text"],
                    "n_tokens": len(ids),
                }) + "\n")
                n_out += 1

    print(f"wrote {out_path}: {n_out} chunks from {n_in} records")


if __name__ == "__main__":
    main()
