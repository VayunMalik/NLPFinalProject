import argparse
import json
import re
import statistics
from collections import Counter
from pathlib import Path

import sacrebleu


def _tokenize(text: str) -> list[str]:
    text = text.lower()
    try:
        from nltk.tokenize import word_tokenize

        return word_tokenize(text)
    except (ImportError, LookupError):
        # nltk punkt data may be missing; regex fallback keeps the harness usable offline.
        return re.findall(r"\w+", text)


def _ngrams(tokens: list[str], n: int) -> Counter:
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def unigram_overlap(pred: str, ref: str) -> dict:
    pred_toks = _tokenize(pred)
    ref_toks = _tokenize(ref)
    pred_counts = Counter(pred_toks)
    ref_counts = Counter(ref_toks)
    overlap = sum((pred_counts & ref_counts).values())
    p = overlap / len(pred_toks) if pred_toks else 0.0
    r = overlap / len(ref_toks) if ref_toks else 0.0
    f = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return {"precision": p, "recall": r, "f1": f}


def ngram_precision(pred: str, ref: str, n: int) -> float:
    pred_ng = _ngrams(_tokenize(pred), n)
    ref_ng = _ngrams(_tokenize(ref), n)
    total = sum(pred_ng.values())
    if total == 0:
        return 0.0
    matched = sum((pred_ng & ref_ng).values())
    return matched / total


def bleu4(pred: str, ref: str) -> float:
    # sentence_bleu returns BLEU on a 0-100 scale; rescale to [0,1] for consistency with other metrics.
    score = sacrebleu.sentence_bleu(pred, [ref]).score
    return score / 100.0


def score_completions(predictions: list[str], references: list[str]) -> dict:
    if len(predictions) != len(references):
        raise ValueError(
            f"predictions/references length mismatch: {len(predictions)} vs {len(references)}"
        )
    if not predictions:
        return {"n": 0}

    uni_p, uni_r, uni_f = [], [], []
    bi_p, tri_p, bleus = [], [], []
    pred_lens, ref_lens = [], []

    for pred, ref in zip(predictions, references):
        u = unigram_overlap(pred, ref)
        uni_p.append(u["precision"])
        uni_r.append(u["recall"])
        uni_f.append(u["f1"])
        bi_p.append(ngram_precision(pred, ref, 2))
        tri_p.append(ngram_precision(pred, ref, 3))
        bleus.append(bleu4(pred, ref))
        pred_lens.append(len(_tokenize(pred)))
        ref_lens.append(len(_tokenize(ref)))

    def _mean(xs: list[float]) -> float:
        return float(statistics.mean(xs)) if xs else 0.0

    return {
        "n": len(predictions),
        "unigram_precision": _mean(uni_p),
        "unigram_recall": _mean(uni_r),
        "unigram_f1": _mean(uni_f),
        "bigram_precision": _mean(bi_p),
        "trigram_precision": _mean(tri_p),
        "bleu4": _mean(bleus),
        "pred_len_mean": _mean(pred_lens),
        "ref_len_mean": _mean(ref_lens),
    }


def _read_jsonl(path: str, field: str = "text") -> list[str]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out.append(row[field])
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=str, required=True)
    parser.add_argument("--references", type=str, required=True)
    parser.add_argument("--field", type=str, default="text")
    args = parser.parse_args()

    preds = _read_jsonl(args.predictions, args.field)
    refs = _read_jsonl(args.references, args.field)
    metrics = score_completions(preds, refs)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
