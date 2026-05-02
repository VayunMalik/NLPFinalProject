import argparse
import json
from pathlib import Path

import numpy as np


_DEFAULT_MODEL = "sentence-transformers/all-mpnet-base-v2"


def embed(texts: list[str], model_name: str = _DEFAULT_MODEL) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    vecs = model.encode(
        texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vecs


def cosine_sim_to_corpus(
    generated: list[str],
    reference_corpus: list[str],
    model_name: str = _DEFAULT_MODEL,
) -> dict:
    if not generated or not reference_corpus:
        return {"mean_max_sim": 0.0, "mean_avg_sim": 0.0, "per_sample": []}

    gen_vecs = embed(generated, model_name=model_name)
    ref_vecs = embed(reference_corpus, model_name=model_name)
    # Embeddings are L2-normalized above, so dot product == cosine similarity.
    sim = gen_vecs @ ref_vecs.T  # (G, R)

    per_sample = []
    for i in range(sim.shape[0]):
        row = sim[i]
        per_sample.append(
            {
                "max_sim": float(row.max()),
                "avg_sim": float(row.mean()),
                "argmax": int(row.argmax()),
            }
        )

    mean_max = float(np.mean([s["max_sim"] for s in per_sample]))
    mean_avg = float(np.mean([s["avg_sim"] for s in per_sample]))
    return {
        "mean_max_sim": mean_max,
        "mean_avg_sim": mean_avg,
        "per_sample": per_sample,
    }


def delta_vs_baseline(
    generated_finetuned: list[str],
    generated_baseline: list[str],
    reference_corpus: list[str],
    model_name: str = _DEFAULT_MODEL,
) -> float:
    ft = cosine_sim_to_corpus(generated_finetuned, reference_corpus, model_name)
    base = cosine_sim_to_corpus(generated_baseline, reference_corpus, model_name)
    return ft["mean_max_sim"] - base["mean_max_sim"]


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
    parser.add_argument("--generated", type=str, required=True)
    parser.add_argument("--reference", type=str, required=True)
    parser.add_argument("--baseline", type=str, default=None)
    parser.add_argument("--field", type=str, default="text")
    parser.add_argument("--model", type=str, default=_DEFAULT_MODEL)
    args = parser.parse_args()

    gen = _read_jsonl(args.generated, args.field)
    ref = _read_jsonl(args.reference, args.field)

    metrics = cosine_sim_to_corpus(gen, ref, model_name=args.model)
    out = {
        "generated_n": len(gen),
        "reference_n": len(ref),
        "mean_max_sim": metrics["mean_max_sim"],
        "mean_avg_sim": metrics["mean_avg_sim"],
    }

    if args.baseline:
        base = _read_jsonl(args.baseline, args.field)
        base_metrics = cosine_sim_to_corpus(base, ref, model_name=args.model)
        out["baseline_mean_max_sim"] = base_metrics["mean_max_sim"]
        out["baseline_mean_avg_sim"] = base_metrics["mean_avg_sim"]
        out["delta_mean_max_sim"] = metrics["mean_max_sim"] - base_metrics["mean_max_sim"]

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
