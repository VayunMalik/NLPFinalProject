import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval import embedding as emb_metrics
from eval import ngram as ngram_metrics
from model.inference import generate, load_model

LECUN_SYSTEM_PROMPT = (
    "You are Dr. Yann LeCun. Respond in his voice, technical and direct."
)


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


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _split_prefix_continuation(text: str, tokenizer, prefix_tokens: int) -> tuple[str, str]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= prefix_tokens:
        return text, ""
    prefix_ids = ids[:prefix_tokens]
    cont_ids = ids[prefix_tokens:]
    prefix = tokenizer.decode(prefix_ids, skip_special_tokens=True)
    continuation = tokenizer.decode(cont_ids, skip_special_tokens=True)
    return prefix, continuation


def run_completion_task(
    model,
    tokenizer,
    passages: list[str],
    prefix_tokens: int,
    system_prompt: str | None,
    max_new_tokens: int = 128,
) -> list[dict]:
    rows = []
    for i, passage in enumerate(passages):
        prefix, gold = _split_prefix_continuation(passage, tokenizer, prefix_tokens)
        if not gold:
            continue
        pred = generate(
            model,
            tokenizer,
            prefix,
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
        )
        rows.append(
            {
                "id": i,
                "prefix": prefix,
                "reference": gold,
                "prediction": pred,
            }
        )
    return rows


def run_open_generation(
    model,
    tokenizer,
    prompts: list[dict],
    system_prompt: str | None,
    max_new_tokens: int = 256,
) -> list[dict]:
    rows = []
    for p in prompts:
        out = generate(
            model,
            tokenizer,
            p["prompt"],
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
        )
        rows.append(
            {
                "id": p["id"],
                "topic": p.get("topic", ""),
                "prompt": p["prompt"],
                "text": out,
            }
        )
    return rows


def _summarize_completion(rows: list[dict]) -> dict:
    preds = [r["prediction"] for r in rows]
    refs = [r["reference"] for r in rows]
    return ngram_metrics.score_completions(preds, refs)


def _summarize_open_gen(rows: list[dict], reference_corpus: list[str]) -> dict:
    gens = [r["text"] for r in rows]
    return emb_metrics.cosine_sim_to_corpus(gens, reference_corpus)


def _markdown_table(summary: dict) -> str:
    runs = ["base", "base_prompted", "finetuned"]
    lines = []
    lines.append("# Eval Summary\n")
    lines.append("## Completion task (n-gram vs. held-out continuation)\n")
    lines.append("| Run | n | Unigram-P | Bigram-P | Trigram-P | BLEU-4 |")
    lines.append("|---|---|---|---|---|---|")
    for r in runs:
        c = summary[r]["completion"]
        lines.append(
            f"| {r} | {c.get('n', 0)} | {c.get('unigram_precision', 0):.4f} | "
            f"{c.get('bigram_precision', 0):.4f} | {c.get('trigram_precision', 0):.4f} | "
            f"{c.get('bleu4', 0):.4f} |"
        )
    lines.append("\n## Open generation (embedding similarity to held-out LeCun corpus)\n")
    lines.append("| Run | mean_max_sim | mean_avg_sim |")
    lines.append("|---|---|---|")
    for r in runs:
        o = summary[r]["open_gen"]
        lines.append(f"| {r} | {o['mean_max_sim']:.4f} | {o['mean_avg_sim']:.4f} |")
    lines.append("\n## Headline deltas\n")
    d = summary["deltas"]
    lines.append(f"- BLEU-4 finetuned vs. base: **{d['bleu4_ft_vs_base']:+.4f}**")
    lines.append(
        f"- BLEU-4 finetuned vs. base_prompted: **{d['bleu4_ft_vs_prompted']:+.4f}**"
    )
    lines.append(
        f"- Unigram-P finetuned vs. base: **{d['unigram_p_ft_vs_base']:+.4f}**"
    )
    lines.append(
        f"- Unigram-P finetuned vs. base_prompted: **{d['unigram_p_ft_vs_prompted']:+.4f}**"
    )
    lines.append(
        f"- mean_max_sim finetuned vs. base: **{d['sim_ft_vs_base']:+.4f}**"
    )
    lines.append(
        f"- mean_max_sim finetuned vs. base_prompted: **{d['sim_ft_vs_prompted']:+.4f}**"
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=str, required=True, help="Path to LoRA adapter dir")
    parser.add_argument("--base", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--test-path", type=str, default="data/splits/test.jsonl")
    parser.add_argument(
        "--baseline-path",
        type=str,
        default="data/splits/baseline.jsonl",
        help="Everyday-speech baseline corpus (currently unused for scoring; kept for parity with CLAUDE.md)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Defaults to eval/results/<timestamp>",
    )
    parser.add_argument("--n-completion-prompts", type=int, default=200)
    parser.add_argument("--prompt-prefix-tokens", type=int, default=32)
    parser.add_argument("--max-new-tokens-completion", type=int, default=128)
    parser.add_argument("--max-new-tokens-open", type=int, default=256)
    parser.add_argument(
        "--prompts-path",
        type=str,
        default=str(Path(__file__).parent / "human_eval" / "prompts.json"),
    )
    args = parser.parse_args()

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(__file__).parent / "results" / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    test_passages = _read_jsonl(args.test_path, field="text")
    completion_passages = test_passages[: args.n_completion_prompts]
    reference_corpus = test_passages

    with open(args.prompts_path, "r", encoding="utf-8") as f:
        open_prompts = json.load(f)

    run_specs = [
        {"name": "base", "adapter": None, "system_prompt": None},
        {"name": "base_prompted", "adapter": None, "system_prompt": LECUN_SYSTEM_PROMPT},
        {"name": "finetuned", "adapter": args.adapter, "system_prompt": None},
    ]

    summary: dict = {}
    for spec in run_specs:
        t0 = time.time()
        model, tokenizer = load_model(args.base, adapter=spec["adapter"])

        comp_rows = run_completion_task(
            model,
            tokenizer,
            completion_passages,
            prefix_tokens=args.prompt_prefix_tokens,
            system_prompt=spec["system_prompt"],
            max_new_tokens=args.max_new_tokens_completion,
        )
        open_rows = run_open_generation(
            model,
            tokenizer,
            open_prompts,
            system_prompt=spec["system_prompt"],
            max_new_tokens=args.max_new_tokens_open,
        )

        run_dir = out_dir / spec["name"]
        _write_jsonl(run_dir / "completion.jsonl", comp_rows)
        _write_jsonl(run_dir / "open_gen.jsonl", open_rows)

        comp_metrics = _summarize_completion(comp_rows)
        open_metrics = _summarize_open_gen(open_rows, reference_corpus)
        # Strip per-sample sims to keep summary.json compact; raw sims are derivable from saved JSONL.
        open_metrics_compact = {
            k: v for k, v in open_metrics.items() if k != "per_sample"
        }

        summary[spec["name"]] = {
            "adapter": spec["adapter"],
            "system_prompt": spec["system_prompt"],
            "elapsed_sec": time.time() - t0,
            "completion": comp_metrics,
            "open_gen": open_metrics_compact,
        }

        # Free GPU memory between runs.
        del model
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass

    base_c = summary["base"]["completion"]
    bp_c = summary["base_prompted"]["completion"]
    ft_c = summary["finetuned"]["completion"]
    base_o = summary["base"]["open_gen"]
    bp_o = summary["base_prompted"]["open_gen"]
    ft_o = summary["finetuned"]["open_gen"]
    deltas = {
        "bleu4_ft_vs_base": ft_c.get("bleu4", 0) - base_c.get("bleu4", 0),
        "bleu4_ft_vs_prompted": ft_c.get("bleu4", 0) - bp_c.get("bleu4", 0),
        "unigram_p_ft_vs_base": ft_c.get("unigram_precision", 0)
        - base_c.get("unigram_precision", 0),
        "unigram_p_ft_vs_prompted": ft_c.get("unigram_precision", 0)
        - bp_c.get("unigram_precision", 0),
        "sim_ft_vs_base": ft_o["mean_max_sim"] - base_o["mean_max_sim"],
        "sim_ft_vs_prompted": ft_o["mean_max_sim"] - bp_o["mean_max_sim"],
    }
    summary["deltas"] = deltas
    summary["config"] = {
        "adapter": args.adapter,
        "base": args.base,
        "test_path": args.test_path,
        "baseline_path": args.baseline_path,
        "n_completion_prompts": args.n_completion_prompts,
        "prompt_prefix_tokens": args.prompt_prefix_tokens,
    }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write(_markdown_table(summary))

    print(f"Wrote results to {out_dir}")
    print(json.dumps(deltas, indent=2))


if __name__ == "__main__":
    main()
