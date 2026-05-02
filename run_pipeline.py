"""
End-to-end pipeline runner.

Stages (in order):
  1. clean       — data/scripts/clean.py        (raw JSON → processed/clean.jsonl)
  2. chunk       — data/scripts/chunk.py         (clean.jsonl → processed/chunked.jsonl)
  3. build       — data/scripts/build_dataset.py (chunked.jsonl → splits/)
  4. train       — model/train.py                (splits/ → model/adapters/)
  5. eval        — eval/run_eval.py              (adapter + test split → eval/results/)

Scraping is intentionally excluded — raw data in data/raw/ is assumed to exist.

Usage examples:
  python run_pipeline.py                          # run all 5 stages
  python run_pipeline.py --from-stage chunk       # resume from chunking
  python run_pipeline.py --stages clean chunk     # run specific stages only
  python run_pipeline.py --config model/config/default.yaml
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

STAGES = ["clean", "chunk", "build", "train", "eval"]

# Defaults that mirror each script's own defaults
DEFAULTS = {
    "clean_output": "data/processed/clean.jsonl",
    "chunk_output": "data/processed/chunked.jsonl",
    "splits_dir": "data/splits",
    "config": "model/config/default.yaml",
    "adapter_dir": "model/adapters/run_default",
    "eval_output": None,  # run_eval.py auto-timestamps
    "tokenizer": "Qwen/Qwen2.5-7B-Instruct",
    "base_model": "Qwen/Qwen2.5-7B-Instruct",
}


def run(cmd: list[str], stage: str) -> None:
    print(f"\n{'='*60}")
    print(f"[{stage}] $ {' '.join(cmd)}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=ROOT)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n[{stage}] FAILED (exit {result.returncode}) after {elapsed:.1f}s", file=sys.stderr)
        sys.exit(result.returncode)
    print(f"\n[{stage}] done in {elapsed:.1f}s")


def stage_clean(args: argparse.Namespace) -> None:
    run([
        sys.executable, "data/scripts/clean.py",
        "--papers", "data/raw/lecun_research.json",
        "--interviews", "data/raw/lecun_interviews.json",
        "--tweets", "data/raw/yann_lecun_tweets.json",
        "--output", args.clean_output,
        "--seed", str(args.seed),
    ], "clean")


def stage_chunk(args: argparse.Namespace) -> None:
    run([
        sys.executable, "data/scripts/chunk.py",
        "--input", args.clean_output,
        "--output", args.chunk_output,
        "--tokenizer", args.tokenizer,
        "--seed", str(args.seed),
    ], "chunk")


def stage_build(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable, "data/scripts/build_dataset.py",
        "--input", args.chunk_output,
        "--splits-dir", args.splits_dir,
        "--seed", str(args.seed),
    ]
    if args.baseline_source:
        cmd += ["--baseline-source", args.baseline_source]
    if args.force_relock:
        cmd.append("--force-relock")
    run(cmd, "build")


def stage_train(args: argparse.Namespace) -> None:
    cmd = [sys.executable, "model/train.py", "--config", args.config]
    if args.max_steps > 0:
        cmd += ["--max-steps", str(args.max_steps)]
    run(cmd, "train")


def stage_eval(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable, "eval/run_eval.py",
        "--adapter", args.adapter_dir,
        "--base", args.base_model,
        "--test-path", str(Path(args.splits_dir) / "test.jsonl"),
        "--baseline-path", str(Path(args.splits_dir) / "baseline.jsonl"),
    ]
    if args.eval_output:
        cmd += ["--output-dir", args.eval_output]
    run(cmd, "eval")


STAGE_FNS = {
    "clean": stage_clean,
    "chunk": stage_chunk,
    "build": stage_build,
    "train": stage_train,
    "eval": stage_eval,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the full LeCun persona fine-tuning pipeline (no scraping).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Stage selection
    sel = p.add_mutually_exclusive_group()
    sel.add_argument(
        "--stages",
        nargs="+",
        choices=STAGES,
        metavar="STAGE",
        help="Run only the named stages (in pipeline order). E.g. --stages clean chunk",
    )
    sel.add_argument(
        "--from-stage",
        choices=STAGES,
        metavar="STAGE",
        help="Skip all stages before this one. E.g. --from-stage train",
    )

    # Paths
    p.add_argument("--clean-output", default=DEFAULTS["clean_output"])
    p.add_argument("--chunk-output", default=DEFAULTS["chunk_output"])
    p.add_argument("--splits-dir", default=DEFAULTS["splits_dir"])
    p.add_argument("--config", default=DEFAULTS["config"], help="Training config YAML")
    p.add_argument("--adapter-dir", default=DEFAULTS["adapter_dir"], help="Where train.py writes the adapter (must match config output_dir)")
    p.add_argument("--eval-output", default=DEFAULTS["eval_output"], help="Eval results directory (auto-timestamped if omitted)")

    # Model
    p.add_argument("--tokenizer", default=DEFAULTS["tokenizer"])
    p.add_argument("--base-model", default=DEFAULTS["base_model"])

    # Data options
    p.add_argument("--baseline-source", default="none", choices=["none", "daily_dialog"])
    p.add_argument("--force-relock", action="store_true", help="Allow rewriting the locked test split")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Cap train stage at this many optimizer steps. Use 5 for a quick smoke test (-1 = full run)",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.stages:
        # Run only the explicitly named stages, preserving pipeline order
        to_run = [s for s in STAGES if s in args.stages]
    elif args.from_stage:
        idx = STAGES.index(args.from_stage)
        to_run = STAGES[idx:]
    else:
        to_run = STAGES

    print(f"Pipeline stages to run: {' → '.join(to_run)}")

    total_t0 = time.time()
    for stage in to_run:
        STAGE_FNS[stage](args)

    total = time.time() - total_t0
    print(f"\n{'='*60}")
    print(f"All stages complete in {total:.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
