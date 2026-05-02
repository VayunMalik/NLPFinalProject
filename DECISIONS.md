# Architectural Decisions

Append-only log. Each entry: date, area, decision, rationale.

---

## 2026-05-02 — Repo layout

- Moved scrapers to `data/scripts/` and raw JSON dumps to `data/raw/` to match the layout in `CLAUDE.md`.
- Added `utils/seed.py` for centralized seeding (torch, numpy, random, PYTHONHASHSEED).
- Pinned versions in `requirements.txt`. `transformers==4.45.2`, `peft==0.13.2`, `trl==0.11.4` chosen because they jointly support Llama 3.1 chat templates and `SFTTrainer` packing.

## 2026-05-02 — Base model and training

- Base model: `meta-llama/Llama-3.1-8B-Instruct`. Per CLAUDE.md.
- Fine-tuning: LoRA via `peft`, attached to attention projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`). Default rank 16, alpha 32, dropout 0.05.
- Loss: causal LM over packed raw text chunks (no instruction template). Rationale: matches the "next-token prediction" framing in CLAUDE.md and the completion eval task. We can revisit with chat-formatted SFT if naive LM yields rigid output (per PersLLM warning).
- Quantization: 4-bit NF4 (bitsandbytes) for QLoRA, so an 8B model fits on a single 24GB consumer GPU. Falls back to fp16/bf16 on larger hardware.

## 2026-05-02 — Data pipeline

- Sources: research papers (title + abstract from Semantic Scholar), tweets (filtered: non-French, no retweets, no @-only replies, full_text length ≥ 40 chars after stripping URLs), interviews.
- **Gap:** `data/raw/lecun_interviews.json` contains only metadata (title, URL, description). Transcripts have not been fetched. Cleaner currently treats descriptions as low-weight snippets; a transcript-fetch step is TODO before training.
- Chunking: 1024-token windows with 128-token stride for paper abstracts (most abstracts fit in one window, but stride handles longer ones). Tweets are kept whole.
- Splits: 80/10/10 train/dev/test, stratified by source. `test.jsonl` is locked — its SHA-256 is recorded in `data/splits/test.sha256` after first generation; subsequent runs verify the hash.
- Everyday-speech baseline (`baseline.jsonl`) is **not yet populated**. Plan: use `daily_dialog` (HuggingFace) as a placeholder; revisit if a better matched corpus emerges.

## 2026-05-02 — Evaluation

- Three metrics, per CLAUDE.md:
  1. N-gram overlap (BLEU-4 + unigram/bigram precision/recall) for the completion task.
  2. Embedding similarity using `sentence-transformers/all-mpnet-base-v2` — chosen as a strong general-purpose encoder; cosine sim is averaged over held-out passages.
  3. Human A/B prompts (20 LeCun-relevant topics in `eval/human_eval/prompts.json`).
- All three baselines tracked: base Llama (no FT), base Llama prompted-as-LeCun, fine-tuned Llama. Targets per CLAUDE.md (≥20% next-token improvement vs. raw base, ≥10% vs. prompted base).
