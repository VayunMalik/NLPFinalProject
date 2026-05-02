# CLAUDE.md

This file provides guidance to Claude Code when working in this repository.

## Project Overview

**Course:** Natural Language Processing
**Project Type:** Final Project — Persona Fine-Tuning
**Team:** Vayun Malik (Data), Jean Park (Modeling), Danielle Copeland (Evaluation & Paper)

This project fine-tunes a large language model to replicate the linguistic style and discourse patterns of **Dr. Yann LeCun**. The goal is to quantify how expert tone differs from everyday language by training on a curated corpus of LeCun's research papers, interviews, blog posts, and tweets, then evaluating how faithfully the fine-tuned model reproduces his voice.

The broader research question: **LLMs can mimic expert tones superficially, but how accurately can they replicate them?** We test this by measuring syntactic structure, lexical distributions, and semantic patterns against both a generic LLM baseline and held-out LeCun passages.

## Problem Statement

LLMs surface-imitate expert tone but the depth of replication is unclear. By fine-tuning on LeCun's writings/speech and comparing against a generic baseline + everyday-speech corpus, we measure the *delta* between expert-tone generation and generic generation. This has implications for technical-domain NLP and persona modeling.

## System Architecture

### Base Model
- **Llama 3.1 8B Instruct** (open-source, pre-trained)
- Chosen so effort concentrates on fine-tuning rather than pretraining

### Fine-Tuning Approach
- **Supervised Fine-Tuning (SFT)** with **LoRA adapters** for parameter-efficient training
- Next-token prediction objective on LeCun corpus chunks
- Tokenizer must match Llama 3.1's tokenizer exactly to keep inputs/outputs aligned

### Data Pipeline
1. **Scrape** — research papers, interviews, blog posts, tweets, talks
2. **Clean** — strip citations, timestamps, speaker labels; standardize formatting
3. **Chunk** — break long documents (papers) into training-sized segments
4. **Tokenize** — using Llama 3.1 tokenizer
5. **Split** — train / dev / test / everyday-speech baseline (test set is **locked**)

### Hyperparameters to Tune
- Learning rate
- Batch size
- LoRA rank / alpha
- Number of epochs

## Evaluation Strategy

Two task types:
1. **Open Generation** — prompt with a LeCun-relevant topic; compare output to held-out LeCun passages
2. **Completion** — provide sentence prefix; measure how the model completes it

### Metrics
| Metric | Use |
|---|---|
| **N-gram overlap** | Baseline metric for completion task |
| **Embedding similarity** | Mean cosine similarity between generated response and held-out LeCun passages; compute *delta* vs. generic LLM (ChatGPT/Claude) |
| **Human evaluation** | A/B test: fine-tuned vs. generic LLM on identical prompts; which "feels more like an expert's voice" |

### Baselines to Beat
1. **Base Llama 3.1 (no fine-tuning)** — target ≥20% improvement on next-token prediction across data types
2. **Base Llama 3.1 prompted to act like Yann LeCun** — target ≥10% improvement on next-token prediction
3. **Blurb-generation A/B** — fine-tuned vs. base, judged on faithfulness

### Guardrails
- **No data leakage** between train/dev/test
- **Locked test set** — never used for tuning
- **Reproducibility** — seed everything, log configs

## Repository Conventions

When adding code, follow this structure:

```
.
├── data/
│   ├── raw/              # Scraped, untouched sources
│   ├── processed/        # Cleaned, chunked text
│   ├── splits/           # train.jsonl, dev.jsonl, test.jsonl (LOCKED), baseline.jsonl
│   └── scripts/          # Scrapers, cleaners, chunkers
├── model/
│   ├── train.py          # SFT + LoRA training entrypoint
│   ├── config/           # Hyperparameter configs (YAML)
│   ├── adapters/         # Saved LoRA weights
│   └── inference.py      # Generation utilities
├── eval/
│   ├── ngram.py          # N-gram overlap metric
│   ├── embedding.py      # Cosine similarity vs. held-out LeCun
│   ├── human_eval/       # A/B prompt sets and result sheets
│   └── run_eval.py       # End-to-end evaluation runner
├── notebooks/            # Exploration only — not authoritative
├── paper/                # LaTeX source, figures
└── CLAUDE.md
```

## Working Conventions for Claude

When asked to help with this project:

1. **Never modify the locked test set** (`data/splits/test.jsonl`) once it has been finalized. If a change is requested, flag it explicitly.
2. **Log every architectural decision** — when changing the model config, hyperparameters, data pipeline, or evaluation methodology, append a dated entry to `DECISIONS.md`.
3. **Prefer reproducibility** — set seeds (`torch`, `numpy`, `random`, `transformers`); pin library versions in `requirements.txt`.
4. **Keep data and model code separate** — Vayun owns `data/`, Jean owns `model/`, Danielle owns `eval/` and `paper/`. Cross-cutting changes require flagging.
5. **Default to LaTeX for math** — equations in markdown/paper should use `$...$` or `$$...$$`.
6. **Cite sources properly in `paper/`** — ACL Anthology format matches the related-works section in the proposal.

## Related Work (Reference)

These papers inform the methodology — consult them when design questions arise:

- **CharacterBot** (Wang et al., ACL 2025) — CharLoRA mechanism for capturing speech + thought patterns from written works
- **RoleLLM** (Wang et al., ACL 2024) — Role-Conditioned Instruction Tuning (RoCIT); compares fine-tuned model vs. vanilla GPT-4
- **Character-LLM** (Shao et al., EMNLP 2023) — Trainable agents for historical figures (Beethoven, Caesar); interview-based eval
- **PersLLM** (Zeng et al., EMNLP 2025) — Multi-source data + CoT + automated preference optimization; warns that standard SFT alone yields rigid personas
- **"Catch Me If You Can?"** (Wang et al., EMNLP 2025) — Shows few-shot prompting fails to replicate real authors; motivates fine-tuning over prompting

**Key takeaway from PersLLM:** if naive SFT produces rigid, unconvincing output, escalate to multi-source data + preference optimization (DPO/CoT prompting during training).

## Stretch Goals (If Time Permits)

- Train parallel models on a **politician**, an **athlete**, and an **everyday individual**
- Compare which persona type the model imitates most faithfully
- Analyze whether technical/expert personas (LeCun) are easier or harder to replicate than conversational personas

## Collaboration Notes

- **Independent phase:** dataset construction (Vayun), training scaffolding with dummy data (Jean), evaluation harness (Danielle) all proceed in parallel
- **Convergence point:** once the corpus and eval harness are ready, full training runs begin
- **Joint phase:** hyperparameter tuning and result analysis are done together