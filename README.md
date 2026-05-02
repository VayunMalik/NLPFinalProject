# Persona Fine-Tuning: Replicating Yann LeCun's Linguistic Voice

A fine-tuned language model project that replicates the linguistic style and discourse patterns of **Dr. Yann LeCun** through supervised fine-tuning with LoRA adapters.

## Overview

**Course:** Natural Language Processing  
**Project Type:** Final Project — Persona Fine-Tuning  
**Team:** 
- Vayun Malik (Data)
- Jean Park (Modeling)
- Danielle Copeland (Evaluation & Paper)

### Research Question

**LLMs can mimic expert tones superficially, but how accurately can they replicate them?**

This project measures the depth of persona replication by training on LeCun's research papers, interviews, blog posts, and tweets, then evaluating how faithfully the fine-tuned model reproduces his distinctive voice compared to a generic baseline.

## Problem Statement

While large language models can surface-imitate expert tone, the fidelity of this replication is unclear. By fine-tuning on a curated LeCun corpus and comparing against both a generic Llama baseline and everyday-speech baseline, we quantify the **delta** between expert-tone and generic-tone generation. This has implications for technical-domain NLP and persona modeling.

## System Architecture

### Base Model
- **Llama 3.1 8B Instruct** (open-source, pre-trained)
- Chosen to concentrate effort on fine-tuning rather than pretraining

### Training Approach
- **Supervised Fine-Tuning (SFT)** with **LoRA adapters** for parameter-efficient training
- Next-token prediction objective on LeCun corpus chunks
- Llama 3.1 tokenizer (exactly matched to maintain input/output alignment)

### Data Pipeline
1. **Scrape** — research papers, interviews, blog posts, tweets, talks
2. **Clean** — strip citations, timestamps, speaker labels; standardize formatting
3. **Chunk** — segment long documents (papers) into training-sized passages
4. **Tokenize** — using Llama 3.1 tokenizer
5. **Split** — train / dev / test / everyday-speech baseline
   - **Test set is locked** — never used for tuning

## Repository Structure

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
└── CLAUDE.md             # Development conventions
```

## Getting Started

### Prerequisites
- Python 3.10+
- CUDA 11.8+ (for GPU training)
- Git LFS (for large model files)

### Installation

1. **Clone the repository:**
```bash
git clone <repo-url>
cd NLPFinalProject
```

2. **Create and activate a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### Data Pipeline

#### 1. Scrape and Collect Raw Data
```bash
python data/scripts/scrape_papers.py
python data/scripts/scrape_interviews.py
python data/scripts/scrape_blog_posts.py
```

#### 2. Clean and Process Data
```bash
python data/scripts/clean_text.py \
  --input data/raw/ \
  --output data/processed/
```

#### 3. Chunk Documents
```bash
python data/scripts/chunk_documents.py \
  --input data/processed/ \
  --output data/processed/ \
  --chunk_size 512 \
  --overlap 50
```

#### 4. Create Train/Dev/Test Splits
```bash
python data/scripts/create_splits.py \
  --input data/processed/ \
  --output data/splits/ \
  --train_ratio 0.7 \
  --dev_ratio 0.15 \
  --test_ratio 0.15 \
  --seed 42
```

### Model Training

#### Training Configuration
Hyperparameters are specified in `model/config/`. Create or modify a config file:

```yaml
# model/config/default.yaml
model_name: "meta-llama/Llama-2-8b-instruct"
learning_rate: 1e-4
batch_size: 16
epochs: 3
lora_rank: 8
lora_alpha: 16
seed: 42
```

#### Start Training
```bash
python model/train.py \
  --config model/config/default.yaml \
  --train_data data/splits/train.jsonl \
  --dev_data data/splits/dev.jsonl \
  --output_dir model/adapters/run_001 \
  --seed 42
```

#### Resume Training
```bash
python model/train.py \
  --config model/config/default.yaml \
  --resume_from model/adapters/run_001/checkpoint-500
```

### Inference

#### Generate Text with Fine-Tuned Model
```bash
python model/inference.py \
  --base_model meta-llama/Llama-2-8b-instruct \
  --adapter_path model/adapters/run_001 \
  --prompt "Yann LeCun's thoughts on deep learning:" \
  --max_tokens 200
```

### Evaluation

#### Run All Evaluations
```bash
python eval/run_eval.py \
  --fine_tuned_model model/adapters/run_001 \
  --base_model meta-llama/Llama-2-8b-instruct \
  --test_data data/splits/test.jsonl \
  --output_dir eval/results/run_001
```

#### Individual Metrics

**N-gram Overlap (Completion Task):**
```bash
python eval/ngram.py \
  --predictions eval/results/run_001/completions.jsonl \
  --references data/splits/test.jsonl \
  --n 2 3 4
```

**Embedding Similarity (Open Generation):**
```bash
python eval/embedding.py \
  --predictions eval/results/run_001/generations.jsonl \
  --references data/splits/test.jsonl \
  --model all-MiniLM-L6-v2
```

**Human Evaluation:**
```bash
# Generate A/B test pairs
python eval/human_eval/generate_pairs.py \
  --fine_tuned model/adapters/run_001 \
  --base_model meta-llama/Llama-2-8b-instruct \
  --prompts eval/human_eval/prompts.txt \
  --output eval/human_eval/pairs.csv
```

### Full Pipeline (One Command)

With raw data already in `data/raw/`, run the entire pipeline (clean → chunk → build splits → train → eval):

```bash
python run_pipeline.py
```

#### Resume from a specific stage

```bash
# Skip data processing, start from training
python run_pipeline.py --from-stage train

# Resume from evaluation only
python run_pipeline.py --from-stage eval
```

#### Run specific stages only

```bash
python run_pipeline.py --stages clean chunk build
python run_pipeline.py --stages train eval
```

#### Use a custom training config

```bash
python run_pipeline.py --config model/config/my_config.yaml --adapter-dir model/adapters/my_run
```

#### Full option reference

```
--stages STAGE [STAGE ...]   Run only these stages (clean chunk build train eval)
--from-stage STAGE           Skip all stages before this one
--config PATH                Training config YAML (default: model/config/default.yaml)
--adapter-dir PATH           Where the LoRA adapter is saved (must match config output_dir)
--eval-output PATH           Eval results directory (auto-timestamped if omitted)
--tokenizer HF_ID            Tokenizer for chunking (default: Llama-3.1-8B-Instruct)
--base-model HF_ID           Base model for eval (default: Llama-3.1-8B-Instruct)
--baseline-source            Everyday-speech corpus: none (default) or daily_dialog
--force-relock               Allow rewriting the locked test split
--seed INT                   Global random seed (default: 42)
```

### Important Notes

- **🔒 Never modify `data/splits/test.jsonl`** once finalized
- **📝 Log all decisions** in `DECISIONS.md` when changing configs, data pipeline, or evaluation
- **🔄 Pin seeds** across all runs for reproducibility
- **⚠️ GPU Requirements:** Training the 8B model requires ~24GB VRAM; use gradient checkpointing for smaller GPUs

## Hyperparameters to Tune

- Learning rate
- Batch size
- LoRA rank / alpha
- Number of epochs

## Evaluation Strategy

### Tasks
1. **Open Generation** — prompt with a LeCun-relevant topic; compare output to held-out passages
2. **Completion** — provide sentence prefix; measure how the model completes it

### Metrics

| Metric | Purpose |
|--------|---------|
| **N-gram overlap** | Baseline metric for completion task |
| **Embedding similarity** | Cosine similarity between generated response and held-out LeCun passages; compute delta vs. generic LLM |
| **Human evaluation** | A/B test: fine-tuned vs. generic LLM on identical prompts |

### Baselines to Beat

1. **Base Llama 3.1 (no fine-tuning)** — target ≥20% improvement on next-token prediction
2. **Base Llama 3.1 prompted to act like Yann LeCun** — target ≥10% improvement on next-token prediction
3. **Blurb-generation A/B** — fine-tuned vs. base, judged on faithfulness to LeCun's voice

## Key Constraints

- ✅ **No data leakage** between train/dev/test splits
- ✅ **Locked test set** — never used for hyperparameter tuning
- ✅ **Reproducibility** — all random seeds pinned; configs logged
- ✅ **Cross-team coordination** — data/model/eval work tracked in DECISIONS.md

## Collaboration Model

- **Independent Phase:** Dataset construction (Vayun), training scaffolding (Jean), evaluation harness (Danielle) proceed in parallel
- **Convergence Point:** Once corpus and eval harness are ready, full training runs begin
- **Joint Phase:** Hyperparameter tuning and result analysis done together

## Related Work

This project builds on recent persona-modeling research:

- **CharacterBot** (Wang et al., ACL 2025) — CharLoRA for capturing speech + thought patterns
- **RoleLLM** (Wang et al., ACL 2024) — Role-Conditioned Instruction Tuning; compares fine-tuned vs. vanilla GPT-4
- **Character-LLM** (Shao et al., EMNLP 2023) — Trainable agents for historical figures
- **PersLLM** (Zeng et al., EMNLP 2025) — Multi-source data + CoT; warns that naive SFT can produce rigid personas
- **"Catch Me If You Can?"** (Wang et al., EMNLP 2025) — Few-shot prompting fails for author replication; motivates fine-tuning

**Key Insight:** If naive SFT produces unconvincing output, escalate to preference optimization (DPO) or multi-source training.

## Stretch Goals

If time permits:
- Train parallel models on a politician, athlete, and everyday individual
- Compare persona replication fidelity across domains
- Analyze whether technical/expert personas are easier or harder to replicate than conversational personas

## Development Notes

- All architectural decisions logged in `DECISIONS.md`
- Prefer LaTeX for math notation in paper (`$...$` or `$$...$$`)
- Cite sources in ACL Anthology format (see `paper/` and related work section)
- Seeds pinned across `torch`, `numpy`, `random`, `transformers`
- Library versions pinned in `requirements.txt`

For detailed development conventions, see [CLAUDE.md](./CLAUDE.md).
