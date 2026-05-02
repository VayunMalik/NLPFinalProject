import argparse
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

from utils.seed import seed_everything


def _to_ns(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_ns(v) for v in obj]
    return obj


def load_config(path: str) -> SimpleNamespace:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return _to_ns(raw)


def _resolve_dtype(name: str) -> torch.dtype:
    name = name.lower()
    if name in ("bfloat16", "bf16"):
        return torch.bfloat16
    if name in ("float16", "fp16", "half"):
        return torch.float16
    if name in ("float32", "fp32"):
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def _cuda_available() -> bool:
    return torch.cuda.is_available()


def build_model_and_tokenizer(cfg: SimpleNamespace):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4-bit quantization (bitsandbytes) requires CUDA; skip on MPS/CPU
    use_4bit = getattr(cfg.model, "use_4bit", False) and _cuda_available()

    if torch.backends.mps.is_available():
        device_map = "mps"
    elif _cuda_available():
        device_map = "auto"
    else:
        device_map = "cpu"

    model_kwargs = {}
    if use_4bit:
        compute_dtype = _resolve_dtype(cfg.model.bnb_compute_dtype)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = bnb_config
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name,
        device_map=device_map,
        **model_kwargs,
    )
    model.config.use_cache = False
    if hasattr(model.config, "pretraining_tp"):
        model.config.pretraining_tp = 1

    if use_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=cfg.train.gradient_checkpointing,
        )

    return model, tokenizer


def build_lora_config(cfg: SimpleNamespace) -> LoraConfig:
    return LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=list(cfg.lora.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )


def build_sft_config(cfg: SimpleNamespace, max_steps: int = -1) -> SFTConfig:
    cuda = _cuda_available()
    mps = torch.backends.mps.is_available()
    # paged_adamw_8bit and bf16 require CUDA; fall back gracefully on MPS/CPU
    optim = cfg.train.optim if cuda else "adamw_torch"
    bf16 = cfg.train.bf16 and cuda
    # 7B model in bf16 is ~14 GB; cap seq length + batch on MPS to leave room for activations
    max_seq_length = min(cfg.data.max_seq_length, 512) if mps else cfg.data.max_seq_length
    batch_size = 1 if mps else cfg.train.per_device_batch_size
    return SFTConfig(
        output_dir=cfg.train.output_dir,
        num_train_epochs=cfg.train.epochs,
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=cfg.train.grad_accum,
        learning_rate=float(cfg.train.lr),
        warmup_ratio=cfg.train.warmup_ratio,
        weight_decay=cfg.train.weight_decay,
        lr_scheduler_type=cfg.train.lr_scheduler,
        optim=optim,
        eval_strategy="steps",
        eval_steps=cfg.train.eval_steps,
        save_strategy="steps",
        save_steps=cfg.train.save_steps,
        save_total_limit=3,
        logging_steps=cfg.train.logging_steps,
        bf16=bf16,
        gradient_checkpointing=cfg.train.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to=["none"],
        seed=cfg.seed,
        load_best_model_at_end=False,
        dataset_text_field=cfg.data.text_field,
        max_seq_length=max_seq_length,
        packing=cfg.data.packing,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="model/config/default.yaml")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Stop after this many optimizer steps (-1 = train for full epochs)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.seed)

    output_dir = Path(cfg.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.config, output_dir / "config.yaml")

    model, tokenizer = build_model_and_tokenizer(cfg)
    lora_cfg = build_lora_config(cfg)

    data_files = {"train": cfg.data.train_path, "validation": cfg.data.dev_path}
    raw = load_dataset("json", data_files=data_files)
    train_ds = raw["train"]
    eval_ds = raw["validation"]

    sft_cfg = build_sft_config(cfg, max_steps=args.max_steps)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=sft_cfg,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=lora_cfg,
    )

    trainer.train()

    trainer.model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))


if __name__ == "__main__":
    main()
