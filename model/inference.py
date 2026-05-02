import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import yaml
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config" / "default.yaml"


def _default_base_model() -> str:
    try:
        with open(DEFAULT_CONFIG, "r") as f:
            return yaml.safe_load(f)["model"]["name"]
    except Exception:
        return "Qwen/Qwen2.5-7B-Instruct"


def _resolve_dtype(name: str) -> torch.dtype:
    name = name.lower()
    if name in ("bfloat16", "bf16"):
        return torch.bfloat16
    if name in ("float16", "fp16", "half"):
        return torch.float16
    if name in ("float32", "fp32"):
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def load_model(
    base: str,
    adapter: Optional[str] = None,
    dtype: str = "bf16",
    device_map: str = "auto",
    load_in_4bit: bool = False,
):
    tokenizer = AutoTokenizer.from_pretrained(base, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {"device_map": device_map}
    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=_resolve_dtype(dtype),
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = bnb_config
    else:
        model_kwargs["torch_dtype"] = _resolve_dtype(dtype)

    model = AutoModelForCausalLM.from_pretrained(base, **model_kwargs)

    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)

    model.eval()
    return model, tokenizer


def _build_chat_inputs(tokenizer, prompt: str, system_prompt: Optional[str]):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    )


def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    system_prompt: Optional[str] = None,
    raw: bool = False,
) -> str:
    device = next(model.parameters()).device

    if raw or system_prompt is None:
        if raw:
            inputs = tokenizer(prompt, return_tensors="pt").input_ids
        else:
            inputs = _build_chat_inputs(tokenizer, prompt, system_prompt=None)
    else:
        inputs = _build_chat_inputs(tokenizer, prompt, system_prompt=system_prompt)

    inputs = inputs.to(device)
    do_sample = temperature is not None and temperature > 0

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    with torch.no_grad():
        output_ids = model.generate(inputs, **gen_kwargs)

    new_tokens = output_ids[0, inputs.shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def _read_prompts(prompt: Optional[str], prompt_file: Optional[str]) -> Iterable[str]:
    if prompt is not None:
        return [prompt]
    if prompt_file is not None:
        path = Path(prompt_file)
        text = path.read_text()
        if path.suffix == ".jsonl":
            out = []
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict):
                    out.append(obj.get("prompt") or obj.get("text") or "")
                else:
                    out.append(str(obj))
            return out
        return [block for block in text.split("\n\n") if block.strip()]
    raise ValueError("must provide --prompt or --prompt-file")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=str, default=None)
    parser.add_argument("--base", type=str, default=_default_base_model())
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--prompt-file", type=str, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--system-prompt", type=str, default=None)
    parser.add_argument("--raw", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--dtype", type=str, default="bf16")
    args = parser.parse_args()

    prompts = _read_prompts(args.prompt, args.prompt_file)

    model, tokenizer = load_model(
        base=args.base,
        adapter=args.adapter,
        dtype=args.dtype,
        load_in_4bit=args.load_in_4bit,
    )

    for i, p in enumerate(prompts):
        out = generate(
            model,
            tokenizer,
            p,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            system_prompt=args.system_prompt,
            raw=args.raw,
        )
        if len(prompts) > 1:
            print(f"=== prompt {i} ===")
        print(out)


if __name__ == "__main__":
    main()
