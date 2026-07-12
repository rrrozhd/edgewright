"""Lazy HuggingFace loading shared by controllers, synthesizers, and trainers.

torch/transformers/peft are imported inside functions so the foundation package
stays importable without the ``.[ml]`` extra. Every entry point that needs them
funnels through ``require_ml`` for one consistent, helpful error.
"""

from __future__ import annotations

from typing import Any

_ML_HINT = (
    "this component needs the ML extra. Install it with:\n"
    "    pip install -e '.[ml]'\n"
    "(on Colab, where torch is preinstalled: pip install -e '.[colab]')"
)


def require_ml() -> None:
    """Raise a helpful ImportError if torch/transformers are missing."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:
        raise ImportError(f"{_ML_HINT}\n(missing: {exc.name})") from exc


def pick_device(preference: str = "auto") -> str:
    """Resolve a device string: explicit value passes through; 'auto' picks
    cuda > mps > cpu."""
    if preference != "auto":
        return preference
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _bnb_available() -> bool:
    try:
        import bitsandbytes  # noqa: F401

        return True
    except ImportError:
        return False


def load_causal_lm(
    hf_id: str,
    *,
    adapter_path: str | None = None,
    device: str = "auto",
    four_bit: str | bool = "auto",
    train_mode: bool = False,
    gradient_checkpointing: bool = False,
) -> tuple[Any, Any, str]:
    """Load a causal LM + tokenizer, optionally 4-bit (QLoRA base) and/or with a
    LoRA adapter.

    Returns ``(model, tokenizer, device_str)``. ``four_bit="auto"`` enables 4-bit
    only when CUDA + bitsandbytes are available; on MPS/CPU the model loads in
    fp16/fp32. With ``train_mode`` a 4-bit model is prepared for k-bit training.
    """
    require_ml()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = pick_device(device)
    use_4bit = (dev == "cuda" and _bnb_available()) if four_bit == "auto" else bool(four_bit)

    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs: dict[str, Any] = {}
    if use_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = {"": 0}
    else:
        kwargs["torch_dtype"] = torch.float16 if dev in ("cuda", "mps") else torch.float32
    if dev == "mps" and train_mode:
        # MPS SDPA lacks dropout support; eager attention keeps training runnable.
        kwargs["attn_implementation"] = "eager"

    model = AutoModelForCausalLM.from_pretrained(hf_id, **kwargs)
    if not use_4bit:
        model = model.to(dev)

    if use_4bit and train_mode:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=gradient_checkpointing
        )
    elif gradient_checkpointing and train_mode:
        model.gradient_checkpointing_enable()

    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=train_mode)

    if not train_mode:
        model.eval()
    return model, tokenizer, dev


# LoRA target modules for the Qwen2/Qwen3 families (and most Llama-style decoders).
DEFAULT_LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def attach_lora(model: Any, *, r: int, alpha: int, dropout: float) -> Any:
    """Wrap a loaded causal LM with a fresh LoRA adapter for training.

    Targets the Qwen/Llama-style projection names when present, otherwise falls
    back to peft's ``"all-linear"`` so GPT-2-style architectures work too.
    """
    require_ml()
    from peft import LoraConfig, get_peft_model

    module_names = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
    matched = [t for t in DEFAULT_LORA_TARGETS if t in module_names]
    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=matched or "all-linear",
    )
    return get_peft_model(model, config)


__all__ = ["require_ml", "pick_device", "load_causal_lm", "attach_lora", "DEFAULT_LORA_TARGETS"]
