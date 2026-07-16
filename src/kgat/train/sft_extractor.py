"""Supervised fine-tuning of the backfill triple extractor (DESIGN-BACKFILL.md).

LoRA/QLoRA SFT of the small decoder on distant-supervision pairs
(``kgat.data.backfill_export``): given a filing chunk + filer, the model learns to
emit the teacher's triples in the constrained triple grammar — `` NONE`` or
`` <relation> :: <target>`` items. Targets are tokenized with
``encode_triples_target`` (the grammar's canonical segment-wise ids, NOT a flat
``tokenizer.encode`` of the joined string), so training and constrained inference
score the identical token sequence. Loss is completion-only, exactly like the
controller SFT; the trainer scaffold is shared via ``kgat.train.sft.fit_lora``.

CLI::

    python -m kgat.data.backfill_export --out data/backfill/synthetic
    python -m kgat.train.sft_extractor train=sft_extractor model=qwen3-0.6b
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kgat.controller.constrained_decoding import (
    TripleGrammar,
    build_triple_grammar,
    encode_triples_target,
)
from kgat.controller.prompting import format_extraction_prompt
from kgat.data.backfill_export import ExtractionPair, read_pairs_jsonl
from kgat.utils.hf import attach_lora, load_causal_lm, require_ml


def load_vocab(data_dir: str | Path) -> dict:
    """Read the ``vocab.json`` written by ``backfill_export.export_dataset``."""
    path = Path(data_dir) / "vocab.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run `python -m kgat.data.backfill_export --out {data_dir}` first"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def encode_extraction_example(
    pair: ExtractionPair,
    tokenizer: Any,
    grammar: TripleGrammar,
    *,
    max_seq_len: int,
) -> dict[str, list[int]]:
    """Tokenize one pair with completion-only labels.

    ``input_ids = prompt + grammar-canonical target``; ``labels`` mask the prompt
    with -100. Overlong prompts truncate from the LEFT so the chunk-text tail and
    the trailing ``extraction:`` cue always survive (mirrors ``sft.encode_example``).
    """
    prompt = format_extraction_prompt(pair.filer, pair.text, grammar.relations)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    target_ids = encode_triples_target(pair.triples, grammar)

    room = max_seq_len - len(target_ids)
    if room <= 0:
        raise ValueError(f"max_seq_len={max_seq_len} too small for target {pair.triples!r}")
    if len(prompt_ids) > room:
        prompt_ids = prompt_ids[-room:]

    input_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + target_ids
    return {"input_ids": input_ids, "labels": labels}


def run_sft_extractor(cfg: Any) -> Path:
    """Train the extractor LoRA adapter and return its output directory."""
    require_ml()

    from kgat.train.sft import fit_lora
    from kgat.utils.paths import resolve_path
    from kgat.utils.seed import set_seed

    if "sft_extractor" not in cfg.train:
        raise ValueError("extractor config missing — run with the train=sft_extractor override")
    sft = cfg.train.sft_extractor
    set_seed(int(cfg.seed))

    data_dir = resolve_path(sft.data_dir)
    pairs = read_pairs_jsonl(
        data_dir / f"{sft.split}.jsonl", max_examples=sft.get("max_examples")
    )
    vocab = load_vocab(data_dir)
    output_dir = resolve_path(sft.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer, device = load_causal_lm(
        cfg.model.hf_id,
        device=cfg.get("device", "auto"),
        four_bit=sft.get("four_bit", "auto"),
        train_mode=True,
        gradient_checkpointing=bool(sft.get("gradient_checkpointing", False)),
    )
    model = attach_lora(
        model, r=int(sft.lora_r), alpha=int(sft.lora_alpha), dropout=float(sft.lora_dropout)
    )
    model.print_trainable_parameters()

    max_seq_len = int(sft.max_seq_len)
    max_triples = int(sft.max_triples)
    targets_mode = str(sft.get("targets_mode", "vocab"))

    if targets_mode == "chunk":
        # Open-vocabulary: candidates are the chunk's own capitalized spans; the
        # grammar is per-example and no corpus-level name list exists. Teacher
        # labels are remapped onto candidate surface forms; unmatchable labels
        # are unlearnable under this constraint and get dropped (counted).
        from dataclasses import replace

        from kgat.data.chunk_targets import chunk_target_candidates, match_candidate

        encoded, dropped = [], 0
        for p in pairs:
            candidates = chunk_target_candidates(p.text, filer=p.filer)
            kept = []
            for relation, target in p.triples:
                surface = match_candidate(target, candidates)
                if surface is None:
                    dropped += 1
                else:
                    kept.append((relation, surface))
            grammar = build_triple_grammar(
                vocab["relations"],
                candidates,
                tokenizer,
                eos_id=tokenizer.eos_token_id,
                max_triples=max_triples,
            )
            encoded.append(
                encode_extraction_example(
                    replace(p, triples=tuple(kept)), tokenizer, grammar,
                    max_seq_len=max_seq_len,
                )
            )
        n_gold = sum(len(p.triples) for p in pairs)
        print(
            f"extractor SFT (chunk-local targets): {len(encoded)} pairs, "
            f"{dropped}/{n_gold} gold edges unmatchable in-chunk (dropped)"
        )
    elif targets_mode == "vocab":
        grammar = build_triple_grammar(
            vocab["relations"],
            vocab["targets"],
            tokenizer,
            eos_id=tokenizer.eos_token_id,
            max_triples=max_triples,
        )
        encoded = [
            encode_extraction_example(p, tokenizer, grammar, max_seq_len=max_seq_len)
            for p in pairs
        ]
        print(
            f"extractor SFT: {len(encoded)} pairs from {data_dir}, "
            f"{len(grammar.targets)} grammar targets (device={device})"
        )
    else:
        raise ValueError(f"targets_mode must be 'vocab' or 'chunk', got {targets_mode!r}")

    return fit_lora(
        model,
        tokenizer,
        device,
        encoded=encoded,
        sft_cfg=sft,
        output_dir=output_dir,
        seed=int(cfg.seed),
    )


def _main() -> None:
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base=None, config_path="../../../configs", config_name="config")
    def main(cfg: DictConfig) -> None:
        run_sft_extractor(cfg)

    main()


if __name__ == "__main__":
    _main()


__all__ = ["run_sft_extractor", "encode_extraction_example", "load_vocab"]
