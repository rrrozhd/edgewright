"""LLM path-reader synthesizer.

Verbalizes the retrieved reasoning paths and asks a causal LM to name the answer
entities. The reader is decoupled from the controller so it can be swapped (a fixed
strong reader isolates the controller's contribution; a small reader measures the
fully-cheap regime).

Parsing is defensive: generated lines/comma-fields are matched (case-insensitively)
against entities that actually appear on the retrieved paths — the reader can only
*select from* the evidence, never introduce an entity the traversal didn't reach.
If nothing matches, falls back to the frontier tails (the dummy behavior), so a
weak reader degrades gracefully rather than answering with hallucinations.
"""

from __future__ import annotations

import re
from typing import Any

from kgat.data.schemas import Entity, Path, Question
from kgat.synthesis.base import AnswerSynthesizer, DummySynthesizer

_PROMPT = (
    "Answer the question using only the knowledge-graph paths below.\n"
    "Question: {question}\n"
    "Paths:\n{paths}\n"
    "Answer entities (comma-separated): "
)


def verbalize_paths(paths: list[Path], max_paths: int = 10) -> str:
    lines: list[str] = []
    for path in paths[:max_paths]:
        if not path.triples:
            continue
        parts = [path.triples[0].head]
        parts.extend(f" -[{t.relation}]-> {t.tail}" for t in path.triples)
        lines.append("- " + "".join(parts))
    return "\n".join(lines) if lines else "- (no paths retrieved)"


class PathReaderSynthesizer(AnswerSynthesizer):
    """Causal-LM reader over verbalized paths. Lazy model load; graceful fallback."""

    def __init__(
        self,
        model_name: str,
        adapter_path: str | None = None,
        max_new_tokens: int = 64,
        max_paths: int = 10,
        device: str = "auto",
        four_bit: str | bool = "auto",
        **kwargs: object,
    ) -> None:
        self.model_name = model_name
        self.adapter_path = adapter_path
        self.max_new_tokens = int(max_new_tokens)
        self.max_paths = int(max_paths)
        self.device = device
        self.four_bit = four_bit
        self._model: Any = None
        self._tokenizer: Any = None
        self._device_str: str | None = None
        self._fallback = DummySynthesizer()
        self._extra = kwargs

    def _ensure_loaded(self) -> None:
        if self._model is None:
            from kgat.utils.hf import load_causal_lm

            self._model, self._tokenizer, self._device_str = load_causal_lm(
                self.model_name,
                adapter_path=self.adapter_path,
                device=self.device,
                four_bit=self.four_bit,
            )

    def synthesize(self, question: Question, paths: list[Path]) -> tuple[Entity, ...]:
        self._ensure_loaded()
        import torch

        tok = self._tokenizer
        prompt = _PROMPT.format(
            question=question.text, paths=verbalize_paths(paths, self.max_paths)
        )
        input_ids = torch.tensor(
            [tok.encode(prompt, add_special_tokens=False)], device=self._device_str
        )
        with torch.no_grad():
            out = self._model.generate(
                input_ids,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
        text = tok.decode(out[0, input_ids.shape[1] :], skip_special_tokens=True)

        # Constrain parsed mentions to entities the traversal actually reached.
        on_path: dict[str, Entity] = {}
        for path in paths:
            for node in path.nodes:
                on_path.setdefault(node.lower(), node)
        picked: dict[Entity, None] = {}
        for piece in re.split(r"[,\n;]+", text):
            key = piece.strip().strip(".").lower()
            if key in on_path:
                picked.setdefault(on_path[key], None)
        if picked:
            return tuple(picked)
        return self._fallback.synthesize(question, paths)


__all__ = ["PathReaderSynthesizer", "verbalize_paths"]
