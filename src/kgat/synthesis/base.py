"""``AnswerSynthesizer`` — turn retrieved paths into answers, plus a dummy impl.

The synthesizer is swappable and independent of the controller: given the question
and the frontier paths the traversal produced, it emits the predicted answer
entities. ``DummySynthesizer`` returns the frontier tails directly, which is enough
to exercise the engine and eval harness end-to-end without a model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from kgat.data.schemas import Entity, Path, Question


class AnswerSynthesizer(ABC):
    """Abstract answer synthesizer."""

    @abstractmethod
    def synthesize(self, question: Question, paths: list[Path]) -> tuple[Entity, ...]:
        """Produce predicted answer entities from the retrieved ``paths``."""
        raise NotImplementedError


class DummySynthesizer(AnswerSynthesizer):
    """Return the distinct current nodes (tails) of the frontier paths as answers.

    Order-preserving and deduped. Empty/rootless paths are skipped. This makes the
    traversal's reached set the prediction — the trivial baseline synthesizer.
    """

    def synthesize(self, question: Question, paths: list[Path]) -> tuple[Entity, ...]:
        seen: dict[Entity, None] = {}
        for path in paths:
            try:
                node = path.current_node
            except ValueError:
                continue
            seen.setdefault(node, None)
        return tuple(seen)


__all__ = ["AnswerSynthesizer", "DummySynthesizer"]
