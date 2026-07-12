"""Oracle controller: replays a precomputed relation script, then stops.

The mining teacher (``kgat.train.mine_trajectories``) finds a shortest relation
path from the topic entities to a gold answer by BFS, then replays it through the
*real* ``TraversalEngine`` with this controller. Replaying through the engine —
rather than synthesizing steps by hand — guarantees the recorded ``state_repr`` and
``candidates`` match exactly what a learned controller will see at inference.
"""

from __future__ import annotations

from collections.abc import Sequence

from kgat.controller.base import TraversalController
from kgat.data.schemas import Action, Relation, TraversalState


class OracleController(TraversalController):
    """Follows ``script`` one relation per hop; STOPs when done or off-script."""

    def __init__(self, script: Sequence[Relation]) -> None:
        self.script = tuple(script)

    def select(self, state: TraversalState, candidates: list[Relation]) -> Action:
        if state.step >= len(self.script):
            return Action.stop(score=1.0)
        relation = self.script[state.step]
        if relation not in candidates:
            # The scripted relation is not reachable from the current frontier
            # (e.g. beam pruning dropped the carrier path) — bail out safely.
            return Action.stop(score=0.0)
        return Action.expand(relation, score=1.0)


__all__ = ["OracleController"]
