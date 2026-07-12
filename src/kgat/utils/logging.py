"""Run logging: local JSONL always, Weights & Biases optionally.

Every run writes newline-delimited JSON records locally (durable, survives session
resets, no external service). W&B is gated behind config and imported lazily so the
foundation never depends on it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class JSONLLogger:
    """Append JSON records to a ``.jsonl`` file, creating parent dirs as needed."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Truncate on open so a re-run starts clean.
        self._fh = self.path.open("w", encoding="utf-8")

    def log(self, record: dict[str, Any]) -> None:
        self._fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> JSONLLogger:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class WandbRun:
    """Thin optional W&B wrapper. No-op unless ``enabled`` and ``wandb`` is installed."""

    def __init__(
        self,
        enabled: bool,
        project: str | None = None,
        config: dict[str, Any] | None = None,
        name: str | None = None,
    ) -> None:
        self._run = None
        if not enabled:
            return
        try:
            import wandb  # noqa: PLC0415
        except ImportError:  # pragma: no cover - optional extra
            return
        self._run = wandb.init(project=project, config=config, name=name)

    def log(self, record: dict[str, Any]) -> None:
        if self._run is not None:
            self._run.log(record)

    def finish(self) -> None:
        if self._run is not None:
            self._run.finish()


def snapshot_config(cfg: Any, path: str | Path) -> None:
    """Write a resolved config snapshot next to a run's logs for reproducibility.

    Uses OmegaConf when available (Hydra configs); falls back to JSON for plain dicts.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from omegaconf import OmegaConf  # noqa: PLC0415

        if OmegaConf.is_config(cfg):
            path.write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
            return
    except ImportError:  # pragma: no cover - omegaconf is a core dep
        pass
    path.write_text(json.dumps(cfg, indent=2, default=str), encoding="utf-8")


__all__ = ["JSONLLogger", "WandbRun", "snapshot_config"]
