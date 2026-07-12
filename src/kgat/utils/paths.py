"""Path resolution for Hydra entrypoints.

Hydra changes the process cwd into its run directory, so relative paths from the
config (datasets, outputs, adapters) must be anchored back to where the user
launched the command. Every CLI in the project resolves through here.
"""

from __future__ import annotations

from pathlib import Path


def resolve_path(path: str | Path) -> Path:
    """Resolve a possibly-relative config path against the original launch dir."""
    p = Path(path)
    if p.is_absolute():
        return p
    try:
        from hydra.utils import get_original_cwd

        base = Path(get_original_cwd())
    except (ImportError, ValueError):  # not inside a Hydra run (tests, plain python)
        base = Path.cwd()
    return base / p


__all__ = ["resolve_path"]
