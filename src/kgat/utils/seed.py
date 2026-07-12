"""Global seeding for reproducibility.

Seeds Python's ``random``, NumPy, and (if installed) Torch. Torch is imported lazily
and guarded so seeding never forces the model dependency onto the foundation.
"""

from __future__ import annotations

import os
import random


def set_seed(seed: int, *, deterministic_torch: bool = True) -> None:
    """Seed all RNGs. ``deterministic_torch`` also sets cuDNN deterministic flags."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np  # noqa: PLC0415

        np.random.seed(seed)
    except ImportError:  # pragma: no cover - numpy is a core dep, but stay defensive
        pass

    try:
        import torch  # noqa: PLC0415
    except ImportError:  # torch is an optional (.[ml]) extra
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


__all__ = ["set_seed"]
