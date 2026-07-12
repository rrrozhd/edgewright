"""Dataset loading + download for WebQSP / CWQ / MetaQA (and the bundled sample).

Loading from local ``*.jsonl`` files uses only stdlib ``json`` so the foundation
stays offline and dependency-light. Downloading the real releases lazily imports
HuggingFace ``datasets`` (the ``.[data]`` extra) so importing this module never
pulls a heavy dependency.

On-disk schema is documented in ``kgat.data.subgraph`` and matches the
``rmanluo/RoG-*`` preprocessed releases (verify at M2).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path as FsPath

from kgat.data.schemas import Question
from kgat.data.subgraph import SubgraphRecord, record_from_raw

# Canonical split filenames looked up under a dataset's data dir.
_SPLIT_FILES: dict[str, tuple[str, ...]] = {
    "train": ("train.jsonl", "train.json"),
    "dev": ("dev.jsonl", "validation.jsonl", "dev.json", "valid.jsonl"),
    "validation": ("validation.jsonl", "dev.jsonl", "valid.jsonl"),
    "test": ("test.jsonl", "test.json"),
}


def _resolve_split_path(data_dir: FsPath, split: str) -> FsPath:
    candidates = _SPLIT_FILES.get(split, (f"{split}.jsonl", f"{split}.json"))
    for name in candidates:
        candidate = data_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"no file for split {split!r} under {data_dir} (looked for {candidates}). "
        f"Run scripts/download_data.sh to fetch the dataset, or point "
        f"dataset.data_dir at an existing preprocessed release."
    )


def _iter_jsonl(path: FsPath) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_records(
    data_dir: str | FsPath,
    split: str,
    dataset: str,
    limit: int | None = None,
) -> list[SubgraphRecord]:
    """Load a split into ``SubgraphRecord``s.

    Args:
        data_dir: directory holding ``<split>.jsonl`` (per the documented schema).
        split: "train" | "dev" | "validation" | "test" (aliases resolved).
        dataset: dataset name stamped onto every ``Question`` ("webqsp", ...).
        limit: if set, load at most this many records (fast smoke runs).
    """
    path = _resolve_split_path(FsPath(data_dir), split)
    records: list[SubgraphRecord] = []
    for i, raw in enumerate(_iter_jsonl(path)):
        if limit is not None and i >= limit:
            break
        records.append(record_from_raw(raw, dataset=dataset))
    return records


def load_questions(
    data_dir: str | FsPath,
    split: str,
    dataset: str,
    limit: int | None = None,
) -> list[Question]:
    """Convenience: load just the ``Question``s for a split."""
    return [rec.question for rec in load_records(data_dir, split, dataset, limit=limit)]


def download_dataset(dataset: str, data_dir: str | FsPath) -> None:
    """Download a preprocessed release into ``data_dir`` as ``<split>.jsonl``.

    Lazily imports HuggingFace ``datasets`` (install ``.[data]``). Maps our dataset
    names to the released HF repos and writes each split in the schema documented in
    ``kgat.data.subgraph``.

    NOTE (M1/M2): the repo ids below are the community-standard RoG mirrors and MUST
    be verified against the live releases before relying on the numbers. Do not
    invent alternate ids.
    """
    hf_repos = {
        "webqsp": "rmanluo/RoG-webqsp",
        "cwq": "rmanluo/RoG-cwq",
        "metaqa": "rmanluo/RoG-metaqa",  # verify: MetaQA hop-split naming differs
    }
    if dataset not in hf_repos:
        raise ValueError(f"unknown dataset {dataset!r}; known: {sorted(hf_repos)}")

    try:
        from datasets import load_dataset  # noqa: PLC0415  (lazy, optional dep)
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "downloading datasets requires the 'data' extra: pip install -e '.[data]'"
        ) from exc

    out_dir = FsPath(data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(hf_repos[dataset])
    for split_name, split_ds in ds.items():  # type: ignore[union-attr]
        out_path = out_dir / f"{split_name}.jsonl"
        with out_path.open("w", encoding="utf-8") as fh:
            for row in split_ds:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")


__all__ = ["load_records", "load_questions", "download_dataset"]
