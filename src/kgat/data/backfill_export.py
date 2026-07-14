"""Distant-supervision pairs for the backfill extractor (DESIGN-BACKFILL.md).

alphina's past backfills pre-paid a labeled dataset: every committed edge stores its
grounded ``evidence_text``, so ``(evidence/chunk text -> schema-valid triple)`` pairs
fall out of the database for free. This module loads that export, and — because the
real export runs later with the user in the loop — also fabricates an equivalent
synthetic dataset from FinKG triples so the whole pilot runs offline end-to-end.

Export contract (read-only against alphina PG; column names verified against
``market_analysis/src/alphina/models/filing.py`` 2026-07-14). Positives — one row
per extracted edge::

    SELECT cr.evidence_text,
           cr.source_company_name AS filer,
           cr.relationship_type,
           cr.target_company_name AS target,
           cr.confidence,
           pf.accession_number
    FROM company_relationships cr
    JOIN processed_filings pf ON cr.filing_id = pf.id
    WHERE cr.confidence >= 0.7;          -- teacher-quality floor; tune

Negatives — chunks from the same filings with no extracted edges (the skip class;
``relationship_type``/``target`` NULL)::

    SELECT fc.text AS evidence_text,
           c.name AS filer,
           NULL AS relationship_type,
           NULL AS target,
           NULL AS confidence,
           pf.accession_number
    FROM filing_chunks fc
    JOIN processed_filings pf ON fc.filing_id = pf.id
    JOIN companies c ON pf.company_id = c.id
    LEFT JOIN company_relationships cr ON cr.chunk_id = fc.id
    WHERE cr.id IS NULL;

Both exported as JSONL (one row-object per line, keys as selected above) into a
single file or several; ``load_export_jsonl`` accepts either. Rows sharing
(accession_number, evidence_text) merge into one multi-triple example. Splits are
by FILING (accession number), never by pair — same-filing evidence in train and
test would leak near-duplicate text.

CLI (synthetic pilot data)::

    python -m kgat.data.backfill_export --out data/backfill/synthetic --filings 150 --seed 42
    # or convert a real export:
    python -m kgat.data.backfill_export --export exports/pairs.jsonl --out data/backfill/real
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

from kgat.data.finkg import FinKG, build_synthetic_kg

# The seven filer-centric relationship types — mirrors alphina's
# relationship_taxonomy.RELATIONSHIP_TYPES (single source of truth over there; this
# copy is the kgat-side contract and is what the triple grammar constrains to).
RELATIONSHIP_TYPES = (
    "supplier",
    "customer",
    "competitor",
    "partner",
    "acquirer",
    "subsidiary",
    "investor",
)

# FinKG forward edge label -> filer-centric taxonomy type. Only forward labels:
# each underlying fact becomes exactly one filer-perspective example. Semantics per
# finkg.py: (X, supplier_to, C) means X supplies C, and alphina's "supplier" means
# the FILER sells/provides to the target — so filer=X, type=supplier, target=C.
FINKG_RELATION_TO_TYPE = {
    "supplier_to": "supplier",
    "customer_of": "customer",
    "competitor_of": "competitor",
    "partner_of": "partner",
    "acquired": "acquirer",
    "subsidiary_of": "subsidiary",
    "invested_in": "investor",
}


@dataclass(frozen=True)
class ExtractionPair:
    """One (chunk text -> triples) training/eval example.

    ``triples`` is filer-centric ``(relationship_type, target_name)`` items; empty
    means the skip class (the grammar's NONE). ``filing`` is the leakage-split unit
    (accession number, or a synthetic stand-in).
    """

    text: str
    filer: str
    triples: tuple[tuple[str, str], ...]
    filing: str
    confidence: float | None = None

    def to_record(self) -> dict:
        return {
            "text": self.text,
            "filer": self.filer,
            "triples": [list(t) for t in self.triples],
            "filing": self.filing,
            "confidence": self.confidence,
        }


def load_export_jsonl(path: str | Path) -> list[ExtractionPair]:
    """Load an alphina export (docstring SQL) into grouped ``ExtractionPair``s.

    Rows with NULL/absent ``relationship_type`` are negatives. Rows sharing
    (accession_number, evidence_text) merge into one multi-triple pair; a pair's
    confidence is the MIN over its rows (weakest teacher edge bounds the example).
    Raises ``ValueError`` on relationship types outside the seven-type taxonomy.
    """
    grouped: dict[tuple[str, str], dict] = {}
    with Path(path).open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = str(row["evidence_text"])
            filer = str(row["filer"])
            filing = str(row["accession_number"])
            rel = row.get("relationship_type")
            key = (filing, text)
            entry = grouped.setdefault(
                key, {"filer": filer, "triples": [], "confidence": None}
            )
            if entry["filer"] != filer:
                raise ValueError(
                    f"line {line_no}: filer {filer!r} != {entry['filer']!r} "
                    f"for the same (filing, evidence_text) group"
                )
            if rel is None:
                continue  # negative row: contributes the group, no triple
            if rel not in RELATIONSHIP_TYPES:
                raise ValueError(
                    f"line {line_no}: relationship_type {rel!r} outside the taxonomy "
                    f"{RELATIONSHIP_TYPES}"
                )
            triple = (str(rel), str(row["target"]))
            if triple not in entry["triples"]:
                entry["triples"].append(triple)
            conf = row.get("confidence")
            if conf is not None:
                conf = float(conf)
                entry["confidence"] = (
                    conf if entry["confidence"] is None else min(entry["confidence"], conf)
                )
    return [
        ExtractionPair(
            text=text,
            filer=e["filer"],
            triples=tuple(e["triples"]),
            filing=filing,
            confidence=e["confidence"],
        )
        for (filing, text), e in grouped.items()
    ]


def split_by_filing(
    pairs: list[ExtractionPair],
    *,
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> dict[str, list[ExtractionPair]]:
    """Filing-disjoint train/dev/test split (the leakage guard).

    Filings are shuffled deterministically and allocated by filing count; every
    pair of one filing lands in exactly one split.
    """
    filings = sorted({p.filing for p in pairs})
    random.Random(seed).shuffle(filings)
    n = len(filings)
    n_train = int(n * fractions[0])
    n_dev = int(n * fractions[1])
    assignment: dict[str, str] = {}
    for i, filing in enumerate(filings):
        split = "train" if i < n_train else ("dev" if i < n_train + n_dev else "test")
        assignment[filing] = split
    out: dict[str, list[ExtractionPair]] = {"train": [], "dev": [], "test": []}
    for p in pairs:
        out[assignment[p.filing]].append(p)
    return out


# ---------------------------------------------------------------------------
# Synthetic pairs — evidence sentences fabricated from FinKG triples so the pilot
# runs offline. Sentence style mimics 10-K filer-perspective prose; the honest
# limits of DESIGN-BACKFILL.md apply doubly here (template text is far easier than
# real filings — this validates the PIPELINE, not the science).
# ---------------------------------------------------------------------------

_TEMPLATES: dict[str, tuple[str, ...]] = {
    "supplier": (
        "We supply our products and related services directly to {t}.",
        "Our customers include {t}, which accounted for a significant portion of net revenue.",
        "We sell components to {t} under a multi-year agreement.",
    ),
    "customer": (
        "We purchase key components from {t}.",
        "Our suppliers include {t}.",
        "We rely on {t} for certain raw materials and manufacturing services.",
    ),
    "competitor": (
        "We compete directly with {t} across several of our product lines.",
        "{t} is one of our principal competitors.",
        "Our industry is highly competitive, and we face significant competition from {t}.",
    ),
    "partner": (
        "We entered into a strategic alliance with {t} to co-develop new products.",
        "We maintain a joint venture with {t}.",
        "Under a licensing agreement, we collaborate with {t} on manufacturing.",
    ),
    "acquirer": (
        "During the year, we completed our acquisition of {t}.",
        "We acquired {t} in a cash-and-stock transaction.",
        "Our acquisition of {t} closed in the fourth quarter.",
    ),
    "subsidiary": (
        "We operate as a wholly owned subsidiary of {t}.",
        "The Company is a subsidiary of {t}.",
        "Our parent company, {t}, controls a majority of our voting stock.",
    ),
    "investor": (
        "We hold a minority equity investment in {t}.",
        "Our investment portfolio includes an equity stake in {t}.",
        "We made a strategic investment in {t} during the period.",
    ),
}

# Boilerplate with no extractable relationship — the negative/skip class. The
# company-mentioning variants are deliberate hard negatives: an entity name appears
# but no taxonomy relationship is stated.
_NOISE = (
    "Our fiscal year ends on December 31.",
    "Our results of operations may fluctuate significantly from period to period.",
    "We are subject to extensive regulation in the jurisdictions in which we operate.",
    "Our common stock is listed on a national securities exchange.",
    "We may require additional capital to fund our operations and growth.",
    "Seasonality has historically affected our quarterly revenue.",
)
_NOISE_WITH_ENTITY = (
    "Securities analysts also publish research regarding {t}.",
    "Broader market conditions, including announcements by {t}, may affect our stock price.",
)
_SECTOR_SENTENCE = "We operate primarily in the {s} sector."


def generate_synthetic_pairs(
    kg: FinKG | None = None,
    *,
    n_filings: int = 150,
    seed: int = 42,
    negatives_per_filing: int = 3,
) -> list[ExtractionPair]:
    """Fabricate filer-perspective evidence chunks from FinKG triples.

    One synthetic filing per company: each of its outgoing forward-label edges
    becomes a positive chunk (occasionally two edges share a chunk — the
    multi-triple case), padded with boilerplate; plus ``negatives_per_filing``
    all-boilerplate chunks. Deterministic in ``seed``.
    """
    rng = random.Random(seed)
    if kg is None:
        kg = build_synthetic_kg(n_companies=n_filings, seed=seed)
    companies = kg.companies[:n_filings]

    # Filer -> its extractable facts, and filer -> sector for negative flavor.
    facts: dict[str, list[tuple[str, str]]] = {c: [] for c in companies}
    sector_of: dict[str, str] = {}
    for t in kg.triples:
        if t.relation in FINKG_RELATION_TO_TYPE and t.head in facts:
            item = (FINKG_RELATION_TO_TYPE[t.relation], t.tail)
            if item not in facts[t.head]:
                facts[t.head].append(item)
        elif t.relation == "in_sector" and t.head in facts:
            sector_of[t.head] = t.tail.split(":", 1)[-1]

    def noise(filer: str, k: int) -> list[str]:
        pool = list(_NOISE)
        if filer in sector_of:
            pool.append(_SECTOR_SENTENCE.format(s=sector_of[filer]))
        other = rng.choice([c for c in companies if c != filer])
        pool.extend(s.format(t=other) for s in _NOISE_WITH_ENTITY)
        return rng.sample(pool, min(k, len(pool)))

    pairs: list[ExtractionPair] = []
    for filer in companies:
        filing = f"synthetic:{filer}"
        edges = list(facts[filer])
        rng.shuffle(edges)
        while edges:
            # ~25% of positive chunks carry two facts (the multi-triple case).
            take = 2 if len(edges) >= 2 and rng.random() < 0.25 else 1
            chunk_triples, edges = tuple(edges[:take]), edges[take:]
            sentences = [
                rng.choice(_TEMPLATES[rel]).format(t=target) for rel, target in chunk_triples
            ]
            sentences.extend(noise(filer, rng.randrange(1, 3)))
            rng.shuffle(sentences)
            pairs.append(
                ExtractionPair(
                    text=" ".join(sentences),
                    filer=filer,
                    triples=chunk_triples,
                    filing=filing,
                    confidence=round(rng.uniform(0.7, 0.99), 3),
                )
            )
        for _ in range(negatives_per_filing):
            pairs.append(
                ExtractionPair(
                    text=" ".join(noise(filer, rng.randrange(2, 4))),
                    filer=filer,
                    triples=(),
                    filing=filing,
                )
            )
    return pairs


# ---------------------------------------------------------------------------
# On-disk pair format (what sft_extractor / extractor_cascade consume)
# ---------------------------------------------------------------------------


def write_pairs_jsonl(pairs: list[ExtractionPair], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for p in pairs:
            fh.write(json.dumps(p.to_record(), ensure_ascii=False) + "\n")


def read_pairs_jsonl(path: str | Path, max_examples: int | None = None) -> list[ExtractionPair]:
    pairs: list[ExtractionPair] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            pairs.append(
                ExtractionPair(
                    text=row["text"],
                    filer=row["filer"],
                    triples=tuple((str(r), str(t)) for r, t in row["triples"]),
                    filing=row["filing"],
                    confidence=row.get("confidence"),
                )
            )
            if max_examples is not None and len(pairs) >= max_examples:
                break
    if not pairs:
        raise ValueError(f"no extraction pairs in {path}")
    return pairs


def target_vocabulary(pairs: list[ExtractionPair]) -> list[str]:
    """Sorted unique target names across all pairs.

    This is the grammar's closed target set — the pilot's stand-in for alphina's
    entity-resolver universe. It is a DECODING constraint (which names are
    emittable), not supervision, so building it over all splits is not leakage;
    what must stay split-disjoint is the filings.
    """
    return sorted({t for p in pairs for _, t in p.triples})


def export_dataset(
    pairs: list[ExtractionPair],
    out_dir: str | Path,
    *,
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> dict[str, int]:
    """Write filing-disjoint train/dev/test JSONLs + ``vocab.json``. Returns counts."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = split_by_filing(pairs, fractions=fractions, seed=seed)
    counts: dict[str, int] = {}
    for split, items in splits.items():
        write_pairs_jsonl(items, out_dir / f"{split}.jsonl")
        counts[split] = len(items)
    vocab = {"relations": list(RELATIONSHIP_TYPES), "targets": target_vocabulary(pairs)}
    (out_dir / "vocab.json").write_text(json.dumps(vocab, indent=2, ensure_ascii=False))
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build backfill extractor pairs (alphina export or synthetic)."
    )
    parser.add_argument("--out", default="data/backfill/synthetic", help="output split dir")
    parser.add_argument(
        "--export", default=None, help="alphina export JSONL (omit for synthetic pairs)"
    )
    parser.add_argument("--filings", type=int, default=150, help="synthetic filings (companies)")
    parser.add_argument("--negatives-per-filing", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.export:
        pairs = load_export_jsonl(args.export)
        source = args.export
    else:
        pairs = generate_synthetic_pairs(
            n_filings=args.filings, seed=args.seed, negatives_per_filing=args.negatives_per_filing
        )
        source = "synthetic FinKG"

    n_pos = sum(1 for p in pairs if p.triples)
    n_multi = sum(1 for p in pairs if len(p.triples) > 1)
    counts = export_dataset(pairs, args.out, seed=args.seed)
    print(
        f"{len(pairs)} pairs from {source}: {n_pos} positive ({n_multi} multi-triple), "
        f"{len(pairs) - n_pos} negative; {len({p.filing for p in pairs})} filings"
    )
    print(f"wrote {counts} + vocab.json -> {args.out}")


if __name__ == "__main__":
    main()


__all__ = [
    "RELATIONSHIP_TYPES",
    "FINKG_RELATION_TO_TYPE",
    "ExtractionPair",
    "load_export_jsonl",
    "split_by_filing",
    "generate_synthetic_pairs",
    "write_pairs_jsonl",
    "read_pairs_jsonl",
    "target_vocabulary",
    "export_dataset",
]
