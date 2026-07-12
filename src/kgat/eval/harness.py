"""Evaluation harness — run a controller+synthesizer over a dataset split.

Hydra entrypoint. Loads a split, builds the in-memory graph store, instantiates the
controller and synthesizer from config, runs the traversal engine per question, and
writes:

* ``<output_dir>/<run_label>.jsonl``        — one record per question (trajectory + cost)
* ``<output_dir>/<run_label>.summary.json`` — aggregate metrics + mean cost (frontier input)
* ``<output_dir>/<run_label>.config.yaml``  — resolved config snapshot

It also prints an aggregate table. Runs fully offline with ``controller=dummy
synth=dummy dataset=sample`` — no GPU, no model deps.

Usage::

    python -m kgat.eval.harness controller=dummy synth=dummy dataset=sample
    python -m kgat.eval.harness experiment=arch_a_flagship model=qwen3-0.6b \
        dataset=webqsp controller=dummy
"""

from __future__ import annotations

import json

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from kgat.data.loaders import load_records
from kgat.eval import metrics as M
from kgat.eval.cost import CostRecord, aggregate, mean_cost
from kgat.governance.policy import build_policies
from kgat.graph.inmemory import InMemoryKGStore
from kgat.traversal.budget import BudgetCaps
from kgat.traversal.engine import TraversalEngine
from kgat.utils.logging import JSONLLogger, WandbRun, snapshot_config
from kgat.utils.paths import resolve_path as _resolve
from kgat.utils.seed import set_seed


def _run_label(cfg: DictConfig) -> str:
    if cfg.get("run_label"):
        return str(cfg.run_label)
    ds = cfg.dataset.name
    model = cfg.get("model", {}).get("name", "model")
    ctrl = str(cfg.controller.get("_target_", "controller")).rsplit(".", 1)[-1]
    hops = cfg.controller.get("max_hops", "na")
    label = f"{ds}_{model}_{ctrl}_hops{hops}"
    # Include lambda when set so a `-m train.grpo.lam=...` sweep gets distinct outputs.
    lam = cfg.get("train", {}).get("grpo", {}).get("lam") if "train" in cfg else None
    if lam is not None:
        label += f"_lam{lam}"
    return label


def evaluate(cfg: DictConfig) -> dict:
    """Run the harness and return the summary dict (also written to disk)."""
    set_seed(int(cfg.seed))

    data_dir = _resolve(cfg.dataset.data_dir)
    records = load_records(
        data_dir,
        split=cfg.dataset.split,
        dataset=cfg.dataset.name,
        limit=cfg.dataset.get("limit"),
    )
    store = InMemoryKGStore.from_records(records)

    controller = instantiate(cfg.controller)
    synthesizer = instantiate(cfg.synth)
    caps = BudgetCaps.from_config(OmegaConf.to_container(cfg.budget, resolve=True))
    gov_cfg = cfg.get("governance", {}) or {}
    gov_enabled = bool(gov_cfg.get("enabled", False))
    policies = build_policies(gov_cfg) if gov_enabled else []
    engine = TraversalEngine(
        store=store,
        controller=controller,
        synthesizer=synthesizer,
        policies=policies,
        budget_caps=caps,
        beam_size=int(cfg.engine.beam_size),
        max_steps=int(cfg.engine.max_steps),
    )

    label = _run_label(cfg)
    out_dir = _resolve(cfg.paths.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / f"{label}.jsonl"

    costs: list[CostRecord] = []
    hits: list[float] = []
    h1s: list[float] = []
    f1s: list[float] = []
    verdicts: list[bool] = []

    with JSONLLogger(jsonl_path) as logger:
        for rec in records:
            q = rec.question
            result = engine.run(q)
            traj = result.trajectory
            pred = traj.predicted_answers
            q_hit = float(M.hit(pred, q.gold_answers))
            q_h1 = M.hits_at_1(pred, q.gold_answers)
            q_f1 = M.f1(pred, q.gold_answers)
            hits.append(q_hit)
            h1s.append(q_h1)
            f1s.append(q_f1)
            costs.append(traj.cost)
            verdicts.append(result.certificate.final_verdict)
            logger.log(
                {
                    "qid": q.qid,
                    "question": q.text,
                    "gold": list(q.gold_answers),
                    "predicted": list(pred),
                    "n_steps": len(traj.steps),
                    "hit": q_hit,
                    "hits_at_1": q_h1,
                    "f1": q_f1,
                    "cost": traj.cost.as_dict(),
                    "audit_verdict": result.certificate.final_verdict,
                }
            )

    n = len(records)
    metrics_agg = {
        "hit": (sum(hits) / n) if n else 0.0,
        "hits_at_1": (sum(h1s) / n) if n else 0.0,
        "f1": (sum(f1s) / n) if n else 0.0,
    }
    summary = {
        "run_label": label,
        "dataset": cfg.dataset.name,
        "split": cfg.dataset.split,
        "n_questions": n,
        "controller": str(cfg.controller.get("_target_", "")),
        "lam": cfg.train.get("grpo", {}).get("lam") if "train" in cfg else None,
        "metrics": metrics_agg,
        "cost": aggregate(costs),
        "mean_cost": {axis: mean_cost(costs, axis) for axis in CostRecord().as_dict()},
        "governance": {
            "enabled": gov_enabled,
            "audit_pass_rate": (sum(verdicts) / n) if n else 0.0,
        },
        "jsonl": str(jsonl_path),
    }

    summary_path = out_dir / f"{label}.summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    snapshot_config(cfg, out_dir / f"{label}.config.yaml")

    wandb_run = WandbRun(
        enabled=bool(cfg.wandb.enabled),
        project=cfg.wandb.get("project"),
        config=OmegaConf.to_container(cfg, resolve=True),
        name=label,
    )
    wandb_run.log({**metrics_agg, **{f"cost/{k}": v for k, v in summary["mean_cost"].items()}})
    wandb_run.finish()

    _print_table(summary)
    return summary


def _print_table(summary: dict) -> None:
    m = summary["metrics"]
    mc = summary["mean_cost"]
    print(f"\n=== {summary['run_label']}  ({summary['n_questions']} questions) ===")
    print(f"  Hit      : {m['hit']:.4f}")
    print(f"  Hits@1   : {m['hits_at_1']:.4f}")
    print(f"  F1       : {m['f1']:.4f}")
    print(f"  mean llm_calls: {mc.get('llm_calls', 0):.3f}   mean hops: {mc.get('hops', 0):.3f}")
    print(f"  wrote {summary['jsonl']}")


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    evaluate(cfg)


if __name__ == "__main__":
    main()


__all__ = ["evaluate"]
