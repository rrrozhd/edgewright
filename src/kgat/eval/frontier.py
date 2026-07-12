"""Cost/quality frontier — the project's headline artifact.

Given several runs at varying ``lam`` or budget caps (each a summary dict written by
``kgat.eval.harness``), build the two frontier views as tidy DataFrames and a plot:

* **accuracy@budget** — the best accuracy reachable at or under a given mean cost.
* **cost@accuracy**   — the minimum mean cost to reach a given accuracy.

The plot is x = mean cost, y = accuracy.

matplotlib uses the headless ``Agg`` backend (set before pyplot import) so this runs
in CI / over SSH with no display.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: must precede pyplot import
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def load_summaries(runs_dir: str | Path, pattern: str = "*.summary.json") -> list[dict]:
    """Load every run-summary JSON under ``runs_dir`` (see harness output format)."""
    runs_dir = Path(runs_dir)
    summaries: list[dict] = []
    for path in sorted(runs_dir.rglob(pattern)):
        summaries.append(json.loads(path.read_text(encoding="utf-8")))
    return summaries


def frontier_dataframe(
    summaries: list[dict],
    *,
    accuracy_metric: str = "hit",
    cost_axis: str = "llm_calls",
) -> pd.DataFrame:
    """Tidy DataFrame with one row per run: label, accuracy, mean_cost (+ raw fields).

    Rows are sorted by ascending mean cost so the frontier reads left-to-right.
    """
    rows: list[dict] = []
    for s in summaries:
        metrics = s.get("metrics", {})
        mean_cost_map = s.get("mean_cost", {})
        rows.append(
            {
                "label": s.get("run_label", s.get("dataset", "run")),
                "lam": s.get("lam"),
                "accuracy": float(metrics.get(accuracy_metric, float("nan"))),
                "mean_cost": float(mean_cost_map.get(cost_axis, float("nan"))),
                "accuracy_metric": accuracy_metric,
                "cost_axis": cost_axis,
                "n_questions": s.get("n_questions"),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("mean_cost", kind="stable").reset_index(drop=True)
    return df


def accuracy_at_budget(df: pd.DataFrame, budget: float) -> float:
    """Best accuracy among runs whose mean cost is <= ``budget`` (NaN if none)."""
    eligible = df[df["mean_cost"] <= budget]
    if eligible.empty:
        return float("nan")
    return float(eligible["accuracy"].max())


def cost_at_accuracy(df: pd.DataFrame, target_accuracy: float) -> float:
    """Minimum mean cost among runs reaching >= ``target_accuracy`` (NaN if none)."""
    eligible = df[df["accuracy"] >= target_accuracy]
    if eligible.empty:
        return float("nan")
    return float(eligible["mean_cost"].min())


def plot_frontier(
    df: pd.DataFrame, out_path: str | Path, *, title: str = "Cost/quality frontier"
) -> Path:
    """Plot accuracy vs mean cost and save a PNG. Returns the output path."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    if not df.empty:
        ax.plot(df["mean_cost"], df["accuracy"], marker="o", linestyle="-")
        for _, row in df.iterrows():
            if pd.notna(row["mean_cost"]) and pd.notna(row["accuracy"]):
                ax.annotate(
                    str(row["label"]),
                    (row["mean_cost"], row["accuracy"]),
                    textcoords="offset points",
                    xytext=(5, 5),
                    fontsize=8,
                )
        cost_axis = df["cost_axis"].iloc[0]
        acc_metric = df["accuracy_metric"].iloc[0]
    else:
        cost_axis, acc_metric = "cost", "accuracy"
        ax.text(0.5, 0.5, "no runs", ha="center", va="center", transform=ax.transAxes)

    ax.set_xlabel(f"mean cost ({cost_axis})")
    ax.set_ylabel(f"accuracy ({acc_metric})")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def build_frontier(
    runs_dir: str | Path,
    out_dir: str | Path,
    *,
    accuracy_metric: str = "hit",
    cost_axis: str = "llm_calls",
) -> tuple[pd.DataFrame, Path, Path]:
    """Load run summaries, write ``frontier.csv`` + ``frontier.png``. Returns them."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = load_summaries(runs_dir)
    df = frontier_dataframe(summaries, accuracy_metric=accuracy_metric, cost_axis=cost_axis)
    csv_path = out_dir / "frontier.csv"
    png_path = out_dir / "frontier.png"
    df.to_csv(csv_path, index=False)
    plot_frontier(df, png_path)
    return df, csv_path, png_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the cost/quality frontier from run summaries."
    )
    parser.add_argument("--runs-dir", required=True, help="dir holding *.summary.json run files")
    parser.add_argument("--out-dir", required=True, help="where to write frontier.csv/.png")
    parser.add_argument("--accuracy-metric", default="hit", choices=["hit", "hits_at_1", "f1"])
    parser.add_argument("--cost-axis", default="llm_calls")
    args = parser.parse_args()

    df, csv_path, png_path = build_frontier(
        args.runs_dir,
        args.out_dir,
        accuracy_metric=args.accuracy_metric,
        cost_axis=args.cost_axis,
    )
    print(df.to_string(index=False) if not df.empty else "(no runs found)")
    print(f"\nwrote {csv_path}\nwrote {png_path}")


if __name__ == "__main__":
    main()


__all__ = [
    "load_summaries",
    "frontier_dataframe",
    "accuracy_at_budget",
    "cost_at_accuracy",
    "plot_frontier",
    "build_frontier",
]
