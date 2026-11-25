"""Build Stage 2 ablation outputs and a velocity-error/training-time Pareto frontier."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


VARIANTS = (
    ("1", "Velocity–pressure, soft, no causality", "velocity_pressure_soft_no_causality", "velocity_pressure", "soft", False),
    ("2", "Velocity–pressure, hard, no causality", "velocity_pressure_hard_no_causality", "velocity_pressure", "hard", False),
    ("3", "Velocity–pressure, hard, causal", "velocity_pressure_hard_causal", "velocity_pressure", "hard", True),
    ("4", "Streamfunction–pressure, soft, no causality", "streamfunction_soft_no_causality", "streamfunction", "soft", False),
    ("5", "Streamfunction–pressure, hard, causal", "streamfunction_hard_causal", "streamfunction", "hard", True),
    ("6", "Velocity–pressure, soft, causal", "velocity_pressure_soft_causal", "velocity_pressure", "soft", True),
    ("7", "Streamfunction–pressure, soft, causal", "streamfunction_soft_causal", "streamfunction", "soft", True),
    ("8", "Streamfunction–pressure, hard, no causality", "streamfunction_hard_no_causality", "streamfunction", "hard", False),
)


def _is_dominated(candidate: dict, rows: list[dict]) -> bool:
    """Whether another variant is no worse in velocity error and time, and better in one."""
    return any(
        other["velocity_relative_l2"] <= candidate["velocity_relative_l2"]
        and other["training_seconds"] <= candidate["training_seconds"]
        and (other["velocity_relative_l2"] < candidate["velocity_relative_l2"]
             or other["training_seconds"] < candidate["training_seconds"])
        for other in rows
        if other["variant"] != candidate["variant"]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs/stage2"), help="Stage 2 run root")
    args = parser.parse_args()
    root = args.runs

    missing = []
    for number, label, dirname, _, _, _ in VARIANTS:
        run_dir = root / dirname / "Re_20"
        for filename in ("metrics.json", "config.json", "error_vs_time.csv"):
            if not (run_dir / filename).is_file():
                missing.append(f"variant {number} ({label}): {run_dir / filename}")
    if missing:
        raise SystemExit("Cannot build the eight-variant comparison; missing run outputs:\n  "
                         + "\n  ".join(missing))

    rows = []
    curves = []
    for number, label, dirname, expected_formulation, expected_periodicity, expected_causality in VARIANTS:
        run_dir = root / dirname / "Re_20"
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        config = metrics["configuration"]
        expected = (expected_formulation, expected_periodicity, expected_causality)
        actual = (config["formulation"], config["periodicity"], config["causality"])
        if actual != expected:
            raise ValueError(f"Variant {number} has configuration {actual}, expected {expected}")
        row = {
            "variant": number,
            "case": label,
            "formulation": config["formulation"],
            "periodicity": config["periodicity"],
            "causality": config["causality"],
            "reynolds": metrics["reynolds"],
            "final_time": metrics["final_time"],
            "u_relative_l2": metrics["errors"]["u"],
            "v_relative_l2": metrics["errors"]["v"],
            "p_relative_l2": metrics["errors"]["p"],
            "velocity_relative_l2": metrics["errors"]["velocity"],
            "training_seconds": metrics["training_seconds"],
            "adam_steps": metrics["training_steps_adam"],
            "lbfgs_max_iter": metrics["lbfgs_steps_requested"],
            "collocation_points": config["collocation_points"],
            "initial_points": config["initial_points"],
            "periodic_pairs": config["periodic_pairs"],
            "layers": config["hidden_layers"],
            "width": config["width"],
            "dtype": config["dtype"],
            "seed": config["seed"],
            "learning_rate": config["learning_rate"],
            "final_learning_rate": config["final_learning_rate"],
            "periodic_features": config["periodic_features"],
            "causal_epsilon": config["causal_epsilon"],
            "causal_time_bins": config["causal_time_bins"],
            "eval_resolution": config["evaluation_spatial_resolution"],
            "eval_times": config["evaluation_time_points"],
            "rar": config["rar"],
        }
        rows.append(row)
        with (run_dir / "error_vs_time.csv").open(newline="", encoding="utf-8") as stream:
            curves.append(list(csv.DictReader(stream)))

    common = ("reynolds", "final_time", "adam_steps", "lbfgs_max_iter", "collocation_points",
              "initial_points", "periodic_pairs", "layers", "width", "dtype", "seed",
              "learning_rate", "final_learning_rate", "periodic_features", "causal_epsilon",
              "causal_time_bins", "eval_resolution", "eval_times", "rar")
    for key in common:
        if len({row[key] for row in rows}) != 1:
            raise ValueError(f"Ablation runs do not share {key}")

    for row in rows:
        row["pareto_efficient"] = not _is_dominated(row, rows)
    frontier_rows = sorted((row for row in rows if row["pareto_efficient"]),
                           key=lambda row: row["training_seconds"])

    with (root / "ablation_table.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    pareto_fields = ("variant", "case", "velocity_relative_l2", "training_seconds", "p_relative_l2")
    with (root / "pareto_frontier.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=pareto_fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in pareto_fields} for row in frontier_rows)

    table = [
        "# Stage 2 Taylor–Green ablation results", "",
        f"All cases use Re={rows[0]['reynolds']:g}, T={rows[0]['final_time']:g}, a {rows[0]['layers']}×{rows[0]['width']} network, {rows[0]['adam_steps']:,} Adam steps, up to {rows[0]['lbfgs_max_iter']:,} L-BFGS iterations, {rows[0]['collocation_points']:,} space-time collocation points per Adam step, {rows[0]['dtype']} precision, and a {rows[0]['eval_resolution']}×{rows[0]['eval_resolution']} grid at {rows[0]['eval_times']} times. Relative pressure error aligns pressure means independently at each evaluation frame. RAR is disabled.",
        "The Pareto flag minimizes combined relative L2 velocity error (εᵤ) and training time; pressure error is reported but is not part of this two-objective frontier.", "",
        "| Variant | Formulation | Periodicity | Causality | Rel. L2 u | Rel. L2 v | Rel. L2 p | Rel. L2 velocity (εᵤ) | Training (s) | Pareto efficient? |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        formulation = "Velocity–pressure" if row["formulation"] == "velocity_pressure" else "Streamfunction–pressure"
        periodicity = row["periodicity"].capitalize()
        causal = "Yes" if row["causality"] else "No"
        table.append(
            f"| {row['variant']} | {formulation} | {periodicity} | {causal} | "
            f"{row['u_relative_l2']:.6g} | {row['v_relative_l2']:.6g} | "
            f"{row['p_relative_l2']:.6g} | {row['velocity_relative_l2']:.6g} | "
            f"{row['training_seconds']:.1f} | {'Yes' if row['pareto_efficient'] else 'No'} |"
        )
    table += ["", "Pareto-efficient variants are also listed in `pareto_frontier.csv` and plotted in `pareto_frontier.png`.",
              "Per-run directories contain config, metrics, history, checkpoints, and individual plots."]
    (root / "ablation_table.md").write_text("\n".join(table) + "\n", encoding="utf-8")

    colors = plt.get_cmap("tab10").colors[:len(rows)]
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.8))
    legend_handles = []
    for axis, field, title in zip(axes, ("u", "v", "p"), ("u", "v", "pressure")):
        for index, ((number, label, *_), samples, color) in enumerate(zip(VARIANTS, curves, colors)):
            time_values = [float(sample["time"]) for sample in samples]
            errors = [float(sample[f"{field}_relative_l2"]) for sample in samples]
            line, = axis.semilogy(time_values, errors, label=f"{number}: {label}", color=color, linewidth=1.8)
            if field == "u":
                legend_handles.append(line)
        axis.set_title(f"Relative L2 error: {title}")
        axis.set_xlabel("time")
        axis.set_ylabel("relative L2 error")
        axis.grid(True, which="both", alpha=0.25)
    fig.legend(handles=legend_handles, labels=[line.get_label() for line in legend_handles],
               loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=2, fontsize=8)
    fig.tight_layout(rect=(0, 0.18, 1, 1))
    fig.savefig(root / "comparison_error_vs_time.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(14, 6.5), constrained_layout=True)
    metric_keys = ("u_relative_l2", "v_relative_l2", "p_relative_l2", "velocity_relative_l2")
    metric_labels = ("u", "v", "p", "velocity")
    groups = len(rows)
    bar_width = 0.18
    centers = list(range(groups))
    for metric_index, (metric_key, metric_label) in enumerate(zip(metric_keys, metric_labels)):
        offsets = [(metric_index - 1.5) * bar_width] * groups
        axis.bar([center + offset for center, offset in zip(centers, offsets)],
                 [row[metric_key] for row in rows], width=bar_width, label=metric_label)
    axis.set_yscale("log")
    axis.set_xticks(centers, [f"{row['variant']}\n{'VP' if row['formulation'] == 'velocity_pressure' else 'Stream'}\n{row['periodicity']} / {'causal' if row['causality'] else 'plain'}" for row in rows], fontsize=8)
    axis.set_ylabel("relative L2 error (log scale)")
    axis.set_title("Ablation accuracy comparison")
    axis.grid(True, axis="y", which="both", alpha=0.25)
    axis.legend()
    fig.savefig(root / "ablation_error_comparison.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
    axis.bar([row["variant"] for row in rows], [row["training_seconds"] / 60 for row in rows], color=colors)
    axis.set_xticks(range(len(rows)), [row["variant"] for row in rows])
    axis.set_xlabel("Variant number (see ablation table)")
    axis.set_ylabel("training time (minutes)")
    axis.set_title("Ablation training cost comparison")
    axis.grid(True, axis="y", alpha=0.25)
    fig.savefig(root / "ablation_training_time.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 6.5), constrained_layout=True)
    dominated_rows = [row for row in rows if not row["pareto_efficient"]]
    axis.scatter([row["training_seconds"] / 60 for row in dominated_rows],
                 [row["velocity_relative_l2"] for row in dominated_rows],
                 color="#aeb4bd", edgecolor="#333333", s=75, label="Dominated")
    axis.scatter([row["training_seconds"] / 60 for row in frontier_rows],
                 [row["velocity_relative_l2"] for row in frontier_rows],
                 color="#d1495b", edgecolor="#333333", s=100, marker="D",
                 label="Pareto efficient")
    if len(frontier_rows) > 1:
        axis.plot([row["training_seconds"] / 60 for row in frontier_rows],
                  [row["velocity_relative_l2"] for row in frontier_rows],
                  color="#d1495b", linewidth=1.5, alpha=0.8)
    for row in rows:
        axis.annotate(row["variant"],
                      (row["training_seconds"] / 60, row["velocity_relative_l2"]),
                      xytext=(5, 5), textcoords="offset points", fontsize=9)
    axis.set_yscale("log")
    axis.set_xlabel("Training time (minutes; lower is better)")
    axis.set_ylabel("Combined relative L2 velocity error εᵤ (lower is better)")
    axis.set_title("Stage 2 Pareto frontier: accuracy versus training time")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    fig.savefig(root / "pareto_frontier.png", dpi=180)
    plt.close(fig)
    print(f"Wrote ablation table and comparison plots to {root}")


if __name__ == "__main__":
    main()
