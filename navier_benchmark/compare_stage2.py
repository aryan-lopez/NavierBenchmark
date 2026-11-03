"""Build the Stage 2 ablation table and cross-variant comparison plots."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


VARIANTS = (
    ("1", "Velocity–pressure, soft, no causality", "velocity_pressure_soft_no_causality"),
    ("2", "Velocity–pressure, hard, no causality", "velocity_pressure_hard_no_causality"),
    ("3", "Velocity–pressure, hard, causal", "velocity_pressure_hard_causal"),
    ("4", "Streamfunction–pressure, soft, no causality", "streamfunction_soft_no_causality"),
    ("5", "Streamfunction–pressure, hard, causal", "streamfunction_hard_causal"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs/stage2"), help="Stage 2 run root")
    args = parser.parse_args()
    root = args.runs

    rows = []
    curves = []
    for number, label, dirname in VARIANTS:
        run_dir = root / dirname / "Re_20"
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        config = metrics["configuration"]
        row = {
            "variant": number,
            "case": label,
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
            "layers": config["hidden_layers"],
            "width": config["width"],
            "dtype": config["dtype"],
            "eval_resolution": config["evaluation_spatial_resolution"],
            "eval_times": config["evaluation_time_points"],
        }
        rows.append(row)
        with (run_dir / "error_vs_time.csv").open(newline="", encoding="utf-8") as stream:
            curves.append(list(csv.DictReader(stream)))

    common = ("reynolds", "final_time", "adam_steps", "lbfgs_max_iter", "collocation_points",
              "layers", "width", "dtype", "eval_resolution", "eval_times")
    for key in common:
        if len({row[key] for row in rows}) != 1:
            raise ValueError(f"Ablation runs do not share {key}")

    with (root / "ablation_table.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    table = [
        "# Stage 2 Taylor–Green ablation results", "",
        f"All cases use Re={rows[0]['reynolds']:g}, T={rows[0]['final_time']:g}, a {rows[0]['layers']}×{rows[0]['width']} network, {rows[0]['adam_steps']:,} Adam steps, up to {rows[0]['lbfgs_max_iter']:,} L-BFGS iterations, {rows[0]['collocation_points']:,} space-time collocation points per Adam step, {rows[0]['dtype']} precision, and a {rows[0]['eval_resolution']}×{rows[0]['eval_resolution']} grid at {rows[0]['eval_times']} times. Relative pressure error aligns pressure means independently at each evaluation frame. RAR is disabled.", "",
        "| Variant | Formulation | Periodicity | Causality | Rel. L2 u | Rel. L2 v | Rel. L2 p | Rel. L2 velocity | Training (s) |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|",
    ]
    labels = (
        ("Velocity–pressure", "Soft", "No"),
        ("Velocity–pressure", "Hard", "No"),
        ("Velocity–pressure", "Hard", "Yes"),
        ("Streamfunction–pressure", "Soft", "No"),
        ("Streamfunction–pressure", "Hard", "Yes"),
    )
    for row, (formulation, periodicity, causal) in zip(rows, labels):
        table.append(
            f"| {row['variant']} | {formulation} | {periodicity} | {causal} | "
            f"{row['u_relative_l2']:.6g} | {row['v_relative_l2']:.6g} | "
            f"{row['p_relative_l2']:.6g} | {row['velocity_relative_l2']:.6g} | "
            f"{row['training_seconds']:.1f} |"
        )
    table += ["", "Per-run directories contain config, metrics, history, checkpoints, and individual plots."]
    (root / "ablation_table.md").write_text("\n".join(table) + "\n", encoding="utf-8")

    colors = ("#1677b8", "#dc5945", "#2c9955", "#9367bd", "#d18f22")
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), constrained_layout=True)
    for axis, field, title in zip(axes, ("u", "v", "p"), ("u", "v", "pressure")):
        for index, ((_, label, _), samples, color) in enumerate(zip(VARIANTS, curves, colors)):
            time_values = [float(sample["time"]) for sample in samples]
            errors = [float(sample[f"{field}_relative_l2"]) for sample in samples]
            axis.semilogy(time_values, errors, label=f"{index + 1}: {label}", color=color, linewidth=1.8)
        axis.set_title(f"Relative L2 error: {title}")
        axis.set_xlabel("time")
        axis.set_ylabel("relative L2 error")
        axis.grid(True, which="both", alpha=0.25)
    axes[0].legend(fontsize=8, loc="best")
    fig.savefig(root / "comparison_error_vs_time.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(11, 6), constrained_layout=True)
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
    axis.set_xticks(centers, [f"{row['variant']}\n{row['case'].replace(', ', '\n')}" for row in rows], fontsize=8)
    axis.set_ylabel("relative L2 error (log scale)")
    axis.set_title("Ablation accuracy comparison")
    axis.grid(True, axis="y", which="both", alpha=0.25)
    axis.legend()
    fig.savefig(root / "ablation_error_comparison.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    axis.bar([row["variant"] for row in rows], [row["training_seconds"] / 60 for row in rows], color=colors)
    axis.set_xticks(range(len(rows)), [f"{row['variant']}: {row['case']}" for row in rows], rotation=20, ha="right")
    axis.set_ylabel("training time (minutes)")
    axis.set_title("Ablation training cost comparison")
    axis.grid(True, axis="y", alpha=0.25)
    fig.savefig(root / "ablation_training_time.png", dpi=180)
    plt.close(fig)
    print(f"Wrote ablation table and comparison plots to {root}")


if __name__ == "__main__":
    main()
