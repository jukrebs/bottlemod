from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt

from experiment_runner import run_experiments


def main() -> None:
    results = run_experiments()
    output_dir = "experiment_plots"
    os.makedirs(output_dir, exist_ok=True)

    labels = [result.name for result in results]
    analytic = [result.analytic_runtime_s for result in results]
    upstream = [result.upstream_runtime_s for result in results]
    sh = [result.sh_runtime_s for result in results]
    upstream_err = [abs(result.upstream_error_pct) for result in results]
    sh_err = [abs(result.sh_error_pct) for result in results]

    x = list(range(len(labels)))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar([i - width for i in x], analytic, width, label="Analytic", color="#4C78A8")
    ax.bar(x, upstream, width, label="Upstream", color="#F58518")
    ax.bar([i + width for i in x], sh, width, label="BottleMod-SH", color="#54A24B")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Runtime (s)")
    ax.set_title("Runtime comparison")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "runtime_comparison.png"), dpi=200)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar([i - width / 2 for i in x], upstream_err, width, label="Upstream", color="#F58518")
    ax.bar([i + width / 2 for i in x], sh_err, width, label="BottleMod-SH", color="#54A24B")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Absolute error (%)")
    ax.set_title("Absolute error vs analytic")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "error_comparison.png"), dpi=200)

    # Prefer fio-based ground truth if present.
    ground_truth_path = "experiment_ground_truth_fio.json"
    if not os.path.exists(ground_truth_path):
        ground_truth_path = "experiment_ground_truth.json"

    if os.path.exists(ground_truth_path):
        with open(ground_truth_path, "r", encoding="utf-8") as handle:
            ground_truth = json.load(handle)
        keys = [k for k in ("sequential", "two_pass") if k in ground_truth]
        gt_labels = [
            ground_truth[key]["experiment"]
            for key in keys
        ]
        gt_actual = [
            ground_truth[key]["actual_runtime_s"]
            for key in keys
        ]
        gt_upstream = [
            ground_truth[key]["upstream_runtime_s"]
            for key in keys
        ]
        gt_sh = [
            ground_truth[key]["sh_runtime_s"]
            for key in keys
        ]
        gt_upstream_err = [
            abs(ground_truth[key]["upstream_error_pct"])
            for key in keys
        ]
        gt_sh_err = [
            abs(ground_truth[key]["sh_error_pct"])
            for key in keys
        ]

        x = list(range(len(gt_labels)))

        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.bar([i - width for i in x], gt_actual, width, label="Actual", color="#4C78A8")
        ax.bar(x, gt_upstream, width, label="Upstream", color="#F58518")
        ax.bar([i + width for i in x], gt_sh, width, label="BottleMod-SH", color="#54A24B")
        ax.set_xticks(x)
        ax.set_xticklabels(gt_labels, rotation=20, ha="right")
        ax.set_ylabel("Runtime (s)")
        ax.set_title("Runtime comparison (ground truth)")
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "runtime_ground_truth.png"), dpi=200)

        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.bar([i - width / 2 for i in x], gt_upstream_err, width, label="Upstream", color="#F58518")
        ax.bar([i + width / 2 for i in x], gt_sh_err, width, label="BottleMod-SH", color="#54A24B")
        ax.set_xticks(x)
        ax.set_xticklabels(gt_labels, rotation=20, ha="right")
        ax.set_ylabel("Absolute error (%)")
        ax.set_title("Absolute error vs actual")
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "error_ground_truth.png"), dpi=200)


if __name__ == "__main__":
    main()
