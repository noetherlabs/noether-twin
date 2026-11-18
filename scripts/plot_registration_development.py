"""Plot every retained development unit from the published registration summary.

No acquisition, fit, random draw or acceptance decision is performed. Means and
fit medians are read from the supplied evidence; no confidence intervals are fit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

GRIDS = (64, 128, 256)
BLUE = "#225b82"
ORANGE = "#ae572b"
BLACK = "#20252b"


def main():
    root = Path(__file__).resolve().parents[1]
    destination = root / "results/registration_development_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=destination / "summary.json")
    parser.add_argument("--output", type=Path, default=destination)
    args = parser.parse_args()
    data = json.loads(args.summary.read_text())
    if data["schema"] != "noether-registration-development-public-summary-v1" or len(data["units"]) != 51:
        raise ValueError("Expected the complete 51-unit registration development summary")
    rows = data["units"]
    if len({row["id"] for row in rows}) != 51:
        raise ValueError("Repeated unit identity")
    groups = [[row for row in rows if row["state_index"] == state] for state in range(3)]
    for group in groups:
        if sum(row["kind"] == "poisson" for row in group) != 16 or len(group) != 17:
            raise ValueError("Each state requires sixteen noisy units and one mean diagnostic")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": BLACK,
            "text.color": BLACK,
            "axes.edgecolor": "#7b8086",
            "savefig.dpi": 180,
            "pdf.fonttype": 42,
        }
    )
    args.output.mkdir(parents=True, exist_ok=True)

    def style(ax):
        ax.axhline(0, color="#969ba0", linewidth=0.7, linestyle="--", zorder=0)
        ax.grid(axis="y", color="#e7e9eb", linewidth=0.5)
        ax.set_axisbelow(True)

    def point(ax, x, y, *, fallback, mean=False, boundary=False):
        if mean:
            ax.scatter(x, y, marker="D", s=34, facecolors="none", edgecolors=BLACK, linewidths=1.2, zorder=4)
        else:
            ax.scatter(
                x,
                y,
                marker="x" if fallback else ("s" if boundary else "o"),
                s=23,
                color=ORANGE if fallback else BLUE,
                linewidths=0.9,
                alpha=0.9,
                zorder=3,
            )

    handles = [
        Line2D([], [], marker="o", linestyle="none", color=BLUE, label="Locally resolved noisy unit"),
        Line2D([], [], marker="x", linestyle="none", color=ORANGE, label="Noisy unit with nominal fallback"),
        Line2D([], [], marker="D", linestyle="none", markerfacecolor="none", color=BLACK, label="Mean diagnostic"),
    ]

    def save(fig, name):
        for suffix in ("png", "pdf"):
            fig.savefig(args.output / f"{name}.{suffix}", bbox_inches="tight")
        plt.close(fig)

    contrasts = ("estimated128_minus_nominal256", "estimated128_minus_estimated256")
    labels = ("Estimated 128 − nominal 256", "Estimated 128 − estimated 256")
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 6.4), sharey="row")
    for ri, (contrast, label) in enumerate(zip(contrasts, labels, strict=True)):
        for state, group in enumerate(groups):
            ax = axes[ri, state]
            noisy = [row for row in group if row["kind"] == "poisson"]
            for index, row in enumerate(noisy):
                grids = (128,) if ri == 0 else (128, 256)
                fallback = any(row["fits"][str(grid)]["status"] != "locally_resolved" for grid in grids)
                point(
                    ax,
                    (index - 7.5) * 0.02,
                    row["scores"]["contrasts"][contrast]["observed_difference"] * 1e5,
                    fallback=fallback,
                )
            mean = next(row for row in group if row["kind"] == "expected_mean")
            point(
                ax, 0.45, mean["scores"]["contrasts"][contrast]["observed_difference"] * 1e5, fallback=False, mean=True
            )
            average = data["stratified_summary"][state]["paired_contrasts"][contrast]["mean"]
            ax.hlines(average * 1e5, -0.18, 0.18, color=BLACK, linewidth=1.5)
            style(ax)
            ax.set_xlim(-0.25, 0.63)
            ax.set_xticks((0, 0.45), ("16 noise draws", "Mean image"))
            ax.set_title(f"Offset {mean['true_registration_pitch']:+.2f} pitch")
            if state == 0:
                ax.set_ylabel(label + "\nLoss difference (×10⁻⁵)")
    fig.suptitle("Calibration did not replace higher grid resolution in the all-unit means", fontsize=12)
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.03))
    fig.text(
        0.5, 0.005, "Horizontal bars: noisy mean. Negative favours the first arm. No confidence intervals.", ha="center"
    )
    fig.tight_layout(rect=(0, 0.11, 1, 0.95))
    save(fig, "predictive-loss")

    fig, axes = plt.subplots(2, 3, figsize=(10.8, 6.7), sharey="row")
    for column, grid in enumerate(GRIDS):
        ax = axes[0, column]
        for state, group in enumerate(groups):
            noisy = [row for row in group if row["kind"] == "poisson"]
            for index, row in enumerate(noisy):
                fit = row["fits"][str(grid)]
                point(
                    ax,
                    state + (index - 7.5) * 0.019,
                    row["registration_errors_pitch"][str(grid)],
                    fallback=fit["status"] != "locally_resolved",
                    boundary=fit["boundary_solution"],
                )
            mean = next(row for row in group if row["kind"] == "expected_mean")
            point(ax, state + 0.23, mean["registration_errors_pitch"][str(grid)], fallback=False, mean=True)
        style(ax)
        ax.set_xticks((0, 1, 2), ("−0.10", "0", "+0.10"))
        ax.set_xlabel("True registration (pitch)")
        ax.set_title(f"{grid}-sample candidate")
        if column == 0:
            ax.set_ylabel("Prediction registration − true shift\n(pitch; fallback is not an estimate)")
    contrast = "estimated128_minus_nominal256"
    for state, group in enumerate(groups):
        ax = axes[1, state]
        noisy = [row for row in group if row["kind"] == "poisson"]
        for index, row in enumerate(noisy):
            c = row["scores"]["contrasts"][contrast]
            fallback = row["fits"]["128"]["status"] != "locally_resolved"
            color = ORANGE if fallback else BLUE
            offset = (index - 7.5) * 0.003
            ax.plot(
                (offset, 1 + offset),
                (c["expected_difference_512"] * 1e5, c["expected_difference_1024"] * 1e5),
                color=color,
                alpha=0.55,
                marker="x" if fallback else "o",
                markersize=2.7,
                linewidth=0.8,
            )
        mean = next(row for row in group if row["kind"] == "expected_mean")
        c = mean["scores"]["contrasts"][contrast]
        ax.plot(
            (0, 1),
            (c["expected_difference_512"] * 1e5, c["expected_difference_1024"] * 1e5),
            color=BLACK,
            marker="D",
            markerfacecolor="none",
            linewidth=1.2,
        )
        style(ax)
        ax.set_xticks((0, 1), ("512", "1024"))
        ax.set_xlim(-0.12, 1.12)
        ax.set_xlabel("Reference samples (horizontal offsets separate repeats)", fontsize=7.5)
        ax.set_title(f"Offset {mean['true_registration_pitch']:+.2f} pitch")
        if state == 0:
            ax.set_ylabel("Estimated 128 − nominal 256\nExpected loss difference (×10⁻⁵)")
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.025))
    fig.text(
        0.5,
        0.005,
        "Reference refinement measures sensitivity; it is not a continuum bound or confidence interval.",
        ha="center",
    )
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    save(fig, "registration-reference")

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.3))
    for column, grid in enumerate(GRIDS):
        for index, row in enumerate(rows):
            fit = row["fits"][str(grid)]
            x = column + (index - 25) * 0.009
            for ax, value in zip(axes, (1000 * fit["fit_with_journal_seconds"], fit["objective_calls"]), strict=True):
                point(ax, x, value, fallback=fit["status"] != "locally_resolved", mean=row["kind"] == "expected_mean")
        median = 1000 * data["costs"][str(grid)]["fit_seconds"]["median"]
        axes[0].hlines(median, column - 0.25, column + 0.25, color=BLACK, linewidth=1.5)
    for ax in axes:
        style(ax)
        ax.set_xticks((0, 1, 2), ("64", "128", "256"))
        ax.set_xlabel("Candidate pupil samples; 51 fits per grid")
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("Fit wall time including evidence I/O (ms)")
    axes[1].set_ylabel("Charged objective/gradient calls")
    fig.suptitle("Every fit remains in the computation ledger", fontsize=12)
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.025))
    fig.text(
        0.5,
        0.005,
        "Bars: median fit time. Compilation is reported separately. These are not control-loop deadlines.",
        ha="center",
    )
    fig.tight_layout(rect=(0, 0.16, 1, 0.94))
    save(fig, "fit-cost")


if __name__ == "__main__":
    main()
