from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = WORKSPACE_ROOT / "figures" / "report_safe"

matplotlib.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)

C_BLUE = "#2563EB"
C_LBLUE = "#93C5FD"
C_GREEN = "#059669"
C_LGREEN = "#6EE7B7"
C_AMBER = "#D97706"
C_RED = "#DC2626"
C_GRAY = "#6B7280"
C_LGRAY = "#E5E7EB"
C_NAVY = "#1E3A5F"
C_PINK = "#EC4899"
C_PURPLE = "#9333EA"
C_SKY = "#0EA5E9"

FAMILY_COLORS = {
    "XGBoost": C_BLUE,
    "CatBoost": C_GREEN,
    "LightGBM": C_AMBER,
    "KNN": C_GRAY,
    "LSTM": C_RED,
    "GRU": C_PURPLE,
    "Prophet": C_PINK,
    "ARIMA": C_SKY,
    "Stacking": C_NAVY,
}

FEATURE_LABELS = {
    "current_only": "Current\nOnly",
    "baseline_limited": "Baseline\nLimited",
    "short_lags": "Short\nLags",
    "short_plus_medium": "Short +\nMedium",
    "short_plus_weekly": "Short +\nWeekly",
    "full_multiscale_no_rolling": "Full MS\nNo Rolling",
    "full_multiscale": "Full\nMulti-Scale",
}


def load_report_data(workspace_root: Path | None = None) -> dict[str, object]:
    root = Path(workspace_root) if workspace_root else WORKSPACE_ROOT

    novelty_dir = root / "outputs" / "training" / "novelty"
    all_model_summary_path = root / "outputs" / "training" / "all_model_comparison_summary.csv"
    dataset_summary_path = root / "outputs" / "ismn" / "final_dataset_summary.json"

    return {
        "workspace_root": root,
        "ablation": pd.read_csv(novelty_dir / "xgboost_ablation_table.csv"),
        "benchmark": pd.read_csv(novelty_dir / "benchmark_table.csv"),
        "common_subset": pd.read_csv(novelty_dir / "common_subset_benchmark_table.csv"),
        "all_model_summary": pd.read_csv(all_model_summary_path),
        "dataset_summary": json.loads(dataset_summary_path.read_text(encoding="utf-8")),
    }


def prepare_common_subset_dual_metrics(common_subset: pd.DataFrame) -> pd.DataFrame:
    order = ["XGBoost", "Stacking", "CatBoost", "LightGBM", "LSTM", "GRU"]
    ranked = common_subset.copy()
    ranked["model"] = pd.Categorical(ranked["model"], categories=order, ordered=True)
    ranked = ranked.sort_values("model").reset_index(drop=True)
    return ranked[["model", "rmse_mean", "mae_mean", "r2_mean"]].copy()


def prepare_catboost_confirmation(all_model_summary: pd.DataFrame) -> pd.DataFrame:
    order = [
        "current_only",
        "baseline_limited",
        "short_lags",
        "short_plus_medium",
        "short_plus_weekly",
        "full_multiscale_no_rolling",
        "full_multiscale",
    ]
    catboost = all_model_summary.loc[
        (all_model_summary["model"].str.lower() == "catboost")
        & (all_model_summary["experiment"].str.endswith("__confirm"))
        & (all_model_summary["feature_group"].notna())
    ].copy()
    catboost["feature_group"] = pd.Categorical(catboost["feature_group"], categories=order, ordered=True)
    catboost = catboost.sort_values("feature_group").reset_index(drop=True)
    return catboost[
        [
            "feature_group",
            "rmse_mean",
            "mae_mean",
            "r2_mean",
        ]
    ].copy()


def prepare_refined_xgboost_ablation(all_model_summary: pd.DataFrame) -> pd.DataFrame:
    order = [
        "current_only",
        "baseline_limited",
        "short_lags",
        "short_plus_medium",
        "short_plus_weekly",
        "full_multiscale_no_rolling",
        "full_multiscale",
    ]
    refined = all_model_summary.loc[
        (all_model_summary["model"].str.lower() == "xgboost")
        & (all_model_summary["experiment"].str.endswith("__refined"))
        & (all_model_summary["feature_group"].notna())
    ].copy()
    refined["feature_group"] = pd.Categorical(refined["feature_group"], categories=order, ordered=True)
    refined = refined.sort_values("feature_group").reset_index(drop=True)
    return refined[["feature_group", "rmse_mean", "r2_mean"]].copy()


def save(fig: plt.Figure, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / name)
    plt.close(fig)


def fig_xgboost_ablation_bar(ablation: pd.DataFrame, output_dir: Path) -> None:
    labels = [FEATURE_LABELS.get(feature, str(feature)) for feature in ablation["feature_group"]]
    rmse = ablation["rmse_mean"].tolist()
    best_feature = ablation.iloc[0]["feature_group"]
    colors = [C_BLUE if feature == best_feature else C_LBLUE for feature in ablation["feature_group"]]
    edges = [C_NAVY if feature == best_feature else C_BLUE for feature in ablation["feature_group"]]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, rmse, color=colors, edgecolor=edges, linewidth=1.2, width=0.55, zorder=3)
    for bar, val in zip(bars, rmse):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.00003, f"{val:.5f}", ha="center", va="bottom", fontsize=9)

    baseline = float(ablation.loc[ablation["feature_group"] == "baseline_limited", "rmse_mean"].iloc[0])
    best = float(ablation["rmse_mean"].min())
    gain_pct = (baseline - best) / baseline * 100.0
    baseline_idx = ablation.index[ablation["feature_group"] == "baseline_limited"][0]
    best_idx = ablation["rmse_mean"].idxmin()
    ax.annotate(
        "",
        xy=(best_idx, best),
        xytext=(baseline_idx, baseline),
        arrowprops=dict(arrowstyle="<->", color=C_RED, lw=1.5),
    )
    ax.text(
        (baseline_idx + best_idx) / 2,
        (baseline + best) / 2 + 0.00008,
        f"RMSE gain\n{baseline - best:.6f}\n({gain_pct:.2f}%)",
        ha="center",
        va="bottom",
        color=C_RED,
        fontsize=8.5,
        fontweight="bold",
    )

    ax.set_ylabel("RMSE (m3/m3)")
    ax.set_xlabel("Feature Group")
    ax.set_title("XGBoost Ablation: Verified Strong-Run Results")
    ax.grid(axis="y", linestyle="--", alpha=0.45, zorder=0)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
    fig.tight_layout()
    save(fig, output_dir, "fig01_xgboost_ablation_bar.png")


def fig_xgboost_ablation_gain(ablation: pd.DataFrame, output_dir: Path) -> None:
    non_baseline = ablation.loc[ablation["feature_group"] != "baseline_limited"].copy()
    non_baseline["gain_pct"] = non_baseline["relative_rmse_gain_pct_vs_baseline"]
    labels = [FEATURE_LABELS.get(feature, str(feature)) for feature in non_baseline["feature_group"]]

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    bars = ax.barh(
        labels,
        non_baseline["gain_pct"],
        color=[C_LBLUE, C_LBLUE, C_BLUE],
        edgecolor=[C_BLUE, C_BLUE, C_NAVY],
        linewidth=1.1,
        height=0.45,
        zorder=3,
    )
    for bar, val in zip(bars, non_baseline["gain_pct"]):
        ax.text(val + 0.02, bar.get_y() + bar.get_height() / 2, f"+{val:.2f}%", va="center", fontsize=9)

    ax.set_xlabel("RMSE Improvement over Baseline (%)")
    ax.set_title("XGBoost Ablation: Relative Improvement vs Baseline")
    ax.axvline(0, color="black", lw=0.8)
    ax.grid(axis="x", linestyle="--", alpha=0.45, zorder=0)
    fig.tight_layout()
    save(fig, output_dir, "fig02_xgboost_ablation_gain.png")


def fig_benchmark_descriptive(benchmark: pd.DataFrame, output_dir: Path) -> None:
    descriptive = benchmark.copy().sort_values("rmse_mean")
    labels = [f"{row.model}\n({int(row.test_unique_keys)} keys)" for row in descriptive.itertuples()]
    colors = [FAMILY_COLORS.get(model, C_GRAY) for model in descriptive["model"]]

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    bars = ax.bar(labels, descriptive["rmse_mean"], color=colors, edgecolor="white", linewidth=0.8, zorder=3)
    for bar, val in zip(bars, descriptive["rmse_mean"]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.002, f"{val:.4f}", ha="center", va="bottom", fontsize=8.5)

    ax.set_ylabel("RMSE (m3/m3)")
    ax.set_xlabel("Model / Saved Benchmark Entry")
    ax.set_title("Descriptive Benchmark Across Saved Experiments\nCoverage differs by model, so this is not a strict ranking")
    ax.set_ylim(0, 0.32)
    ax.grid(axis="y", linestyle="--", alpha=0.45, zorder=0)
    ax.text(
        0.5,
        -0.18,
        "Labels show the number of unique test keys used by each saved benchmark entry.",
        transform=ax.transAxes,
        ha="center",
        fontsize=8,
        color=C_GRAY,
    )
    fig.tight_layout()
    save(fig, output_dir, "fig03_benchmark_descriptive.png")


def fig_common_subset_bar(common_subset: pd.DataFrame, output_dir: Path) -> None:
    subset = common_subset.copy()
    labels = subset["model"].tolist()
    colors = [FAMILY_COLORS.get(model, C_GRAY) for model in subset["model"]]

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    bars = ax.bar(labels, subset["rmse_mean"], color=colors, edgecolor="white", linewidth=0.8, width=0.56, zorder=3)
    for bar, val in zip(bars, subset["rmse_mean"]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.0002, f"{val:.5f}", ha="center", va="bottom", fontsize=9)

    best_bar = bars[0]
    ax.text(best_bar.get_x() + best_bar.get_width() / 2, subset["rmse_mean"].iloc[0] + 0.0013, "Best", ha="center", color=C_BLUE, fontweight="bold")

    ax.set_ylabel("RMSE (m3/m3)")
    ax.set_xlabel("Model")
    ax.set_title("Common-Subset Benchmark: Strict Apples-to-Apples RMSE")
    ax.set_ylim(0.019, 0.030)
    ax.grid(axis="y", linestyle="--", alpha=0.45, zorder=0)
    fig.tight_layout()
    save(fig, output_dir, "fig04_common_subset_bar.png")


def fig_common_subset_dual(common_subset: pd.DataFrame, output_dir: Path) -> None:
    dual = prepare_common_subset_dual_metrics(common_subset)
    x = np.arange(len(dual))
    width = 0.36
    colors = [FAMILY_COLORS.get(model, C_GRAY) for model in dual["model"]]

    fig, ax1 = plt.subplots(figsize=(9.8, 5.0))
    ax2 = ax1.twinx()
    ax2.spines["right"].set_visible(True)

    bars1 = ax1.bar(x - width / 2, dual["rmse_mean"], width=width, color=colors, alpha=0.88, edgecolor="white", zorder=3)
    bars2 = ax2.bar(x + width / 2, dual["r2_mean"], width=width, color=colors, alpha=0.45, edgecolor="white", hatch="///", zorder=3)

    for bar, val in zip(bars1, dual["rmse_mean"]):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.00015, f"{val:.5f}", ha="center", va="bottom", fontsize=8.5)
    for bar, val in zip(bars2, dual["r2_mean"]):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.0004, f"{val:.4f}", ha="center", va="bottom", fontsize=8, color=C_GRAY)

    ax1.set_xticks(x)
    ax1.set_xticklabels(dual["model"])
    ax1.set_ylabel("RMSE (m3/m3) ↓")
    ax2.set_ylabel("R2 Score ↑")
    ax1.set_ylim(0.018, 0.032)
    ax2.set_ylim(0.87, 0.935)
    ax1.set_title("Common-Subset Benchmark: RMSE and R2\nUsing the aligned exported comparison table")
    ax1.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)

    rmse_patch = mpatches.Patch(color=C_GRAY, alpha=0.88, label="RMSE")
    r2_patch = mpatches.Patch(color=C_GRAY, alpha=0.45, hatch="///", label="R2")
    ax1.legend(handles=[rmse_patch, r2_patch], loc="upper left")

    fig.tight_layout()
    save(fig, output_dir, "fig05_common_subset_dual.png")


def fig_catboost_confirmation(all_model_summary: pd.DataFrame, output_dir: Path) -> None:
    catboost = prepare_catboost_confirmation(all_model_summary)
    labels = [FEATURE_LABELS.get(feature, str(feature)) for feature in catboost["feature_group"]]
    colors = [C_GREEN if feature == "baseline_limited" else C_LGREEN for feature in catboost["feature_group"]]

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    bars = ax.bar(labels, catboost["rmse_mean"], color=colors, edgecolor="#065F46", linewidth=1.0, width=0.58, zorder=3)
    for bar, val in zip(bars, catboost["rmse_mean"]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.00005, f"{val:.5f}", ha="center", va="bottom", fontsize=8)

    best_idx = catboost["rmse_mean"].idxmin()
    ax.annotate(
        "Best CatBoost\nconfiguration",
        xy=(best_idx, catboost.loc[best_idx, "rmse_mean"]),
        xytext=(best_idx + 0.65, catboost["rmse_mean"].min() + 0.0005),
        color=C_GREEN,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=C_GREEN, lw=1.2),
    )

    ax.set_ylabel("RMSE (m3/m3)")
    ax.set_xlabel("Feature Group")
    ax.set_title("CatBoost Confirmation Study: Verified Confirmation Runs")
    ax.grid(axis="y", linestyle="--", alpha=0.45, zorder=0)
    fig.tight_layout()
    save(fig, output_dir, "fig06_catboost_confirmation.png")


def fig_refined_xgboost_ablation(all_model_summary: pd.DataFrame, output_dir: Path) -> None:
    refined = prepare_refined_xgboost_ablation(all_model_summary)
    if refined.empty:
        return
    labels = [FEATURE_LABELS.get(feature, str(feature)) for feature in refined["feature_group"]]
    colors = [C_BLUE if feature == "full_multiscale" else C_LBLUE for feature in refined["feature_group"]]

    fig, ax = plt.subplots(figsize=(10.2, 5.0))
    bars = ax.bar(labels, refined["rmse_mean"], color=colors, edgecolor=C_BLUE, linewidth=1.0, width=0.58, zorder=3)
    for bar, val in zip(bars, refined["rmse_mean"]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.00004, f"{val:.5f}", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("RMSE (m3/m3)")
    ax.set_xlabel("Feature Group")
    ax.set_title("Refined XGBoost Ablation: Current, Weekly, and Rolling Contributions")
    ax.grid(axis="y", linestyle="--", alpha=0.45, zorder=0)
    ax.text(
        0.02,
        0.96,
        "Uses active `__refined` XGBoost runs from the current workspace.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color=C_GRAY,
    )
    fig.tight_layout()
    save(fig, output_dir, "fig07_refined_xgboost_ablation.png")


def fig_dataset_summary(dataset_summary: dict[str, object], output_dir: Path) -> None:
    station_counts = dataset_summary["ready_stations"]
    labels = list(station_counts.keys())
    values = list(station_counts.values())

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5))

    bars = axes[0].bar(labels, values, color=[C_BLUE, C_GREEN, C_AMBER], edgecolor="white", linewidth=0.8, zorder=3)
    for bar, val in zip(bars, values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, val + 350, f"{val:,}", ha="center", va="bottom", fontsize=9)
    axes[0].set_title("Ready 48h Dataset by Station")
    axes[0].set_ylabel("Model-Ready Rows")
    axes[0].grid(axis="y", linestyle="--", alpha=0.45, zorder=0)

    full_rows = int(dataset_summary["full_shape"]["rows"])
    ready_rows = int(dataset_summary["ready_shape"]["rows"])
    filtered_out = full_rows - ready_rows
    funnel_labels = ["Full hourly grid", "Model-ready 48h", "Filtered out"]
    funnel_values = [full_rows, ready_rows, filtered_out]
    funnel_colors = [C_NAVY, C_BLUE, C_LGRAY]
    bars2 = axes[1].bar(funnel_labels, funnel_values, color=funnel_colors, edgecolor="white", linewidth=0.8, zorder=3)
    for bar, val in zip(bars2, funnel_values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, val + 4000, f"{val:,}", ha="center", va="bottom", fontsize=9)
    axes[1].set_title("48h Dataset Funnel")
    axes[1].set_ylabel("Rows")
    axes[1].grid(axis="y", linestyle="--", alpha=0.45, zorder=0)

    time_start = str(dataset_summary["ready_time_start"]).replace("T", " ")
    time_end = str(dataset_summary["ready_time_end"]).replace("T", " ")
    fig.suptitle(f"Dataset Snapshot: target_48h, {time_start} to {time_end}", fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, output_dir, "fig08_dataset_summary.png")


def fig_feature_framework_diagram(output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5.3))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def box(x: float, y: float, w: float, h: float, color: str, label: str, sub: str = "") -> None:
        rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="white", linewidth=1.5, alpha=0.94, zorder=3)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h * (0.62 if sub else 0.5), label, ha="center", va="center", fontsize=10, fontweight="bold", color="white")
        if sub:
            ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", fontsize=8, color="white", alpha=0.92)

    ax.text(6, 5.65, "Multi-Scale Temporal Feature Framework for 48-Hour Soil Moisture Forecasting", ha="center", va="top", fontsize=12, fontweight="bold", color=C_NAVY)

    box(0.2, 3.8, 2.0, 1.2, C_NAVY, "Current State", "soil + weather variables")
    box(0.2, 2.35, 2.0, 1.2, C_GRAY, "Calendar", "hour, dow, month\nsin/cos terms")
    box(0.2, 0.35, 2.0, 1.8, C_SKY, "Short-Term Lags", "1, 3, 6, 12, 24h")
    box(2.55, 0.35, 2.0, 1.8, C_AMBER, "Medium-Term Lags", "48, 72h")
    box(4.9, 0.35, 2.0, 1.8, C_RED, "Weekly Lag", "168h")
    box(7.25, 0.35, 2.0, 1.8, C_GREEN, "Rolling Features", "mean, std, precip sum\n6, 24, 48, 168h")
    box(9.65, 1.4, 2.05, 1.8, C_BLUE, "XGBoost\nForecaster", "feature vector X_t")
    box(9.65, 3.65, 2.05, 1.1, C_NAVY, "y-hat(t + 48h)", "soil moisture forecast")

    ax.add_patch(plt.Rectangle((0.0, 2.25), 9.3, 0.08, facecolor=C_LGRAY, zorder=2))
    for x in [1.2, 3.55, 5.9, 8.25]:
        ax.annotate("", xy=(x, 2.33), xytext=(x, 2.15), arrowprops=dict(arrowstyle="->", color=C_GRAY, lw=1.2), zorder=5)
    ax.annotate("", xy=(1.2, 2.33), xytext=(1.2, 3.8), arrowprops=dict(arrowstyle="->", color=C_NAVY, lw=1.2), zorder=5)
    ax.annotate("", xy=(1.2, 2.33), xytext=(1.2, 2.35), arrowprops=dict(arrowstyle="->", color=C_GRAY, lw=1.2), zorder=5)
    ax.annotate("", xy=(9.65, 2.29), xytext=(9.3, 2.29), arrowprops=dict(arrowstyle="->", color=C_BLUE, lw=1.5), zorder=5)
    ax.annotate("", xy=(10.68, 3.65), xytext=(10.68, 3.2), arrowprops=dict(arrowstyle="->", color=C_NAVY, lw=1.5), zorder=5)
    ax.text(4.7, 2.52, "Structured feature vector X_t", ha="center", va="bottom", fontsize=9, color=C_GRAY, style="italic")

    fig.tight_layout()
    save(fig, output_dir, "fig09_feature_framework.png")


def fig_pipeline_workflow(output_dir: Path) -> None:
    steps = [
        ("Read raw\n.stm files", C_NAVY),
        ("Quality filter\n(flag = G)", "#1D4ED8"),
        ("Median-aggregate\nduplicates", C_BLUE),
        ("Expand to true\nhourly grid", C_AMBER),
        ("Create future\ntargets", C_GREEN),
        ("Lag + rolling\nfeatures", "#059669"),
        ("Ablation\nstudy", C_RED),
        ("Benchmarks\nand exports", C_GRAY),
        ("Common-subset\nanalysis", C_NAVY),
    ]

    fig, ax = plt.subplots(figsize=(13, 3.4))
    ax.set_xlim(-0.5, len(steps) - 0.5)
    ax.set_ylim(-0.55, 2.4)
    ax.axis("off")
    ax.set_title("Data Pipeline and Reporting Workflow", fontsize=13, fontweight="bold", pad=12)

    for i, (label, color) in enumerate(steps):
        rect = plt.Rectangle((i - 0.42, 0.5), 0.84, 1.16, facecolor=color, edgecolor="white", linewidth=1.5, alpha=0.92, zorder=3)
        ax.add_patch(rect)
        ax.text(i, 1.08, label, ha="center", va="center", fontsize=8.5, fontweight="bold", color="white", zorder=4)
        if i < len(steps) - 1:
            ax.annotate("", xy=(i + 0.54, 1.08), xytext=(i + 0.42, 1.08), arrowprops=dict(arrowstyle="->", color=C_GRAY, lw=1.4), zorder=5)

    highlight = plt.Rectangle((3 - 0.45, 0.45), 0.9, 1.28, facecolor="none", edgecolor=C_RED, linewidth=2.4, linestyle="--", zorder=5)
    ax.add_patch(highlight)
    ax.text(3, 0.18, "Key repair:\ntrue clock-hour continuity", ha="center", fontsize=7.5, color=C_RED, fontweight="bold")

    fig.tight_layout()
    save(fig, output_dir, "fig10_pipeline_workflow.png")


def fig_stacking_comparison(common_subset: pd.DataFrame, output_dir: Path) -> None:
    subset = common_subset.loc[common_subset["model"].isin(["XGBoost", "Stacking", "CatBoost"])].copy()
    order = ["XGBoost", "Stacking", "CatBoost"]
    subset["model"] = pd.Categorical(subset["model"], categories=order, ordered=True)
    subset = subset.sort_values("model").reset_index(drop=True)

    x = np.arange(len(subset))
    width = 0.35
    colors = [FAMILY_COLORS.get(model, C_GRAY) for model in subset["model"]]

    fig, ax1 = plt.subplots(figsize=(8.2, 4.9))
    ax2 = ax1.twinx()
    ax2.spines["right"].set_visible(True)

    bars1 = ax1.bar(x - width / 2, subset["rmse_mean"], width=width, color=colors, alpha=0.88, edgecolor="white", zorder=3)
    bars2 = ax2.bar(x + width / 2, subset["r2_mean"], width=width, color=colors, alpha=0.45, edgecolor="white", hatch="///", zorder=3)

    for bar, val in zip(bars1, subset["rmse_mean"]):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.00005, f"{val:.5f}", ha="center", va="bottom", fontsize=9)
    for bar, val in zip(bars2, subset["r2_mean"]):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.0005, f"{val:.4f}", ha="center", va="bottom", fontsize=8, color=C_GRAY)

    gap = float(subset.loc[subset["model"] == "Stacking", "rmse_mean"].iloc[0] - subset.loc[subset["model"] == "XGBoost", "rmse_mean"].iloc[0])
    ax1.annotate(
        "",
        xy=(1 - width / 2, subset.loc[1, "rmse_mean"]),
        xytext=(0 - width / 2, subset.loc[0, "rmse_mean"]),
        arrowprops=dict(arrowstyle="<->", color=C_RED, lw=1.3),
    )
    ax1.text(0.5, (subset.loc[0, "rmse_mean"] + subset.loc[1, "rmse_mean"]) / 2 + 0.0001, f"Delta={gap:.5f}", ha="center", color=C_RED, fontsize=8.5, fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels(["XGBoost", "Stacking", "CatBoost"])
    ax1.set_ylabel("RMSE (m3/m3) ↓")
    ax2.set_ylabel("R2 Score ↑")
    ax1.set_ylim(0.020, 0.024)
    ax2.set_ylim(0.920, 0.932)
    ax1.set_title("Strict Common-Subset Comparison: XGBoost vs Stacking vs CatBoost")
    ax1.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    fig.tight_layout()
    save(fig, output_dir, "fig11_stacking_comparison.png")


def generate_all_figures(workspace_root: Path | None = None, output_dir: Path | None = None) -> Path:
    data = load_report_data(workspace_root)
    destination = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR

    fig_xgboost_ablation_bar(data["ablation"], destination)
    fig_xgboost_ablation_gain(data["ablation"], destination)
    fig_benchmark_descriptive(data["benchmark"], destination)
    fig_common_subset_bar(data["common_subset"], destination)
    fig_common_subset_dual(data["common_subset"], destination)
    fig_catboost_confirmation(data["all_model_summary"], destination)
    fig_refined_xgboost_ablation(data["all_model_summary"], destination)
    fig_dataset_summary(data["dataset_summary"], destination)
    fig_feature_framework_diagram(destination)
    fig_pipeline_workflow(destination)
    fig_stacking_comparison(data["common_subset"], destination)

    return destination


def main() -> Path:
    output_dir = generate_all_figures()
    print(output_dir)
    return output_dir


if __name__ == "__main__":
    main()
