#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   analysis.py
@Time    :   2025/09/26 00:23:38
@Author  :   Dawn Li
@Version :   1.0
@Contact :   dawnli619215645@gmail.com
@Desc    :   Aggregates evaluation metrics for model size comparison experiments
             and exports both CSV and LaTeX summaries for reporting.
"""

import argparse
import json
import logging
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots
import torch
from matplotlib.patches import Patch
from safetensors import safe_open

from flow.utils import setup_logging
from flow.utils.plot_utils import COLOR_PALETTE

if scienceplots:
    plt.style.use(["science"])

DEFAULT_BASE_DIR = Path(
    "/mnt/local_data1/liweiguo/experiments/model_size_comparison/work_dir"
)
DEFAULT_OUTPUT_DIR = DEFAULT_BASE_DIR / "analysis"
DEFAULT_TYPES: List[str] = ["Small", "Medium", "Large"]
DEFAULT_ALGORITHMS: List[str] = [
    "Concat-BPE-1K",
    "Seg-BPE-1K",
    "Concat-BBPE-1K",
    "Seg-BBPE-1K",
    "DecimalWordLevel",
]

RADAR_ALGORITHM_GROUPS: Dict[str, tuple[str, ...]] = {
    "Concat*": ("Concat-BPE-1K", "Concat-BBPE-1K"),
    "Seg*": ("Seg-BPE-1K", "Seg-BBPE-1K"),
    "DecimalWordLevel": ("DecimalWordLevel",),
}
RADAR_ALGORITHM_ORDER: List[str] = ["Concat*", "Seg*", "DecimalWordLevel"]
_BPE_ZERO_EPS = 1e-12

MEAN_METRIC_MAPPING: Dict[str, str] = {
    "is_connected_all_loads": "Connected",
    "is_graceful": "Graceful",
    "is_perfect_match": "Perfect",
    "red_similarity_score": "RED Score",
    "bleu_4": "BLEU-4",
    "rougeL_f": "ROUGE-L F1",
}

RATIO_METRIC_MAPPING: Dict[str, str] = {
    "wirelength": "Wirelength",
    "via": "Via",
}

RATIO_COLUMN_PAIRS: Dict[str, tuple[str, str]] = {
    "wirelength": ("wirelength_pred", "wirelength_true"),
    "via": ("num_vias_pred", "num_vias_true"),
}

SUMMARY_COLUMNS: List[str] = [
    "Type",
    "Connected",
    "Graceful",
    "Perfect",
    "RED Score",
    "BLEU-4",
    "ROUGE-L F1",
    "Wirelength",
    "Via",
]

CHECKPOINT_SUBDIR = Path("stage_training") / "model"
CHECKPOINT_PATTERN = re.compile(r"checkpoint-(\d+)$")

MODEL_PARAMETER_COLUMNS: List[tuple[str, str]] = [
    ("Type", r"\textbf{Type}"),
    ("total_parameters", r"\textbf{Total Parameters}"),
    ("encoder_parameters", r"\textbf{Encoder Parameters}"),
    ("decoder_parameters", r"\textbf{Decoder Parameters}"),
    ("d_model", r"\textbf{\(d_{model}\)}"),
    ("n_layer", r"\textbf{\(n_{layer}\)}"),
    ("n_head", r"\textbf{\(n_{head}\)}"),
    ("d_cross", r"\textbf{\(d_{cross}\)}"),
    ("d_ff", r"\textbf{\(d_{ff}\)}"),
    ("d_que", r"\textbf{\(d_{que}\)}"),
]

CONFIG_VALUE_PATHS: Dict[str, List[str]] = {
    "d_model": ["hidden_size", "decoder.hidden_size", "encoder.hidden_size"],
    "n_layer": [
        "num_hidden_layers",
        "decoder.num_hidden_layers",
        "encoder.num_hidden_layers",
    ],
    "n_head": [
        "num_attention_heads",
        "decoder.num_attention_heads",
        "encoder.num_attention_heads",
    ],
    "d_cross": [
        "cross_attention_hidden_size",
        "decoder.cross_attention_hidden_size",
        "encoder.cross_attention_hidden_size",
    ],
    "d_ff": [
        "intermediate_size",
        "decoder.intermediate_size",
        "encoder.intermediate_size",
    ],
    "d_que": [
        "query_pre_attn_scalar",
        "decoder.query_pre_attn_scalar",
        "encoder.query_pre_attn_scalar",
    ],
}

def _base_type_name(value: object) -> str:
    text = str(value)
    return text.split("-", 1)[0]


def _algorithm_variant_name(value: object) -> str:
    text = str(value)
    return text.split("-", 1)[1] if "-" in text else text


def _prepare_sorted_display_df(df: pd.DataFrame) -> pd.DataFrame:
    type_order = ["Small", "Medium", "Large"]
    order_map = {name: index for index, name in enumerate(type_order)}
    algorithm_order = {name: index for index, name in enumerate(DEFAULT_ALGORITHMS)}

    sorted_df = df.copy()
    sorted_df["_type_order"] = sorted_df["Type"].map(
        lambda value: order_map.get(_base_type_name(value), len(type_order))
    )
    sorted_df["_algo_order"] = sorted_df["Type"].map(
        lambda value: algorithm_order.get(
            _algorithm_variant_name(value), len(algorithm_order)
        )
    )
    sorted_df["_base_type"] = sorted_df["Type"].map(_base_type_name)
    sorted_df = sorted_df.sort_values(["_type_order", "_algo_order", "Type"])
    sorted_df = sorted_df.drop(columns=["_type_order", "_algo_order"], errors="ignore")
    sorted_df = sorted_df.reset_index(drop=True)
    return sorted_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate model size comparison metrics into CSV and LaTeX outputs",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=DEFAULT_BASE_DIR,
        help="Root directory containing experiment sub-folders",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to store aggregated analysis artifacts",
    )
    parser.add_argument(
        "--types",
        nargs="+",
        default=DEFAULT_TYPES,
        help="Model size variants to include in the analysis",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=DEFAULT_ALGORITHMS,
        help="Tokenization algorithms to evaluate",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging verbosity level",
    )
    return parser.parse_args()


def read_metrics_csv(csv_path: Path) -> pd.DataFrame | None:
    if not csv_path.exists():
        logging.warning("Metrics CSV not found: %s", csv_path)
        return None
    try:
        return pd.read_csv(csv_path)
    except Exception as exc:  # pragma: no cover - defensive guard
        logging.error("Failed to read metrics CSV at %s: %s", csv_path, exc)
        return None


def calculate_mean_metrics(df: pd.DataFrame) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for column, label in MEAN_METRIC_MAPPING.items():
        if column not in df.columns:
            logging.warning("Column '%s' missing in metrics data", column)
            metrics[label] = float("nan")
            continue
        metrics[label] = df[column].mean()
    return metrics


def calculate_ratio_metrics(df: pd.DataFrame) -> Dict[str, float]:
    ratios: Dict[str, float] = {}
    for key, (pred_col, true_col) in RATIO_COLUMN_PAIRS.items():
        label = RATIO_METRIC_MAPPING[key]
        if pred_col not in df.columns or true_col not in df.columns:
            missing = [col for col in (pred_col, true_col) if col not in df.columns]
            logging.warning(
                "Missing column(s) %s required for ratio '%s'",
                ", ".join(missing),
                label,
            )
            ratios[label] = float("nan")
            continue
        numerator = df[pred_col].sum()
        denominator = df[true_col].sum()
        if denominator == 0:
            logging.warning(
                "Denominator sums to zero for ratio '%s' (columns: %s/%s)",
                label,
                pred_col,
                true_col,
            )
            ratios[label] = float("nan")
            continue
        ratios[label] = numerator / denominator
    return ratios


def aggregate_metrics(
    base_dir: Path, types: Iterable[str], algorithms: Iterable[str]
) -> pd.DataFrame:
    records: List[Dict[str, float]] = []
    for size in types:
        for algorithm in algorithms:
            experiment_id = f"{size}-{algorithm}"
            data_dir = base_dir / experiment_id
            if not data_dir.exists():
                logging.warning(
                    "Data directory not found for '%s': %s", experiment_id, data_dir
                )
                continue
            csv_path = (
                data_dir / "stage_evaluation" / "metrics" / "evaluation_metrics.csv"
            )
            metrics_df = read_metrics_csv(csv_path)
            if metrics_df is None:
                continue
            mean_metrics = calculate_mean_metrics(metrics_df)
            ratio_metrics = calculate_ratio_metrics(metrics_df)
            record = {"Type": experiment_id}
            record.update(mean_metrics)
            record.update(ratio_metrics)
            records.append(record)
            logging.info("Aggregated metrics for '%s'", experiment_id)

    if not records:
        logging.warning(
            "No metrics were aggregated. Check input directories and filters."
        )
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    summary_df = pd.DataFrame(records)
    summary_df = summary_df.reindex(columns=SUMMARY_COLUMNS)
    return summary_df


def format_value(value: float, precision: int = 3) -> str:
    if pd.isna(value):
        return "--"
    return f"{value:.{precision}f}"


def dataframe_to_latex_table(df: pd.DataFrame) -> str:
    # Build table columns dynamically from SUMMARY_COLUMNS while preserving order
    available_cols = [c for c in SUMMARY_COLUMNS if c in df.columns]
    if "Type" not in available_cols:
        raise ValueError("DataFrame must contain a 'Type' column for table generation")

    # Build LaTeX column specification. First column is centered (c), rest use C (tabularx custom column)
    ncols = len(available_cols)
    col_spec = "{|" + "|".join(["c"] + ["C"] * (ncols - 1)) + "|}"

    ratio_labels = set(RATIO_METRIC_MAPPING.values())

    def header_label(column_name: str) -> str:
        label = r"\textbf{" + column_name + r"}"
        if column_name in ratio_labels:
            return r"\textcolor{gray}{" + label + r"}"
        return label

    header = [
        r"\begin{table*}[ht]",
        r"    \centering",
        r"    \caption{Ablation model metrics.}",
        r"    \label{tab:model_size_metrics}",
        f"    \\begin{{tabularx}}{{0.98\\linewidth}}{col_spec}",
        r"        \hline",
        "        " + " & ".join([header_label(col) for col in available_cols]) + r" \\",
        r"        \hline",
    ]

    sorted_df = _prepare_sorted_display_df(df[[col for col in available_cols]])

    # Determine which metric columns should be considered for bolding (exclude Type and ratios)
    metric_cols_to_bold = [
        c for c in available_cols if c != "Type" and c not in ratio_labels
    ]

    if metric_cols_to_bold:
        max_per_base = sorted_df.groupby("_base_type")[metric_cols_to_bold].max()
    else:
        max_per_base = pd.DataFrame()

    rows: List[str] = []
    previous_base: str | None = None
    # Tolerance for floating comparison
    EPS = 1e-9
    for _, row in sorted_df.iterrows():
        current_base = row.get("_base_type", _base_type_name(row["Type"]))
        if previous_base is not None and current_base != previous_base:
            rows.append(r"        \hline")

        formatted_values: List[str] = []
        for col in available_cols:
            if col == "Type":
                formatted_values.append(str(row["Type"]))
                continue

            val = row.get(col, float("nan"))
            # Format the value (handles NaN)
            formatted = format_value(val)

            # Bold if this column is eligible and equals the per-base max
            if col in metric_cols_to_bold and current_base in max_per_base.index:
                max_val = max_per_base.at[current_base, col]
                if (
                    not pd.isna(val)
                    and not pd.isna(max_val)
                    and abs(float(val) - float(max_val)) <= EPS
                ):
                    formatted = r"\textbf{" + formatted + r"}"

            if col in ratio_labels:
                formatted = r"\textcolor{gray}{" + formatted + r"}"

            formatted_values.append(formatted)

        row_line = "        " + " & ".join(formatted_values) + " \\\\"
        rows.append(row_line)
        previous_base = current_base

    footer = [
        r"        \hline",
        r"    \end{tabularx}",
        r"\end{table*}",
    ]

    table_lines = header + rows + footer
    return "\n".join(table_lines) + "\n"


def save_metrics_bar_chart(
    df: pd.DataFrame, output_path: Path, metric_names: List[str] | None = None
) -> None:
    if metric_names is not None:
        metrics = [col for col in metric_names if col in df.columns and col != "Type"]
    else:
        metrics = [
            col for col in SUMMARY_COLUMNS if col in df.columns and col != "Type"
        ]
    # metrics already follow SUMMARY_COLUMNS order
    if not metrics:
        logging.warning("No numeric metrics available for bar chart generation.")
        return

    sorted_df = _prepare_sorted_display_df(df[["Type"] + metrics])
    if sorted_df.empty:
        logging.warning("Summary DataFrame is empty; skipping bar chart export.")
        return

    numeric_df = sorted_df[metrics].apply(pd.to_numeric, errors="coerce")
    type_list = sorted_df["Type"].tolist()
    if not type_list:
        logging.warning("No model types available for plotting.")
        return

    value_lookup: Dict[tuple[str, str], float] = {}
    for idx, type_name in enumerate(type_list):
        for metric in metrics:
            val = numeric_df.iloc[idx][metric]
            value_lookup[(type_name, metric)] = float(val) if pd.notna(val) else 0.0

    available_types = set(type_list)
    algorithm_colors = {
        algo: COLOR_PALETTE[idx % len(COLOR_PALETTE)]
        for idx, algo in enumerate(DEFAULT_ALGORITHMS)
    }
    alpha_levels = {"Small": 0.5, "Medium": 0.75, "Large": 1.0}

    bar_width = 0.12
    type_gap = bar_width * 0.8
    metric_gap = bar_width * 1.8

    fig_width = max(8.0, len(metrics) * 1.6)
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    current_x = 0.0
    xticks: List[float] = []
    labels: List[str] = []
    used_algorithms: set[str] = set()
    used_sizes: set[str] = set()
    bar_drawn = False

    for metric in metrics:
        metric_start = current_x
        first_type_drawn = False
        for base_type in DEFAULT_TYPES:
            entries: List[tuple[str, str]] = []
            for algo in DEFAULT_ALGORITHMS:
                type_id = f"{base_type}-{algo}"
                if type_id in available_types:
                    entries.append((type_id, algo))
            if not entries:
                continue
            if first_type_drawn:
                current_x += type_gap
            first_type_drawn = True
            used_sizes.add(base_type)
            for type_id, algo in entries:
                value = value_lookup.get((type_id, metric), 0.0)
                color = algorithm_colors.get(algo, COLOR_PALETTE[0])
                alpha = alpha_levels.get(base_type, 1.0)
                ax.bar(
                    current_x,
                    value,
                    width=bar_width,
                    color=color,
                    alpha=alpha,
                    edgecolor="black",
                    linewidth=0.4,
                )
                used_algorithms.add(algo)
                bar_drawn = True
                current_x += bar_width
        metric_end = current_x
        if metric_end == metric_start:
            xticks.append(metric_start)
        else:
            xticks.append((metric_start + metric_end) / 2)
        labels.append(metric)
        current_x += metric_gap

    if not bar_drawn:
        logging.warning("No bars were drawn for the bar chart; skipping export.")
        plt.close(fig)
        return

    ax.set_xticks(xticks)
    ax.set_xticklabels(labels, rotation=25, ha="center", fontsize=16)
    ax.set_ylabel("Score", fontsize=16)
    # ax.set_xlabel("Metrics", fontsize=12)
    ax.tick_params(axis="y", labelsize=16)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    ax.grid(axis="y", linestyle="--", alpha=0.35, linewidth=0.6)
    ax.set_xlim(-bar_width, max(current_x, 1.0))

    algorithm_handles = [
        Patch(
            facecolor=algorithm_colors[algo],
            edgecolor="black",
            linewidth=0.5,
            label=algo,
        )
        for algo in DEFAULT_ALGORITHMS
        if algo in used_algorithms
    ]
    size_handles = [
        Patch(
            facecolor="gray",
            edgecolor="black",
            linewidth=0.5,
            alpha=alpha_levels[base_type],
            label=base_type,
        )
        for base_type in DEFAULT_TYPES
        if base_type in used_sizes
    ]

    legend1 = ax.legend(
        handles=algorithm_handles,
        loc="upper left",
        bbox_to_anchor=(0.5, 1.0),
        fontsize=12,
        frameon=True,
        fancybox=True,
        shadow=True,
        title="Algorithms",
        title_fontsize=12,
    )
    ax.add_artist(legend1)
    ax.legend(
        handles=size_handles,
        loc="upper left",
        bbox_to_anchor=(0.3, 1.0),
        fontsize=12,
        frameon=True,
        fancybox=True,
        shadow=True,
        title="Sizes",
        title_fontsize=12,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logging.info("Ablation bar chart saved to %s", output_path)


def _compute_group_averages(
    df: pd.DataFrame,
    metrics: List[str],
    group_func,
    category_order: List[str],
) -> pd.DataFrame:
    working = df.copy()
    working["__group"] = working["Type"].map(group_func)
    working = working.dropna(subset=["__group"])
    if working.empty:
        return pd.DataFrame(columns=metrics)
    numeric = working[metrics].apply(pd.to_numeric, errors="coerce")
    grouped = numeric.assign(__group=working["__group"]).groupby("__group").mean()
    if category_order:
        grouped = grouped.reindex(category_order)
    return grouped


def _safe_metric_value(row: pd.Series | None, metric: str) -> float:
    if row is None:
        return float("nan")
    value = row.get(metric, float("nan"))
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _combine_bpe_metric_values(bpe_val: float, bbpe_val: float) -> float:
    bpe_na = pd.isna(bpe_val)
    bbpe_na = pd.isna(bbpe_val)
    if bpe_na and bbpe_na:
        return float("nan")
    if bpe_na:
        return bbpe_val
    if bbpe_na:
        return bpe_val
    if abs(bpe_val) <= _BPE_ZERO_EPS:
        return bbpe_val
    return (bpe_val + bbpe_val) / 2.0


def _prepare_algorithm_radar_df(
    grouped_df: pd.DataFrame, metrics: List[str]
) -> pd.DataFrame:
    if grouped_df.empty:
        return pd.DataFrame(columns=metrics)

    combined: Dict[str, List[float]] = {}
    for label, variants in RADAR_ALGORITHM_GROUPS.items():
        if len(variants) == 2:
            bpe_key, bbpe_key = variants
            row_bpe = grouped_df.loc[bpe_key] if bpe_key in grouped_df.index else None
            row_bbpe = (
                grouped_df.loc[bbpe_key] if bbpe_key in grouped_df.index else None
            )
            if row_bpe is None and row_bbpe is None:
                continue
            values: List[float] = []
            for metric in metrics:
                bpe_val = _safe_metric_value(row_bpe, metric)
                bbpe_val = _safe_metric_value(row_bbpe, metric)
                values.append(_combine_bpe_metric_values(bpe_val, bbpe_val))
        else:
            key = variants[0]
            if key not in grouped_df.index:
                continue
            row = grouped_df.loc[key]
            values = [_safe_metric_value(row, metric) for metric in metrics]
        combined[label] = values

    if not combined:
        return pd.DataFrame(columns=metrics)

    aggregated = pd.DataFrame(combined, index=metrics).T
    aggregated = aggregated.reindex(RADAR_ALGORITHM_ORDER)
    return aggregated


def _plot_group_averages(
    grouped_df: pd.DataFrame,
    metrics: List[str],
    category_order: List[str],
    color_map: Dict[str, tuple[float, float, float]],
    output_path: Path,
    legend_title: str,
) -> None:
    categories = [cat for cat in category_order if cat in grouped_df.index]
    if not categories:
        logging.warning("No categories available for %s chart.", legend_title)
        return

    x = np.arange(len(metrics))
    bar_width = min(0.3, 1.5 / max(len(categories), 1))
    fig, ax = plt.subplots(figsize=(8, 6))
    for idx, category in enumerate(categories):
        offset = (idx - (len(categories) - 1) / 2) * bar_width
        heights = grouped_df.loc[category, metrics].to_numpy(dtype=float)
        heights = np.nan_to_num(heights, nan=0.0)
        color = color_map.get(category, COLOR_PALETTE[idx % len(COLOR_PALETTE)])
        ax.bar(
            x + offset,
            heights,
            width=bar_width,
            color=color,
            linewidth=0.4,
            label=category,
        )
        # Add the corresponding value directly above the bar.
        for metric_idx, height in enumerate(heights):
            if math.isclose(height, 0.0):
                continue
            ax.text(
                x[metric_idx] + offset,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xlim(-0.5, len(metrics) - 0.5)
    ax.set_xticklabels(metrics, rotation=15, ha="center", fontsize=20)
    ax.set_ylabel("Score", fontsize=20)
    ax.tick_params(axis="y", labelsize=20)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    ax.grid(axis="y", linestyle="--", alpha=0.35, linewidth=0.6)
    ncol = 3 if len(categories) > 3 else len(categories)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=ncol,
        fontsize=16,
        frameon=True,
        fancybox=True,
        shadow=True,
        title=legend_title,
        title_fontsize=16,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logging.info("Grouped bar chart saved to %s", output_path)


def _plot_group_radar(
    grouped_df: pd.DataFrame,
    metrics: List[str],
    category_order: List[str],
    color_map: Dict[str, tuple[float, float, float]],
    output_path: Path,
    legend_title: str,
) -> None:
    categories = [cat for cat in category_order if cat in grouped_df.index]
    if not categories:
        logging.warning("No categories available for %s radar chart.", legend_title)
        return
    metrics_available = [metric for metric in metrics if metric in grouped_df.columns]
    if not metrics_available:
        logging.warning("No metrics provided for %s radar chart.", legend_title)
        return

    numeric = grouped_df.loc[categories, metrics_available].apply(
        pd.to_numeric, errors="coerce"
    )

    metric_max_pairs: List[tuple[float, str]] = []
    for metric in metrics_available:
        column = numeric[metric].to_numpy(dtype=float)
        valid = column[~np.isnan(column)]
        if valid.size == 0:
            continue
        metric_max_pairs.append((float(np.max(valid)), metric))
    if not metric_max_pairs:
        logging.warning("No valid metric values for %s radar chart.", legend_title)
        return

    metric_max_pairs.sort(key=lambda item: (-item[0], item[1]))
    ordered_metrics = [metric for _, metric in metric_max_pairs]

    values_matrix = numeric[ordered_metrics].to_numpy(dtype=float)
    values_matrix = np.nan_to_num(values_matrix, nan=0.0)
    max_value = float(np.max(values_matrix)) if values_matrix.size else 0.0
    if not np.isfinite(max_value) or max_value <= 0:
        max_value = 1.0

    metric_angles = np.linspace(
        0, 2 * np.pi, len(ordered_metrics), endpoint=False
    ).tolist()
    metric_angles += metric_angles[:1]

    fig, ax = plt.subplots(figsize=(8, 7), subplot_kw={"polar": True})
    for idx, category in enumerate(categories):
        values = values_matrix[idx].tolist()
        if not values:
            continue
        values.append(values[0])
        color = color_map.get(category, COLOR_PALETTE[idx % len(COLOR_PALETTE)])
        ax.plot(metric_angles, values, color=color, linewidth=2.0, label=category)
        ax.fill(metric_angles, values, color=color, alpha=0.18)

    ax.set_xticks(metric_angles[:-1])
    ax.set_xticklabels(ordered_metrics, fontsize=16)
    ax.set_ylim(0.0, max_value * 1.05)

    tick_values = np.linspace(0, max_value, num=5)[1:]
    if max_value <= 1:
        tick_labels = [f"{tick:.2f}" for tick in tick_values]
    elif max_value <= 10:
        tick_labels = [f"{tick:.1f}" for tick in tick_values]
    else:
        tick_labels = [f"{tick:.0f}" for tick in tick_values]
    ax.set_yticks(tick_values)
    ax.set_yticklabels(tick_labels)

    ax.set_rlabel_position(0)
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.4)

    legend = ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.15, 1.1),
        fontsize=14,
        frameon=True,
        fancybox=True,
        shadow=True,
        title=legend_title,
        title_fontsize=14,
    )
    legend.get_frame().set_alpha(0.9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logging.info("Grouped radar chart saved to %s", output_path)


def _is_encoder_key(param_name: str) -> bool:
    lower_name = param_name.lower()
    return lower_name.startswith("encoder.") or ".encoder." in lower_name


def _is_decoder_key(param_name: str) -> bool:
    lower_name = param_name.lower()
    return lower_name.startswith("decoder.") or ".decoder." in lower_name


def _count_parameters_from_state_dict(
    state_dict: Mapping[str, torch.Tensor],
) -> tuple[int, int, int]:
    total = 0
    encoder = 0
    decoder = 0
    for name, tensor in state_dict.items():
        if not hasattr(tensor, "numel"):
            continue
        numel = int(tensor.numel())
        total += numel
        if _is_encoder_key(name):
            encoder += numel
        elif _is_decoder_key(name):
            decoder += numel
    return total, encoder, decoder


def _count_parameters_from_safetensors(files: Iterable[Path]) -> tuple[int, int, int]:
    total = 0
    encoder = 0
    decoder = 0
    for file_path in sorted(files):
        if not file_path.exists():
            continue
        try:
            with safe_open(str(file_path), framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    tensor_slice = handle.get_slice(key)
                    shape = tensor_slice.get_shape()
                    numel = int(math.prod(shape) if shape else 1)
                    total += numel
                    if _is_encoder_key(key):
                        encoder += numel
                    elif _is_decoder_key(key):
                        decoder += numel
        except Exception as exc:  # pragma: no cover - best effort logging
            logging.error("Failed to parse safetensors file %s: %s", file_path, exc)
    return total, encoder, decoder


def load_parameter_counts(checkpoint_dir: Path) -> tuple[int, int, int] | None:
    safetensors_files = [
        path
        for path in checkpoint_dir.glob("*.safetensors")
        if path.name.startswith("model")
    ]
    if safetensors_files:
        total, encoder, decoder = _count_parameters_from_safetensors(safetensors_files)
        if total > 0:
            return total, encoder, decoder

    bin_files = sorted(checkpoint_dir.glob("pytorch_model*.bin"))
    if bin_files:
        total = 0
        encoder = 0
        decoder = 0
        for file_path in bin_files:
            try:
                state_dict = torch.load(file_path, map_location="cpu")
            except Exception as exc:  # pragma: no cover - defensive logging
                logging.error(
                    "Failed to load checkpoint weights %s: %s", file_path, exc
                )
                continue
            part_total, part_encoder, part_decoder = _count_parameters_from_state_dict(
                state_dict
            )
            total += part_total
            encoder += part_encoder
            decoder += part_decoder
            del state_dict
        if total > 0:
            return total, encoder, decoder

    logging.warning("No supported model weight files found in %s", checkpoint_dir)
    return None


def find_latest_checkpoint(model_dir: Path) -> Path | None:
    if not model_dir.exists():
        logging.warning("Model directory does not exist: %s", model_dir)
        return None
    candidates: List[tuple[int, Path]] = []
    for child in model_dir.iterdir():
        if not child.is_dir():
            continue
        match = CHECKPOINT_PATTERN.match(child.name)
        if match is None:
            continue
        step = int(match.group(1))
        candidates.append((step, child))
    if not candidates:
        logging.warning("No checkpoints found under %s", model_dir)
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _get_nested_config_value(config: Mapping[str, object], dotted_path: str):
    current: object = config
    for key in dotted_path.split("."):
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def load_config_metadata(config_path: Path) -> Dict[str, object]:
    if not config_path.exists():
        logging.warning("Config file not found: %s", config_path)
        return {key: None for key in CONFIG_VALUE_PATHS}
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
    except Exception as exc:  # pragma: no cover - defensive guard
        logging.error("Failed to read config at %s: %s", config_path, exc)
        return {key: None for key in CONFIG_VALUE_PATHS}

    values: Dict[str, object] = {}
    for field, paths in CONFIG_VALUE_PATHS.items():
        value = None
        for dotted_path in paths:
            candidate = _get_nested_config_value(config, dotted_path)
            if candidate is None:
                continue
            if isinstance(candidate, list):
                candidate = candidate[-1] if candidate else None
            value = candidate
        values[field] = value
    return values


def gather_model_parameter_records(
    base_dir: Path, types: Iterable[str], algorithms: Iterable[str]
) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for size in types:
        for algorithm in algorithms:
            experiment_id = f"{size}-{algorithm}"
            experiment_dir = base_dir / experiment_id
            if not experiment_dir.exists():
                logging.warning(
                    "Experiment directory missing for '%s': %s",
                    experiment_id,
                    experiment_dir,
                )
                continue
            model_dir = experiment_dir / CHECKPOINT_SUBDIR
            checkpoint_dir = find_latest_checkpoint(model_dir)
            if checkpoint_dir is None:
                continue
            counts = load_parameter_counts(checkpoint_dir)
            if counts is None:
                continue
            total, encoder, decoder = counts
            config_values = load_config_metadata(checkpoint_dir / "config.json")
            record = {
                "Type": experiment_id,
                "total_parameters": total,
                "encoder_parameters": encoder,
                "decoder_parameters": decoder,
            }
            record.update(config_values)
            records.append(record)
            logging.info("Recorded parameter statistics for '%s'", experiment_id)
    return records


def _format_parameter_cell(value: object, use_commas: bool = False) -> str:
    if value is None:
        return "--"
    if isinstance(value, (int, float)):
        if use_commas:
            return f"{int(value):,}"
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value)
    if isinstance(value, str):
        return value
    return str(value)


def model_parameters_to_latex(records: List[Dict[str, object]]) -> str:
    header = [
        r"\begin{table}[!t]",
        r"    \centering",
        r"    \caption{Model configurations comparison.}",
        r"    \label{tab:model_size_comparison}",
        r"\resizebox{\columnwidth}{!} {",
        r"    \begin{tabular}{c|ccccccccc}",
        r"        \hline",
        "        " + " & ".join(label for _, label in MODEL_PARAMETER_COLUMNS) + r" \\",
        r"        \hline",
    ]

    rows: List[str] = []
    for record in records:
        row_values: List[str] = []
        for key, _ in MODEL_PARAMETER_COLUMNS:
            value = record.get(key)
            use_commas = key in {
                "total_parameters",
                "encoder_parameters",
                "decoder_parameters",
            }
            row_values.append(_format_parameter_cell(value, use_commas=use_commas))
        rows.append("        " + " & ".join(row_values) + r" \\")

    footer = [
        r"        \hline",
        r"    \end{tabular}",
        r"}",
        r"\end{table}",
    ]

    return "\n".join(header + rows + footer) + "\n"


def save_model_parameter_table(
    base_dir: Path, types: Iterable[str], algorithms: Iterable[str], output_dir: Path
) -> None:
    records = gather_model_parameter_records(base_dir, types, algorithms)
    if not records:
        logging.warning(
            "No model parameter data collected; skipping parameter table export."
        )
        return
    latex_table = model_parameters_to_latex(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    tex_path = output_dir / "model_parameters_table.tex"
    tex_path.write_text(latex_table, encoding="utf-8")
    logging.info("Model parameter table saved to %s", tex_path)


def save_outputs(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "model_size_metrics_summary.csv"
    df.to_csv(csv_path, index=False)
    logging.info("Aggregated metrics CSV saved to %s", csv_path)

    latex_table = dataframe_to_latex_table(df)
    tex_path = output_dir / "model_size_metrics_table.tex"
    tex_path.write_text(latex_table, encoding="utf-8")
    logging.info("LaTeX table saved to %s", tex_path)

    available_metrics = [
        col for col in SUMMARY_COLUMNS if col in df.columns and col != "Type"
    ]

    ratio_metric_labels = set(RATIO_METRIC_MAPPING.values())
    value_metrics = [m for m in available_metrics if m not in ratio_metric_labels]

    bar_chart_path = output_dir / "model_size_metrics_bar.pdf"
    save_metrics_bar_chart(df, bar_chart_path, metric_names=value_metrics)

    if value_metrics:
        df_subset = df[["Type"] + value_metrics].copy()

        size_decimal_subset = df_subset[
            df_subset["Type"].map(_algorithm_variant_name) == "DecimalWordLevel"
        ].copy()
        size_color_map = {
            "Small": COLOR_PALETTE[0],
            "Medium": COLOR_PALETTE[3 % len(COLOR_PALETTE)],
            "Large": COLOR_PALETTE[6 % len(COLOR_PALETTE)],
        }
        if size_decimal_subset.empty:
            logging.warning(
                "No DecimalWordLevel entries found for size-level charts; skipping."
            )
        else:
            size_avg = _compute_group_averages(
                size_decimal_subset,
                value_metrics,
                _base_type_name,
                DEFAULT_TYPES,
            )
            size_chart_path = output_dir / "model_size_metrics_bar_size_level.pdf"
            _plot_group_averages(
                size_avg,
                value_metrics,
                DEFAULT_TYPES,
                size_color_map,
                size_chart_path,
                legend_title="Sizes",
            )

            size_radar_path = output_dir / "model_size_metrics_radar_size_level.pdf"
            _plot_group_radar(
                size_avg,
                value_metrics,
                DEFAULT_TYPES,
                size_color_map,
                size_radar_path,
                legend_title="Sizes",
            )

        alg_avg = _compute_group_averages(
            df_subset,
            value_metrics,
            _algorithm_variant_name,
            DEFAULT_ALGORITHMS,
        )
        alg_radar_df = _prepare_algorithm_radar_df(alg_avg, value_metrics)
        if alg_radar_df.empty:
            logging.warning(
                "No aggregated algorithm data available for algorithm-level charts; skipping."
            )
        else:
            alg_chart_path = output_dir / "model_size_metrics_bar_alg_level.pdf"
            palette_len = len(COLOR_PALETTE)
            alg_radar_color_map = {
                "Concat*": COLOR_PALETTE[0],
                "Seg*": COLOR_PALETTE[3 % palette_len],
                "DecimalWordLevel": COLOR_PALETTE[6 % palette_len],
            }
            _plot_group_averages(
                alg_radar_df,
                value_metrics,
                RADAR_ALGORITHM_ORDER,
                alg_radar_color_map,
                alg_chart_path,
                legend_title="Algorithms",
            )

            alg_radar_path = output_dir / "model_size_metrics_radar_alg_level.pdf"
            _plot_group_radar(
                alg_radar_df,
                value_metrics,
                RADAR_ALGORITHM_ORDER,
                alg_radar_color_map,
                alg_radar_path,
                legend_title="Algorithms",
            )
    else:
        logging.warning(
            "No eligible metrics available for grouped visualizations; skipping."
        )


def main() -> None:
    args = parse_args()
    setup_logging(level=args.log_level.upper())
    summary_df = aggregate_metrics(args.base_dir, args.types, args.algorithms)
    if summary_df.empty:
        logging.warning("No data available for metric export; skipping metrics table.")
    else:
        save_outputs(summary_df, args.output_dir)
    save_model_parameter_table(
        args.base_dir, args.types, args.algorithms, args.output_dir
    )


if __name__ == "__main__":
    main()
