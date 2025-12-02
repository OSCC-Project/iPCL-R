#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   analysis.py
@Time    :   2025/08/05 11:26:59
@Author  :   Dawn Li
@Version :   1.0
@Contact :   dawnli619215645@gmail.com
@Desc    :   Analysis tool for tokenizer comparison experiments that generates symbol
             frequency ranking plots with consistent color schemes, supporting
             DecimalWordLevel, BPE, and BBPE tokenization methods
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import scienceplots

from flow import FlowConfig
from flow.utils import setup_logging

from .init_env import create_alias

FIG_SIZE = (16, 14)
FONT_SIZE = 36
LEGEND_SIZE = 36
MARKER_SIZE = 32


def extract_metadata_set(
    work_dir: Path,
    vocab_sizes: List[int],
    tokenizer_algorithms: List[str],
) -> Dict[str, dict]:
    """Create the workspace and FlowConfig for the tokenizer comparison experiment"""
    metadata_set: Dict[str, dict] = {}
    for vocab_size in vocab_sizes:
        for tokenizer_algorithm in tokenizer_algorithms:
            alias = create_alias(tokenizer_algorithm, vocab_size)
            if alias in metadata_set:
                # Avoid duplicate configurations for 'DecimalWordLevel'
                continue
            # Dummy FlowConfig to get directory
            flow_config = FlowConfig()
            sub_work_dir = work_dir / alias
            flow_config.replace_path_prefixes(sub_work_dir)
            metadata_json_path = Path(
                flow_config.tokenization.paths.output_metadata_path
            )
            if not metadata_json_path.exists():
                logging.warning(
                    f"Metadata JSON not found for {alias} at {metadata_json_path}"
                )
                continue
            metadata_set[alias] = json.loads(metadata_json_path.read_text())
            logging.info(f"Found metadata JSON for '{alias}': {metadata_json_path}")
    logging.info(f"Found {len(metadata_set)} metadata JSON files in {work_dir}")
    return metadata_set


def plot_symbol_frequency_ranking(metadata_set: Dict[str, dict], output_dir: Path):
    """Plot symbol frequency ranking for all tokenization methods"""
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    # Get unified style mapping for consistent colors and markers
    method_color_mapping, vocab_style_mapping = get_unified_style_mapping(metadata_set)

    # 1. Plot DecimalWordLevel first (highest priority)
    plot_method1_symbol_frequency(
        ax, metadata_set, method_color_mapping, vocab_style_mapping
    )

    # 2. Plot BPE methods group
    plot_bpe_symbol_frequency_group(
        ax, metadata_set, method_color_mapping, vocab_style_mapping
    )

    # 3. Plot BBPE methods group
    plot_bbpe_symbol_frequency_group(
        ax, metadata_set, method_color_mapping, vocab_style_mapping
    )

    # Configure plot appearance
    configure_symbol_frequency_plot(fig, ax)

    # Save plot
    output_path = output_dir / "symbol_frequency_rank.pdf"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logging.info(f"Symbol frequency rank comparison plot saved: {output_path}")


def calculate_symbol_frequencies(token_counts: Dict[str, int]) -> Tuple:
    """Calculate symbol frequencies from token counts"""
    if not token_counts:
        return None, None, 0, 0

    # Sort by frequency (descending)
    sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
    frequencies = [count for _, count in sorted_tokens]

    if not frequencies:
        return None, None, 0, 0

    # Calculate cumulative percentage
    total_tokens = sum(frequencies)
    cumulative_percentages = []
    cumulative_sum = 0
    for freq in frequencies:
        cumulative_sum += freq
        cumulative_percentages.append(cumulative_sum / total_tokens * 100)

    actual_vocab_size = len(sorted_tokens)
    return cumulative_percentages, frequencies, actual_vocab_size, total_tokens


def get_unified_style_mapping(metadata_set: Dict[str, dict]) -> Tuple[Dict, Dict]:
    """Get unified style mapping for all methods with consistent colors and markers"""
    # Count BPE and BBPE methods separately
    bpe_methods = []
    bbpe_methods = []

    for alias in metadata_set.keys():
        if alias == "DecimalWordLevel":
            continue
        if "BBPE" in alias:
            bbpe_methods.append(alias)
        elif "BPE" in alias:
            bpe_methods.append(alias)

    n_bpe = len(bpe_methods)
    n_bbpe = len(bbpe_methods)

    # Create color palettes: get 2n+1 colors, skip first n colors
    bpe_cmap = plt.get_cmap("Greens", 2 * n_bpe + 1) if n_bpe > 0 else None
    bbpe_cmap = plt.get_cmap("Blues", 2 * n_bbpe + 1) if n_bbpe > 0 else None

    # Get all unique configurations (excluding specific algorithm)
    configurations = set()
    for alias in metadata_set.keys():
        if alias == "DecimalWordLevel":
            continue

        # Extract configuration from alias
        if "Concat-" in alias:
            config_base = "Concat"
        elif "Seg-" in alias:
            config_base = "Seg"
        else:
            config_base = "Other"

        # Extract vocab size from alias
        parts = alias.split("-")
        if len(parts) > 1:
            vocab_suffix = parts[-1]
        else:
            vocab_suffix = "N/A"

        config_key = f"{config_base} + {vocab_suffix}"
        configurations.add(config_key)

    # Sort configurations for consistent assignment
    sorted_configs = sorted(list(configurations))

    # Marker list for different configurations
    markers = ["o", "^", "s", "D", "p", "h", "H", "x", "v", "<", ">", "+"]

    # Create mappings
    method_color_mapping = {}
    vocab_style_mapping = {}

    # Special handling for DecimalWordLevel
    method_color_mapping["DecimalWordLevel"] = "#C41E3A"
    vocab_style_mapping["DecimalWordLevel"] = ("*", 0.85)

    # Assign colors and markers to BPE and BBPE methods
    config_marker_mapping = {}
    for i, config in enumerate(sorted_configs):
        marker = markers[i % len(markers)]
        config_marker_mapping[config] = marker

    # Assign colors to BPE methods
    bpe_color_index = 0
    for alias in bpe_methods:
        # Extract configuration
        if "Concat-" in alias:
            config_base = "Concat"
        elif "Seg-" in alias:
            config_base = "Seg"
        else:
            config_base = "Other"

        parts = alias.split("-")
        vocab_suffix = parts[-1] if len(parts) > 1 else "N/A"
        config_key = f"{config_base} + {vocab_suffix}"

        # Use color from position n_bpe + bpe_color_index
        color = mcolors.to_hex(bpe_cmap(n_bpe + bpe_color_index))
        bpe_color_index += 1

        # Assign marker based on configuration
        marker = config_marker_mapping.get(config_key, "o")
        opacity = 0.3

        method_color_mapping[alias] = color
        vocab_style_mapping[alias] = (marker, opacity)

    # Assign colors to BBPE methods
    bbpe_color_index = 0
    for alias in bbpe_methods:
        # Extract configuration
        if "Concat-" in alias:
            config_base = "Concat"
        elif "Seg-" in alias:
            config_base = "Seg"
        else:
            config_base = "Other"

        parts = alias.split("-")
        vocab_suffix = parts[-1] if len(parts) > 1 else "N/A"
        config_key = f"{config_base} + {vocab_suffix}"

        # Use color from position n_bbpe + bbpe_color_index
        color = mcolors.to_hex(bbpe_cmap(n_bbpe + bbpe_color_index))
        bbpe_color_index += 1

        # Assign marker based on configuration
        marker = config_marker_mapping.get(config_key, "o")
        opacity = 0.3

        method_color_mapping[alias] = color
        vocab_style_mapping[alias] = (marker, opacity)

    return method_color_mapping, vocab_style_mapping


def plot_method1_symbol_frequency(
    ax,
    metadata_set: Dict[str, dict],
    method_color_mapping: Dict,
    vocab_style_mapping: Dict,
):
    """Plot DecimalWordLevel symbol frequency ranking"""
    if "DecimalWordLevel" not in metadata_set:
        return

    metadata = metadata_set["DecimalWordLevel"]
    frequency_distribution = metadata.get("frequency_distribution", {})
    cumulative_percentages, frequencies, actual_vocab_size, _ = (
        calculate_symbol_frequencies(frequency_distribution)
    )

    if cumulative_percentages is None:
        return

    color = method_color_mapping["DecimalWordLevel"]
    marker, alpha = vocab_style_mapping["DecimalWordLevel"]
    label = "DecimalWordLevel"

    # Calculate marker positions based on uniform x-axis distribution
    marker_positions = calculate_uniform_marker_positions(cumulative_percentages)

    ax.plot(
        cumulative_percentages,
        frequencies,
        color=color,
        alpha=alpha,
        linewidth=3.0,
        marker=marker,
        linestyle="--",
        markersize=MARKER_SIZE,
        markeredgewidth=1,
        markeredgecolor=color,
        markevery=marker_positions,
        label=label,
        zorder=15,
    )


def plot_bpe_symbol_frequency_group(
    ax,
    metadata_set: Dict[str, dict],
    method_color_mapping: Dict,
    vocab_style_mapping: Dict,
):
    """Plot BPE methods group: mean lines, confidence bands, then individual methods"""
    bpe_aliases = [
        alias for alias in metadata_set.keys() if "BPE" in alias and "BBPE" not in alias
    ]

    if not bpe_aliases:
        return

    # Collect data for mean and confidence calculation
    bpe_data, bpe_results = collect_symbol_frequency_data(metadata_set, bpe_aliases)

    # Get mean color from colormap
    n_bpe = len(bpe_aliases)
    if n_bpe > 0:
        bpe_cmap = plt.get_cmap("Greens", 2 * n_bpe + 1)
        bpe_mean_color = mcolors.to_hex(bpe_cmap(2 * n_bpe))

        # Plot mean and confidence band
        plot_symbol_frequency_mean_and_confidence(
            ax, bpe_data, bpe_mean_color, "BPE", bpe_results
        )

    # Plot individual BPE methods
    plot_individual_symbol_frequency_methods(
        ax, metadata_set, bpe_aliases, method_color_mapping, vocab_style_mapping
    )


def plot_bbpe_symbol_frequency_group(
    ax,
    metadata_set: Dict[str, dict],
    method_color_mapping: Dict,
    vocab_style_mapping: Dict,
):
    """Plot BBPE methods group: mean lines, confidence bands, then individual methods"""
    bbpe_aliases = [alias for alias in metadata_set.keys() if "BBPE" in alias]

    if not bbpe_aliases:
        return

    # Collect data for mean and confidence calculation
    bbpe_data, bbpe_results = collect_symbol_frequency_data(metadata_set, bbpe_aliases)

    # Get mean color from colormap
    n_bbpe = len(bbpe_aliases)
    if n_bbpe > 0:
        bbpe_cmap = plt.get_cmap("Blues", 2 * n_bbpe + 1)
        bbpe_mean_color = mcolors.to_hex(bbpe_cmap(2 * n_bbpe))

        # Plot mean and confidence band
        plot_symbol_frequency_mean_and_confidence(
            ax, bbpe_data, bbpe_mean_color, "BBPE", bbpe_results
        )

    # Plot individual BBPE methods
    plot_individual_symbol_frequency_methods(
        ax, metadata_set, bbpe_aliases, method_color_mapping, vocab_style_mapping
    )


def collect_symbol_frequency_data(
    metadata_set: Dict[str, dict], method_aliases: List[str]
):
    """Collect and interpolate symbol frequency data for given methods"""
    method_data = []
    method_results = []

    for alias in method_aliases:
        metadata = metadata_set[alias]
        frequency_distribution = metadata.get("frequency_distribution", {})
        cumulative_percentages, frequencies, _, _ = calculate_symbol_frequencies(
            frequency_distribution
        )

        if cumulative_percentages is None:
            continue

        # Get actual x-axis range for consistent interpolation
        x_min, x_max = min(cumulative_percentages), max(cumulative_percentages)
        # Use the same x-axis range as the actual data
        common_x = np.linspace(x_min, x_max, 100)
        log_frequencies = np.log10(np.array(frequencies))
        interpolated_log_y = np.interp(
            common_x, cumulative_percentages, log_frequencies
        )
        interpolated_y = 10**interpolated_log_y

        method_data.append(interpolated_y)
        # Store common_x for each method
        method_results.append((alias, metadata, common_x))

    return np.array(method_data) if method_data else None, method_results


def plot_symbol_frequency_mean_and_confidence(
    ax,
    method_data,
    color: str,
    label_prefix: str,
    method_results,
    zorder_mean=10,
    zorder_band=1,
):
    """Plot mean line and confidence band for symbol frequency method group"""
    if method_data is None or len(method_data) == 0:
        return

    # Use the x-axis range from the first method
    if method_results:
        common_x = method_results[0][2]
    else:
        common_x = np.linspace(0, 100, 100)  # Fallback

    # Calculate geometric mean for log scale data
    log_mean = np.mean(np.log10(method_data), axis=0)
    log_std = np.std(np.log10(method_data), axis=0)
    mean_data = 10**log_mean
    lower_data = 10 ** (log_mean - log_std)
    upper_data = 10 ** (log_mean + log_std)

    # Mean line
    ax.plot(
        common_x,
        mean_data,
        color=color,
        linewidth=4,
        linestyle="-",
        alpha=0.8,
        label=f"${label_prefix}-Mean$",
        zorder=zorder_mean,
    )

    # Confidence band
    ax.fill_between(
        common_x,
        lower_data,
        upper_data,
        color=color,
        alpha=0.1,
        label=f"${label_prefix} \\pm 1 \\sigma$",
        zorder=zorder_band,
    )


def plot_individual_symbol_frequency_methods(
    ax,
    metadata_set: Dict[str, dict],
    method_aliases: List[str],
    method_color_mapping: Dict,
    vocab_style_mapping: Dict,
):
    """Plot individual symbol frequency method lines"""
    for alias in method_aliases:
        metadata = metadata_set[alias]
        frequency_distribution = metadata.get("frequency_distribution", {})
        cumulative_percentages, frequencies, actual_vocab_size, _ = (
            calculate_symbol_frequencies(frequency_distribution)
        )

        if cumulative_percentages is None:
            continue

        color = method_color_mapping.get(alias, "#666666")
        marker, alpha = vocab_style_mapping.get(alias, ("o", 0.3))

        # Create label with original and actual vocab sizes
        # tokenizer_info = metadata.get("tokenizer_info", {})
        # original_vocab_size = tokenizer_info.get("original_vocab_size", -1)
        label = f"{alias}"

        # Calculate marker positions based on uniform x-axis distribution
        marker_positions = calculate_uniform_marker_positions(cumulative_percentages)

        ax.plot(
            cumulative_percentages,
            frequencies,
            color=color,
            alpha=alpha,
            linewidth=3.0,
            marker=marker,
            linestyle="--",
            markersize=MARKER_SIZE,
            markeredgewidth=1,
            markeredgecolor=color,
            markevery=marker_positions,
            label=label,
            zorder=5,
        )


def calculate_uniform_marker_positions(
    cumulative_percentages: List[float], num_markers: int = 5
) -> List[int]:
    """Calculate marker positions based on uniform distribution along x-axis"""
    if not cumulative_percentages or len(cumulative_percentages) < 2:
        return []

    x_min, x_max = min(cumulative_percentages), max(cumulative_percentages)
    # Create uniform x positions
    target_x_positions = np.linspace(x_min, x_max, num_markers)

    marker_indices = []
    for target_x in target_x_positions:
        # Find the closest index to the target x position
        closest_idx = np.argmin(np.abs(np.array(cumulative_percentages) - target_x))
        marker_indices.append(closest_idx)

    # Remove duplicates while preserving order
    marker_indices = list(dict.fromkeys(marker_indices))

    return marker_indices


def configure_symbol_frequency_plot(fig, ax):
    """Configure the appearance of symbol frequency ranking plot"""
    # Labels and scales
    ax.set_xlabel(r"Cumulative Percentage (\%)", fontsize=FONT_SIZE, fontweight="bold")
    ax.set_ylabel("Frequency", fontsize=FONT_SIZE, fontweight="bold")
    ax.set_yscale("log")  # Set y-axis to log scale

    # Grid and axis styling
    ax.grid(True, linestyle="--")
    ax.tick_params(axis="both", labelsize=FONT_SIZE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend styling - place outside the plot area on the right side
    legend = ax.legend(
        bbox_to_anchor=(1.05, 0.5),
        loc="center left",
        frameon=True,
        fancybox=True,
        shadow=True,
        fontsize=LEGEND_SIZE,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_alpha(0.9)

    # Adjust layout to accommodate the external legend
    # fig.tight_layout()
    # fig.subplots_adjust(right=0.8)  # Make room for the legend on the right


def main():
    setup_logging()
    """Main entry point for pipeline initialization"""
    parser = argparse.ArgumentParser(
        description="Analyze tokenizer comparison experiments and generate symbol frequency ranking plots"
    )

    # Configuration generation
    parser.add_argument(
        "--work-dir",
        type=Path,
        metavar="WORK_DIR",
        default=Path(
            "/mnt/local_data1/liweiguo/experiments/tokenizer_comparison/work_dir"
        ),
        help="Create the corresponding directory and configuration in WORK_DIR based on the experiment setup",
    )

    parser.add_argument(
        "--target-vocab-size",
        type=int,
        nargs="+",
        default=[1000, 4000, 16000],
        help="List of target vocabulary sizes for tokenization",
    )

    parser.add_argument(
        "--tokenizer-algorithm",
        type=str,
        nargs="+",
        default=[
            "DecimalWordLevel",
            "Seg-BPE",
            "Concat-BPE",
            "Seg-BBPE",
            "Concat-BBPE",
        ],
        help="List of tokenizer algorithms to compare",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/mnt/local_data1/liweiguo/experiments/tokenizer_comparison/work_dir/analysis"
        ),
        help="Directory to save analysis results",
    )

    args = parser.parse_args()

    if scienceplots:
        plt.style.use(["science"])
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3

    # Extract metadata from all experiments
    metadata_set = extract_metadata_set(
        work_dir=args.work_dir,
        vocab_sizes=args.target_vocab_size,
        tokenizer_algorithms=args.tokenizer_algorithm,
    )

    if not metadata_set:
        logging.error(
            "No metadata files found. Make sure tokenization experiments have been run."
        )
        return

    # Generate the symbol frequency ranking plot
    plot_symbol_frequency_ranking(metadata_set, args.output_dir)

    logging.info(f"Analysis complete! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
