#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   collect_drc.py
@Time    :   2025/09/06 16:31:55
@Author  :   Dawn Li
@Version :   1.0
@Contact :   dawnli619215645@gmail.com
@Desc    :   Collect DRC total violation counts from output/<design>/<meta_type>/drc.rpt and drc_eco.rpt
"""

import argparse
import csv
import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import scienceplots

from flow.utils import setup_logging
from flow.utils.plot_utils import palette_slice

if scienceplots:
    plt.style.use(["science"])


class DRCParser:
    """DRC report parser for extracting violation counts."""

    def __init__(
        self,
        violation_re: re.Pattern[str] = re.compile(
            r"Total\s+Violations\s*:\s*(\d+)", re.IGNORECASE
        ),
    ) -> None:
        self.violation_re = violation_re

    def parse_drc_num(self, path: Path) -> Optional[int]:
        """Parse DRC violation count from report file."""
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
        except OSError:
            return None

        match = self.violation_re.search(text)
        if not match:
            return None

        try:
            return int(match.group(1))
        except Exception:
            return None


class DRCDataCollector:
    """Collector for DRC data from directory structure."""

    def __init__(self, parser: DRCParser):
        self.parser = parser

    def collect(self, base_dir: Path) -> List[Tuple[str, str, str, str]]:
        """Collect DRC data from base directory structure."""
        if not base_dir.is_dir():
            raise SystemExit(f"Base directory does not exist: {base_dir}")

        rows = []
        for design in sorted(base_dir.iterdir()):
            if not design.is_dir():
                continue
            rows.extend(self._collect_design_data(design))
        return rows

    def _collect_design_data(
        self, design_path: Path
    ) -> List[Tuple[str, str, str, str]]:
        """Collect DRC data for a specific design."""
        rows = []
        for meta in sorted(design_path.iterdir()):
            if not meta.is_dir():
                continue

            if meta.name not in {"GT", "PRED", "POST"}:
                continue

            drc_data = self._parse_meta_drc_data(meta)
            rows.append((design_path.name, meta.name, drc_data[0], drc_data[1]))
        return rows

    def _parse_meta_drc_data(self, meta_path: Path) -> Tuple[str, str]:
        """Parse DRC data for a meta directory."""
        # Parse original drc.rpt
        drc_path = meta_path / "drc.rpt"
        drc_num = self.parser.parse_drc_num(drc_path)
        drc_str = "0" if drc_num is None else str(drc_num)

        # Parse drc_eco.rpt
        drc_eco_path = meta_path / "drc_eco.rpt"
        drc_eco_num = self.parser.parse_drc_num(drc_eco_path)
        drc_eco_str = "0" if drc_eco_num is None else str(drc_eco_num)

        return drc_str, drc_eco_str


class CSVExporter:
    """CSV file exporter for DRC data."""

    @staticmethod
    def export(data: List[Tuple[str, str, str, str]], output_path: Path) -> None:
        """Export DRC data to CSV file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", newline="", encoding="utf-8") as csvf:
            writer = csv.writer(csvf)
            writer.writerow(["design", "meta_type", "drc_num", "drc_eco_num"])
            writer.writerows(data)


class DRCPlotter:
    """DRC data visualization plotter."""

    def __init__(
        self,
        fig_size: Tuple[int, int] = (24, 8),
        text_size: int = 24,
        font_size: int = 36,
        legend_size: int = 36,
    ):
        meta_types = ["GT", "PRED", "POST"]
        colors = palette_slice(len(meta_types))
        self.color_map = {meta: colors[idx] for idx, meta in enumerate(meta_types)}
        self.fig_size = fig_size
        self.text_size = text_size
        self.font_size = font_size
        self.legend_size = legend_size

    def create_plot(self, csv_path: Path, output_path: Path) -> None:
        """Create and save DRC distribution plot."""
        df = self._load_and_prepare_data(csv_path)
        fig, ax = self._create_base_plot()
        self._add_bars_and_labels(ax, df)
        self._customize_plot(ax, df)
        self._save_plot(fig, output_path)

    def _load_and_prepare_data(self, csv_path: Path) -> pd.DataFrame:
        """Load and prepare data for plotting."""
        df = pd.read_csv(csv_path)
        df["drc_num"] = pd.to_numeric(df["drc_num"]).astype(int)
        df["drc_eco_num"] = pd.to_numeric(df["drc_eco_num"]).astype(int)
        return df.sort_values(by="drc_num", ascending=True)

    def _create_base_plot(self) -> Tuple[plt.Figure, plt.Axes]:
        """Create base plot configuration."""
        return plt.subplots(figsize=self.fig_size)

    def _add_bars_and_labels(self, ax: plt.Axes, df: pd.DataFrame) -> None:
        """Add bars and labels to the plot."""
        designs = df["design"].unique()
        meta_types = ["GT", "PRED", "POST"]

        # Calculate layout parameters
        num_designs, num_meta_types = len(designs), len(meta_types)
        total_width = self._calculate_bar_width(num_designs)
        bar_width = total_width / num_meta_types
        group_spacing = 1.0
        x_pos = [i * group_spacing for i in range(len(designs))]

        min_y = self._get_min_positive_value(df)

        for i, meta_type in enumerate(meta_types):
            self._add_meta_type_bars(
                ax, df, designs, meta_type, i, num_meta_types, bar_width, x_pos, min_y
            )

    def _calculate_bar_width(self, num_designs: int) -> float:
        """Calculate optimal bar width based on number of designs."""
        if num_designs <= 5:
            return 0.8
        elif num_designs <= 10:
            return 0.7
        return 0.6

    def _get_min_positive_value(self, df: pd.DataFrame) -> float:
        """Get minimum positive value for label positioning."""
        return min(
            df[df["drc_num"] > 0]["drc_num"].min(),
            df[df["drc_eco_num"] > 0]["drc_eco_num"].min(),
        )

    def _add_meta_type_bars(
        self,
        ax: plt.Axes,
        df: pd.DataFrame,
        designs: List[str],
        meta_type: str,
        meta_index: int,
        num_meta_types: int,
        bar_width: float,
        x_pos: List[float],
        min_y: float,
    ) -> None:
        """Add bars for a specific meta type."""
        meta_data = (
            df[df["meta_type"] == meta_type].set_index("design").reindex(designs)
        )
        drc_original = meta_data["drc_num"].values
        drc_eco = meta_data["drc_eco_num"].values

        x_offset = (meta_index - (num_meta_types - 1) / 2) * bar_width
        x_positions = [x + x_offset for x in x_pos]
        base_color = self.color_map.get(meta_type, "#888888")

        # Add bars
        bars_eco = ax.bar(
            x_positions,
            drc_eco,
            bar_width,
            label=f"{meta_type} (ECO)",
            color=base_color,
            alpha=0.5,
        )
        bars_orig = ax.bar(
            x_positions,
            drc_original,
            bar_width,
            bottom=drc_eco,
            label=f"{meta_type}",
            color=base_color,
            alpha=1.0,
        )

        # Add labels
        self._add_bar_labels(
            ax, bars_eco, bars_orig, drc_eco, drc_original, min_y, meta_type
        )

    def _add_bar_labels(
        self,
        ax: plt.Axes,
        bars_eco,
        bars_orig,
        drc_eco: List[int],
        drc_original: List[int],
        min_y: float,
        meta_type: str,
    ) -> None:
        """Add value labels to bars."""
        for j, (bar_eco, bar_orig) in enumerate(zip(bars_eco, bars_orig)):
            # ECO label
            ax.text(
                bar_eco.get_x() + bar_eco.get_width() / 2,
                bar_eco.get_height() + 5 if drc_eco[j] > 0 else min_y * 0.80,
                f"{int(drc_eco[j])}",
                ha="center",
                va="bottom",
                fontsize=self.text_size,
                fontweight="bold",
            )

            # Original label with adjusted positioning for PRED and POST
            total_height = bar_eco.get_height() + bar_orig.get_height()
            original_text = (
                f"{int(drc_original[j])}"
                if drc_original[j] < 1000
                else f"{int(drc_original[j] / 1000 * 10) / 10}K"
            )

            # Adjust horizontal position and alignment based on meta_type
            if meta_type == "PRED":
                # Position at left upper corner
                x_pos = bar_orig.get_x() + bar_orig.get_width() * 0.1
            elif meta_type == "POST":
                # Position at right upper corner
                x_pos = bar_orig.get_x() + bar_orig.get_width() * 0.9
            else:
                # GT: keep center position
                x_pos = bar_orig.get_x() + bar_orig.get_width() * 0.1

            ax.text(
                x_pos,
                total_height + 5 if drc_original[j] > 0 else min_y + 5,
                original_text,
                ha="center",
                va="bottom",
                fontsize=self.text_size,
                fontweight="bold",
            )

    def _customize_plot(self, ax: plt.Axes, df: pd.DataFrame) -> None:
        """Customize plot appearance and legend."""
        ax.set_yscale("log")
        ax.set_ylabel("DRC Count", fontsize=self.font_size)
        ax.tick_params(axis="y", labelsize=self.font_size)

        # Rotate x-axis labels if too many designs
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=15)
        designs = df["design"].unique()
        x_pos = [i * 1.0 for i in range(len(designs))]
        ax.set_xticks(x_pos)
        ax.set_xticklabels(designs, ha="center", fontsize=self.font_size)
        ax.set_xlim(-0.5, len(designs) - 0.5)

        self._add_legend(ax)
        ax.grid(True, alpha=0.3)

    def _add_legend(self, ax: plt.Axes) -> None:
        """Add and customize legend."""
        handles, labels = ax.get_legend_handles_labels()
        legend_order = self._get_legend_order(labels)

        ordered_handles = [handles[i] for i in legend_order]
        ordered_labels = [labels[i] for i in legend_order]

        legend = ax.legend(
            ordered_handles,
            ordered_labels,
            loc="upper left",
            bbox_to_anchor=(0.02, 0.98),
            frameon=True,
            fancybox=True,
            shadow=True,
            ncol=3,
            columnspacing=0.8,
            fontsize=self.legend_size,
        )
        legend.get_frame().set_facecolor("white")
        legend.get_frame().set_alpha(0.9)

    def _get_legend_order(self, labels: List[str]) -> List[int]:
        """Get legend order for proper grouping."""
        legend_order = []
        for meta_type in ["GT", "PRED", "POST"]:
            for label_suffix in ["", " (ECO)"]:
                target_label = f"{meta_type}{label_suffix}"
                for i, label in enumerate(labels):
                    if label == target_label:
                        legend_order.append(i)
        return legend_order

    def _save_plot(self, fig: plt.Figure, output_path: Path) -> None:
        """Save plot to file."""
        fig.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches="tight")


class DRCProcessor:
    """Main processor for DRC data collection and visualization."""

    def __init__(
        self,
        parser: Optional[DRCParser] = None,
        plotter: Optional[DRCPlotter] = None,
    ):
        self.parser = parser or DRCParser()
        self.collector = DRCDataCollector(self.parser)
        self.exporter = CSVExporter()
        self.plotter = plotter or DRCPlotter()

    def process(self, base_dir: Path, output_dir: Path) -> None:
        """Process DRC data: collect, export to CSV, and create visualization."""
        # Collect data
        data = self.collector.collect(base_dir)
        logging.info(f"Collected {len(data)} rows of DRC data")

        # Export to CSV
        csv_path = output_dir / "drc_summary.csv"
        self.exporter.export(data, csv_path)
        logging.info(f"Wrote {len(data)} rows to {csv_path}")

        # Create visualization
        plot_path = output_dir / "drc_distribution.pdf"
        self.plotter.create_plot(csv_path, plot_path)
        logging.info(f"Created visualization at {plot_path}")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Collect DRC totals into CSV")
    parser.add_argument(
        "--base",
        "-b",
        default=Path("/data2/project_share/liweiguo/rt_gen/output"),
        type=Path,
        help="base output directory",
    )
    parser.add_argument(
        "--out",
        "-o",
        default=Path("/data2/project_share/liweiguo/rt_gen/output"),
        type=Path,
        help="output path",
    )
    return parser.parse_args()


def main():
    setup_logging()

    """Main entry point for DRC data processing."""
    args = parse_arguments()
    processor = DRCProcessor()
    processor.process(args.base, args.out)


if __name__ == "__main__":
    main()
