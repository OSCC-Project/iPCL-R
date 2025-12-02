#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   main.py
@Time    :   2025/10/29 00:21:48
@Author  :   Dawn Li
@Version :   1.0
@Contact :   dawnli619215645@gmail.com
@Desc    :   Extract design statistics from check_place.log files and generate LaTeX tables
"""

import argparse
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

from flow.utils import setup_logging


class DesignLogParser:
    """Parser for extracting design metrics from check_place.log files."""

    def __init__(self):
        # Regex patterns for extracting different metrics
        self.routable_nets_pattern = re.compile(
            r"#Total number of routable nets = (\d+)"
        )
        self.nets_pattern = re.compile(r"#Total number of nets in the design = (\d+)")
        self.density_pattern = re.compile(
            r"Placement Density: ?([\d.]+)%\((\d+)/(\d+)\)"
        )
        self.std_cell_insts_pattern = re.compile(
            r"\*\* info: there are (\d+) stdCell insts\."
        )

    def extract_routable_nets(self, log_content: str) -> Optional[str]:
        """Extract routable nets count from log content."""
        match = self.routable_nets_pattern.search(log_content)
        if match:
            return match.group(1)
        return None

    def extract_nets(self, log_content: str) -> Optional[str]:
        """Extract total nets count from log content."""
        match = self.nets_pattern.search(log_content)
        if match:
            return match.group(1)
        return None

    def extract_density(self, log_content: str) -> Optional[str]:
        """Extract placement density from log content."""
        match = self.density_pattern.search(log_content)
        if match:
            return match.group(1)  # Return the percentage value
        return None

    def extract_std_cell_insts(self, log_content: str) -> Optional[str]:
        """Extract std cell instances count from log content."""
        match = self.std_cell_insts_pattern.search(log_content)
        if match:
            return match.group(1)
        return None

    def parse_log_file(self, log_file: Path) -> Dict[str, Optional[str]]:
        """Parse a single check_place.log file and extract all metrics."""
        if not log_file.exists():
            logging.warning(f"      - Log file not found: {log_file}")
            return self._empty_metrics()

        try:
            with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            # Extract all metrics
            routable_nets = self.extract_routable_nets(content)
            nets = self.extract_nets(content)
            density = self.extract_density(content)
            std_cell_insts = self.extract_std_cell_insts(content)

            return {
                "routable_nets": routable_nets,
                "nets": nets,
                "density": density,
                "std_cell_insts": std_cell_insts,
            }

        except Exception as e:
            logging.error(f"      - Error parsing {log_file}: {e}")
            return self._empty_metrics()

    def _empty_metrics(self) -> Dict[str, Optional[str]]:
        """Return empty metrics dictionary."""
        return {
            "routable_nets": None,
            "nets": None,
            "density": None,
            "std_cell_insts": None,
        }


class DesignStatsProcessor:
    """Main processor for generating design statistics LaTeX tables."""

    def __init__(self, base_dir: Path, output_dir: Path):
        self.base_dir = base_dir
        self.output_dir = output_dir
        self.parser = DesignLogParser()

        # Default design lists
        self.default_train_designs = [
            "nvdla",
            "shanghai_MS",
            "ZJU_asic_top",
            "iEDA_2023",
            "ysyx4_SoC2",
            "BEIHAI_2.0",
            "AIMP",
            "AIMP_2.0",
            "ysyx6",
            "ysyx4_SoC1",
            "ysyx4_2",
            "s44",
            "gcd",
            "s1488",
            "apb4_ps2",
            "apb4_timer",
            "apb4_i2c",
            "apb4_pwm",
            "apb4_clint",
            "s15850",
            "s38417",
            "s35932",
            "s38584",
            "BM64",
            "picorv32",
            "PPU",
            "blabla",
            "aes_core",
            "aes",
            "salsa20",
            "jpeg_encoder",
            "eth_top",
            "yadan_riscv_sopc",
        ]

        self.default_val_designs = [
            "apb4_archinfo",
            "apb4_rng",
            "apb4_uart",
            "apb4_wdg",
            "ASIC",
            "s1238",
            "s5378",
            "s713",
            "s9234",
            "s13207",
        ]

    def process_designs(
        self, train_designs: List[str], val_designs: List[str], show_util: bool = False
    ):
        """Process train and validation designs and generate LaTeX tables."""
        logging.info("Processing design statistics from check_place.log files...")

        if not self.base_dir.exists():
            logging.error(f"Error: Base directory does not exist: {self.base_dir}")
            return

        # Process training designs
        train_results = self._process_design_list(train_designs, "Training")

        # Process validation designs
        val_results = self._process_design_list(val_designs, "Validation")

        logging.info(
            f"Total Routable Nets: {sum(int(r['routable_nets']) for r in train_results if r.get('routable_nets') and r['routable_nets'].isdigit()) + sum(int(r['routable_nets']) for r in val_results if r.get('routable_nets') and r['routable_nets'].isdigit())}"
        )

        # Generate LaTeX output
        if train_results or val_results:
            self._export_latex_tables(train_results, val_results, show_util)
            logging.info(
                f"\nDesign statistics LaTeX saved to: {self.output_dir / 'design_stats.tex'}"
            )
        else:
            logging.warning("\nNo data to export.")

    def _process_design_list(self, designs: List[str], design_type: str) -> List[Dict]:
        """Process a list of designs and return their metrics."""
        results = []
        logging.info(f"\n  Processing {design_type} designs:")

        for design in designs:
            logging.info(f"    - {design}")

            # Construct log file path: /base/design/check_place.log
            log_file = self.base_dir / design / "check_place.log"
            metrics = self.parser.parse_log_file(log_file)

            # Create result record
            result = {
                "design": design,
                "nets": metrics.get("nets"),
                "routable_nets": metrics.get("routable_nets"),
                "std_cell_insts": metrics.get("std_cell_insts"),
                "density": metrics.get("density"),
            }
            results.append(result)

            # Show extracted metrics
            if metrics.get("nets"):
                log_parts = []
                log_parts.append(f"nets={metrics['nets']}")

                if metrics.get("routable_nets"):
                    log_parts.append(f"routable_nets={metrics['routable_nets']}")
                if metrics.get("std_cell_insts"):
                    log_parts.append(f"std_cell_insts={metrics['std_cell_insts']}")
                if metrics.get("density"):
                    log_parts.append(f"density={metrics['density']}%")

                logging.info(f"      - Extracted: {', '.join(log_parts)}")
            else:
                logging.warning("      - No complete metrics extracted")

        return results

    def _export_latex_tables(
        self,
        train_results: List[Dict],
        val_results: List[Dict],
        show_util: bool = False,
    ):
        """Export design statistics as LaTeX tables."""
        output_file = self.output_dir / "design_stats.tex"

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Filter results with valid data and sort by nets
        valid_train_results = [
            r for r in train_results if r.get("nets") and r.get("nets").isdigit()
        ]
        valid_train_results.sort(key=lambda x: int(x["nets"]))

        valid_val_results = [
            r for r in val_results if r.get("nets") and r.get("nets").isdigit()
        ]
        valid_val_results.sort(key=lambda x: int(x["nets"]))

        # Generate LaTeX content
        latex_content = self._generate_training_table(valid_train_results, show_util)
        latex_content += "\n\n"
        latex_content += self._generate_validation_table(valid_val_results, show_util)

        # Write LaTeX file
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(latex_content)

    def _generate_training_table(
        self, train_results: List[Dict], show_util: bool = False
    ) -> str:
        """Generate LaTeX table for training designs."""
        if not train_results:
            return "% No training design data available"

        # Determine column specification based on show_util flag
        if show_util:
            col_spec = "{CCCCC}"
            header = "\\textbf{Case} & \\textbf{\\#Nets} & \\textbf{\\#Routable Nets} & \\textbf{\\#StdCell Insts} & \\textbf{Util} \\\\"
        else:
            col_spec = "{CCCC}"
            header = "\\textbf{Case} & \\textbf{\\#Nets} & \\textbf{\\#Routable Nets} & \\textbf{\\#StdCell Insts} \\\\"

        latex_content = f"""\\begin{{table}}[ht]
    \\centering
    \\caption{{Training design statistics.}}
    \\label{{tab:design_stat_train}}
    \\begin{{tabularx}}{{0.98\\linewidth}}{col_spec}
        \\hline
        {header}
        \\hline
"""

        for result in train_results:
            design = result["design"].replace("_", r"\_")
            nets = result["nets"]
            routable_nets = result.get("routable_nets", "")
            std_cell_insts = result.get("std_cell_insts", "")
            density = result.get("density", "")

            # Convert density percentage to decimal (divide by 100) if showing util
            if density and show_util:
                try:
                    density_val = f"{float(density) / 100:.2f}"
                except (ValueError, TypeError):
                    density_val = density
            else:
                density_val = ""

            if show_util:
                latex_content += f"        {design:<15} & {nets:>8} & {routable_nets:>15} & {std_cell_insts:>14} & {density_val:>10} \\\\\n"
            else:
                latex_content += f"        {design:<15} & {nets:>8} & {routable_nets:>15} & {std_cell_insts:>14} \\\\\n"

        latex_content += r"""        \hline
    \end{tabularx}
\end{table}"""

        return latex_content

    def _generate_validation_table(
        self, val_results: List[Dict], show_util: bool = False
    ) -> str:
        """Generate LaTeX table for validation designs."""
        if not val_results:
            return "% No validation design data available"

        # Determine column specification based on show_util flag
        if show_util:
            col_spec = "{CCCCC}"
            header = "\\textbf{Case} & \\textbf{\\#Nets} & \\textbf{\\#Routable Nets} & \\textbf{\\#StdCell Insts} & \\textbf{Util} \\\\"
        else:
            col_spec = "{CCCC}"
            header = "\\textbf{Case} & \\textbf{\\#Nets} & \\textbf{\\#Routable Nets} & \\textbf{\\#StdCell Insts} \\\\"

        latex_content = f"""\\begin{{table}}[ht]
    \\centering
    \\caption{{Validation design statistics.}}
    \\label{{tab:design_stat_val}}
    \\begin{{tabularx}}{{0.98\\linewidth}}{col_spec}
        \\hline
        {header}
        \\hline
"""

        for result in val_results:
            design = result["design"].replace("_", r"\_")
            nets = result["nets"]
            routable_nets = result.get("routable_nets", "")
            std_cell_insts = result.get("std_cell_insts", "")
            density = result.get("density", "")

            # Convert density percentage to decimal (divide by 100) if showing util
            if density and show_util:
                try:
                    density_val = f"{float(density) / 100:.2f}"
                except (ValueError, TypeError):
                    density_val = density
            else:
                density_val = ""

            if show_util:
                latex_content += f"        {design:<15} & {nets:>8} & {routable_nets:>15} & {std_cell_insts:>14} & {density_val:>10} \\\\\n"
            else:
                latex_content += f"        {design:<15} & {nets:>8} & {routable_nets:>15} & {std_cell_insts:>14} \\\\\n"

        latex_content += r"""        \hline
    \end{tabularx}
\end{table}"""

        return latex_content


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract design statistics from check_place.log files and generate LaTeX tables"
    )
    parser.add_argument(
        "--base",
        "-b",
        default="/data2/project_share/liweiguo/rt_gen/pl_summary",
        type=str,
        help="Base directory containing design folders (default: /data2/project_share/liweiguo/rt_gen/pl_summary)",
    )
    parser.add_argument(
        "--out",
        "-o",
        default="/data2/project_share/liweiguo/rt_gen/output",
        type=str,
        help="Output directory for generated LaTeX file (default: /data2/project_share/liweiguo/rt_gen/output)",
    )
    parser.add_argument(
        "--train_designs",
        nargs="+",
        default=[
            "nvdla",
            "shanghai_MS",
            "ZJU_asic_top",
            "iEDA_2023",
            "ysyx4_SoC2",
            "BEIHAI_2.0",
            "AIMP",
            "AIMP_2.0",
            "ysyx6",
            "ysyx4_SoC1",
            "ysyx4_2",
            "s44",
            "gcd",
            "s1488",
            "apb4_ps2",
            "apb4_timer",
            "apb4_i2c",
            "apb4_pwm",
            "apb4_clint",
            "s15850",
            "s38417",
            "s35932",
            "s38584",
            "BM64",
            "picorv32",
            "PPU",
            "blabla",
            "aes_core",
            "aes",
            "salsa20",
            "jpeg_encoder",
            "eth_top",
            "yadan_riscv_sopc",
        ],
        help="List of training designs to process",
    )
    parser.add_argument(
        "--val_designs",
        nargs="+",
        default=[
            "apb4_archinfo",
            "apb4_rng",
            "apb4_uart",
            "apb4_wdg",
            "ASIC",
            "s1238",
            "s5378",
            "s713",
            "s9234",
        ],
        help="List of validation designs to process",
    )
    parser.add_argument(
        "--show-util",
        action="store_true",
        default=False,
        help="Include utilization (density) column in the LaTeX table (default: False)",
    )
    return parser.parse_args()


def main():
    """Main entry point for the design statistics script."""
    setup_logging()
    args = parse_arguments()

    base_dir = Path(args.base)
    output_dir = Path(args.out)

    processor = DesignStatsProcessor(base_dir, output_dir)
    processor.process_designs(args.train_designs, args.val_designs, args.show_util)


if __name__ == "__main__":
    main()
