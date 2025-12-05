#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   summary_design.py
@Time    :   2025/09/08 17:30:00
@Author  :   Dawn Li
@Version :   1.0
@Contact :   dawnli619215645@gmail.com
@Desc    :   Extract design metrics from innovus.log files and generate summary CSV
"""

import argparse
import csv
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots

from flow.utils import setup_logging

if scienceplots:
    plt.style.use(["science", "ieee", "no-latex"])


class InnovusLogParser:
    """Parser for extracting design metrics from innovus.log files."""

    def __init__(self):
        # Regex patterns for extracting different metrics
        self.eco_cpu_pattern = re.compile(
            r"#% End globalDetailRoute.*total cpu=(\d+:\d+:[\d.]+).*real=(\d+:\d+:[\d.]+)"
        )

        self.timedesign_pattern = re.compile(
            r"timeDesign Summary.*?Setup views included:.*?all.*?reg2reg.*?default.*?"
            r"\|\s*WNS \(ns\):\|\s*([-\d.]+)\s*\|.*?\|\s*TNS \(ns\):\|\s*([-\d.]+)\s*\|.*?"
            r"Density:\s*([\d.]+)%",
            re.DOTALL,
        )

        self.power_pattern = re.compile(
            r"Total Power\s*\n-+\s*\n"
            r"Total Internal Power:\s*([\d.]+).*?\n"
            r"Total Switching Power:\s*([\d.]+).*?\n"
            r"Total Leakage Power:\s*([\d.]+).*?\n"
            r"Total Power:\s*([\d.]+)",
            re.DOTALL,
        )

        self.wirelength_pattern = re.compile(
            r"#Complete Post Route Wire Spread\..*?"
            r"#Total wire length = (\d+) um\..*?"
            r"#Total number of vias = (\d+)",
            re.DOTALL,
        )

        # Patterns for routing metrics from innovus_rt.log
        self.route_design_pattern = re.compile(
            r"#% End routeDesign.*total cpu=(\d+:\d+:[\d.]+).*real=(\d+:\d+:[\d.]+)"
        )

        self.opt_design_pattern = re.compile(
            r"\*\*optDesign.*cpu = (\d+:\d+:[\d.]+), real = (\d+:\d+:[\d.]+)"
        )

        # Patterns for cell and net metrics from innovus_rt.log
        self.combinational_cells_pattern = re.compile(
            r"Total number of combinational cells:\s*(\d+)"
        )

        self.sequential_cells_pattern = re.compile(
            r"Total number of sequential cells:\s*(\d+)"
        )

        self.net_num_pattern = re.compile(
            r"#Total number of nets in the design = (\d+)"
        )

    def extract_eco_timing(
        self, log_content: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """Extract ECO Total CPU and ECO Real time from globalDetailRoute."""
        match = self.eco_cpu_pattern.search(log_content)
        if match:
            cpu_time = self._convert_time_to_seconds(match.group(1))
            real_time = self._convert_time_to_seconds(match.group(2))
            return cpu_time, real_time
        return None, None

    def _convert_time_to_seconds(self, time_str: str) -> str:
        """Convert time format 'H:MM:SS.S' to seconds as string."""
        try:
            parts = time_str.split(":")
            if len(parts) == 3:
                hours = int(parts[0])
                minutes = int(parts[1])
                seconds = float(parts[2])
                total_seconds = hours * 3600 + minutes * 60 + seconds
                return f"{total_seconds:.1f}"
            else:
                return time_str  # Return original if format is unexpected
        except (ValueError, IndexError):
            return time_str  # Return original if parsing fails

    def extract_timing_metrics(
        self, log_content: str
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Extract WNS, TNS, and Density from timeDesign Summary."""
        match = self.timedesign_pattern.search(log_content)
        if match:
            return match.group(1), match.group(2), match.group(3)
        return None, None, None

    def extract_power_metrics(
        self, log_content: str
    ) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
        """Extract power metrics from Total Power section."""
        match = self.power_pattern.search(log_content)
        if match:
            return match.group(1), match.group(2), match.group(3), match.group(4)
        return None, None, None, None

    def extract_wire_metrics(
        self, log_content: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """Extract Total Wirelength and Total Vias from Post Route Wire Spread."""
        match = self.wirelength_pattern.search(log_content)
        if match:
            return match.group(1), match.group(2)
        return None, None

    def parse_innovus_log(self, log_file: Path) -> Dict[str, Optional[str]]:
        """Parse a single innovus.log file and extract all metrics."""
        if not log_file.exists():
            logging.warning(f"      - Log file not found: {log_file}")
            return self._empty_metrics()

        try:
            with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            # Extract all metrics
            eco_total_cpu, eco_real = self.extract_eco_timing(content)
            wns, tns, density = self.extract_timing_metrics(content)
            internal_power, switching_power, leakage_power, total_power = (
                self.extract_power_metrics(content)
            )
            total_wirelength, total_vias = self.extract_wire_metrics(content)

            return {
                "ECO Total CPU": eco_total_cpu,
                "ECO Real": eco_real,
                "WNS": wns,
                "TNS": tns,
                "Density": density,
                "Internal Power": internal_power,
                "Switching Power": switching_power,
                "Leakage Power": leakage_power,
                "Total Power": total_power,
                "Total Wirelength": total_wirelength,
                "Total Vias": total_vias,
            }

        except Exception as e:
            logging.error(f"      - Error parsing {log_file}: {e}")
            return self._empty_metrics()

    def _empty_metrics(self) -> Dict[str, Optional[str]]:
        """Return empty metrics dictionary."""
        return {
            "ECO Total CPU": None,
            "ECO Real": None,
            "WNS": None,
            "TNS": None,
            "Density": None,
            "Internal Power": None,
            "Switching Power": None,
            "Leakage Power": None,
            "Total Power": None,
            "Total Wirelength": None,
            "Total Vias": None,
        }

    def extract_routing_metrics(self, rt_log_content: str) -> Dict[str, Optional[str]]:
        """Extract routing metrics from innovus_rt.log file."""
        # 1. Extract RouteDesign metrics from '#% End routeDesign'
        route_design_match = self.route_design_pattern.search(rt_log_content)
        route_design_cpu = None
        route_design_real = None

        if route_design_match:
            route_design_cpu = self._convert_time_to_seconds(
                route_design_match.group(1)
            ).split(".")[0]
            route_design_real = self._convert_time_to_seconds(
                route_design_match.group(2)
            )

        # 2. Extract OptDesign metrics from '**optDesign ... cpu' (take the last one)
        opt_design_matches = self.opt_design_pattern.findall(rt_log_content)
        opt_design_cpu = None
        opt_design_real = None

        if opt_design_matches:
            # Take the last match
            last_match = opt_design_matches[-1]
            opt_design_cpu = self._convert_time_to_seconds(last_match[0]).split(".")[0]
            opt_design_real = self._convert_time_to_seconds(last_match[1])

        # 3. Calculate total routing time
        route_total_cpu = None
        route_total_real = None

        if route_design_cpu and opt_design_cpu:
            try:
                total_cpu = float(route_design_cpu) + float(opt_design_cpu)
                route_total_cpu = f"{total_cpu:.1f}"
            except (ValueError, TypeError):
                pass

        if route_design_real and opt_design_real:
            try:
                total_real = float(route_design_real) + float(opt_design_real)
                route_total_real = f"{total_real:.1f}"
            except (ValueError, TypeError):
                pass

        # 4. Extract cell and net metrics
        combinational_cells = None
        sequential_cells = None
        net_num = None

        # Extract combinational cells
        comb_match = self.combinational_cells_pattern.search(rt_log_content)
        if comb_match:
            combinational_cells = comb_match.group(1)

        # Extract sequential cells
        seq_match = self.sequential_cells_pattern.search(rt_log_content)
        if seq_match:
            sequential_cells = seq_match.group(1)

        # Extract net number (take the first occurrence)
        net_match = self.net_num_pattern.search(rt_log_content)
        if net_match:
            net_num = net_match.group(1)

        return {
            "RouteDesign Total CPU": route_design_cpu,
            "RouteDesign Real": route_design_real,
            "OptDesign Total CPU": opt_design_cpu,
            "OptDesign Real": opt_design_real,
            "Route Total CPU": route_total_cpu,
            "Route Total Real": route_total_real,
            "Combinational Cells": combinational_cells,
            "Sequential Cells": sequential_cells,
            "Net Num": net_num,
        }

    def parse_innovus_rt_log(self, rt_log_file: Path) -> Dict[str, Optional[str]]:
        """Parse a single innovus_rt.log file and extract routing metrics."""
        if not rt_log_file.exists():
            logging.warning(f"      - RT log file not found: {rt_log_file}")
            return self._empty_routing_metrics()

        try:
            with open(rt_log_file, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            return self.extract_routing_metrics(content)

        except Exception as e:
            logging.error(f"      - Error parsing {rt_log_file}: {e}")
            return self._empty_routing_metrics()

    def _empty_routing_metrics(self) -> Dict[str, Optional[str]]:
        """Return empty routing metrics dictionary."""
        return {
            "RouteDesign Total CPU": None,
            "RouteDesign Real": None,
            "OptDesign Total CPU": None,
            "OptDesign Real": None,
            "Route Total CPU": None,
            "Route Total Real": None,
            "Combinational Cells": None,
            "Sequential Cells": None,
            "Net Num": None,
        }


class DesignSummaryProcessor:
    """Main processor for generating design summary CSV."""

    def __init__(self, base_dir: Path, output_dir: Path):
        self.base_dir = base_dir
        self.output_dir = output_dir
        self.parser = InnovusLogParser()
        self.meta_types = ["ORIGINAL", "GT", "PRED", "POST"]

    def process_all_designs(self):
        """Process all designs and generate summary CSV."""
        logging.info("Processing design metrics from innovus.log files...")

        if not self.base_dir.exists():
            logging.error(f"Error: Base directory does not exist: {self.base_dir}")
            return

        all_results = []
        all_routing_results = []

        # Process each design directory
        for design_dir in self.base_dir.iterdir():
            if not design_dir.is_dir():
                continue

            logging.info(f"  Processing design: {design_dir.name}")

            # Process each meta_type (ORIGINAL, GT, PRED, POST)
            for meta_type in self.meta_types:
                meta_type_dir = design_dir / meta_type
                if not meta_type_dir.exists():
                    logging.warning(f"    - {meta_type} directory not found")
                    continue

                logging.info(f"    Processing meta_type: {meta_type}")

                # Look for innovus.log file
                log_file = meta_type_dir / "innovus.log"
                metrics = self.parser.parse_innovus_log(log_file)

                # Create result record
                result = {"design": design_dir.name, "meta_type": meta_type, **metrics}
                all_results.append(result)

                # Show extracted metrics
                if any(v is not None for v in metrics.values()):
                    logging.info(
                        f"      - Extracted metrics: {sum(1 for v in metrics.values() if v is not None)}/11 fields"
                    )
                else:
                    logging.warning("      - No metrics extracted")

                # Process routing metrics for ORIGINAL meta_type only
                if meta_type == "ORIGINAL":
                    rt_log_file = meta_type_dir / "innovus_rt.log"
                    routing_metrics = self.parser.parse_innovus_rt_log(rt_log_file)

                    # Create routing result record
                    routing_result = {"Design": design_dir.name, **routing_metrics}
                    all_routing_results.append(routing_result)

                    # Show extracted routing metrics
                    if any(v is not None for v in routing_metrics.values()):
                        logging.info(
                            f"      - Extracted routing metrics: {sum(1 for v in routing_metrics.values() if v is not None)}/9 fields"
                        )
                    else:
                        logging.warning("      - No routing metrics extracted")

        # Generate CSV outputs
        if all_results:
            self._export_to_csv(all_results)
            logging.info(
                f"\nDesign metrics summary saved to: {self.output_dir / 'design_summary.csv'}"
            )

            # Generate flattened CSV (no routing data)
            self._export_flattened_csv(all_results)
            logging.info(
                f"Flattened metrics summary saved to: {self.output_dir / 'design_summary_flatten.csv'}"
            )

            # Generate routing CSV
            if all_routing_results:
                self._export_routing_csv(all_routing_results)
                logging.info(
                    f"Routing metrics summary saved to: {self.output_dir / 'design_summary_route.csv'}"
                )

                # Generate design summary table CSV
                self._export_design_summary_table(all_routing_results, all_results)
                logging.info(
                    f"Design summary table saved to: {self.output_dir / 'design_summary_table.csv'}"
                )

                # Generate design comparison table CSV
                self._export_design_comparison_table(all_results, all_routing_results)
                logging.info(
                    f"Design comparison table saved to: {self.output_dir / 'design_comparison_table.csv'}"
                )

                # Generate LaTeX tables
                self._export_design_summary_table_tex(all_routing_results, all_results)
                logging.info(
                    f"Design summary table LaTeX saved to: {self.output_dir / 'design_summary_table.tex'}"
                )

                self._export_design_comparison_table_tex(
                    all_results, all_routing_results
                )
                logging.info(
                    f"Design comparison table LaTeX saved to: {self.output_dir / 'design_comparison_table.tex'}"
                )

            # Generate radar chart
            self._generate_radar_chart(all_results, all_routing_results)
            logging.info(
                f"Radar chart saved to: {self.output_dir / 'design_summary_radar.pdf'}"
            )
        else:
            logging.warning("\nNo data to export.")

    def _export_to_csv(self, results: List[Dict]):
        """Export results to CSV file."""
        output_file = self.output_dir / "design_summary.csv"

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Define column order
        columns = [
            "design",
            "meta_type",
            "ECO Total CPU",
            "ECO Real",
            "WNS",
            "TNS",
            "Density",
            "Internal Power",
            "Switching Power",
            "Leakage Power",
            "Total Power",
            "Total Wirelength",
            "Total Vias",
        ]

        # Export regular CSV
        with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=columns)
            writer.writeheader()
            writer.writerows(results)

    def _export_routing_csv(self, routing_results: List[Dict]):
        """Export routing results to CSV file."""
        output_file = self.output_dir / "design_summary_route.csv"

        # Define column order for routing CSV
        columns = [
            "Design",
            "RouteDesign Total CPU",
            "RouteDesign Real",
            "OptDesign Total CPU",
            "OptDesign Real",
            "Route Total CPU",
            "Route Total Real",
            "Combinational Cells",
            "Sequential Cells",
            "Net Num",
        ]

        # Export routing CSV
        with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=columns)
            writer.writeheader()
            writer.writerows(routing_results)

    def _export_design_summary_table(
        self, routing_results: List[Dict], all_results: List[Dict]
    ):
        """Export design summary table CSV with [Design, #Combinational Cells, #Sequential Cells, #Nets, Density]."""
        output_file = self.output_dir / "design_summary_table.csv"

        # Get density from ORIGINAL meta_type results
        density_lookup = {}
        for result in all_results:
            if result["meta_type"] == "ORIGINAL" and result["Density"]:
                density_lookup[result["design"]] = result["Density"]

        # Create summary table data
        table_data = []
        for routing_result in routing_results:
            design = routing_result["Design"]
            row = {
                "Design": design,
                "#Combinational Cells": routing_result.get("Combinational Cells", ""),
                "#Sequential Cells": routing_result.get("Sequential Cells", ""),
                "#Nets": routing_result.get("Net Num", ""),
                "Density": density_lookup.get(design, ""),
            }
            table_data.append(row)

        # Sort by #Nets in ascending order
        table_data.sort(
            key=lambda x: int(x["#Nets"])
            if x["#Nets"] and str(x["#Nets"]).isdigit()
            else 0
        )

        # Define column order
        columns = [
            "Design",
            "#Combinational Cells",
            "#Sequential Cells",
            "#Nets",
            "Density",
        ]

        # Export table CSV
        with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=columns)
            writer.writeheader()
            writer.writerows(table_data)

    def _export_design_comparison_table(
        self, all_results: List[Dict], routing_results: List[Dict]
    ):
        """Export design comparison table CSV."""
        output_file = self.output_dir / "design_comparison_table.csv"

        # Create routing lookup
        routing_lookup = {r["Design"]: r for r in routing_results}

        # Group all_results by design
        design_data = {}
        for result in all_results:
            design = result["design"]
            meta_type = result["meta_type"]
            if design not in design_data:
                design_data[design] = {}
            design_data[design][meta_type] = result

        # Create comparison table data
        table_data = []
        for design, meta_data in design_data.items():
            routing_data = routing_lookup.get(design, {})

            # Calculate inference CPU (net_num * 0.015923)
            inf_cpu = None
            if routing_data.get("Net Num"):
                try:
                    net_num = float(routing_data["Net Num"])
                    inf_cpu = f"{net_num * 0.015923:.2f}"
                except (ValueError, TypeError):
                    inf_cpu = ""

            row = {"Design": design}

            # Add data for each meta_type
            for meta_type in ["ORIGINAL", "GT", "PRED", "POST"]:
                data = meta_data.get(meta_type, {})

                row[f"#Vias ({meta_type})"] = data.get("Total Vias", "")
                row[f"Wirelength ({meta_type})"] = data.get("Total Wirelength", "")
                row[f"Switching Power ({meta_type})"] = data.get("Switching Power", "")
                row[f"Power ({meta_type})"] = data.get("Total Power", "")
                row[f"WNS ({meta_type})"] = data.get("WNS", "")
                row[f"TNS ({meta_type})"] = data.get("TNS", "")

                # Add runtime columns
                if meta_type == "ORIGINAL":
                    row["Route CPU"] = routing_data.get("Route Total CPU", "")
                    row["Inf. CPU"] = inf_cpu
                    row["ECO CPU (ORIGINAL)"] = data.get("ECO Total CPU", "")
                elif meta_type in ["GT", "PRED", "POST"]:
                    eco_cpu = data.get("ECO Total CPU", "")
                    if inf_cpu and eco_cpu:
                        try:
                            total_cpu = float(inf_cpu) + float(eco_cpu)
                            row[f"Inf. & ECO CPU ({meta_type})"] = f"{total_cpu:.2f}"
                        except (ValueError, TypeError):
                            row[f"Inf. & ECO CPU ({meta_type})"] = ""
                    else:
                        row[f"Inf. & ECO CPU ({meta_type})"] = ""

            # Add Net Num for sorting
            row["_Net_Num_"] = routing_data.get("Net Num", "")
            table_data.append(row)

        # Sort by Net Num in ascending order
        table_data.sort(
            key=lambda x: int(x["_Net_Num_"])
            if x["_Net_Num_"] and str(x["_Net_Num_"]).isdigit()
            else 0
        )

        # Remove the temporary sorting field
        for row in table_data:
            row.pop("_Net_Num_", None)

        # Define column order as specified
        columns = [
            "Design",
            "#Vias (ORIGINAL)",
            "#Vias (GT)",
            "#Vias (PRED)",
            "#Vias (POST)",
            "Wirelength (ORIGINAL)",
            "Wirelength (GT)",
            "Wirelength (PRED)",
            "Wirelength (POST)",
            "Switching Power (ORIGINAL)",
            "Switching Power (GT)",
            "Switching Power (PRED)",
            "Switching Power (POST)",
            "Power (ORIGINAL)",
            "Power (GT)",
            "Power (PRED)",
            "Power (POST)",
            "WNS (ORIGINAL)",
            "WNS (GT)",
            "WNS (PRED)",
            "WNS (POST)",
            "TNS (ORIGINAL)",
            "TNS (GT)",
            "TNS (PRED)",
            "TNS (POST)",
            "Route CPU",
            "Inf. CPU",
            "ECO CPU (ORIGINAL)",
            "Inf. & ECO CPU (GT)",
            "Inf. & ECO CPU (PRED)",
            "Inf. & ECO CPU (POST)",
        ]

        # Calculate totals for the last row
        total_row = {"Design": "Total"}

        # Get ORIGINAL baseline values for normalization
        original_totals = {}
        route_cpu_total = 0

        for design, meta_data in design_data.items():
            original_data = meta_data.get("ORIGINAL", {})
            routing_data = routing_lookup.get(design, {})

            # Sum ORIGINAL values for normalization base
            for metric in [
                "#Vias",
                "Wirelength",
                "Switching Power",
                "Power",
                "WNS",
                "TNS",
            ]:
                col_name = f"{metric} (ORIGINAL)"
                if metric == "#Vias":
                    value = original_data.get("Total Vias")
                elif metric == "Wirelength":
                    value = original_data.get("Total Wirelength")
                elif metric == "Switching Power":
                    value = original_data.get("Switching Power")
                elif metric == "Power":
                    value = original_data.get("Total Power")
                else:
                    value = original_data.get(metric)

                if value:
                    try:
                        original_totals[metric] = original_totals.get(
                            metric, 0
                        ) + float(value)
                    except (ValueError, TypeError):
                        pass

            # Sum Route CPU for runtime normalization
            route_cpu = routing_data.get("Route Total CPU")
            if route_cpu:
                try:
                    route_cpu_total += float(route_cpu)
                except (ValueError, TypeError):
                    pass

        # Calculate normalized totals
        for meta_type in ["ORIGINAL", "GT", "PRED", "POST"]:
            for metric in [
                "#Vias",
                "Wirelength",
                "Switching Power",
                "Power",
                "WNS",
                "TNS",
            ]:
                col_name = f"{metric} ({meta_type})"
                metric_total = 0

                for design, meta_data in design_data.items():
                    data = meta_data.get(meta_type, {})
                    if metric == "#Vias":
                        value = data.get("Total Vias")
                    elif metric == "Wirelength":
                        value = data.get("Total Wirelength")
                    elif metric == "Switching Power":
                        value = data.get("Switching Power")
                    elif metric == "Power":
                        value = data.get("Total Power")
                    else:
                        value = data.get(metric)

                    if value:
                        try:
                            metric_total += float(value)
                        except (ValueError, TypeError):
                            pass

                # Normalize to ORIGINAL = 1.00
                if original_totals.get(metric, 0) > 0:
                    normalized = metric_total / original_totals[metric]
                    total_row[col_name] = f"{normalized:.2f}"
                else:
                    total_row[col_name] = ""

        # Calculate runtime totals (normalized to Route CPU = 1.00)
        if route_cpu_total > 0:
            total_row["Route CPU"] = "1.00"

            # Calculate total inf CPU
            total_inf_cpu = 0
            for design, _ in design_data.items():
                routing_data = routing_lookup.get(design, {})
                if routing_data.get("Net Num"):
                    try:
                        net_num = float(routing_data["Net Num"])
                        total_inf_cpu += net_num * 0.015923
                    except (ValueError, TypeError):
                        pass

            if total_inf_cpu > 0:
                inf_normalized = total_inf_cpu / route_cpu_total
                total_row["Inf. CPU"] = f"{inf_normalized:.2f}"
            else:
                total_row["Inf. CPU"] = ""

            # Calculate total ECO CPU for ORIGINAL
            total_eco_cpu_original = 0
            for design, meta_data in design_data.items():
                original_data = meta_data.get("ORIGINAL", {})
                eco_cpu = original_data.get("ECO Total CPU")
                if eco_cpu:
                    try:
                        total_eco_cpu_original += float(eco_cpu)
                    except (ValueError, TypeError):
                        pass

            if total_eco_cpu_original > 0:
                eco_original_normalized = total_eco_cpu_original / route_cpu_total
                total_row["ECO CPU (ORIGINAL)"] = f"{eco_original_normalized:.2f}"
            else:
                total_row["ECO CPU (ORIGINAL)"] = ""

            # Calculate total inf + ECO CPU for each meta_type
            for meta_type in ["GT", "PRED", "POST"]:
                total_inf_eco_cpu = total_inf_cpu
                for design, meta_data in design_data.items():
                    data = meta_data.get(meta_type, {})
                    eco_cpu = data.get("ECO Total CPU")
                    if eco_cpu:
                        try:
                            total_inf_eco_cpu += float(eco_cpu)
                        except (ValueError, TypeError):
                            pass

                if total_inf_eco_cpu > 0:
                    inf_eco_normalized = total_inf_eco_cpu / route_cpu_total
                    total_row[f"Inf. & ECO CPU ({meta_type})"] = (
                        f"{inf_eco_normalized:.2f}"
                    )
                else:
                    total_row[f"Inf. & ECO CPU ({meta_type})"] = ""
        else:
            for col in [
                "Route CPU",
                "Inf. CPU",
                "ECO CPU (ORIGINAL)",
                "Inf. & ECO CPU (GT)",
                "Inf. & ECO CPU (PRED)",
                "Inf. & ECO CPU (POST)",
            ]:
                total_row[col] = ""

        # Add total row
        table_data.append(total_row)

        # Export comparison table CSV
        with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=columns)
            writer.writeheader()
            writer.writerows(table_data)

    def _export_design_summary_table_tex(
        self, routing_results: List[Dict], all_results: List[Dict]
    ):
        """Export design summary table as LaTeX file."""
        output_file = self.output_dir / "design_summary_table.tex"

        # Get density from ORIGINAL meta_type results
        density_lookup = {}
        for result in all_results:
            if result["meta_type"] == "ORIGINAL" and result["Density"]:
                density_lookup[result["design"]] = result["Density"]

        # Create LaTeX content
        latex_content = r"""\begin{table}[ht]
	\centering
	\caption{Design statistics.}
	\label{tab:design_stat}
	\begin{tabularx}{0.98\linewidth}{cCCCC}
		\hline
		\textbf{Designs} & \textbf{\#Comb.} & \textbf{\#FFs} & \textbf{\#Nets} & \textbf{Util} \\
		\hline
"""

        # Add data rows
        # Sort routing_results by Net Num in ascending order
        sorted_routing_results = sorted(
            routing_results,
            key=lambda x: int(x.get("Net Num", "0"))
            if x.get("Net Num", "").isdigit()
            else 0,
        )

        for routing_result in sorted_routing_results:
            design = routing_result["Design"]
            comb_cells = routing_result.get("Combinational Cells", "")
            seq_cells = routing_result.get("Sequential Cells", "")
            nets = routing_result.get("Net Num", "")
            density = density_lookup.get(design, "")

            # Format density to show as decimal (divide by 100)
            if density:
                try:
                    density_val = f"{float(density) / 100:.2f}"
                except (ValueError, TypeError):
                    density_val = density
            else:
                density_val = ""

            safe_design = design.replace("_", "\_")
            latex_content += f"\t\t{safe_design} & {comb_cells} & {seq_cells} & {nets} & {density_val} \\\n"

        latex_content += r"""		\midrule
	\end{tabularx}
\end{table}"""

        # Write LaTeX file
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(latex_content)

    def _export_design_comparison_table_tex(
        self, all_results: List[Dict], routing_results: List[Dict]
    ):
        """Export design comparison table as LaTeX file."""
        output_file = self.output_dir / "design_comparison_table.tex"

        # Check if TNS columns are all zero
        tns_all_zero = True
        for result in all_results:
            if result["design"] != "Total":
                for meta_type in ["ORIGINAL", "GT", "PRED", "POST"]:
                    if result["meta_type"] == meta_type:
                        tns_val = result.get("TNS", "")
                        if tns_val and tns_val != "0.000":
                            try:
                                if float(tns_val) != 0.0:
                                    tns_all_zero = False
                                    break
                            except (ValueError, TypeError):
                                pass
            if not tns_all_zero:
                break

        if tns_all_zero:
            logging.warning(
                "All TNS values are zero, omitting TNS columns from LaTeX table."
            )

        # Create routing lookup
        routing_lookup = {r["Design"]: r for r in routing_results}

        # Group all_results by design
        design_data = {}
        for result in all_results:
            design = result["design"]
            meta_type = result["meta_type"]
            if design not in design_data:
                design_data[design] = {}
            design_data[design][meta_type] = result

        def colored_header_label(label: str) -> str:
            color_map = {
                "Com.": r"\cellcolor{gray!6} Com.",
                "iPCL-R": r"\cellcolor{blue!7} iPCL-R",
                "PO": r"\cellcolor{green!6} PO",
            }
            return color_map.get(label, label)

        # Define column structure - without TNS since all values are zero
        if tns_all_zero:
            primary_cols = [
                r"\textbf{\#Vias}",
                r"\textbf{WL (\textit{\textmu m})}",
                r"\textbf{Sw. Pwr (\textit{\textmu W})}",
                r"\textbf{Pwr (\textit{\textmu W})}",
                r"\textbf{WNS (\textit{ps})}",
                r"\textbf{Runtime (s)}",
            ]
        else:
            primary_cols = [
                r"\textbf{\#Vias}",
                r"\textbf{WL (\textit{\textmu m})}",
                r"\textbf{Sw. Pwr (\textit{\textmu W})}",
                r"\textbf{Pwr (\textit{\textmu W})}",
                r"\textbf{WNS (\textit{ps})}",
                r"\textbf{TNS (\textit{ps})}",
                r"Runtime (s)",
            ]

        runtime_headers = [
            r"\textcolor{gray}{Inf.}",
            colored_header_label("Com."),
            colored_header_label("iPCL-R"),
            colored_header_label("PO"),
        ]
        runtime_col_count = len(runtime_headers)

        col_sections = ["c"]
        col_sections.extend(["ccc"] * (len(primary_cols) - 1))
        col_sections.append("c" * runtime_col_count)
        col_specs = "|".join(col_sections)

        # Create LaTeX content
        latex_content = (
            r"""\begin{table*}[!t]
	\setlength{\tabcolsep}{2.65pt}
	\centering
	\caption{{Comparison of design metrics.
 ``Com.'', ``iPCL-R'', and ``PO'' represent the Innovus routing flow, iPCL-R-0.4B, and the results of iPCL-R-0.4B with post-optimization, respectively.
 ``Inf.'' denotes model inference time.}}
	\label{tab:result_comparison}
	\resizebox{\textwidth}{!} { \footnotesize
		\begin{tabularx}{\linewidth}{"""
            + col_specs
            + r"""}
		\toprule
		\textbf{Unseen}"""
        )

        runtime_span = runtime_col_count

        # Add primary column headers (first header row)
        for col in primary_cols:
            if "Runtime" in col:
                latex_content += f" & \\multicolumn{{{runtime_span}}}{{c}}{{{col}}}"
            else:
                latex_content += f" & \\multicolumn{{3}}{{c|}}{{{col}}}"

        total_columns = 1 + 3 * (len(primary_cols) - 1) + runtime_span
        latex_content += (
            r""" \\\cline{2-"""
            + str(total_columns)
            + r"""}
		"""
        )

        # Add secondary column headers
        latex_content += r"\textbf{Designs}"
        header_segments = []
        for col in primary_cols:
            if "Runtime" in col:
                header_segments.append(" & ".join(runtime_headers))
            else:
                trio = [
                    colored_header_label("Com."),
                    colored_header_label("iPCL-R"),
                    colored_header_label("PO"),
                ]
                header_segments.append(" & ".join(trio))

        latex_content += " & " + " & ".join(header_segments)
        latex_content += r""" \\
		\midrule
"""
        # Add data rows (excluding Total row for now)
        # Sort designs by Net Num in ascending order
        sorted_designs = []
        for design, meta_data in design_data.items():
            if design == "Total":
                continue
            routing_data = routing_lookup.get(design, {})
            net_num = routing_data.get("Net Num", "0")
            net_num_int = int(net_num) if net_num.isdigit() else 0
            sorted_designs.append((design, meta_data, net_num_int))

        sorted_designs.sort(key=lambda x: x[2])  # Sort by net_num_int

        # Collect all values for finding best values
        all_values = {
            "vias_orig": [],
            "vias_pred": [],
            "vias_post": [],
            "wl_orig": [],
            "wl_pred": [],
            "wl_post": [],
            "sw_orig": [],
            "sw_pred": [],
            "sw_post": [],
            "pwr_orig": [],
            "pwr_pred": [],
            "pwr_post": [],
            "wns_orig": [],
            "wns_pred": [],
            "wns_post": [],
            "route_cpu": [],
            "inf_eco_pred": [],
            "inf_eco_post": [],
        }

        # Helper functions for consistent formatting
        def format_power(val):
            if val:
                try:
                    return int(float(val) * 1000)
                except (ValueError, TypeError):
                    return None
            return None

        def format_wns(val):
            if val:
                try:
                    return int(float(val) * 1000)
                except (ValueError, TypeError):
                    return None
            return None

        def format_runtime_for_comparison(val):
            if val:
                try:
                    num = float(val)
                    formatted = f"{num:.2f}"
                    # Remove trailing zeros and decimal point if necessary
                    if "." in formatted:
                        formatted = formatted.rstrip("0").rstrip(".")
                    return float(formatted)
                except (ValueError, TypeError):
                    return None
            return None

        # Collect values for comparison based on formatted values
        for design, meta_data, _ in sorted_designs:
            routing_data = routing_lookup.get(design, {})

            # Collect metric values for comparison (using same formatting as display)
            vias_orig = meta_data.get("ORIGINAL", {}).get("Total Vias", "")
            vias_pred = meta_data.get("PRED", {}).get("Total Vias", "")
            vias_post = meta_data.get("POST", {}).get("Total Vias", "")

            wl_orig = meta_data.get("ORIGINAL", {}).get("Total Wirelength", "")
            wl_pred = meta_data.get("PRED", {}).get("Total Wirelength", "")
            wl_post = meta_data.get("POST", {}).get("Total Wirelength", "")

            sw_orig = meta_data.get("ORIGINAL", {}).get("Switching Power", "")
            sw_pred = meta_data.get("PRED", {}).get("Switching Power", "")
            sw_post = meta_data.get("POST", {}).get("Switching Power", "")

            pwr_orig = meta_data.get("ORIGINAL", {}).get("Total Power", "")
            pwr_pred = meta_data.get("PRED", {}).get("Total Power", "")
            pwr_post = meta_data.get("POST", {}).get("Total Power", "")

            wns_orig = meta_data.get("ORIGINAL", {}).get("WNS", "")
            wns_pred = meta_data.get("PRED", {}).get("WNS", "")
            wns_post = meta_data.get("POST", {}).get("WNS", "")

            route_cpu = routing_data.get("Route Total CPU", "")

            # Calculate inf+eco times for PRED and POST
            inf_cpu = None
            if routing_data.get("Net Num"):
                try:
                    net_num = float(routing_data["Net Num"])
                    inf_cpu = net_num * 0.015923
                except (ValueError, TypeError):
                    pass

            inf_eco_pred = None
            inf_eco_post = None
            eco_pred = meta_data.get("PRED", {}).get("ECO Total CPU", "")
            eco_post = meta_data.get("POST", {}).get("ECO Total CPU", "")

            if inf_cpu and eco_pred:
                try:
                    inf_eco_pred = inf_cpu + float(eco_pred)
                except (ValueError, TypeError):
                    pass

            if inf_cpu and eco_post:
                try:
                    inf_eco_post = inf_cpu + float(eco_post)
                except (ValueError, TypeError):
                    pass

            # Store values for comparison using formatted values
            def safe_int(val):
                try:
                    return int(val) if val else None
                except (ValueError, TypeError):
                    return None

            # Store formatted values for comparison
            all_values["vias_orig"].append(safe_int(vias_orig))
            all_values["vias_pred"].append(safe_int(vias_pred))
            all_values["vias_post"].append(safe_int(vias_post))
            all_values["wl_orig"].append(safe_int(wl_orig))
            all_values["wl_pred"].append(safe_int(wl_pred))
            all_values["wl_post"].append(safe_int(wl_post))
            all_values["sw_orig"].append(format_power(sw_orig))
            all_values["sw_pred"].append(format_power(sw_pred))
            all_values["sw_post"].append(format_power(sw_post))
            all_values["pwr_orig"].append(format_power(pwr_orig))
            all_values["pwr_pred"].append(format_power(pwr_pred))
            all_values["pwr_post"].append(format_power(pwr_post))
            all_values["wns_orig"].append(format_wns(wns_orig))
            all_values["wns_pred"].append(format_wns(wns_pred))
            all_values["wns_post"].append(format_wns(wns_post))
            all_values["route_cpu"].append(format_runtime_for_comparison(route_cpu))
            all_values["inf_eco_pred"].append(
                format_runtime_for_comparison(inf_eco_pred)
            )
            all_values["inf_eco_post"].append(
                format_runtime_for_comparison(inf_eco_post)
            )

        # Find best values for each metric
        def find_best_indices(values, lower_is_better=True):
            valid_values = [v for v in values if v is not None]
            if not valid_values:
                return set()
            best_val = min(valid_values) if lower_is_better else max(valid_values)
            return {i for i, v in enumerate(values) if v == best_val}

        # For each row metric, find which columns have the best values
        best_indices = {}
        for i in range(len(sorted_designs)):
            best_indices[i] = {
                "vias": find_best_indices(
                    [
                        all_values["vias_orig"][i],
                        all_values["vias_pred"][i],
                        all_values["vias_post"][i],
                    ],
                    True,
                ),
                "wl": find_best_indices(
                    [
                        all_values["wl_orig"][i],
                        all_values["wl_pred"][i],
                        all_values["wl_post"][i],
                    ],
                    True,
                ),
                "sw": find_best_indices(
                    [
                        all_values["sw_orig"][i],
                        all_values["sw_pred"][i],
                        all_values["sw_post"][i],
                    ],
                    True,
                ),
                "pwr": find_best_indices(
                    [
                        all_values["pwr_orig"][i],
                        all_values["pwr_pred"][i],
                        all_values["pwr_post"][i],
                    ],
                    True,
                ),
                "wns": find_best_indices(
                    [
                        all_values["wns_orig"][i],
                        all_values["wns_pred"][i],
                        all_values["wns_post"][i],
                    ],
                    False,
                ),
                "runtime": find_best_indices(
                    [
                        all_values["route_cpu"][i],
                        all_values["inf_eco_pred"][i],
                        all_values["inf_eco_post"][i],
                    ],
                    True,
                ),
            }

        def make_bold_if_best(value, is_best):
            if is_best and value:
                return f"\\textbf{{{value}}}"
            return str(value)

        for row_idx, (design, meta_data, _) in enumerate(sorted_designs):
            routing_data = routing_lookup.get(design, {})
            safe_design = design.replace("_", "\\_")
            latex_content += f"\t\t{safe_design}"

            # #Vias (remove GT column)
            vias_orig = meta_data.get("ORIGINAL", {}).get("Total Vias", "")
            vias_pred = meta_data.get("PRED", {}).get("Total Vias", "")
            vias_post = meta_data.get("POST", {}).get("Total Vias", "")

            best_vias = best_indices[row_idx]["vias"]
            vias_orig_fmt = make_bold_if_best(vias_orig, 0 in best_vias)
            vias_pred_fmt = make_bold_if_best(vias_pred, 1 in best_vias)
            vias_post_fmt = make_bold_if_best(vias_post, 2 in best_vias)

            latex_content += f" & {vias_orig_fmt} & {vias_pred_fmt} & {vias_post_fmt}"

            # Wirelength (remove GT column)
            wl_orig = meta_data.get("ORIGINAL", {}).get("Total Wirelength", "")
            wl_pred = meta_data.get("PRED", {}).get("Total Wirelength", "")
            wl_post = meta_data.get("POST", {}).get("Total Wirelength", "")

            best_wl = best_indices[row_idx]["wl"]
            wl_orig_fmt = make_bold_if_best(wl_orig, 0 in best_wl)
            wl_pred_fmt = make_bold_if_best(wl_pred, 1 in best_wl)
            wl_post_fmt = make_bold_if_best(wl_post, 2 in best_wl)

            latex_content += f" & {wl_orig_fmt} & {wl_pred_fmt} & {wl_post_fmt}"

            # Switching Power (multiply by 1000 and show as integer, remove GT column)
            sw_orig = meta_data.get("ORIGINAL", {}).get("Switching Power", "")
            sw_pred = meta_data.get("PRED", {}).get("Switching Power", "")
            sw_post = meta_data.get("POST", {}).get("Switching Power", "")

            def format_power(val):
                if val:
                    try:
                        return f"{int(float(val) * 1000)}"
                    except (ValueError, TypeError):
                        return val
                return ""

            sw_orig_val = format_power(sw_orig)
            sw_pred_val = format_power(sw_pred)
            sw_post_val = format_power(sw_post)

            # Compare original float values for determining best, not formatted strings
            best_sw = best_indices[row_idx]["sw"]
            sw_orig_fmt = make_bold_if_best(sw_orig_val, 0 in best_sw)
            sw_pred_fmt = make_bold_if_best(sw_pred_val, 1 in best_sw)
            sw_post_fmt = make_bold_if_best(sw_post_val, 2 in best_sw)

            latex_content += f" & {sw_orig_fmt} & {sw_pred_fmt} & {sw_post_fmt}"

            # Total Power (multiply by 1000 and show as integer, remove GT column)
            pwr_orig = meta_data.get("ORIGINAL", {}).get("Total Power", "")
            pwr_pred = meta_data.get("PRED", {}).get("Total Power", "")
            pwr_post = meta_data.get("POST", {}).get("Total Power", "")

            pwr_orig_val = format_power(pwr_orig)
            pwr_pred_val = format_power(pwr_pred)
            pwr_post_val = format_power(pwr_post)

            # Compare original float values for determining best, not formatted strings
            best_pwr = best_indices[row_idx]["pwr"]
            pwr_orig_fmt = make_bold_if_best(pwr_orig_val, 0 in best_pwr)
            pwr_pred_fmt = make_bold_if_best(pwr_pred_val, 1 in best_pwr)
            pwr_post_fmt = make_bold_if_best(pwr_post_val, 2 in best_pwr)

            latex_content += f" & {pwr_orig_fmt} & {pwr_pred_fmt} & {pwr_post_fmt}"

            # WNS (multiply by 1000 and show as integer, remove GT column)
            wns_orig = meta_data.get("ORIGINAL", {}).get("WNS", "")
            wns_pred = meta_data.get("PRED", {}).get("WNS", "")
            wns_post = meta_data.get("POST", {}).get("WNS", "")

            def format_wns(val):
                if val:
                    try:
                        return f"{int(float(val) * 1000)}"
                    except (ValueError, TypeError):
                        return val
                return ""

            wns_orig_val = format_wns(wns_orig)
            wns_pred_val = format_wns(wns_pred)
            wns_post_val = format_wns(wns_post)

            # Compare original float values for determining best, not formatted strings
            best_wns = best_indices[row_idx]["wns"]
            wns_orig_fmt = make_bold_if_best(wns_orig_val, 0 in best_wns)
            wns_pred_fmt = make_bold_if_best(wns_pred_val, 1 in best_wns)
            wns_post_fmt = make_bold_if_best(wns_post_val, 2 in best_wns)

            latex_content += f" & {wns_orig_fmt} & {wns_pred_fmt} & {wns_post_fmt}"

            # Runtime columns (Inf., Com., iPCL-R, PO)
            def format_runtime(val):
                if val:
                    try:
                        num = float(val)
                        formatted = f"{num:.2f}"
                        # Remove trailing zeros and decimal point if necessary
                        if "." in formatted:
                            formatted = formatted.rstrip("0").rstrip(".")
                        return formatted
                    except (ValueError, TypeError):
                        return val
                return ""

            route_cpu = routing_data.get("Route Total CPU", "")
            inf_cpu = ""
            if routing_data.get("Net Num"):
                try:
                    net_num = float(routing_data["Net Num"])
                    inf_cpu = f"{net_num * 0.015923:.2f}"
                    # Format inf_cpu with the same rule
                    inf_cpu = format_runtime(inf_cpu)
                except (ValueError, TypeError):
                    pass

            # PRED, POST inf+eco times
            eco_pred = meta_data.get("PRED", {}).get("ECO Total CPU", "")
            eco_post = meta_data.get("POST", {}).get("ECO Total CPU", "")

            inf_eco_pred = ""
            inf_eco_post = ""

            if inf_cpu and eco_pred:
                try:
                    total_cpu = float(inf_cpu) + float(eco_pred)
                    inf_eco_pred = format_runtime(str(total_cpu))
                except (ValueError, TypeError):
                    pass

            if inf_cpu and eco_post:
                try:
                    total_cpu = float(inf_cpu) + float(eco_post)
                    inf_eco_post = format_runtime(str(total_cpu))
                except (ValueError, TypeError):
                    pass

            # Runtime formatting with bold for best values (only comparing Com., iPCL-R, PO)
            route_cpu_val = format_runtime(route_cpu)
            best_runtime = best_indices[row_idx]["runtime"]
            route_cpu_fmt = make_bold_if_best(route_cpu_val, 0 in best_runtime)
            inf_eco_pred_fmt = make_bold_if_best(inf_eco_pred, 1 in best_runtime)
            inf_eco_post_fmt = make_bold_if_best(inf_eco_post, 2 in best_runtime)

            # Format gray values
            inf_cpu_gray = f"\\textcolor{{gray}}{{{inf_cpu}}}" if inf_cpu else ""
            latex_content += (
                f" & {inf_cpu_gray} & {route_cpu_fmt}"
                f" & {inf_eco_pred_fmt} & {inf_eco_post_fmt}"
            )
            latex_content += r" \\" + "\n"

        # Calculate Total row with actual data
        # Get ORIGINAL baseline values for normalization
        original_totals = {}
        route_cpu_total = 0

        for design, meta_data in design_data.items():
            if design == "Total":
                continue
            original_data = meta_data.get("ORIGINAL", {})
            routing_data = routing_lookup.get(design, {})

            # Sum ORIGINAL values for normalization base
            for metric in ["#Vias", "Wirelength", "Switching Power", "Power", "WNS"]:
                if metric == "#Vias":
                    value = original_data.get("Total Vias")
                elif metric == "Wirelength":
                    value = original_data.get("Total Wirelength")
                elif metric == "Switching Power":
                    value = original_data.get("Switching Power")
                elif metric == "Power":
                    value = original_data.get("Total Power")
                else:
                    value = original_data.get(metric)

                if value:
                    try:
                        original_totals[metric] = original_totals.get(
                            metric, 0
                        ) + float(value)
                    except (ValueError, TypeError):
                        pass

            # Sum Route CPU for runtime normalization
            route_cpu = routing_data.get("Route Total CPU")
            if route_cpu:
                try:
                    route_cpu_total += float(route_cpu)
                except (ValueError, TypeError):
                    pass

        # Calculate normalized totals for #Vias
        vias_totals = {}
        for meta_type in ["ORIGINAL", "PRED", "POST"]:
            metric_total = 0
            for design, meta_data in design_data.items():
                if design == "Total":
                    continue
                data = meta_data.get(meta_type, {})
                value = data.get("Total Vias")
                if value:
                    try:
                        metric_total += float(value)
                    except (ValueError, TypeError):
                        pass
            if original_totals.get("#Vias", 0) > 0:
                vias_totals[meta_type] = metric_total / original_totals["#Vias"]
            else:
                vias_totals[meta_type] = 0

        # Calculate normalized totals for Wirelength
        wl_totals = {}
        for meta_type in ["ORIGINAL", "PRED", "POST"]:
            metric_total = 0
            for design, meta_data in design_data.items():
                if design == "Total":
                    continue
                data = meta_data.get(meta_type, {})
                value = data.get("Total Wirelength")
                if value:
                    try:
                        metric_total += float(value)
                    except (ValueError, TypeError):
                        pass
            if original_totals.get("Wirelength", 0) > 0:
                wl_totals[meta_type] = metric_total / original_totals["Wirelength"]
            else:
                wl_totals[meta_type] = 0

        # Calculate normalized totals for Switching Power
        sw_pwr_totals = {}
        for meta_type in ["ORIGINAL", "PRED", "POST"]:
            metric_total = 0
            for design, meta_data in design_data.items():
                if design == "Total":
                    continue
                data = meta_data.get(meta_type, {})
                value = data.get("Switching Power")
                if value:
                    try:
                        metric_total += float(value)
                    except (ValueError, TypeError):
                        pass
            if original_totals.get("Switching Power", 0) > 0:
                sw_pwr_totals[meta_type] = (
                    metric_total / original_totals["Switching Power"]
                )
            else:
                sw_pwr_totals[meta_type] = 0

        # Calculate normalized totals for Total Power
        pwr_totals = {}
        for meta_type in ["ORIGINAL", "PRED", "POST"]:
            metric_total = 0
            for design, meta_data in design_data.items():
                if design == "Total":
                    continue
                data = meta_data.get(meta_type, {})
                value = data.get("Total Power")
                if value:
                    try:
                        metric_total += float(value)
                    except (ValueError, TypeError):
                        pass
            if original_totals.get("Power", 0) > 0:
                pwr_totals[meta_type] = metric_total / original_totals["Power"]
            else:
                pwr_totals[meta_type] = 0

        # Calculate normalized totals for WNS
        wns_totals = {}
        for meta_type in ["ORIGINAL", "PRED", "POST"]:
            metric_total = 0
            for design, meta_data in design_data.items():
                if design == "Total":
                    continue
                data = meta_data.get(meta_type, {})
                value = data.get("WNS")
                if value:
                    try:
                        metric_total += float(value)
                    except (ValueError, TypeError):
                        pass
            if original_totals.get("WNS", 0) > 0:
                wns_totals[meta_type] = metric_total / original_totals["WNS"]
            else:
                wns_totals[meta_type] = 0

        # Calculate runtime totals
        runtime_totals = {}
        if route_cpu_total > 0:
            runtime_totals["route"] = 1.00

            # Calculate total inf CPU
            total_inf_cpu = 0
            for design, _ in design_data.items():
                if design == "Total":
                    continue
                routing_data = routing_lookup.get(design, {})
                if routing_data.get("Net Num"):
                    try:
                        net_num = float(routing_data["Net Num"])
                        total_inf_cpu += net_num * 0.015923
                    except (ValueError, TypeError):
                        pass

            runtime_totals["inf"] = (
                total_inf_cpu / route_cpu_total if total_inf_cpu > 0 else 0
            )

            # Calculate total inf + ECO CPU for each meta_type (Com., iPCL-R, PO)
            for meta_type in ["PRED", "POST"]:
                total_inf_eco_cpu = total_inf_cpu
                for design, meta_data in design_data.items():
                    if design == "Total":
                        continue
                    data = meta_data.get(meta_type, {})
                    eco_cpu = data.get("ECO Total CPU")
                    if eco_cpu:
                        try:
                            total_inf_eco_cpu += float(eco_cpu)
                        except (ValueError, TypeError):
                            pass

                key = f"inf_eco_{meta_type.lower()}"
                runtime_totals[key] = (
                    total_inf_eco_cpu / route_cpu_total if total_inf_eco_cpu > 0 else 0
                )

        # Format totals for LaTeX (3 significant digits)
        def format_total(val):
            if val == 0:
                return "0.00"
            # Format to 3 significant digits
            if abs(val) >= 100:
                return f"{val:.1f}"
            elif abs(val) >= 10:
                return f"{val:.2f}"
            else:
                return f"{val:.3f}"

        # Find best values for Total row (comparing ORIGINAL, PRED, POST)
        total_best = {
            "vias": find_best_indices(
                [
                    vias_totals.get("ORIGINAL", 1.0),
                    vias_totals.get("PRED", 0),
                    vias_totals.get("POST", 0),
                ],
                True,
            ),
            "wl": find_best_indices(
                [
                    wl_totals.get("ORIGINAL", 1.0),
                    wl_totals.get("PRED", 0),
                    wl_totals.get("POST", 0),
                ],
                True,
            ),
            "sw": find_best_indices(
                [
                    sw_pwr_totals.get("ORIGINAL", 1.0),
                    sw_pwr_totals.get("PRED", 0),
                    sw_pwr_totals.get("POST", 0),
                ],
                True,
            ),
            "pwr": find_best_indices(
                [
                    pwr_totals.get("ORIGINAL", 1.0),
                    pwr_totals.get("PRED", 0),
                    pwr_totals.get("POST", 0),
                ],
                True,
            ),
            "wns": find_best_indices(
                [
                    wns_totals.get("ORIGINAL", 1.0),
                    wns_totals.get("PRED", 0),
                    wns_totals.get("POST", 0),
                ],
                False,
            ),
            "runtime": find_best_indices(
                [
                    runtime_totals.get("route", 1.0),
                    runtime_totals.get("inf_eco_pred", 0),
                    runtime_totals.get("inf_eco_post", 0),
                ],
                True,
            ),
        }

        # Add Total row with calculated values
        latex_content += r"""		\midrule
		Avg."""

        # Add Vias columns (Com., iPCL-R, PO)
        vias_orig_total = make_bold_if_best(
            format_total(vias_totals.get("ORIGINAL", 1.0)), 0 in total_best["vias"]
        )
        vias_pred_total = make_bold_if_best(
            format_total(vias_totals.get("PRED", 0)), 1 in total_best["vias"]
        )
        vias_post_total = make_bold_if_best(
            format_total(vias_totals.get("POST", 0)), 2 in total_best["vias"]
        )
        latex_content += f" & {vias_orig_total} & {vias_pred_total} & {vias_post_total}"

        # Add Wirelength columns (Com., iPCL-R, PO)
        wl_orig_total = make_bold_if_best(
            format_total(wl_totals.get("ORIGINAL", 1.0)), 0 in total_best["wl"]
        )
        wl_pred_total = make_bold_if_best(
            format_total(wl_totals.get("PRED", 0)), 1 in total_best["wl"]
        )
        wl_post_total = make_bold_if_best(
            format_total(wl_totals.get("POST", 0)), 2 in total_best["wl"]
        )
        latex_content += f" & {wl_orig_total} & {wl_pred_total} & {wl_post_total}"

        # Add Switching Power columns (Com., iPCL-R, PO)
        sw_orig_total = make_bold_if_best(
            format_total(sw_pwr_totals.get("ORIGINAL", 1.0)), 0 in total_best["sw"]
        )
        sw_pred_total = make_bold_if_best(
            format_total(sw_pwr_totals.get("PRED", 0)), 1 in total_best["sw"]
        )
        sw_post_total = make_bold_if_best(
            format_total(sw_pwr_totals.get("POST", 0)), 2 in total_best["sw"]
        )
        latex_content += f" & {sw_orig_total} & {sw_pred_total} & {sw_post_total}"

        # Add Total Power columns (Com., iPCL-R, PO)
        pwr_orig_total = make_bold_if_best(
            format_total(pwr_totals.get("ORIGINAL", 1.0)), 0 in total_best["pwr"]
        )
        pwr_pred_total = make_bold_if_best(
            format_total(pwr_totals.get("PRED", 0)), 1 in total_best["pwr"]
        )
        pwr_post_total = make_bold_if_best(
            format_total(pwr_totals.get("POST", 0)), 2 in total_best["pwr"]
        )
        latex_content += f" & {pwr_orig_total} & {pwr_pred_total} & {pwr_post_total}"

        # Add WNS columns (Com., iPCL-R, PO)
        wns_orig_total = make_bold_if_best(
            format_total(wns_totals.get("ORIGINAL", 1.0)), 0 in total_best["wns"]
        )
        wns_pred_total = make_bold_if_best(
            format_total(wns_totals.get("PRED", 0)), 1 in total_best["wns"]
        )
        wns_post_total = make_bold_if_best(
            format_total(wns_totals.get("POST", 0)), 2 in total_best["wns"]
        )
        latex_content += f" & {wns_orig_total} & {wns_pred_total} & {wns_post_total}"

        # Add Runtime columns (Inf., Com., iPCL-R, PO)
        inf_total_gray = (
            f"\\textcolor{{gray}}{{{format_total(runtime_totals.get('inf', 0))}}}"
        )

        route_total_fmt = make_bold_if_best(
            format_total(runtime_totals.get("route", 1.0)), 0 in total_best["runtime"]
        )
        inf_eco_pred_total_fmt = make_bold_if_best(
            format_total(runtime_totals.get("inf_eco_pred", 0)),
            1 in total_best["runtime"],
        )
        inf_eco_post_total_fmt = make_bold_if_best(
            format_total(runtime_totals.get("inf_eco_post", 0)),
            2 in total_best["runtime"],
        )

        latex_content += (
            f" & {inf_total_gray} & {route_total_fmt}"
            f" & {inf_eco_pred_total_fmt} & {inf_eco_post_total_fmt}"
        )

        latex_content += r" \\" + "\n"
        latex_content += r"""		\bottomrule
	\end{tabularx}
}
\end{table*}"""

        # Write LaTeX file
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(latex_content)

    def _export_flattened_csv(self, results: List[Dict]):
        """Export flattened CSV file without meta_type column."""
        output_file = self.output_dir / "design_summary_flatten.csv"

        # Define metric columns (excluding 'design' and 'meta_type')
        metric_columns = [
            "ECO Total CPU",
            "ECO Real",
            "WNS",
            "TNS",
            "Density",
            "Internal Power",
            "Switching Power",
            "Leakage Power",
            "Total Power",
            "Total Wirelength",
            "Total Vias",
        ]

        # Group data by design
        design_data = {}
        for result in results:
            design = result["design"]
            meta_type = result["meta_type"]

            if design not in design_data:
                design_data[design] = {}

            # Store metrics with meta_type suffix
            for metric in metric_columns:
                flattened_key = f"{metric} ({meta_type})"
                design_data[design][flattened_key] = result[metric]

        # Create flattened results
        flattened_results = []
        for design, metrics in design_data.items():
            row = {"design": design}
            row.update(metrics)
            flattened_results.append(row)

        # Determine column order for flattened CSV
        if flattened_results:
            # Get all metric columns with meta_type suffixes, sorted
            all_metric_keys = set()
            for row in flattened_results:
                all_metric_keys.update(key for key in row.keys() if key != "design")

            # Sort by metric name first, then by meta_type
            def sort_key(col_name):
                if " (" not in col_name:
                    return (col_name, "")
                metric_name, meta_type = col_name.rsplit(" (", 1)
                meta_type = meta_type.rstrip(")")
                # Order meta_types: ORIGINAL, GT, PRED, POST
                meta_order = {"ORIGINAL": 0, "GT": 1, "PRED": 2, "POST": 3}
                return (metric_name, meta_order.get(meta_type, 999))

            sorted_metric_keys = sorted(all_metric_keys, key=sort_key)
            flattened_columns = ["design"] + sorted_metric_keys

            # Write flattened CSV
            with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=flattened_columns)
                writer.writeheader()
                writer.writerows(flattened_results)

    def _generate_radar_chart(self, results: List[Dict], routing_results: List[Dict]):
        """Generate radar chart comparing ORIGINAL, PRED, POST across designs."""
        # Filter for ORIGINAL, PRED, POST only
        filtered_results = [
            r for r in results if r["meta_type"] in ["ORIGINAL", "PRED", "POST"]
        ]

        if not filtered_results:
            logging.warning("No ORIGINAL, PRED, POST data found for radar chart.")
            return

        # Create routing lookup for Route CPU data
        routing_lookup = {r["Design"]: r for r in routing_results}

        # Convert to DataFrame
        df = pd.DataFrame(filtered_results)

        # Define metrics to compare - updated list
        metrics = [
            "Total Vias",
            "Total Wirelength",
            "Switching Power",
            "Total Power",
            "WNS",
            "Total CPU",
        ]

        # Create a modified dataframe with Total CPU column
        df_modified = df.copy()

        # Add Total CPU column
        df_modified["Total CPU"] = None
        for idx, row in df_modified.iterrows():
            if row["meta_type"] == "ORIGINAL":
                # Use Route CPU for ORIGINAL
                routing_data = routing_lookup.get(row["design"], {})
                route_cpu = routing_data.get("Route Total CPU")
                if route_cpu:
                    df_modified.loc[idx, "Total CPU"] = route_cpu
            else:
                # Use ECO Total CPU for PRED and POST
                eco_cpu = row.get("ECO Total CPU")
                if eco_cpu:
                    df_modified.loc[idx, "Total CPU"] = eco_cpu

        # Convert string values to float
        df_converted = df_modified.copy()
        for metric in metrics:
            df_converted[metric] = pd.to_numeric(df_converted[metric], errors="coerce")

        # Aggregate by meta_type (sum across designs for each meta_type)
        agg_data = df_converted.groupby("meta_type")[metrics].sum().reset_index()

        # Normalize data: scale each metric to [0, 1] with max value = 1.0
        normalized_data = agg_data.copy()
        for metric in metrics:
            max_val = agg_data[metric].max()
            if max_val > 0:
                normalized_data[metric] = agg_data[metric] / max_val

        # Set up the radar chart
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection="polar"))

        # Calculate angles for each metric
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle

        # Define colors for each meta_type
        meta_type_order = ["ORIGINAL", "PRED", "POST"]

        # Plot each meta_type
        for i, meta_type in enumerate(meta_type_order):
            if meta_type in normalized_data["meta_type"].values:
                # Get normalized values for this meta_type
                values = (
                    normalized_data[normalized_data["meta_type"] == meta_type][metrics]
                    .iloc[0]
                    .tolist()
                )
                values += values[:1]  # Complete the circle

                # Plot the line and fill
                ax.plot(angles, values, linewidth=2, label=meta_type)
                ax.fill(angles, values, alpha=0.4)

        # Determine appropriate minimum scale based on data
        all_values = normalized_data[metrics].values.flatten()
        min_data = np.min(all_values)

        # Choose minimum scale from candidates [0, 0.25, 0.5, 0.75]
        scale_candidates = [0, 0.25, 0.5, 0.75]
        min_scale = 0
        for candidate in scale_candidates:
            if min_data > candidate:
                min_scale = candidate
            else:
                break

        # Set up scale ticks
        if min_scale == 0:
            y_ticks = [0.2, 0.4, 0.6, 0.8, 1.0]
            y_labels = ["0.2", "0.4", "0.6", "0.8", "1.0"]
        elif min_scale == 0.25:
            y_ticks = [0.25, 0.4, 0.6, 0.8, 1.0]
            y_labels = ["0.25", "0.4", "0.6", "0.8", "1.0"]
        elif min_scale == 0.5:
            y_ticks = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            y_labels = ["0.5", "0.6", "0.7", "0.8", "0.9", "1.0"]
        else:  # min_scale == 0.75
            y_ticks = [0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
            y_labels = ["0.75", "0.8", "0.85", "0.9", "0.95", "1.0"]

        # Update metric labels for display (remove # prefix for cleaner display)
        display_metrics = [
            "#Vias",
            "Wirelength",
            "Switching Power",
            "Total Power",
            "WNS",
            "Total CPU",
        ]

        # Customize the chart
        ax.set_ylim(min_scale, 1)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels, fontsize=20)

        # Improve aesthetic
        ax.set_thetagrids(np.degrees(angles[:-1]), display_metrics, fontsize=20)

        # Add title and legend
        plt.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1), fontsize=20)

        # Save the plot
        output_file = self.output_dir / "design_summary_radar.pdf"
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()  # Close to free memory


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract design metrics from innovus.log files and generate summary CSV"
    )
    parser.add_argument(
        "--base",
        "-b",
        default=Path("/data2/project_share/liweiguo/rt_gen/output"),
        type=Path,
        help="Base directory containing design folders (default: output)",
    )
    parser.add_argument(
        "--out",
        "-o",
        default=Path("/data2/project_share/liweiguo/rt_gen/output"),
        type=Path,
        help="Output directory for generated CSV file (default: output)",
    )
    return parser.parse_args()


def main():
    setup_logging()

    """Main entry point for the design summary script."""
    args = parse_arguments()

    processor = DesignSummaryProcessor(args.base, args.out)
    processor.process_all_designs()


if __name__ == "__main__":
    main()
