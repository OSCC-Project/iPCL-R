#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   main.py
@Time    :   2025/09/02 16:39:44
@Author  :   Dawn Li
@Version :   1.0
@Contact :   dawnli619215645@gmail.com
@Desc    :   Main flow for DRC analysis
"""

import argparse
import logging
import os
from abc import ABC
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

from data_synthesis import setup_aieda
from flow.utils import setup_logging

if setup_aieda():
    from aieda.flows import DataGeneration, DbFlow
    from aieda.third_party.tools import (
        DataFeatureInnovus,
        InnovusDRC,
        WorkspaceInner,
    )


class MetadataType(Enum):
    GT = "GT"
    PRED = "PRED"
    POST = "POST"


# dataclass
@dataclass
class Metadata(ABC):
    design: str
    metadata_type: MetadataType
    metadata_json_path: Path
    convert_def_path: Path
    convert_verilog_path: Path
    output_drc_json_path: Path
    meta_workspace: WorkspaceInner
    analysis_workspace: WorkspaceInner
    flow: DbFlow


def parse_metadata(
    metadata_dir: Path, output_dir: Path, base_disk_path: Path
) -> List[Metadata]:
    design_list = metadata_dir.glob("*")
    os.environ["iEDA"] = "ON"
    logging.info(f"iEDA environment variable set to: {os.environ['iEDA']}")
    metadata = []
    for design in design_list:
        result_list = design.glob("*")
        metadata_type = None
        for result_path in result_list:
            if result_path.name == f"{design.name}_ground_truth.json":
                metadata_type = MetadataType.GT
            elif result_path.name == f"{design.name}_predictions.json":
                metadata_type = MetadataType.PRED
            elif result_path.name == f"{design.name}_post_predictions.json":
                metadata_type = MetadataType.POST
            else:
                logging.warning(f"Unknown result file: {result_path}")
                continue

            convert_result_dir = (
                output_dir / "convert" / design.name / metadata_type.value
            )
            convert_result_dir.mkdir(parents=True, exist_ok=True)

            report_result_dir = (
                output_dir / "report" / design.name / metadata_type.value
            )
            report_result_dir.mkdir(parents=True, exist_ok=True)

            output_def_path = (
                convert_result_dir / f"{design.name}_{metadata_type.value}.def.gz"
            )
            output_verilog_path = (
                convert_result_dir / f"{design.name}_{metadata_type.value}.v.gz"
            )
            output_drc_json_path = (
                report_result_dir / f"{design.name}_{metadata_type.value}_drc.json"
            )
            meta_workspace = WorkspaceInner(
                directory=str(base_disk_path / design.name / "workspace"),
                design=design.name,
            )

            # update script_dir & log_dir
            analysis_workspace = WorkspaceInner(
                directory=str(base_disk_path / design.name / "workspace"),
                design=design.name,
            )
            analysis_workspace.directory = str(report_result_dir)
            analysis_workspace.paths_table_inner.directory = str(report_result_dir)

            flow = DbFlow(
                eda_tool="innovus",
                step=DbFlow.FlowStep.route,
                input_def=output_def_path,
                input_verilog=output_verilog_path,
            )

            metadata.append(
                Metadata(
                    design=design.name,
                    metadata_type=metadata_type,
                    metadata_json_path=result_path,
                    convert_def_path=output_def_path,
                    convert_verilog_path=output_verilog_path,
                    output_drc_json_path=output_drc_json_path,
                    meta_workspace=meta_workspace,
                    analysis_workspace=analysis_workspace,
                    flow=flow,
                )
            )
    return metadata


def convert_generation_to_def(
    meta_workspace: WorkspaceInner,
    generation_json_path: Path,
    output_def: Path,
    output_verilog: Path,
):
    data_gen = DataGeneration(workspace=meta_workspace)

    input_def = meta_workspace.configs_inner.get_output_def(
        DbFlow(eda_tool="innovus", step=DbFlow.FlowStep.route)
    )
    if not os.path.exists(input_def):
        input_def = meta_workspace.configs_inner.get_output_def(
            DbFlow(eda_tool="innovus", step=DbFlow.FlowStep.route), compressed=False
        )
    input_verilog = meta_workspace.configs_inner.get_output_verilog(
        DbFlow(eda_tool="innovus", step=DbFlow.FlowStep.route)
    )
    if not os.path.exists(input_verilog):
        input_verilog = meta_workspace.configs_inner.get_output_verilog(
            DbFlow(eda_tool="innovus", step=DbFlow.FlowStep.route), compressed=False
        )
    data_gen.vectors_nets_patterns_to_def(
        pattern_path=str(generation_json_path),
        input_def=input_def,
        input_verilog=input_verilog,
        output_def=str(output_def),
        output_verilog=str(output_verilog),
    )


def convert(metadata: List[Metadata]):
    for metadata_obj in tqdm(metadata, desc="Converting generation to DEF format"):
        convert_generation_to_def(
            meta_workspace=metadata_obj.meta_workspace,
            generation_json_path=metadata_obj.metadata_json_path,
            output_def=metadata_obj.convert_def_path,
            output_verilog=metadata_obj.convert_verilog_path,
        )


def report(metadata: List[Metadata]):
    for metadata_obj in tqdm(metadata, desc="Generating DRC report & JSON"):
        innovus_drc = InnovusDRC(
            workspace=metadata_obj.analysis_workspace, flow=metadata_obj.flow
        )
        innovus_drc.generate_tcl()
        innovus_drc.run()
        innovus_drc.generate_json(json_path=metadata_obj.output_drc_json_path)


def analysis(metadata: List[Metadata]) -> Dict[str, Dict[str, Any]]:
    drc_data = {}
    for metadata_obj in tqdm(metadata, desc="Loading DRC data"):
        feature_gen = DataFeatureInnovus(workspace=metadata_obj.analysis_workspace)
        data = feature_gen.load_drc(json_path=metadata_obj.output_drc_json_path)
        drc_data.setdefault(metadata_obj.design, {})[
            metadata_obj.metadata_type.value
        ] = data

    return drc_data


def main():
    setup_logging()

    parser = argparse.ArgumentParser(description="DRC report")
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=Path(
            "/mnt/local_data1/liweiguo/experiments/model_size_comparison/work_dir/Large-DecimalWordLevel/stage_evaluation/def_inference_metadata"
        ),
        help="Path to metadata directory, routing generation results at the design level",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/mnt/local_data1/liweiguo/experiments/drc_analysis"),
        help="Path to output directory",
    )

    parser.add_argument(
        "--base_disk_path",
        type=Path,
        default=Path("/data2/project_share/dataset_baseline"),
        help="Base path for design data",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="convert",
        choices=["all", "convert", "report", "analysis"],
        help="Mode to run",
    )

    args = parser.parse_args()

    metadata_dir = args.metadata_dir
    output_dir = args.output_dir
    base_disk_path = args.base_disk_path
    mode = args.mode

    logging.info(f"Metadata directory: {metadata_dir}")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Base disk path: {base_disk_path}")
    metadata = parse_metadata(metadata_dir, output_dir, base_disk_path)
    if mode == "convert":
        logging.info("Converting generation results to DEF format")
        convert(metadata)
    if mode == "report":
        logging.info("Generating DRC report & JSON")
        report(metadata)
    if mode == "analysis":
        logging.info("Running DRC analysis")
        analysis(metadata)
    if mode == "all":
        logging.info("Converting generation results to DEF format")
        convert(metadata)
        logging.info("Generating DRC report & JSON")
        report(metadata)
        logging.info("Running DRC analysis")
        analysis(metadata)


if __name__ == "__main__":
    main()
