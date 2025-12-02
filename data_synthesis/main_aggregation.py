#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   main_aggregation.py
@Time    :   2025/08/02 12:04:37
@Author  :   Dawn Li
@Version :   1.0
@Contact :   dawnli619215645@gmail.com
@Desc    :   Main aggregation
"""

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List

from .aggregator import DatasetAggregator
from .base import ConfigurationManager
from .design_processor import BatchDesignProcessor


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def create_designs_config(
    design_list: List[str], base_disk_path: Path, base_output_dir: Path
) -> Dict[str, Dict[str, Any]]:
    """Create design configuration dictionary"""
    designs_config = {}

    for design in design_list:
        workspace = base_disk_path / design / "workspace"
        output_dir = base_output_dir / "individual_designs" / design

        designs_config[design] = {"workspace": workspace, "output_dir": output_dir}

    return designs_config


def main_process_individual_designs(
    config: ConfigurationManager, args
) -> Dict[str, Dict[str, Any]]:
    """Process individual designs and generate Parquet files"""
    logging.info("=== Processing Individual Designs ===")

    # Create designs configuration
    designs_config = create_designs_config(
        design_list=args.design_list,
        base_disk_path=args.base_disk_path,
        base_output_dir=args.output_dir,
    )

    # Process designs in batch
    batch_processor = BatchDesignProcessor(
        designs_config=designs_config,
        base_output_dir=args.output_dir / "individual_designs",
        config=config,
    )

    results = batch_processor.process_all_designs()

    logging.info(f"Processed {len(results)} designs")
    return results


def main_aggregate_datasets(config: ConfigurationManager, args) -> Any:
    """Aggregate individual design datasets into consolidated DatasetDict"""
    logging.info("=== Aggregating Datasets ===")

    aggregator = DatasetAggregator(
        base_data_dir=args.output_dir / "individual_designs",
        output_dir=args.output_dir / "aggregated",
        config=config,
    )

    # Aggregate datasets
    dataset_dict = aggregator.aggregate_datasets(
        designs_to_include=args.design_list,
        types_to_include=config.data_types,
    )

    # Create flat corpus for training
    logging.info("Creating flat corpus for training...")
    flat_corpus = aggregator.create_flat_corpus(
        dataset_dict=dataset_dict,
        data_type="net_seqs",
        output_path=args.output_dir / "aggregated" / "flat_corpus",
    )
    logging.info(f"Created flat corpus with {len(flat_corpus)} records")

    # Generate statistics
    stats = aggregator.get_statistics(dataset_dict)
    logging.info(f"Dataset statistics: {stats}")

    return dataset_dict


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Data synthesis and aggregation for pattern sequence generation"
    )

    # Input/Output paths
    parser.add_argument(
        "--base_disk_path",
        type=Path,
        default=Path("/data2/project_share/dataset_baseline"),
        help="Base path for design data",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("/mnt/local_data1/liweiguo/dataset/20250913/val"),
        help="Output directory for processed data",
    )

    # Design selection (Training)
    # parser.add_argument(
    #     "--design_list",
    #     nargs="+",
    #     default=[
    #         "nvdla",
    #         "shanghai_MS",
    #         "ZJU_asic_top",
    #         "iEDA_2023",
    #         "ysyx4_SoC2",
    #         "BEIHAI_2.0",
    #         "AIMP",
    #         "AIMP_2.0",
    #         "ysyx6",
    #         "ysyx4_SoC1",
    #         "ysyx4_2",
    #         "s44",
    #         "gcd",
    #         "s1488",
    #         "apb4_ps2",
    #         "apb4_timer",
    #         "apb4_i2c",
    #         "apb4_pwm",
    #         "apb4_clint",
    #         "s15850",
    #         "s38417",
    #         "s35932",
    #         "s38584",
    #         "BM64",
    #         "picorv32",
    #         "PPU",
    #         "blabla",
    #         "aes_core",
    #         "aes",
    #         "salsa20",
    #         "jpeg_encoder",
    #         "eth_top",
    #         "yadan_riscv_sopc",
    #     ],
    #     help="List of designs to process",
    # )

    # Design selection (Evaluation)
    parser.add_argument(
        "--design_list",
        nargs="+",
        default=[
            "s713",
            "apb4_rng",
            "s1238",
            "apb4_archinfo",
            "s9234",
            "s13207",
            "s5378",
            "apb4_wdg",
            "ASIC",
            "apb4_uart",
        ],
        help="List of designs to process",
    )

    # Data types to process
    parser.add_argument(
        "--data_types",
        nargs="+",
        default=[
            "net_seqs",
            "pin2pin_pattern_seqs",
            "pin2pin_loc_seqs",
            "design_graph",
        ],
        help="List of data types to process",
    )

    # Configuration
    parser.add_argument(
        "--max_turn_num",
        type=int,
        default=1000,
        help="Maximum number of turns for pattern processing",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        default=False,
        help="Force rebuild of existing data",
    )
    parser.add_argument(
        "--enable_dataset_logs",
        action="store_true",
        default=False,
        help="Enable detailed logging for dataset processing",
    )
    parser.add_argument(
        "--disable_dataset_logs",
        action="store_true",
        help="Disable detailed logging for dataset processing",
    )

    # Processing options
    parser.add_argument(
        "--skip_individual",
        action="store_true",
        help="Skip individual design processing",
    )
    parser.add_argument(
        "--skip_aggregation", action="store_true", help="Skip dataset aggregation"
    )

    # Logging
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "CRITICAL", "WARNING", "ERROR"],
        help="Logging level",
    )

    return parser.parse_args()


def main():
    """Main execution function"""
    args = parse_arguments()
    setup_logging(args.log_level)

    logging.info("Starting data synthesis and aggregation pipeline...")

    # Create configuration
    enable_logs = args.enable_dataset_logs and not args.disable_dataset_logs
    config = ConfigurationManager(
        max_turn_num=args.max_turn_num,
        rebuild=args.rebuild,
        enable_dataset_logs=enable_logs,
        data_types=args.data_types,
    )

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Step 1: Process individual designs
        if not args.skip_individual:
            individual_results = main_process_individual_designs(config, args)
            if not individual_results:
                logging.error("No individual designs processed successfully")
                return 1

        # Step 3: Aggregate datasets
        if not args.skip_aggregation:
            dataset_dict = main_aggregate_datasets(config, args)
            if not dataset_dict:
                logging.error("Dataset aggregation failed")
                return 1

        logging.info("=== Pipeline completed successfully ===")

        # Print final summary
        final_aggregated_output_path = (
            args.output_dir / "aggregated" / "aggregated_dataset"
        )
        if final_aggregated_output_path.exists():
            logging.info(
                f"Final aggregated dataset available at: {final_aggregated_output_path}"
            )

        final_flat_corpus_path = args.output_dir / "aggregated" / "flat_corpus"
        if final_flat_corpus_path.exists():
            logging.info(f"Final flat corpus available at: {final_flat_corpus_path}")

        return 0

    except Exception as e:
        logging.error(f"Pipeline failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    main()
