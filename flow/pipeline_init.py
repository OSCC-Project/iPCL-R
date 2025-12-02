#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   pipeline_init.py
@Time    :   2025/08/01 11:16:37
@Author  :   Dawn Li
@Version :   1.0
@Contact :   dawnli619215645@gmail.com
@Desc    :   Pipeline initialization utility for generating FlowConfig JSON files with
             project-specific path configuration and command-line interface for
             creating flow configuration templates
"""

import argparse
import logging
import sys
from pathlib import Path

from .config import FlowConfig
from .utils.logging_utils import setup_logging


def main():
    setup_logging()
    """Main entry point for pipeline initialization"""
    parser = argparse.ArgumentParser(
        description="Initialize flow pipeline configuration files"
    )

    # Configuration generation
    parser.add_argument(
        "--create-flow-config",
        type=Path,
        metavar="OUTPUT_PATH",
        help="Create a flow configuration file at the specified path",
    )

    # Output directory for path prefixes
    parser.add_argument(
        "--project-output-dir",
        type=Path,
        default=Path("./"),
        help="Project output directory to replace /path/to/ prefixes (default: ./)",
    )

    args = parser.parse_args()

    # Check if config generation is requested
    if not args.create_flow_config:
        logging.error(
            "No action specified. Use --create-flow-config to generate a configuration file."
        )
        parser.print_help()
        return 1

    # Create flow configuration
    logging.info("Starting flow configuration file creation process")

    # Create default config
    config = FlowConfig()

    output_path = args.create_flow_config
    project_output_dir = args.project_output_dir

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logging.info(f"Creating flow configuration at: {output_path}")
    logging.info(f"Using project output directory: {project_output_dir}")

    # Generate configuration with proper path replacement
    config_path = config.create_flow_config(output_path, project_output_dir)

    # Success messages
    logging.info(f"✅ Flow configuration created successfully at: {config_path}")

    print("Configuration file generated successfully!")
    print(f"📁 Location: {config_path}")
    print(f"🔧 Project output directory: {project_output_dir}")
    print()
    print("Next steps:")
    print("1. Edit the configuration file to customize settings")
    print("2. Use the configuration with launch scripts:")
    print(f"   python -m flow.launch_tokenization --flow-config {config_path}")
    print(f"   accelerate launch -m flow.launch_training --flow-config {config_path}")
    print(f"   accelerate launch -m flow.launch_evaluation --flow-config {config_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
