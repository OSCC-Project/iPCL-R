#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   launch_training.py
@Time    :   2025/08/01 11:16:30
@Author  :   Dawn Li
@Version :   1.0
@Contact :   dawnli619215645@gmail.com
@Desc    :   Command-line launcher for model training using Accelerate
             framework, handles distributed training setup and hyperparameter
             configuration with FlowConfig integration
"""

import argparse
import logging
import sys
import time
from pathlib import Path

from accelerate import Accelerator

from flow.config import FlowConfig
from flow.training import TrainingPipeline
from flow.utils import setup_logging


def main():
    setup_logging()
    """Main entry point for training launcher"""
    parser = argparse.ArgumentParser(description="Launch training using FlowConfig")
    parser.add_argument(
        "--flow-config", type=str, required=True, help="Path to FlowConfig JSON file"
    )

    args = parser.parse_args()

    try:
        accelerator = Accelerator()
        # Load FlowConfig from JSON file
        logging.info(f"🔧 Loading configuration from: {args.flow_config}")
        flow_config = FlowConfig.from_config_file(Path(args.flow_config))

        # Extract training stage configuration directly
        tokenization_config = flow_config.tokenization
        training_config = flow_config.training

        logging.info("🚀 Starting training pipeline")
        logging.info(f"   Token dataset: {tokenization_config.paths.token_dataset_dir}")
        logging.info(f"   Model save dir: {training_config.paths.model_save_dir}")
        logging.info(
            f"   Training epochs: {training_config.hyperparameters.num_train_epochs}"
        )
        logging.info(
            f"   Batch size per device: {training_config.hyperparameters.batch_size_per_device}"
        )
        logging.info(
            f"   Learning rate: {training_config.hyperparameters.learning_rate}"
        )
        logging.info(f"   Optimizer: {training_config.hyperparameters.optimizer_type}")

        # Create and run training pipeline
        pipeline = TrainingPipeline(accelerator, flow_config)

        start_time = time.time()
        pipeline.run_training()
        end_time = time.time()

        # Report results
        execution_time = end_time - start_time
        logging.info("✅ Training completed successfully!")
        logging.info(f"   Execution time: {execution_time:.2f} seconds")
        logging.info(f"   Model saved to: {training_config.paths.model_save_dir}")
        logging.info(
            f"   Split dataset saved to: {training_config.paths.split_dataset_dir}"
        )

        logging.info(f"   Logs saved to: {training_config.paths.logging_dir}")
        logging.info("💡 Next step: Run evaluation with:")
        logging.info(
            f"   accelerate launch -m flow.launch_evaluation --flow-config {args.flow_config}"
        )

    except FileNotFoundError as e:
        logging.info(f"❌ Error: Configuration file not found: {e}")
        return 1
    except KeyError as e:
        logging.info(f"❌ Error: Missing configuration key: {e}")
        return 1
    except Exception as e:
        logging.info(f"❌ Training failed with error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
