#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   serialization.py
@Time    :   2025/08/06 15:07:52
@Author  :   Dawn Li
@Version :   1.0
@Contact :   dawnli619215645@gmail.com
@Desc    :   Serialization demo for Net Routing Tokenization
"""

import argparse
import json
import logging
from pathlib import Path

from datasets import Dataset

from flow import FlowConfig
from flow.tokenization import TokenizationPipeline
from flow.utils import load_corpus_dataset, setup_logging


def create_flow_config() -> FlowConfig:
    """Create a FlowConfig instance with default settings"""
    flow_config = FlowConfig()
    flow_config.tokenization.workflow.tokenizer_algorithm = "None"
    return flow_config


def serialization(dataset: Dataset, flow_config: FlowConfig) -> Dataset:
    """Serialize the entire dataset using the UnifiedTokenizer"""
    pipeline = TokenizationPipeline(flow_config)
    dataset = pipeline.preprocess_corpus(dataset)
    dataset = pipeline.build_token_dataset(dataset)
    return dataset


def save_demo(serialized_dataset: Dataset, output_dir: Path, k: int = 5) -> None:
    """Save a demo of the serialized dataset"""
    output_dir.mkdir(parents=True, exist_ok=True)
    demo_dataset = serialized_dataset.select(range(k))
    for i, example in enumerate(demo_dataset):
        save_dir = output_dir / f"example_{i + 1}"
        save_dir.mkdir(parents=True, exist_ok=True)
        input = {
            "Driver": example["driver"],
            "Loads": example["loads"],
            "Routing": example["tree_seq"],
        }
        tokenization_result = {
            "Source Tokens": example["source_tokens"],
            "Target Tokens": example["target_tokens"],
        }

        with open(save_dir / "input.json", "w") as f:
            json.dump(input, f, indent=4)
        with open(save_dir / "tokenization_result.json", "w") as f:
            json.dump(tokenization_result, f, indent=4)
        logging.info(f"Saved example {i + 1} to {save_dir}")


def main():
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Serialization demo for Net Routing Tokenization"
    )

    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("/mnt/local_data1/liweiguo/dataset/experiments/val"),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/mnt/local_data1/liweiguo/experiments/demo/serialization"),
    )

    args = parser.parse_args()

    flow_config = FlowConfig()
    tokenization_config = flow_config.tokenization
    tokenization_config.workflow.tokenizer_algorithm = "None"

    dataset = load_corpus_dataset(args.dataset_dir)
    demo_dataset = dataset.filter(
        lambda x: len(x["tree_seq"]) <= 12 and len(x["loads"]) == 2
    )
    logging.info(f"Filtered dataset size: {len(demo_dataset)}")
    serialized_dataset = serialization(demo_dataset, flow_config)
    save_demo(serialized_dataset, args.output_dir)


if __name__ == "__main__":
    main()
