#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   dataset_processor.py
@Time    :   2025/08/15 11:17:00
@Author  :   Dawn Li
@Version :   1.0
@Contact :   dawnli619215645@gmail.com
@Desc    :   Dataset processor for symbol analysis experiments.
"""

import logging
from pathlib import Path
from typing import Any, Dict

import pyarrow.compute as pc
from datasets import Dataset

from flow import FlowConfig
from flow.tokenization import TokenizationPipeline
from flow.utils import load_corpus_dataset


class DatasetProcessor:
    """Handles dataset loading and corpus extraction for symbol analysis"""

    def __init__(self, dataset_dir: Path):
        """
        Initialize dataset processor

        Args:
            dataset_dir: Path to dataset directory
        """
        self.dataset_dir = dataset_dir
        self.dataset = None

    def load_dataset(self) -> Dataset:
        """
        Load dataset using the same logic as serialization.py

        Returns:
            Loaded HuggingFace Dataset

        Raises:
            FileNotFoundError: If dataset cannot be found
            Exception: For other loading errors
        """
        try:
            dataset = load_corpus_dataset(self.dataset_dir)
            logging.info(
                f"Dataset loaded with {len(dataset)} samples from {self.dataset_dir}"
            )
            self.dataset = dataset
            return dataset
        except FileNotFoundError:
            logging.error(f"Dataset not found at: {self.dataset_dir}")
        except Exception as e:
            logging.error(f"Failed to load dataset: {e}")
            raise

    def extract_corpus(self):
        """
        Extract target_tokens corpus for domain-specific tokenization
        Uses the same serialization preprocessing as in experiments/demo/serialization.py

        Returns:
            List of target_tokens strings from the dataset
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")

        # Create UnifiedTokenizer for preprocessing (same as serialization.py)
        flow_config = FlowConfig()
        flow_config.tokenization.workflow.tokenizer_algorithm = "None"
        pipeline = TokenizationPipeline(flow_config)

        self.dataset = pipeline.preprocess_corpus(self.dataset)
        self.dataset = pipeline.build_token_dataset(self.dataset)

        def concat_tree_seq(batch: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "tree_seq": [
                    " ".join(seq) if isinstance(seq, list) else seq
                    for seq in batch["tree_seq"]
                ]
            }

        self.dataset = self.dataset.map(
            concat_tree_seq,
            batched=True,
            num_proc=flow_config.tokenization.performance.num_workers,
            desc="Concatenating tree_seq",
        )

    def get_dataset_statistics(self) -> Dict[str, Any]:
        """
        Get basic statistics about the dataset using PyArrow for fast computation.
        This avoids converting large columns into Python lists, which is slow.
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")

        stats = {
            "total_samples": len(self.dataset),
            "features": list(self.dataset.features.keys()),
            "sample_example": dict(self.dataset[0]) if len(self.dataset) > 0 else {},
        }

        # Get the underlying Arrow table (zero-copy from Dataset)
        table = self.dataset.data.table

        # ----- Compute tree_seq_length -----
        if "tree_seq" in table.column_names:
            # If the feature is a list/sequence type, we can directly get its length
            if hasattr(self.dataset.features["tree_seq"], "feature"):
                tree_seq_length = pc.list_value_length(table["tree_seq"])
            else:
                # If it's a string: split by whitespace and then count
                split_tree_seq = pc.utf8_split_whitespace(table["tree_seq"])
                tree_seq_length = pc.list_value_length(split_tree_seq)
        else:
            tree_seq_length = None

        # ----- Compute target_tokens_length -----
        if "target_tokens" in table.column_names:
            if hasattr(self.dataset.features["target_tokens"], "feature"):
                target_tokens_length = pc.list_value_length(table["target_tokens"])
            else:
                split_target_tokens = pc.utf8_split_whitespace(table["target_tokens"])
                target_tokens_length = pc.list_value_length(split_target_tokens)
        else:
            target_tokens_length = None

        # ----- Aggregate statistics -----
        if tree_seq_length is not None:
            tree_stats = pc.min_max(tree_seq_length)
            stats["tree_seq_avg_length"] = pc.mean(tree_seq_length).as_py()
            stats["tree_seq_min_length"] = tree_stats["min"].as_py()
            stats["tree_seq_max_length"] = tree_stats["max"].as_py()

        if target_tokens_length is not None:
            target_stats = pc.min_max(target_tokens_length)
            stats["target_tokens_avg_length"] = pc.mean(target_tokens_length).as_py()
            
            stats["target_tokens_min_length"] = target_stats["min"].as_py()
            stats["target_tokens_max_length"] = target_stats["max"].as_py()

        return stats
