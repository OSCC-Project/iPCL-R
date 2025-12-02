#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   pipeline.py
@Time    :   2025/08/01 11:15:04
@Author  :   Dawn Li
@Version :   1.0
@Contact :   dawnli619215645@gmail.com
@Desc    :   Tokenization pipeline implementing UnifiedTokenizer integration,
             data synthesis processing, token dataset generation, and metadata
             collection for routing pattern sequences
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict

import pyarrow as pa
import pyarrow.compute as pc
from datasets import Dataset, Sequence, Value
from tqdm import tqdm

from flow.config import FlowConfig
from flow.utils import load_corpus_dataset

from .tokenizer import UnifiedTokenizer


class TokenizationPipeline:
    """Streamlined tokenization pipeline that directly calls tokenizer.py interfaces"""

    def __init__(self, flow_config: FlowConfig):
        self.flow_config = flow_config
        self.dataset_config = flow_config.dataset
        self.tokenization_config = flow_config.tokenization
        self.paths_config = self.tokenization_config.paths
        self.workflow_config = self.tokenization_config.workflow
        self.performance_config = self.tokenization_config.performance
        self.advanced_config = self.tokenization_config.advanced
        self.unified_tokenizer = UnifiedTokenizer(self.tokenization_config)

    def run_tokenization(self) -> Dict[str, Any]:
        """Run complete streamlined tokenization pipeline"""
        start_time = time.time()
        logging.info("Starting streamlined tokenization pipeline")
        logging.info(f"Algorithm: {self.workflow_config.tokenizer_algorithm}")

        # Step 1: Read data_synthesis results
        corpus_dataset = self.read_data_synthesis()

        # Step 2: Preprocess and refine tree_seq in corpus dataset
        corpus_dataset = self.preprocess_corpus(corpus_dataset)

        # Step 3: Extract training tokens using existing interface
        corpus_dataset = self.build_training_text(corpus_dataset)

        # Step 4: Train tokenizer using existing interface
        self.train_tokenizer(corpus_dataset)

        # Step 5: Print token sequences of first 3 training samples
        self.print_sample_sequence(corpus_dataset)

        # Step 6: Create token datasets with statistics and cleaning
        token_dataset = self.build_token_dataset(corpus_dataset)

        # Step 7: Save final dataset and tokenizer
        self.save_result(token_dataset)

        # Step 8: Save metadata with tokenizer and dataset statistics
        metadata = self.save_metadata(corpus_dataset)

        total_time = time.time() - start_time
        logging.info(f"Streamlined tokenization completed in {total_time:.2f}s")

        return {
            "tokenizer": self.unified_tokenizer,
            "metadata": metadata,
            "execution_time": total_time,
        }

    def read_data_synthesis(self) -> Dataset:
        """Read data_synthesis results"""
        target_split = self.dataset_config.train_split
        if self.dataset_config.use_hub():
            logging.info(
                f"Reading dataset from Hugging Face Hub: {self.dataset_config.hub_id} (split: {target_split})"
            )
        else:
            local_path = self.dataset_config.local_path_for_split(target_split)
            logging.info(f"Reading dataset from local path: {local_path}")

        corpus_dataset = load_corpus_dataset(self.dataset_config, split=target_split)

        logging.info(f"Loaded corpus with {len(corpus_dataset)} samples")
        logging.info(f"Corpus columns: {corpus_dataset.column_names}")

        return corpus_dataset

    def preprocess_corpus(self, corpus_dataset: Dataset) -> Dataset:
        """Preprocess corpus dataset to ensure it has required columns"""
        logging.info("Preprocessing corpus dataset")

        def reorganize_dataset(batch: Dict[str, Any]) -> Dict[str, Any]:
            """Refine tree_seq to ensure it is simplified"""

            batch_driver, batch_loads, batch_tree_seq = (
                batch["driver"],
                batch["loads"],
                batch["tree_seq"],
            )

            simplified_tree_seq = [
                self.unified_tokenizer.simplify_coordinate_sequence(tree_seq)
                for tree_seq in batch_tree_seq
            ]

            batch_relative_loads = [
                self.unified_tokenizer.convert_loads_to_relative_loads(driver, loads)
                for driver, loads in zip(batch_driver, batch_loads)
            ]

            batch_relative_tree_seq = [
                self.unified_tokenizer.convert_tree_seq_to_relative_tree_seq(
                    driver, tree_seq
                )
                for driver, tree_seq in zip(batch_driver, simplified_tree_seq)
            ]

            return {
                "tree_seq": simplified_tree_seq,
                "relative_loads": batch_relative_loads,
                "relative_tree_seq": batch_relative_tree_seq,
            }

        corpus_dataset = corpus_dataset.map(
            reorganize_dataset,
            batched=True,
            num_proc=self.performance_config.num_workers,
            desc="Reorganizing the dataset",
        )
        logging.info(f"Corpus dataset preprocessed with {len(corpus_dataset)} samples")
        return corpus_dataset

    def build_training_text(self, corpus_dataset: Dataset) -> Dataset:
        """Extract training tokens using existing tokenizer.py interface"""
        logging.info("Extracting training tokens using UnifiedTokenizer")

        # Call existing extract_training_tokens interface
        def build_text(batch: Dict[str, Any]) -> Dict[str, Any]:
            """Build training texts from source and target sequences in batch"""
            batch_driver, batch_loads = batch["driver"], batch["loads"]
            batch_overlap_info = batch.get("overlap_info", [{}] * len(batch_driver))
            batch_connected_info = batch.get("connected_info", [{}] * len(batch_driver))

            batch_source_text = [
                self.unified_tokenizer.remove_special_token(
                    self.unified_tokenizer.convert_source_to_directional_token(
                        driver, loads, overlap_info, connected_info
                    )
                )
                for driver, loads, overlap_info, connected_info in zip(
                    batch_driver, batch_loads, batch_overlap_info, batch_connected_info
                )
            ]

            batch_relative_tree_seq = batch["relative_tree_seq"]
            batch_target_text = [
                self.unified_tokenizer.remove_special_token(
                    self.unified_tokenizer.convert_relative_target_to_directional_token(
                        relative_tree_seq
                    )
                )
                for relative_tree_seq in batch_relative_tree_seq
            ]

            batch_text = [
                f"{source_text} {target_text}"
                for source_text, target_text in zip(
                    batch_source_text, batch_target_text
                )
            ]

            return {
                "source_text": batch_source_text,
                "target_text": batch_target_text,
                "text": batch_text,
            }

        corpus_dataset = corpus_dataset.map(
            build_text,
            batched=True,
            num_proc=self.performance_config.num_workers,
            desc="Building training texts from source and target",
        )

        logging.info(f"Built {len(corpus_dataset)} training texts")
        return corpus_dataset

    def train_tokenizer(self, corpus_dataset: Dataset):
        """Train tokenizer using existing tokenizer.py interface"""
        logging.info("Training tokenizer using UnifiedTokenizer")

        training_texts = corpus_dataset["text"]

        # Call existing train interface
        tokenizer = self.unified_tokenizer.train(training_texts)

        logging.info("Tokenizer trained successfully")
        logging.info(f"Vocabulary size: {len(tokenizer.get_vocab())}")

        return tokenizer

    def print_debug_sample(self, corpus_dataset: Dataset):
        """Print token sequences of first 3 training samples (source and target)"""
        logging.info("=== SAMPLE TOKENIZED SEQUENCES ===")

        tokenizer = self.unified_tokenizer.tokenizer
        for i in tqdm(
            range(len(corpus_dataset)), desc="Processing samples", unit="sample"
        ):
            sample = corpus_dataset[i]

            # Get source and target sequences using UnifiedTokenizer
            try:
                driver, loads, overlap_info, connected_info = (
                    sample["driver"],
                    sample["loads"],
                    sample.get("overlap_info", {}),
                    sample.get("connected_info", {}),
                )
                source_text = (
                    self.unified_tokenizer.convert_source_to_directional_token(
                        driver, loads, overlap_info, connected_info
                    )
                )

                relative_tree_seq = sample["relative_tree_seq"]
                target_text = (
                    self.unified_tokenizer.convert_relative_target_to_directional_token(
                        relative_tree_seq
                    )
                )

                source_directional_tokens = source_text.split()
                target_directional_tokens = target_text.split()
                source_tokenized_tokens = tokenizer.tokenize(source_text)
                target_tokenized_tokens = tokenizer.tokenize(target_text)

                reverse_routing_text = self.unified_tokenizer.convert_tokens_to_routing(
                    target_text
                )
                if reverse_routing_text == relative_tree_seq:
                    continue

                logging.info(f"Sample {i + 1} Net {sample.get('net_name')}:")
                logging.info(f"  [Source] Driver: {sample.get('driver', 'N/A')}")
                logging.info(
                    f"  [Source] Loads: {sample.get('loads', 'N/A')[:4]}{'...' if len(sample.get('loads', [])) > 4 else ''}"
                )
                logging.info(
                    f"  [Source -> Directional Token]: {source_directional_tokens[:10]}{'...' if len(source_directional_tokens) > 10 else ''}"
                )
                logging.info(
                    f"  [Directional Token -> Tokenized Token] : {source_tokenized_tokens[:10]}{'...' if len(source_tokenized_tokens) > 10 else ''}"
                )
                logging.info(
                    f"  # Source tokenized length: {len(source_tokenized_tokens)}"
                )
                logging.info(f"  [Target] Tree Sequence: {relative_tree_seq}")
                logging.info(
                    f"  [Target -> Directional Token]: {target_directional_tokens[:10]}{'...' if len(target_directional_tokens) > 10 else ''}"
                )
                logging.info(
                    f"  [Directional Token -> Tokenized Token]: {target_tokenized_tokens[:10]}{'...' if len(target_tokenized_tokens) > 10 else ''}"
                )
                logging.info(f"  # Target length: {len(relative_tree_seq)}")
                logging.info(
                    f"  # Target tokenized length: {len(target_tokenized_tokens)}"
                )
                logging.info(f"  [Target -> Reverse Routing]: {reverse_routing_text}")
                logging.info(f"  # Reverse routing length: {len(reverse_routing_text)}")
                logging.info(
                    f"  # Is Equal of (Tree Sequence, Reverse Routing): {relative_tree_seq == reverse_routing_text}"
                )
            except Exception as e:
                logging.error(f"Error tokenizing sample {i + 1}: {e}")

    def print_sample_sequence(self, corpus_dataset: Dataset):
        """Print token sequences of first 3 training samples (source and target)"""
        logging.info("=== SAMPLE TOKENIZED SEQUENCES ===")

        tokenizer = self.unified_tokenizer.tokenizer
        for i in range(min(3, len(corpus_dataset))):
            sample = corpus_dataset[i]
            # Get source and target sequences using UnifiedTokenizer
            try:
                driver, loads, overlap_info, connected_info = (
                    sample["driver"],
                    sample["loads"],
                    sample.get("overlap_info", {}),
                    sample.get("connected_info", {}),
                )
                source_text = (
                    self.unified_tokenizer.convert_source_to_directional_token(
                        driver, loads, overlap_info, connected_info
                    )
                )

                relative_tree_seq = sample["relative_tree_seq"]
                target_text = (
                    self.unified_tokenizer.convert_relative_target_to_directional_token(
                        relative_tree_seq
                    )
                )
                source_directional_tokens = source_text.split()
                target_directional_tokens = target_text.split()
                source_tokenized_tokens = tokenizer.tokenize(source_text)
                target_tokenized_tokens = tokenizer.tokenize(target_text)

                reverse_routing_text = self.unified_tokenizer.convert_tokens_to_routing(
                    target_text
                )

                logging.info(f"Sample {i + 1} Net {sample.get('net_name')}:")
                logging.info(f"  [Source] Driver: {sample.get('driver', 'N/A')}")
                logging.info(
                    f"  [Source] Loads: {sample.get('loads', 'N/A')[:4]}{'...' if len(sample.get('loads', [])) > 4 else ''}"
                )
                logging.info(
                    f"  [Source -> Directional Token]: {source_directional_tokens[:10]}{'...' if len(source_directional_tokens) > 10 else ''}"
                )
                logging.info(
                    f"  [Directional Token -> Tokenized Token] : {source_tokenized_tokens[:10]}{'...' if len(source_tokenized_tokens) > 10 else ''}"
                )
                logging.info(
                    f"  # Source tokenized length: {len(source_tokenized_tokens)}"
                )
                logging.info(
                    f"  [Target] Tree Sequence: {relative_tree_seq[:4]}{'...' if len(relative_tree_seq) > 4 else ''}"
                )
                logging.info(
                    f"  [Target -> Directional Token]: {target_directional_tokens[:10]}{'...' if len(target_directional_tokens) > 10 else ''}"
                )
                logging.info(
                    f"  [Directional Token -> Tokenized Token]: {target_tokenized_tokens[:10]}{'...' if len(target_tokenized_tokens) > 10 else ''}"
                )
                logging.info(f"  # Target length: {len(relative_tree_seq)}")
                logging.info(
                    f"  # Target tokenized length: {len(target_tokenized_tokens)}"
                )
                logging.info(
                    f"  [Target -> Reverse Routing]: {reverse_routing_text[:4]}{'...' if len(reverse_routing_text) > 4 else ''}"
                )
                logging.info(f"  # Reverse routing length: {len(reverse_routing_text)}")
                logging.info(
                    f"  # Is Equal of (Tree Sequence, Reverse Routing): {relative_tree_seq == reverse_routing_text}"
                )
            except Exception as e:
                logging.error(f"Error tokenizing sample {i + 1}: {e}")

    def build_token_dataset(
        self, corpus_dataset: Dataset, remove_columns: bool = True
    ) -> Dataset:
        """Create token ID datasets"""
        logging.info("Creating token ID datasets")

        def add_tokens(batch):
            batch_driver, batch_loads = batch["driver"], batch["loads"]
            batch_overlap_info = batch.get("overlap_info", [{}] * len(batch_driver))
            batch_connected_info = batch.get("connected_info", [{}] * len(batch_driver))

            batch_source_tokens = [
                self.unified_tokenizer.convert_source_to_directional_token(
                    driver, loads, overlap_info, connected_info
                )
                for driver, loads, overlap_info, connected_info in zip(
                    batch_driver, batch_loads, batch_overlap_info, batch_connected_info
                )
            ]

            batch_relative_tree_seq = batch["relative_tree_seq"]
            batch_target_tokens = [
                self.unified_tokenizer.convert_relative_target_to_directional_token(
                    relative_tree_seq
                )
                for relative_tree_seq in batch_relative_tree_seq
            ]
            return {
                "source_tokens": batch_source_tokens,
                "target_tokens": batch_target_tokens,
            }

        corpus_dataset = corpus_dataset.map(
            add_tokens,
            batched=True,
            num_proc=self.performance_config.num_workers,
            desc="Convert coordinate sequences to directional tokens",
        )

        # Compute statistics using pyarrow.compute
        logging.info("=== DATASET STATISTICS ===")
        logging.info(f"Data volume: {len(corpus_dataset)} samples")

        # Convert to pyarrow table for efficient computation
        table = (
            corpus_dataset.data.table
            if hasattr(corpus_dataset.data, "table")
            else corpus_dataset.to_arrow()
        )

        # Calculate lengths using pyarrow
        source_tokens_array = table.column("source_tokens")
        target_tokens_array = table.column("target_tokens")

        # Compute lengths for each token string
        source_lengths = pc.list_value_length(
            pc.split_pattern(source_tokens_array, pattern=" ")
        ).to_pylist()
        target_lengths = pc.list_value_length(
            pc.split_pattern(target_tokens_array, pattern=" ")
        ).to_pylist()

        # Filter out invalid entries (empty tokens)
        source_lengths = [len for len in source_lengths if len > 0]
        target_lengths = [len for len in target_lengths if len > 0]

        # Compute statistics using pyarrow
        source_lengths_array = pa.array(source_lengths)
        target_lengths_array = pa.array(target_lengths)

        # Source statistics
        logging.info("Source length distribution:")
        logging.info(
            f"  Min: {pc.min(source_lengths_array).as_py()}, Max: {pc.max(source_lengths_array).as_py()}"
        )
        logging.info(
            f"  Mean: {pc.mean(source_lengths_array).as_py():.2f}, "
            f"Median: {pc.approximate_median(source_lengths_array).as_py():.2f}, "
            f"95% Quantile: {pc.quantile(source_lengths_array, q=[0.95])[0].as_py():.2f}"
        )

        # Target statistics
        logging.info("Target length distribution:")
        logging.info(
            f"  Min: {pc.min(target_lengths_array).as_py()}, Max: {pc.max(target_lengths_array).as_py()}"
        )
        logging.info(
            f"  Mean: {pc.mean(target_lengths_array).as_py():.2f}, "
            f"Median: {pc.approximate_median(target_lengths_array).as_py():.2f}, "
            f"95% Quantile: {pc.quantile(target_lengths_array, q=[0.95])[0].as_py():.2f}"
        )

        # Remove columns if needed
        if remove_columns:
            # For training and evaluation, we only need the relevant columns
            remaining_columns = [
                "source_tokens",
                "target_tokens",
                "net_name",
                "driver",
                "loads",
                "tree_seq",
                "relative_loads",
                "relative_tree_seq",
            ]
            columns_to_remove = [
                col
                for col in corpus_dataset.column_names
                if col not in remaining_columns
            ]
            corpus_dataset = corpus_dataset.remove_columns(columns_to_remove)

        return corpus_dataset

    def save_result(self, token_dataset: Dataset):
        """Save final dataset and tokenizer"""
        logging.info("Saving dataset and tokenizer")

        # Save tokenizer
        tokenizer_dir = Path(self.paths_config.tokenizer_save_dir)
        tokenizer_dir.mkdir(parents=True, exist_ok=True)
        self.unified_tokenizer.save_pretrained(str(tokenizer_dir))
        logging.info(f"Unified Tokenizer saved to: {tokenizer_dir}")

        # Save token dataset
        dataset_dir = Path(self.paths_config.token_dataset_dir)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        token_dataset.save_to_disk(str(dataset_dir))
        logging.info(f"Token dataset saved to: {dataset_dir}")

    def save_metadata(self, corpus_dataset) -> Dict[str, Any]:
        """Save metadata with tokenizer and dataset statistics using fast Arrow compute"""
        logging.info("=== FREQUENCY LOGGING ===")

        table = corpus_dataset.data.table
        feat = corpus_dataset.features["text"]
        col = table["text"]

        if (
            isinstance(feat, Sequence)
            and isinstance(feat.feature, Value)
            and feat.feature.dtype.startswith("string")
        ):
            words = pc.list_flatten(col)
        else:
            words = pc.list_flatten(pc.utf8_split_whitespace(col))

        words = pc.drop_null(words)
        original_vocab_size = int(pc.count_distinct(words).as_py())

        tokenizer = self.unified_tokenizer.tokenizer

        def build_tokens(batch):
            batch_source_text = batch["source_text"]
            batch_target_text = batch["target_text"]

            batch_source_tokenized_tokens = [
                tokenizer.tokenize(text) for text in batch_source_text
            ]
            batch_target_tokenized_tokens = [
                tokenizer.tokenize(text) for text in batch_target_text
            ]

            return {
                "source_tokenized_tokens": batch_source_tokenized_tokens,
                "target_tokenized_tokens": batch_target_tokenized_tokens,
            }

        tokenized_dataset = corpus_dataset.map(
            build_tokens,
            batched=True,
            num_proc=self.performance_config.num_workers,
            desc="Building tokenized tokens from text",
        )

        # --- Arrow table access ---
        try:
            table = tokenized_dataset.data.table
        except AttributeError:
            table = tokenized_dataset._data.table

        # Helper to flatten tokens and compute counts
        def token_freq_and_lengths(col_name):
            col = table[col_name]  # list<utf8>
            flat = pc.list_flatten(col)
            flat = pc.drop_null(flat)
            freq_struct = pc.value_counts(flat)
            tokens = freq_struct.field("values").to_pylist()
            counts = freq_struct.field("counts").to_pylist()
            freq_dict = dict(zip(tokens, counts))
            lengths = pc.list_value_length(col)
            return freq_dict, lengths

        source_counter, source_lengths = token_freq_and_lengths(
            "source_tokenized_tokens"
        )
        target_counter, target_lengths = token_freq_and_lengths(
            "target_tokenized_tokens"
        )

        # Merge counters
        total_counter = source_counter.copy()
        for tok, cnt in target_counter.items():
            total_counter[tok] = total_counter.get(tok, 0) + cnt

        # Stats for lengths
        def length_stats(lengths_array):
            return {
                "min_token_length": pc.min(lengths_array).as_py(),
                "max_token_length": pc.max(lengths_array).as_py(),
                "mean_token_length": pc.mean(lengths_array).as_py(),
                "median_token_length": pc.quantile(lengths_array, q=0.5).to_pylist()[0],
                "quantile_95_token_length": pc.quantile(
                    lengths_array, q=0.95
                ).to_pylist()[0],
            }

        src_len_stats = length_stats(source_lengths)
        tgt_len_stats = length_stats(target_lengths)

        # Vocabulary stats
        vocab_dict = tokenizer.get_vocab()
        full_vocab = set(vocab_dict.keys())
        used_tokens = set(total_counter.keys())
        unused_tokens = list(full_vocab - used_tokens)

        total_vocab_size = len(full_vocab)
        vocab_utilization_rate = (
            (len(used_tokens) / total_vocab_size * 100.0) if total_vocab_size else 0.0
        )

        metadata = {
            "tokenizer_info": {
                "type": self.workflow_config.tokenizer_algorithm,
                "original_vocab_size": original_vocab_size,
                "expected_vocab_size": self.workflow_config.target_vocab_size,
                "final_vocab_size": len(vocab_dict),
            },
            "source_token_stats": {
                "total_tokens": sum(source_counter.values()),
                "unique_tokens": len(source_counter),
                **src_len_stats,
            },
            "target_token_stats": {
                "total_tokens": sum(target_counter.values()),
                "unique_tokens": len(target_counter),
                **tgt_len_stats,
            },
            "global_token_stats": {
                "total_tokens": sum(total_counter.values()),
                "unique_tokens": len(total_counter),
                "avg_token_length_per_seq": sum(total_counter.values())
                / len(corpus_dataset),
            },
            "frequency_distribution": total_counter,
            "timestamp": time.time(),
            "used_tokens": list(used_tokens),
            "unused_tokens": unused_tokens,
            "vocab_utilization_rate": vocab_utilization_rate,
        }

        if (
            self.workflow_config.save_metadata
            and self.paths_config.output_metadata_path
        ):
            # Create metadata directory
            metadata_path = Path(self.paths_config.output_metadata_path)
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            logging.info(f"Metadata saved to: {metadata_path}")

        return metadata
