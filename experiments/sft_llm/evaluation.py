#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   evaluation.py
@Time    :   2025/08/18 22:50:12
@Author  :   Dawn Li
@Version :   1.0
@Contact :   dawnli619215645@gmail.com
@Desc    :   Evaluation
"""

import argparse
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd
import torch
from accelerate import PartialState
from datasets import Dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from flow import FlowConfig
from flow.evaluation import EvaluationPipeline
from flow.tokenization import TokenizationPipeline, UnifiedTokenizer
from flow.utils import setup_logging

from .training import build_sft_dataset, create_flow_config


def get_components(
    args: argparse.Namespace,
) -> Tuple[FlowConfig, Dataset, AutoTokenizer]:
    # Dataset
    dataset_dir = args.eval_dataset_dir
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Evaluation dataset not found at: {dataset_dir}")
    with PartialState().main_process_first():
        dataset = build_sft_dataset(dataset_dir, remove_columns=False)

    logging.info(f"Loaded evaluation dataset with {len(dataset)} samples")

    # Model and Tokenizer
    flow_config = create_flow_config(args.output_dir)
    model_dir = Path(flow_config.training.paths.model_save_dir)
    checkpoints = list(model_dir.glob("checkpoint-*"))
    if not checkpoints:
        raise FileNotFoundError("No checkpoints found in model directory")
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.name.split("-")[-1]))
    with PartialState().main_process_first():
        tokenizer = AutoTokenizer.from_pretrained(
            latest_checkpoint,
            use_fast=True,
            padding_side="left",  # Required for decoder-only models
        )

    logging.info(f"Loaded tokenizer from last checkpoint: {latest_checkpoint}")
    return flow_config, dataset, tokenizer


def generate_llm_predictions(
    model: str,
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    flow_config: FlowConfig,
    skip_inference: bool = False,
) -> Dataset:
    """
    Generate LLM predictions using vLLM for high-throughput inference (multi-GPU supported).

    Args:
        model: Name of the model to use for generation
        dataset: Dataset with prompt-completion format from build_sft_dataset
        tokenizer: HF tokenizer instance, used to fetch eos_token_id
        flow_config: FlowConfig containing generation parameters and model path
        skip_inference: If True, skip inference and load llm_coords from CSV file

    Returns:
        Dataset with 'llm_coords' column containing generated coordinate sequences
    """
    num_samples = len(dataset)

    # Skip inference mode: load llm_coords from saved CSV metrics file
    if skip_inference:
        metrics_csv_path = Path(flow_config.evaluation.paths.metrics_dir) / "evaluation_metrics.csv"

        if not metrics_csv_path.exists():
            raise FileNotFoundError(
                f"Cannot skip inference: evaluation_metrics.csv not found at {metrics_csv_path}"
            )

        logging.info(f"⚡ Skipping inference - loading llm_coords from {metrics_csv_path}")

        # Read CSV file
        df = pd.read_csv(metrics_csv_path)

        if "llm_coords" not in df.columns:
            raise ValueError(
                f"llm_coords column not found in CSV file at {metrics_csv_path}"
            )

        if len(df) != len(dataset):
            raise ValueError(
                f"CSV dataset size ({len(df)}) does not match current dataset size ({len(dataset)})"
            )

        # Add llm_coords from CSV to current dataset
        dataset = dataset.add_column("llm_coords", df["llm_coords"].tolist())
        logging.info(f"✅ Loaded llm_coords for {len(dataset)} samples from CSV file")
        return dataset

    logging.info(f"Starting vLLM inference on {num_samples} samples")

    # Resolve latest checkpoint directory from flow_config
    model_dir = Path(flow_config.training.paths.model_save_dir)
    checkpoints = list(model_dir.glob("checkpoint-*"))
    if not checkpoints:
        raise FileNotFoundError("No checkpoints found in model directory for vLLM load")
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.name.split("-")[-1]))

    # Extract generation parameters
    generation_config = flow_config.evaluation.generation

    # Build vLLM sampling params
    sampling_kwargs = {
        "max_tokens": generation_config.max_new_tokens,
        "temperature": generation_config.temperature
        if generation_config.do_sample
        else 0.0,
        "top_p": generation_config.top_p if generation_config.do_sample else 1.0,
        "top_k": generation_config.top_k if generation_config.do_sample else -1,
    }

    # Add stop token ids if available from tokenizer
    stop_token_ids = []
    if tokenizer is not None and getattr(tokenizer, "eos_token_id", None) is not None:
        stop_token_ids.append(int(tokenizer.eos_token_id))
    if stop_token_ids:
        sampling_kwargs["stop_token_ids"] = stop_token_ids

    sampling_params = SamplingParams(**sampling_kwargs)

    # Infer tensor parallel size from available GPUs
    tp_size = max(1, torch.cuda.device_count())
    logging.info(
        f"Initializing vLLM engine from checkpoint: {latest_checkpoint} (TP={tp_size})"
    )
    llm = LLM(
        model=model,
        tensor_parallel_size=tp_size,
        trust_remote_code=True,
        dtype="auto",
        gpu_memory_utilization=0.5,
        enable_lora=True
    )

    # Collect prompts
    prompts = dataset["prompt"]
    logging.info("🎯 Generation parameters (vLLM):")
    logging.info(f"   Max new tokens: {generation_config.max_new_tokens}")
    logging.info(f"   Beam search: {generation_config.num_beams}")
    logging.info(f"   Do sample: {generation_config.do_sample}")
    if generation_config.do_sample:
        logging.info(f"   Temperature: {generation_config.temperature}")
        logging.info(f"   Top-p: {generation_config.top_p}")
        logging.info(f"   Top-k: {generation_config.top_k}")
    logging.info(f"   TP size: {tp_size}")

    # Run inference
    logging.info("⚡ Running vLLM inference...")
    inference_start_time = time.time()
    outputs = llm.generate(
        prompts,
        sampling_params=sampling_params,
        use_tqdm=True,
        lora_request=LoRARequest("sql_adapter", 1, str(latest_checkpoint)),
    )

    # Extract top hypothesis text for each prompt
    predictions = []
    for out in outputs:
        if not out.outputs:
            predictions.append("")
            continue
        predictions.append(out.outputs[0].text.strip())

    inference_time = time.time() - inference_start_time

    logging.info(f"✅ vLLM inference completed in {inference_time:.2f}s")
    if inference_time > 0:
        logging.info(f"   Throughput: {num_samples / inference_time:.2f} samples/sec")

    # Ensure predictions match dataset size
    if len(predictions) != len(dataset):
        logging.error(
            f"Prediction count ({len(predictions)}) != dataset size ({len(dataset)}). Adjusting..."
        )
        raise ValueError("Prediction count does not match dataset size.")

    # Add predictions to dataset
    dataset = dataset.add_column("llm_coords", predictions)

    # Log sample prediction
    if len(predictions) > 0:
        sample_prediction = predictions[0]
        logging.info(
            f"Sample LLM prediction: {sample_prediction[:100]}{'...' if len(sample_prediction) > 100 else ''}"
        )

    logging.info(f"Generated predictions for {len(dataset)} samples")
    return dataset


def remove_tail(dataset: Dataset) -> Dataset:
    """
    Find the last ')', and remove all text after that, to avoid truncation interference at the end due to insufficient context length

    Args:
        dataset: Dataset with 'llm_coords' column containing generated sequences

    Returns:
        Dataset with 'llm_coords' column containing preprocessed sequences
    """
    logging.info("Starting tail removal from sequences...")

    dataset = dataset.map(
        lambda x: {
            "llm_coords": (
                x["llm_coords"][: x["llm_coords"].rfind(")") + 1]
                if ")" in x["llm_coords"]
                else x["llm_coords"]
            )
        },
        desc="Removing tail from sequences",
    )

    return dataset


def filter_completion_purity(dataset: Dataset) -> Dataset:
    """
    Filter sequences to ensure completion purity - only valid characters allowed.

    Valid characters: 0-9, comma, parentheses, and direction symbols (> for BRANCH, < for END).
    Invalid sequences are replaced with empty strings to maintain dataset structure.

    Args:
        dataset: Dataset with 'llm_coords' column containing generated sequences

    Returns:
        Dataset with 'completion_valid_coords' column containing filtered sequences
    """
    logging.info("Starting completion purity filtering...")

    def validate_purity(batch):
        """Validate that sequences contain only allowed characters."""
        valid_coords = []

        for seq in batch["llm_coords"]:
            if not seq or not isinstance(seq, str):
                valid_coords.append("")
                continue

            # Check if sequence contains only valid characters: digits, comma, parentheses, > and <
            if re.match(r"^[0-9,()<>\s]*$", seq.strip()):
                valid_coords.append(seq.strip())
            else:
                valid_coords.append("")

        return {"completion_valid_coords": valid_coords}

    with PartialState().main_process_first():
        dataset_with_purity = dataset.map(
            validate_purity,
            batched=True,
            desc="Filtering completion purity",
        )

    # Count valid sequences
    valid_count = sum(
        1 for x in dataset_with_purity["completion_valid_coords"] if x and x.strip()
    )
    total_count = len(dataset_with_purity)
    logging.info(
        f"Completion purity filtering: {valid_count}/{total_count} ({valid_count / total_count:.2%}) sequences passed"
    )

    return dataset_with_purity


def filter_coordinate_completeness(dataset: Dataset) -> Dataset:
    """
    Filter sequences to ensure coordinate completeness - all coordinate patterns are valid.

    Verifies that all patterns matching '({x},{y},{z})' have numeric x, y, z values.
    Invalid sequences are replaced with empty strings.

    Args:
        dataset: Dataset with 'completion_valid_coords' column

    Returns:
        Dataset with 'coord_valid_coords' column containing validated sequences
    """
    logging.info("Starting coordinate completeness filtering...")

    # Coordinate pattern: (x, y, z) where x, y, z are integers (can be negative)
    coord_pattern = re.compile(r"\((-?\d+),\s*(-?\d+),\s*(-?\d+)\)")

    def validate_coordinates(batch):
        """Validate that all coordinate patterns in sequences are well-formed."""
        valid_coords = []

        for seq in batch["completion_valid_coords"]:
            if not seq or not isinstance(seq, str):
                valid_coords.append("")
                continue

            # Find all coordinate patterns in the sequence
            coord_matches = coord_pattern.findall(seq)

            if not coord_matches:
                # No coordinates found - this might be valid for empty sequences
                valid_coords.append(seq)
                continue

            # Check if all found coordinates are at expected positions
            # Reconstruct the sequence with found coordinates to verify completeness
            reconstructed_parts = []
            last_end = 0

            try:
                for match in coord_pattern.finditer(seq):
                    # Add text before coordinate
                    reconstructed_parts.append(seq[last_end : match.start()])
                    # Add the coordinate
                    x, y, z = (
                        int(match.group(1)),
                        int(match.group(2)),
                        int(match.group(3)),
                    )
                    reconstructed_parts.append(f"({x},{y},{z})")
                    last_end = match.end()

                # Add remaining text
                reconstructed_parts.append(seq[last_end:])

                # If reconstruction matches original (ignoring whitespace), it's valid
                original_clean = re.sub(r"\s+", "", seq)
                reconstructed_clean = re.sub(r"\s+", "", "".join(reconstructed_parts))

                if original_clean == reconstructed_clean:
                    valid_coords.append(seq)
                else:
                    valid_coords.append("")

            except (ValueError, IndexError):
                # Parsing error - invalid coordinate
                valid_coords.append("")

        return {"coord_valid_coords": valid_coords}

    with PartialState().main_process_first():
        dataset_with_coords = dataset.map(
            validate_coordinates,
            batched=True,
            desc="Filtering coordinate completeness",
        )

    # Count valid sequences
    valid_count = sum(
        1 for x in dataset_with_coords["coord_valid_coords"] if x and x.strip()
    )
    total_count = len(dataset_with_coords)
    logging.info(
        f"Coordinate completeness filtering: {valid_count}/{total_count} ({valid_count / total_count:.2%}) sequences passed"
    )

    return dataset_with_coords


def filter_usable_sequences(dataset: Dataset) -> Dataset:
    """
    Combine both validation criteria to identify usable sequences.

    A sequence is usable if it passes both completion purity and coordinate completeness checks.

    Args:
        dataset: Dataset with both 'completion_valid_coords' and 'coord_valid_coords' columns

    Returns:
        Dataset with 'valid_coords' column containing sequences that meet both criteria
    """
    logging.info("Starting usability filtering...")

    def combine_validation_criteria(batch):
        """Combine purity and completeness validation results."""
        valid_coords = []

        for purity_seq, coord_seq in zip(
            batch["completion_valid_coords"], batch["coord_valid_coords"]
        ):
            # A sequence is valid if it passes both purity and coordinate checks
            if (
                purity_seq
                and purity_seq.strip()
                and coord_seq
                and coord_seq.strip()
                and purity_seq.strip() == coord_seq.strip()
            ):
                valid_coords.append(coord_seq.strip())
            else:
                valid_coords.append("")

        return {"valid_coords": valid_coords}

    with PartialState().main_process_first():
        dataset_with_valid = dataset.map(
            combine_validation_criteria,
            batched=True,
            desc="Combining validation criteria",
        )

    # Count final valid sequences
    valid_count = sum(1 for x in dataset_with_valid["valid_coords"] if x and x.strip())
    total_count = len(dataset_with_valid)
    logging.info(
        f"Usability filtering: {valid_count}/{total_count} ({valid_count / total_count:.2%}) sequences are usable"
    )

    return dataset_with_valid


def restore_coordinates(dataset: Dataset) -> Dataset:
    """
    Restore coordinate sequences by reversing the completion processing from training.py.

    Reverses the following transformations:
    1. Replace '>' with '[BRANCH]'
    2. Replace '<' with '[END]'
    3. Add spaces back after commas in coordinates

    Args:
        dataset: Dataset with 'valid_coords' column containing filtered sequences

    Returns:
        Dataset with 'restored_coords' column containing restored coordinate sequences
    """
    logging.info("Starting coordinate restoration...")

    def reverse_completion_processing(batch):
        """Reverse the completion processing applied in training.py."""
        restored = []

        for coords in batch["valid_coords"]:
            if not coords or not isinstance(coords, str):
                restored.append("")
                continue

            # Step 1: Replace direction symbols with branch/end tokens
            # Reverse training.py: replace '>' with '[BRANCH]'
            # Reverse training.py: replace '<' with '[END]'
            restored_seq = coords.replace(">", "[BRANCH]").replace("<", "[END]")

            # Step 2: Add spaces back after commas in coordinates
            # Reverse training.py: add spaces back after commas
            # Use regex to find coordinates and add spaces: (x,y,z) -> (x, y, z)
            restored_seq = re.sub(
                r"\((-?\d+),(-?\d+),(-?\d+)\)", r"(\1, \2, \3)", restored_seq
            )

            restored.append(restored_seq)

        return {"restored_coords": restored}

    with PartialState().main_process_first():
        dataset_with_restored = dataset.map(
            reverse_completion_processing,
            batched=True,
            desc="Restoring coordinates",
        )

    # Count successfully restored sequences
    restored_count = sum(
        1 for x in dataset_with_restored["restored_coords"] if x and x.strip()
    )
    total_count = len(dataset_with_restored)
    logging.info(
        f"Coordinate restoration: {restored_count}/{total_count} ({restored_count / total_count:.2%}) sequences restored"
    )

    return dataset_with_restored


def validate_tree_structure(
    dataset: Dataset, unified_tokenizer: UnifiedTokenizer
) -> Dataset:
    """
    Validate tree structure and check displacement patterns.

    Builds tree structure from restored coordinates and validates that displacements between
    adjacent coordinates have only one positive value among x, y, z (Manhattan distance movement).

    Args:
        dataset: Dataset with 'restored_coords' column
        unified_tokenizer: UnifiedTokenizer instance for building tree structures

    Returns:
        Dataset with 'available_coords' column containing structurally valid sequences
    """
    logging.info("Starting tree structure validation...")

    def is_valid_displacement_pattern(tree_root) -> bool:
        """
        Check if tree has valid displacement patterns between adjacent coordinates.

        Valid displacement: only one coordinate (x, y, or z) should have positive change
        between adjacent nodes, representing Manhattan distance routing.
        """
        if not tree_root or not hasattr(tree_root, "coord") or not tree_root.coord:
            return False

        def validate_node_and_children(node) -> bool:
            """Recursively validate displacement patterns in tree."""
            if not node or not hasattr(node, "coord") or not node.coord:
                return False

            # Check displacement from parent to this node
            if node.parent and hasattr(node.parent, "coord") and node.parent.coord:
                parent_coord = node.parent.coord
                current_coord = node.coord

                # Calculate displacement
                dx = abs(current_coord.x - parent_coord.x)
                dy = abs(current_coord.y - parent_coord.y)
                dz = abs(current_coord.m - parent_coord.m)  # metal layer difference

                # Valid displacement: only one dimension should change (Manhattan distance)
                positive_changes = sum([1 for d in [dx, dy, dz] if d > 0])
                if positive_changes != 1:
                    return False

            # Recursively validate all children
            for child in node.children:
                if not validate_node_and_children(child):
                    return False

            return True

        return validate_node_and_children(tree_root)

    def check_displacement(batch):
        """Validate displacement patterns for batch of coordinate sequences."""
        available = []

        for coords in batch["restored_coords"]:
            if not coords or not isinstance(coords, str):
                available.append([])
                continue

            try:
                # Split the restored coordinate sequence into tokens
                coord_tokens = coords.strip()
                coord_tokens = re.findall(
                    r"\(-?\d+\s*,\s*-?\d+\s*,\s*-?\d+\)|[<>]", coords
                )

                mapping = {
                    ">": unified_tokenizer.get_special_token("BRANCH_TOKEN"),
                    "<": unified_tokenizer.get_special_token("END_TOKEN"),
                }
                coord_tokens = [mapping.get(tok, tok) for tok in coord_tokens]

                if not coord_tokens:
                    available.append([])
                    continue

                # Build tree structure using UnifiedTokenizer
                tree_root = unified_tokenizer.build_tree_structure(coord_tokens)

                # Validate displacement patterns
                if is_valid_displacement_pattern(tree_root):
                    available.append(coord_tokens)
                else:
                    available.append([])

            except Exception as e:
                # Handle parsing errors gracefully
                logging.debug(
                    f"Tree structure validation error for sequence '{coords[:50]}...': {e}"
                )
                available.append([])

        return {"available_coords": available}

    with PartialState().main_process_first():
        dataset_with_available = dataset.map(
            check_displacement,
            batched=True,
            desc="Validating tree structure",
        )

    # Count structurally valid sequences
    available_count = sum(
        1 for coords_list in dataset_with_available["available_coords"] if coords_list
    )
    total_count = len(dataset_with_available)
    logging.info(
        f"Tree structure validation: {available_count}/{total_count} ({available_count / total_count:.2%}) sequences are structurally valid"
    )

    return dataset_with_available


def convert_to_predictions(
    dataset: Dataset, unified_tokenizer: UnifiedTokenizer
) -> Dataset:
    """
    Convert available_coords to predictions format using UnifiedTokenizer.

    Uses convert_relative_target_to_directional_tokens to transform coordinate sequences
    into directional token format expected by flow evaluation pipeline.

    Args:
        dataset: Dataset with 'available_coords' column containing valid coordinate sequences
        unified_tokenizer: UnifiedTokenizer instance for coordinate conversion

    Returns:
        Dataset with 'predictions' column containing directional token sequences
    """
    logging.info("Converting coordinates to prediction format...")

    def convert_coords(batch):
        """Convert coordinate sequences to directional token predictions."""
        predictions = []

        for coords_list in batch["available_coords"]:
            if not coords_list:
                predictions.append("")
                continue

            try:
                # Parse first coordinate to get reference point
                first_coord_str = None
                for token in coords_list:
                    if token not in ["[BRANCH]", "[END]"]:
                        first_coord_str = token
                        break

                if first_coord_str is None:
                    predictions.append("")
                    continue

                # Parse the first coordinate as reference point
                first_coord = unified_tokenizer.parse_coord(first_coord_str)

                # Convert to relative coords_list: each 'coord' = 'coord' - first_coord
                relative_coords_list = []
                for token in coords_list:
                    if token in ["[BRANCH]", "[END]"]:
                        # Keep special tokens as-is
                        relative_coords_list.append(token)
                    else:
                        # Parse coordinate string, compute relative, convert back to string
                        coord = unified_tokenizer.parse_coord(token)
                        relative_coord = coord - first_coord
                        relative_coords_list.append(str(relative_coord))

                # Use UnifiedTokenizer method to convert to directional tokens
                pred_tokens = (
                    unified_tokenizer.convert_relative_target_to_directional_token(
                        relative_coords_list
                    )
                )
                predictions.append(pred_tokens)

            except Exception as e:
                # Handle conversion errors gracefully
                logging.debug(
                    f"Conversion error for sequence '{coords_list[:50] if isinstance(coords_list, str) else coords_list}...': {e}"
                )
                predictions.append("")

        return {"predictions": predictions}

    with PartialState().main_process_first():
        dataset_with_predictions = dataset.map(
            convert_coords,
            batched=True,
            desc="Converting to predictions format",
        )

    # Count successful conversions
    prediction_count = sum(
        1 for x in dataset_with_predictions["predictions"] if x and x.strip()
    )
    total_count = len(dataset_with_predictions)
    logging.info(
        f"Coordinate conversion: {prediction_count}/{total_count} ({prediction_count / total_count:.2%}) sequences converted"
    )

    return dataset_with_predictions


def preprocess_for_flow_evaluation(
    dataset: Dataset, flow_config: FlowConfig
) -> Dataset:
    """
    Preprocess dataset using TokenizationPipeline to prepare for flow evaluation.

    Uses TokenizationPipeline.preprocess_corpus to ensure dataset format matches
    flow evaluation pipeline expectations.

    Args:
        dataset: Dataset with predictions column
        flow_config: FlowConfig instance for tokenization pipeline

    Returns:
        Dataset preprocessed for flow evaluation pipeline
    """
    logging.info("Preprocessing dataset for flow evaluation...")

    # Initialize TokenizationPipeline with flow configuration
    tokenization_pipeline = TokenizationPipeline(flow_config)

    # Use existing preprocess_corpus method
    dataset = tokenization_pipeline.preprocess_corpus(dataset)
    dataset = tokenization_pipeline.build_token_dataset(dataset, remove_columns=False)

    logging.info(f"Dataset preprocessing completed for {len(dataset)} samples")
    return dataset


def run_flow_evaluation(dataset: Dataset, flow_config: FlowConfig) -> Dataset:
    """
    Run comprehensive flow evaluation pipeline using existing EvaluationPipeline.

    Follows the exact pattern from flow/launch_evaluation.py lines 194-199 to ensure
    seamless integration with existing evaluation infrastructure.

    Args:
        dataset: Dataset with predictions column and preprocessed format
        flow_config: FlowConfig instance with properly configured evaluation paths

    Returns:
        Dataset with comprehensive routing evaluation metrics calculated
    """
    logging.info("Starting flow evaluation pipeline...")

    evaluation_pipeline = EvaluationPipeline(flow_config)

    # Create output directories
    with PartialState().main_process_first():
        Path(flow_config.evaluation.paths.output_dir).mkdir(parents=True, exist_ok=True)
        Path(flow_config.evaluation.paths.metrics_dir).mkdir(
            parents=True, exist_ok=True
        )

    logging.info("Calculating comprehensive routing metrics...")
    dataset = evaluation_pipeline.calculate_metrics(dataset)

    logging.info("Flow evaluation pipeline completed successfully")
    logging.info(
        f"Dataset now contains comprehensive metrics for {len(dataset)} samples"
    )

    return dataset


def analyze_filtering_statistics(dataset: Dataset) -> Dict[str, Any]:
    """
    Analyze filtering pipeline statistics across all stages.

    Counts valid sequences at each filtering stage and calculates success rates
    for comprehensive statistical analysis.

    Args:
        dataset: Dataset containing all filtering stage columns

    Returns:
        Dictionary containing counts, rates, and statistical metrics for each stage
    """
    logging.info("Analyzing filtering pipeline statistics...")

    stats = {}
    total_samples = len(dataset)
    stats["total_samples"] = total_samples

    # Define filtering stages in order
    filtering_stages = [
        "llm_coords",
        "completion_valid_coords",
        "coord_valid_coords",
        "valid_coords",
        "available_coords",
    ]

    # Count non-empty sequences at each stage
    for column in filtering_stages:
        if column in dataset.column_names:
            valid_count = sum(1 for x in dataset[column] if x)
            stats[f"{column}_count"] = valid_count
            stats[f"{column}_rate"] = (
                valid_count / total_samples if total_samples > 0 else 0.0
            )
        else:
            stats[f"{column}_count"] = 0
            stats[f"{column}_rate"] = 0.0

    # Calculate stage-to-stage retention rates
    for i in range(1, len(filtering_stages)):
        prev_stage = filtering_stages[i - 1]
        curr_stage = filtering_stages[i]

        prev_count = stats.get(f"{prev_stage}_count", 0)
        curr_count = stats.get(f"{curr_stage}_count", 0)

        retention_rate = curr_count / prev_count if prev_count > 0 else 0.0
        stats[f"{curr_stage}_retention_rate"] = retention_rate

    # Calculate cumulative filtering efficiency
    initial_count = stats.get("llm_coords_count", 0)
    final_count = stats.get("available_coords_count", 0)
    overall_efficiency = final_count / initial_count if initial_count > 0 else 0.0
    stats["overall_filtering_efficiency"] = overall_efficiency

    # Extract quality metrics from flow evaluation if available
    if len(dataset) > 0 and "is_perfect_match" in dataset.column_names:
        # Calculate quality metrics
        quality_columns = [
            "is_perfect_match",
            "is_branch_struct_match",
            "is_leaf_set_match",
            "leaf_accuracy",
            "leaf_precision",
            "leaf_recall",
            "leaf_iou",
            "edge_accuracy",
            "edge_precision",
            "edge_recall",
        ]

        for col in quality_columns:
            if col in dataset.column_names:
                if col.startswith("is_"):
                    # Boolean metrics - calculate percentage
                    stats[f"quality_{col}"] = sum(dataset[col]) / len(dataset)
                else:
                    # Continuous metrics - calculate mean
                    stats[f"quality_{col}"] = sum(dataset[col]) / len(dataset)

    logging.info(f"Statistical analysis completed: {len(stats)} metrics calculated")
    return stats


def generate_scientific_report(statistics: Dict[str, Any]) -> str:
    """
    Generate comprehensive scientific publication-ready analysis report.

    Creates detailed report with filtering statistics, quality metrics, and
    comparative analysis suitable for research documentation.

    Args:
        statistics: Statistical metrics from analyze_filtering_statistics

    Returns:
        Formatted scientific report string
    """
    logging.info("Generating scientific publication report...")

    total_samples = statistics.get("total_samples", 0)

    # Calculate key insights
    llm_success_rate = statistics.get("llm_coords_rate", 0)
    final_success_rate = statistics.get("available_coords_rate", 0)
    overall_efficiency = statistics.get("overall_filtering_efficiency", 0)

    report = f"""
SFT LLM Routing Generation Evaluation Report
============================================

## Executive Summary
This report presents a comprehensive evaluation of Supervised Fine-Tuning (SFT) Large Language Models for routing generation tasks, analyzing the quality and validity of generated coordinate sequences through a multi-stage filtering and validation pipeline.

## Dataset Overview
- **Total Evaluation Samples**: {total_samples:,}
- **Initial LLM Predictions**: {statistics.get("llm_coords_count", 0):,} ({llm_success_rate:.2%})
- **Final Valid Sequences**: {statistics.get("available_coords_count", 0):,} ({final_success_rate:.2%})
- **Overall Pipeline Efficiency**: {overall_efficiency:.2%}

## Filtering Pipeline Analysis

### Stage 1: Completion Purity Filtering
- **Valid Character Sequences**: {statistics.get("completion_valid_coords_count", 0):,} ({statistics.get("completion_valid_coords_rate", 0):.2%})
- **Retention Rate**: {statistics.get("completion_valid_coords_retention_rate", 0):.2%}
- **Purpose**: Validates sequences contain only allowed characters (digits, coordinates, direction symbols)

### Stage 2: Coordinate Completeness Filtering  
- **Valid Coordinate Patterns**: {statistics.get("coord_valid_coords_count", 0):,} ({statistics.get("coord_valid_coords_rate", 0):.2%})
- **Retention Rate**: {statistics.get("coord_valid_coords_retention_rate", 0):.2%}
- **Purpose**: Ensures all coordinate patterns (x,y,z) have proper numeric values

### Stage 3: Usability Filtering
- **Usable Sequences**: {statistics.get("valid_coords_count", 0):,} ({statistics.get("valid_coords_rate", 0):.2%})
- **Retention Rate**: {statistics.get("valid_coords_retention_rate", 0):.2%}  
- **Purpose**: Combines purity and completeness criteria for integrated validation

### Stage 4: Tree Structure Validation
- **Structurally Valid Sequences**: {statistics.get("available_coords_count", 0):,} ({statistics.get("available_coords_rate", 0):.2%})
- **Retention Rate**: {statistics.get("available_coords_retention_rate", 0):.2%}
- **Purpose**: Validates Manhattan distance routing patterns and tree structure integrity

## Quality Metrics Analysis"""

    # Add quality metrics if available
    if any(k.startswith("quality_") for k in statistics.keys()):
        report += f"""

### Routing Quality Assessment
- **Perfect Match Rate**: {statistics.get("quality_is_perfect_match", 0):.2%}
- **Branch Structure Match**: {statistics.get("quality_is_branch_struct_match", 0):.2%}  
- **Leaf Set Match**: {statistics.get("quality_is_leaf_set_match", 0):.2%}

### Detailed Quality Metrics
- **Leaf Accuracy**: {statistics.get("quality_leaf_accuracy", 0):.3f}
- **Leaf Precision**: {statistics.get("quality_leaf_precision", 0):.3f}
- **Leaf Recall**: {statistics.get("quality_leaf_recall", 0):.3f}
- **Leaf IoU**: {statistics.get("quality_leaf_iou", 0):.3f}

- **Edge Accuracy**: {statistics.get("quality_edge_accuracy", 0):.3f}
- **Edge Precision**: {statistics.get("quality_edge_precision", 0):.3f}
- **Edge Recall**: {statistics.get("quality_edge_recall", 0):.3f}"""

    report += f"""

## Key Findings

### 1. Filtering Pipeline Effectiveness
The multi-stage filtering pipeline demonstrates {"high" if overall_efficiency > 0.5 else "moderate" if overall_efficiency > 0.2 else "low"} effectiveness with an overall retention rate of {overall_efficiency:.2%}. The most significant filtering occurs at the {"completion purity" if statistics.get("completion_valid_coords_retention_rate", 1) < 0.8 else "coordinate completeness" if statistics.get("coord_valid_coords_retention_rate", 1) < 0.8 else "tree structure validation"} stage.

### 2. SFT LLM Performance Assessment  
The SFT LLM successfully generates valid routing sequences in {final_success_rate:.1%} of cases, indicating {"strong" if final_success_rate > 0.7 else "moderate" if final_success_rate > 0.3 else "limited"} capability in learning routing pattern generation from natural language instructions.

### 3. Quality Distribution
{"High-quality routing patterns are consistently generated" if statistics.get("quality_is_perfect_match", 0) > 0.5 else "Moderate routing quality with room for improvement" if statistics.get("quality_is_perfect_match", 0) > 0.2 else "Significant quality challenges identified"} based on comprehensive routing metrics evaluation.

## Conclusions

This evaluation provides comprehensive insights into SFT LLM performance for routing generation tasks. The multi-stage validation pipeline effectively identifies high-quality routing sequences while providing detailed analysis of failure modes and quality characteristics.

### Recommendations
1. **Pipeline Optimization**: {"Focus on improving LLM instruction following" if statistics.get("completion_valid_coords_retention_rate", 1) < 0.8 else "Enhance coordinate format training" if statistics.get("coord_valid_coords_retention_rate", 1) < 0.8 else "Strengthen tree structure learning"}
2. **Quality Enhancement**: Target improvements in {"leaf accuracy" if statistics.get("quality_leaf_accuracy", 1) < 0.7 else "edge precision" if statistics.get("quality_edge_precision", 1) < 0.7 else "overall routing structure"} 
3. **Future Work**: Investigate correlation between filtering stage failures and final routing quality

---
*Report generated by SFT LLM Routing Evaluation Pipeline*
*Total samples analyzed: {total_samples:,}*
"""

    logging.info("Scientific report generation completed")
    return report


def evaluation(args: argparse.Namespace):
    """
    Main evaluation function orchestrating the complete SFT LLM evaluation pipeline.

    Executes the full pipeline from LLM inference to final analysis, including:
    1. LLM inference and prediction generation
    2. Multi-stage coordinate filtering and validation
    3. Coordinate restoration and tree structure validation
    4. Flow evaluation pipeline integration
    5. Comprehensive analysis and scientific reporting

    Args:
        args: Command-line arguments containing paths and configuration
    """
    logging.info("🚀 Starting SFT LLM Routing Generation Evaluation")
    logging.info(f"   Output directory: {args.output_dir}")
    logging.info(f"   Evaluation dataset: {args.eval_dataset_dir}")

    try:
        # Load components using existing get_components pattern
        logging.info("📁 Loading evaluation components...")
        flow_config, dataset, tokenizer = get_components(args)

        # Generate predictions using multi-GPU accelerate
        if args.skip_inference:
            logging.info("🤖 Skipping LLM inference (loading from saved dataset)...")
        else:
            logging.info("🤖 Starting multi-GPU LLM inference...")
        dataset = generate_llm_predictions(
            args.pretrained_model,
            dataset,
            tokenizer,
            flow_config,
            skip_inference=args.skip_inference,
        )

        # Apply multi-stage filtering pipeline
        logging.info("🔍 Applying coordinate filtering pipeline...")
        dataset = remove_tail(dataset)
        dataset = filter_completion_purity(dataset)
        dataset = filter_coordinate_completeness(dataset)
        dataset = filter_usable_sequences(dataset)

        # Restore coordinates and validate tree structure
        logging.info("🔧 Restoring coordinates and validating tree structure...")
        unified_tokenizer = UnifiedTokenizer()
        dataset = restore_coordinates(dataset)
        dataset = validate_tree_structure(dataset, unified_tokenizer)

        # Convert to flow format and run evaluation
        logging.info("⚡ Converting to flow evaluation format...")
        dataset = convert_to_predictions(dataset, unified_tokenizer)
        dataset = preprocess_for_flow_evaluation(dataset, flow_config)
        dataset = run_flow_evaluation(dataset, flow_config)

        # Generate comprehensive analysis and save results
        logging.info("📊 Generating comprehensive analysis...")
        statistics = analyze_filtering_statistics(dataset)
        report = generate_scientific_report(statistics)

        # Save results to structured files
        logging.info("💾 Saving evaluation results...")
        output_dir = Path(flow_config.evaluation.paths.output_dir)
        with PartialState().main_process_first():
            output_dir.mkdir(parents=True, exist_ok=True)

        # Save detailed statistics and report
        with PartialState().main_process_first():
            results_path = output_dir / "evaluation_results.json"
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "statistics": statistics,
                        "report": report,
                        "total_samples": len(dataset),
                        "evaluation_timestamp": str(logging.root.name),
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

            # Save report as separate markdown file
            report_path = output_dir / "evaluation_report.md"
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report)

        # Save dataset for further analysis
        dataset_path = output_dir / "evaluated_dataset"
        with PartialState().main_process_first():
            dataset.save_to_disk(dataset_path)

        # Log completion summary
        logging.info("✅ SFT LLM Routing Generation Evaluation completed successfully!")
        logging.info(f"   Results saved to: {results_path}")
        logging.info(f"   Report saved to: {report_path}")
        logging.info(f"   Dataset saved to: {dataset_path}")
        logging.info(f"   Total samples evaluated: {len(dataset)}")
        logging.info(
            f"   Final success rate: {statistics.get('available_coords_rate', 0):.2%}"
        )

    except Exception as e:
        logging.error(f"❌ Evaluation failed: {e}")
        import traceback

        traceback.print_exc()
        raise


def main():
    setup_logging()
    parser = argparse.ArgumentParser(
        description="SFT LLM Routing Generation Evaluation Pipeline",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/mnt/local_data1/liweiguo/experiments/sft_llm"),
        help="Output directory for evaluation results and reports",
    )

    parser.add_argument(
        "--eval-dataset-dir",
        type=Path,
        default=Path("/mnt/local_data1/liweiguo/dataset/experiments/val_bak0926"),
        help="Path to evaluation dataset directory",
    )

    parser.add_argument(
        "--pretrained-model", type=str, default="Qwen/Qwen3-4B-Instruct-2507"
    )

    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Skip LLM inference and load llm_coords from evaluation_metrics.csv",
    )

    args = parser.parse_args()

    logging.info("🔬 SFT LLM Routing Generation Evaluation")
    logging.info(f"   Configuration: {args}")

    evaluation(args)


if __name__ == "__main__":
    main()
