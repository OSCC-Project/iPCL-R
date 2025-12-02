#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   pipeline.py
@Time    :   2025/08/01 11:14:51
@Author  :   Dawn Li
@Version :   1.0
@Contact :   dawnli619215645@gmail.com
@Desc    :   Evaluation pipeline, comprehensive metric calculation (NLP, routing, metrics),
             UnifiedTokenizer integration, and RED distance evaluation for routing patterns
"""

import json
import logging
from collections import Counter, defaultdict
from itertools import chain
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import networkx as nx
import numpy as np
import pandas as pd
import scienceplots
import seaborn as sns
from datasets import Dataset
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer

from flow.config import FlowConfig
from flow.tokenization import Node, UnifiedTokenizer
from flow.utils import CoordinatePoint

FIG_SIZE = (16, 10)
FONT_SIZE = 56
LEGEND_SIZE = 48
LINE_WIDTH = 4
COLOR_PALETTE: List[tuple[float, float, float]] = [
    (249 / 255, 199 / 255, 79 / 255),
    (196 / 255, 194 / 255, 94 / 255),
    (144 / 255, 190 / 255, 109 / 255),
    (106 / 255, 180 / 255, 124 / 255),
    (67 / 255, 170 / 255, 139 / 255),
    (77 / 255, 144 / 255, 142 / 255),
    (87 / 255, 117 / 255, 144 / 255),
]


def _palette_slice(count: int) -> List[tuple[float, float, float]]:
    if count <= 0:
        return []
    palette_len = len(COLOR_PALETTE)
    if count >= palette_len:
        return [COLOR_PALETTE[i % palette_len] for i in range(count)]
    if count == 1:
        return [COLOR_PALETTE[0]]
    indices = [
        int(round(i * (palette_len - 1) / (count - 1)))
        for i in range(count)
    ]
    return [COLOR_PALETTE[idx] for idx in indices]


class EvaluationPipeline:
    """Streamlined evaluation pipeline with UnifiedTokenizer integration"""

    def __init__(self, flow_config: FlowConfig):
        self.flow_config = flow_config
        self.tokenization_config = flow_config.tokenization
        self.evaluation_config = flow_config.evaluation
        self.paths_config = self.evaluation_config.paths

        self.output_dir = Path(self.paths_config.output_dir)
        self.metrics_dir = Path(self.paths_config.metrics_dir)
        self.plots_dir = Path(self.paths_config.plots_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        # Initialize tokenizer
        logging.info("Loading UnifiedTokenizer...")
        self.tokenizer = UnifiedTokenizer.from_pretrained(
            self.tokenization_config.paths.tokenizer_save_dir
        )
        logging.info(
            f"UnifiedTokenizer loaded from: {self.tokenization_config.paths.tokenizer_save_dir}"
        )

    def save_def_inference_metadata_txt(self, dataset: Dataset):
        """Save DEF inference metadata as txt per design."""
        # Group by 'source_design'
        design_list = dataset.unique("source_design")

        logging.info(f"Saving DEF inference metadata for designs: {design_list}")

        for design in design_list:
            design_data = dataset.filter(lambda x: x["source_design"] == design)
            save_dir = self.output_dir / "def_inference_metadata_txt" / design
            save_dir.mkdir(parents=True, exist_ok=True)

            with (
                open(save_dir / f"{design}_ground_truth.txt", "w") as gt_file,
                open(save_dir / f"{design}_predictions.txt", "w") as pred_file,
                open(
                    save_dir / f"{design}_post_predictions.txt", "w"
                ) as post_pred_file,
            ):
                for data in design_data:
                    net_name = data["net_name"]
                    tree_seq = data["tree_seq"]

                    pred_tree = self.tokenizer.build_tree_structure(tree_seq)
                    pred_edges = get_edges(pred_tree)
                    gt_file.write(f"net_name\n{net_name}\n")
                    for start, end in pred_edges:
                        gt_file.write(
                            f"{start.x} {start.y} {start.m // 2} {end.x} {end.y} {end.m // 2}\n"
                        )

                    driver = data["driver"]
                    driver_coord = self.tokenizer.parse_coord(driver)
                    pred_relative_tree_seq = data["prediction_tree_seq"]
                    pred_relative_tree = self.tokenizer.build_tree_structure(
                        pred_relative_tree_seq
                    )
                    pred_relative_edges = get_edges(pred_relative_tree)
                    pred_file.write(f"net_name\n{net_name}\n")
                    for start, end in pred_relative_edges:
                        abs_start = start + driver_coord
                        abs_end = end + driver_coord
                        pred_file.write(
                            f"{abs_start.x} {abs_start.y} {abs_start.m // 2} {abs_end.x} {abs_end.y} {abs_end.m // 2}\n"
                        )

                    post_pred_relative_tree_seq = data["post_opt_tree_seq"]
                    post_pred_relative_tree = self.tokenizer.build_tree_structure(
                        post_pred_relative_tree_seq
                    )
                    post_pred_relative_edges = get_edges(post_pred_relative_tree)
                    post_pred_file.write(f"net_name\n{net_name}\n")
                    for start, end in post_pred_relative_edges:
                        abs_start = start + driver_coord
                        abs_end = end + driver_coord
                        post_pred_file.write(
                            f"{abs_start.x} {abs_start.y} {abs_start.m // 2} {abs_end.x} {abs_end.y} {abs_end.m // 2}\n"
                        )

            gt_file.close()
            pred_file.close()
            post_pred_file.close()

        logging.info(
            f"DEF inference metadata saved to: {self.output_dir / 'def_inference_metadata'}"
        )

    def save_def_inference_metadata(self, dataset: Dataset):
        """Save DEF inference metadata as JSON per design."""
        # Group by 'source_design'
        design_list = dataset.unique("source_design")

        logging.info(
            f"Saving DEF inference metadata as JSON for designs: {design_list}"
        )

        for design in design_list:
            design_data = dataset.filter(lambda x: x["source_design"] == design)

            save_dir = self.output_dir / "def_inference_metadata" / design
            save_dir.mkdir(parents=True, exist_ok=True)

            # Prepare data structures for each file type
            ground_truth_data = []
            predictions_data = []
            post_predictions_data = []

            for data in design_data:
                net_name = data["net_name"]
                tree_seq = data["tree_seq"]
                driver = data["driver"]
                driver_coord = self.tokenizer.parse_coord(driver)

                # Ground Truth Data (from tree_seq)
                pred_tree = self.tokenizer.build_tree_structure(tree_seq)
                pred_edges = get_edges(pred_tree)

                gt_edges = []
                for start, end in pred_edges:
                    gt_edges.append(
                        {
                            "start": {
                                "x": start.x,
                                "y": start.y,
                                "layer": start.m,
                            },
                            "end": {"x": end.x, "y": end.y, "layer": end.m},
                        }
                    )

                ground_truth_data.append(
                    {
                        "net_name": net_name,
                        "edges": gt_edges,
                    }
                )

                # Predictions Data (from prediction_tree_seq)
                pred_relative_tree_seq = data["prediction_tree_seq"]
                pred_relative_tree = self.tokenizer.build_tree_structure(
                    pred_relative_tree_seq
                )
                pred_relative_edges = get_edges(pred_relative_tree)

                pred_edges = []
                for start, end in pred_relative_edges:
                    abs_start = start + driver_coord
                    abs_end = end + driver_coord
                    pred_edges.append(
                        {
                            "start": {
                                "x": abs_start.x,
                                "y": abs_start.y,
                                "layer": abs_start.m,
                            },
                            "end": {
                                "x": abs_end.x,
                                "y": abs_end.y,
                                "layer": abs_end.m,
                            },
                        }
                    )

                predictions_data.append(
                    {
                        "net_name": net_name,
                        "edges": pred_edges,
                    }
                )

                # Post-Optimization Data (from post_opt_tree_seq)
                post_pred_relative_tree_seq = data["post_opt_tree_seq"]
                post_pred_relative_tree = self.tokenizer.build_tree_structure(
                    post_pred_relative_tree_seq
                )
                post_pred_relative_edges = get_edges(post_pred_relative_tree)

                post_pred_edges = []
                for start, end in post_pred_relative_edges:
                    abs_start = start + driver_coord
                    abs_end = end + driver_coord
                    post_pred_edges.append(
                        {
                            "start": {
                                "x": abs_start.x,
                                "y": abs_start.y,
                                "layer": abs_start.m,
                            },
                            "end": {
                                "x": abs_end.x,
                                "y": abs_end.y,
                                "layer": abs_end.m,
                            },
                        }
                    )

                post_predictions_data.append(
                    {
                        "net_name": net_name,
                        "edges": post_pred_edges,
                    }
                )

            # Save JSON files with proper formatting
            gt_file_path = save_dir / f"{design}_ground_truth.json"
            pred_file_path = save_dir / f"{design}_predictions.json"
            post_pred_file_path = save_dir / f"{design}_post_predictions.json"

            with open(gt_file_path, "w") as gt_file:
                json.dump(
                    {
                        "design": design,
                        "type": "ground_truth",
                        "description": "Ground truth routing data from tree_seq",
                        "nets": ground_truth_data,
                    },
                    gt_file,
                    indent=2,
                )

            with open(pred_file_path, "w") as pred_file:
                json.dump(
                    {
                        "design": design,
                        "type": "predictions",
                        "description": "Predicted routing data from prediction_tree_seq",
                        "nets": predictions_data,
                    },
                    pred_file,
                    indent=2,
                )

            with open(post_pred_file_path, "w") as post_pred_file:
                json.dump(
                    {
                        "design": design,
                        "type": "post_predictions",
                        "description": "Post-optimized routing data from post_opt_tree_seq",
                        "nets": post_predictions_data,
                    },
                    post_pred_file,
                    indent=2,
                )

        logging.info(
            f"DEF inference metadata JSON saved to: {self.output_dir / 'def_inference_metadata_json'}"
        )

    def evaluation(self, batch: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        batch = convert_prediction_to_routing(batch, self.tokenizer)
        batch = add_nlp_metrics(batch, self.tokenizer)
        batch = add_routing_metrics(batch, self.tokenizer)
        batch = add_post_opt_metrics(batch, self.tokenizer)

        return batch

    def calculate_metrics(self, dataset: Dataset) -> Dataset:
        """
        Calculate evaluation metrics.

        Args:
            dataset: Dataset containing predictions and ground truth data

        Returns:
            Dataset with calculated metrics attached
        """
        logging.info("Starting evaluation metrics calculation...")

        # Step 1: Convert predictions to routing sequences
        logging.info("Converting predictions to routing sequences...")
        dataset = dataset.map(
            self.evaluation,
            batched=True,
            num_proc=self.evaluation_config.performance.num_workers,
            load_from_cache_file=False,
            desc="Evaluating samples",
        )
        # Log completion summary
        logging.info("Evaluation metrics calculation completed successfully")
        logging.info(
            f"Dataset now contains {len(dataset)} samples with comprehensive metrics"
        )

        # Log sample of calculated metrics
        if len(dataset) > 0:
            sample_fields = list(dataset[0].keys())
            metric_fields = [
                f
                for f in sample_fields
                if any(
                    keyword in f.lower()
                    for keyword in ["rouge", "bleu", "exact_match", "red"]
                )
            ]
            logging.info(
                f"Calculated metrics include: {len(sample_fields)} metric fields"
            )
            logging.info(
                f"Sample metric fields: {metric_fields[:10]}{'...' if len(metric_fields) > 10 else ''}"
            )

        # Convert to df
        logging.info("Converting dataset to DataFrame...")
        df = pd.DataFrame(dataset)
        save_path = self.metrics_dir / "evaluation_metrics.csv"
        df.to_csv(save_path, index=False)
        logging.info(f"Saving evaluation metrics to: {save_path}")

        # Plot
        self.metrics_plot(df)

        # Logging stats
        self.stats_logging(df)

        return dataset

    def metrics_plot(self, df: pd.DataFrame):
        """
        Plot evaluation metrics from the dataset.

        Args:
            df: Dataset containing evaluation metrics
        """
        if scienceplots:
            plt.style.use(["science"])

        # PDF
        plot_PDF(
            {
                "GT": df["wirelength_true"],
                "PRED": df["wirelength_pred"],
                "PO": df["post_wirelength_pred"],
            },
            alias="Wirelength",
            save_path=self.plots_dir / "pdf_wirelength.pdf",
        )
        plot_PDF(
            {
                "GT": df["num_vias_true"],
                "PRED": df["num_vias_pred"],
                "PO": df["post_num_vias_pred"],
            },
            alias="Number of Vias",
            save_path=self.plots_dir / "pdf_num_vias.pdf",
        )

        # KDE
        plot_KDE(
            {
                "GT": df["wirelength_true"],
                "PRED": df["wirelength_pred"],
                "PO": df["post_wirelength_pred"],
            },
            alias="Wirelength",
            save_path=self.plots_dir / "kde_wirelength.pdf",
        )
        plot_KDE(
            {
                "GT": df["num_vias_true"],
                "PRED": df["num_vias_pred"],
                "PO": df["post_num_vias_pred"],
            },
            alias="Number of Vias",
            save_path=self.plots_dir / "kde_num_vias.pdf",
        )

        # KDE Cumulative
        plot_KDE_cumulative(
            {
                "GT": df["wirelength_true"],
                "PRED": df["wirelength_pred"],
                "PO": df["post_wirelength_pred"],
            },
            alias="Wirelength",
            save_path=self.plots_dir / "kde_wirelength_cumulative.pdf",
        )
        plot_KDE_cumulative(
            {
                "GT": df["num_vias_true"],
                "PRED": df["num_vias_pred"],
                "PO": df["post_num_vias_pred"],
            },
            alias="Number of Vias",
            save_path=self.plots_dir / "kde_num_vias_cumulative.pdf",
        )
        # CDF
        plot_CDF(
            {
                "GT": df["wirelength_true"],
                "PRED": df["wirelength_pred"],
                "PO": df["post_wirelength_pred"],
            },
            alias="Wirelength",
            save_path=self.plots_dir / "cdf_wirelength.pdf",
        )
        plot_CDF(
            {
                "GT": df["num_vias_true"],
                "PRED": df["num_vias_pred"],
                "PO": df["post_num_vias_pred"],
            },
            alias="Number of Vias",
            save_path=self.plots_dir / "cdf_num_vias.pdf",
        )

    def stats_logging(self, df: pd.DataFrame):
        """
        Log evaluation statistics from the dataset.

        Args:
            df: Dataset containing evaluation metrics
        """
        if len(df) < 5:
            logging.warning("Dataset is too small for meaningful stats")
            return
        # Calculate stats
        total_samples = len(df)
        num_perfect_matches = sum(df["is_perfect_match"])
        num_branch_struct_match = sum(df["is_branch_struct_match"])
        num_leaf_set_match = sum(df["is_leaf_set_match"])

        avg_leaf_acc = df["leaf_accuracy"].mean()
        avg_leaf_precision = df["leaf_precision"].mean()
        avg_leaf_recall = df["leaf_recall"].mean()
        avg_leaf_iou = df["leaf_iou"].mean()

        avg_edge_acc = df["edge_accuracy"].mean()
        avg_edge_precision = df["edge_precision"].mean()
        avg_edge_recall = df["edge_recall"].mean()
        avg_edge_iou = df["edge_iou"].mean()

        via_pred = sum(df["num_vias_pred"])
        via_true = sum(df["num_vias_true"])

        wirelength_pred = sum(df["wirelength_pred"])
        wirelength_true = sum(df["wirelength_true"])

        via_ratio = via_pred / via_true if via_true > 0 else 0.0
        wirelength_ratio = (
            wirelength_pred / wirelength_true if wirelength_true > 0 else 0.0
        )

        avg_red_score = df["red_similarity_score"].mean()

        num_is_connected_all_loads = sum(df["is_connected_all_loads"])
        num_is_connected_all_loads_post = sum(df["post_is_connected_all_loads"])
        num_is_graceful = sum(df["is_graceful"])
        num_is_graceful_post = sum(df["post_is_graceful"])
        post_remove_wirelength_cost = sum(df["post_remove_wirelength_cost"])
        post_remove_via_cost = sum(df["post_remove_via_cost"])
        post_add_wirelength_cost = sum(df["post_add_wirelength_cost"])
        post_add_via_cost = sum(df["post_add_via_cost"])
        post_required_df = df[~df["is_graceful"]]
        post_required_wirelength = sum(post_required_df["wirelength_pred"])
        post_required_via = sum(post_required_df["num_vias_pred"])

        post_via_pred = sum(df["post_num_vias_pred"])
        post_wirelength_pred = sum(df["post_wirelength_pred"])
        post_via_ratio = post_via_pred / via_true if via_true > 0 else 0.0
        post_wirelength_ratio = (
            post_wirelength_pred / wirelength_true if wirelength_true > 0 else 0.0
        )

        # Log stats
        def log_percent(count, total):
            return (
                f"{count}/{total} ({100 * count / total:.2f}%)"
                if total > 0
                else "0/0 (N/A)"
            )

        def log_int(numerator, denominator):
            return (
                f"{numerator}/{denominator} ({numerator / denominator:.2f} per net)"
                if denominator > 0
                else "0/0 (N/A)"
            )

        logging.info(f"--- Evaluation Summary (on {total_samples} samples) ---")
        logging.info(
            f"Perfect Sequence Match : {log_percent(num_perfect_matches, total_samples)}"
        )
        logging.info(
            f"Branch Structure Match : {log_percent(num_branch_struct_match, total_samples)}"
        )
        logging.info(
            f"Leaf Set Match         : {log_percent(num_leaf_set_match, total_samples)}"
        )
        logging.info(f"Avg. Leaf Accuracy     : {avg_leaf_acc * 100:.2f}%")
        logging.info(f"Avg. Leaf Precision    : {avg_leaf_precision * 100:.2f}%")
        logging.info(f"Avg. Leaf Recall       : {avg_leaf_recall * 100:.2f}%")
        logging.info(f"Avg. Leaf IoU          : {avg_leaf_iou * 100:.2f}%")
        logging.info(f"Avg. Edge Accuracy     : {avg_edge_acc * 100:.2f}%")
        logging.info(f"Avg. Edge Precision    : {avg_edge_precision * 100:.2f}%")
        logging.info(f"Avg. Edge Recall       : {avg_edge_recall * 100:.2f}%")
        logging.info(f"Avg. Edge IoU          : {avg_edge_iou * 100:.2f}%")
        logging.info(f"Via (Pred / GT)        : {via_ratio * 100:.2f}%")
        logging.info(f"Wirelength (Pred / GT) : {wirelength_ratio * 100:.2f}%")
        logging.info(f"RED Similarity Score : {avg_red_score:.4f}")

        print("\n")

        logging.info(
            f"--- Post Opt Summary (on {total_samples - num_is_graceful} samples) ---"
        )
        logging.info(
            f"All Loads Connected        : {log_percent(num_is_connected_all_loads, total_samples)}"
        )
        logging.info(
            f"Post All Loads Connected   : {log_percent(num_is_connected_all_loads_post, total_samples)}"
        )
        logging.info(
            f"Graceful Routing           : {log_percent(num_is_graceful, total_samples)}"
        )
        logging.info(
            f"Post Graceful Routing      : {log_percent(num_is_graceful_post, total_samples)}"
        )
        logging.info(
            f"Remove Wirelength          : Avg. Cost - {log_int(post_remove_wirelength_cost, total_samples - num_is_graceful)}, Avg. Ratio - {log_percent(post_remove_wirelength_cost, post_required_wirelength)}"
        )
        logging.info(
            f"Remove Via                 : Avg. Cost - {log_int(post_remove_via_cost, total_samples - num_is_graceful)}, Avg. Ratio - {log_percent(post_remove_via_cost, post_required_via)}"
        )
        logging.info(
            f"Add Wirelength             : Avg. Cost - {log_int(post_add_wirelength_cost, total_samples - num_is_graceful)}, Avg. Ratio - {log_percent(post_add_wirelength_cost, post_required_wirelength)}"
        )
        logging.info(
            f"Add Via                    : Avg. Cost - {log_int(post_add_via_cost, total_samples - num_is_graceful)}, Avg. Ratio - {log_percent(post_add_via_cost, post_required_via)}"
        )
        logging.info(f"Via (Pred* / GT)           : {post_via_ratio * 100:.2f}%")
        logging.info(f"Wirelength (Pred* / GT)    : {post_wirelength_ratio * 100:.2f}%")
        print("\n")

        # Logging demo
        logging.info("--- Evaluation Demos (First 5) ---")
        for i in range(5):
            entry = df.iloc[i]
            print("-" * 80)
            print(f"DEMO {i + 1}")
            print(f"Loads        : {entry['relative_loads']}")
            print(f"Ground Truth : {entry['relative_tree_seq']}")
            print(f"Prediction   : {entry['prediction_tree_seq']}")
            print(f"Post Opt     : {entry['post_opt_tree_seq']}")
            print(
                f"Metrics      : Perfect={entry['is_perfect_match']}, BranchOK={entry['is_branch_struct_match']}, LeafOK={entry['is_leaf_set_match']}, "
                f"LeafAcc={entry['leaf_accuracy']:.2f}, EdgeAcc={entry['edge_accuracy']:.2f}"
            )
            print(
                f"ROUGE1: {entry['rouge1_f']:.4f}, ROUGE2: {entry['rouge2_f']:.4f}, ROUGEL: {entry['rougeL_f']:.4f}, "
            )
            print(
                f"BLEU1: {entry['bleu_1']:.4f}, BLEU2: {entry['bleu_2']:.4f}, BLEU4: {entry['bleu_4']:.4f}, "
            )
            print(
                f"Routing RED: {entry['red_total_cost']:.2f}, RED Similarity: {entry['red_similarity_score']:.4f}"
            )


# ===========================================================================================
# DATASET CONVERSION FUNCTIONS
# ===========================================================================================
def convert_prediction_to_routing(
    batch: Dict[str, List[Any]], unified_tokenizer: UnifiedTokenizer
) -> Dict[str, List[Any]]:
    """
    Convert prediction tokens to routing coordinate sequences for a batch.

    Args:
        batch: Batched dataset dict containing predictions/target_tokens
        unified_tokenizer: UnifiedTokenizer instance for token processing

    Returns:
        Modified batch with cleaned tokens and prediction_tree_seq list added
    """

    def clean_special_tokens(tokens: List[str]) -> List[str]:
        """Remove special tokens like PAD/EOS/UNK from token list"""
        special_tokens = {
            unified_tokenizer.special_token_manager.get_token_by_name(name)
            for name in ("BOS_TOKEN", "PAD_TOKEN", "EOS_TOKEN", "UNKNOWN_TOKEN")
        }
        return [tok for tok in tokens if tok not in special_tokens]

    preds_list = batch["predictions"]
    targets_list = batch["target_tokens"]

    cleaned_preds: List[str] = []
    cleaned_targets: List[str] = []
    pred_tree_seqs: List[Any] = []

    for preds, targets in zip(preds_list, targets_list):
        pred_tokens = preds.split() if isinstance(preds, str) else preds
        pred_tokens = clean_special_tokens(pred_tokens)
        pred_str = " ".join(pred_tokens)
        cleaned_preds.append(pred_str)

        target_tokens = targets.split() if isinstance(targets, str) else targets
        target_tokens = clean_special_tokens(target_tokens)
        target_str = " ".join(target_tokens)
        cleaned_targets.append(target_str)

        if not pred_str:
            pred_tree_seqs.append(["(0, 0, 0)"])
        else:
            pred_tree_seqs.append(unified_tokenizer.convert_tokens_to_routing(pred_str))

    batch["predictions"] = cleaned_preds
    batch["target_tokens"] = cleaned_targets
    batch["prediction_tree_seq"] = pred_tree_seqs

    return batch


def add_nlp_metrics(
    batch: Dict[str, List[Any]], unified_tokenizer: UnifiedTokenizer
) -> Dict[str, List[Any]]:
    """Add NLP metrics to a batch without intermediate shape transposes."""
    metrics = calculate_batch_nlp_metrics(batch, unified_tokenizer.tokenizer)

    batch.update(metrics)

    return batch


def add_routing_metrics(
    batch: Dict[str, List[Any]], tokenizer: UnifiedTokenizer
) -> Dict[str, List[Any]]:
    """Add routing metrics to a batch without intermediate shape transposes."""
    metrics = calculate_batch_routing_metrics(batch, tokenizer)

    batch.update(metrics)

    return batch


def add_post_opt_metrics(
    batch: Dict[str, List[Any]], tokenizer: UnifiedTokenizer
) -> Dict[str, List[Any]]:
    """Post-process the batch after all metrics have been added."""
    metrics = calculate_batch_post_opt_metrics(batch, tokenizer)

    batch.update(metrics)

    return batch


# ===========================================================================================
# PER-SAMPLE METRIC CALCULATION FUNCTIONS
# ===========================================================================================
def calculate_batch_nlp_metrics(
    batch: Dict[str, List[Any]], tokenizer
) -> Dict[str, List[float]]:
    """Calculate NLP metrics for a batch."""
    batch_target_tokens = batch["target_tokens"]
    batch_predictions = batch["predictions"]

    if not batch_target_tokens or not batch_predictions:
        return {
            "rouge1_f": [],
            "rouge2_f": [],
            "rougeL_f": [],
            "bleu_1": [],
            "bleu_2": [],
            "bleu_4": [],
            "exact_match": [],
        }

    rouge_scorer_obj = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL", "rougeLsum"],
        use_stemmer=True,
        tokenizer=tokenizer,
    )
    bleu_smoother = SmoothingFunction()

    rouge1_f: List[float] = []
    rouge2_f: List[float] = []
    rougeL_f: List[float] = []
    bleu_1_l: List[float] = []
    bleu_2_l: List[float] = []
    bleu_4_l: List[float] = []
    exact_match_l: List[float] = []

    for target_tokens, predictions in zip(batch_target_tokens, batch_predictions):
        target_tokens_str = (
            " ".join(target_tokens)
            if isinstance(target_tokens, list)
            else target_tokens
        )
        predictions_str = (
            " ".join(predictions) if isinstance(predictions, list) else predictions
        )
        target_tokens = (
            target_tokens_str.split()
            if isinstance(target_tokens_str, str)
            else target_tokens
        )
        predictions = (
            predictions_str.split() if isinstance(predictions_str, str) else predictions
        )

        r = rouge_scorer_obj.score(target_tokens_str, predictions_str)

        if target_tokens and predictions:
            b1 = sentence_bleu(
                [target_tokens],
                predictions,
                weights=(1, 0, 0, 0),
                smoothing_function=bleu_smoother.method1,
            )
            b2 = sentence_bleu(
                [target_tokens],
                predictions,
                weights=(0.5, 0.5, 0, 0),
                smoothing_function=bleu_smoother.method1,
            )
            b4 = sentence_bleu(
                [target_tokens],
                predictions,
                weights=(0.25, 0.25, 0.25, 0.25),
                smoothing_function=bleu_smoother.method1,
            )
        else:
            b1 = b2 = b4 = 0.0

        # Normalize all metrics to Python floats to avoid pyarrow type inference issues
        rouge1_f.append(float(r["rouge1"].fmeasure))
        rouge2_f.append(float(r["rouge2"].fmeasure))
        rougeL_f.append(float(r["rougeL"].fmeasure))
        bleu_1_l.append(float(b1))
        bleu_2_l.append(float(b2))
        bleu_4_l.append(float(b4))
        exact_match_l.append(
            float(1.0 if target_tokens_str.strip() == predictions_str.strip() else 0.0)
        )

    return {
        "rouge1_f": rouge1_f,
        "rouge2_f": rouge2_f,
        "rougeL_f": rougeL_f,
        "bleu_1": bleu_1_l,
        "bleu_2": bleu_2_l,
        "bleu_4": bleu_4_l,
        "exact_match": exact_match_l,
    }


def calculate_batch_routing_metrics(
    batch: Dict[str, List[Any]], tokenizer: UnifiedTokenizer
) -> Dict[str, List[float]]:
    """Calculate routing metrics for a batch."""
    batch_relative_loads = batch["relative_loads"]
    batch_predictions = batch["prediction_tree_seq"]
    batch_ground_truth = batch["relative_tree_seq"]

    # Accumulators
    accumulator = defaultdict(list)
    for relative_loads, prediction_tree_seq, relative_tree_seq in zip(
        batch_relative_loads, batch_predictions, batch_ground_truth
    ):
        metrics = calculate_routing_metrics(
            relative_loads, prediction_tree_seq, relative_tree_seq, tokenizer
        )
        for k, v in metrics.items():
            accumulator[k].append(v)

    return accumulator


def calculate_batch_post_opt_metrics(
    batch: Dict[str, List[Any]], tokenizer: UnifiedTokenizer
) -> Dict[str, List[Any]]:
    """Calculate post-optimization metrics for a batch."""
    batch_post_opt_metrics = {}

    batch_relative_loads = batch["relative_loads"]
    batch_pred_relative_tree_seq = batch["prediction_tree_seq"]

    batch_post_opt_tree_seq = []
    batch_remove_wirelength_cost = []
    batch_remove_via_cost = []
    batch_add_wirelength_cost = []
    batch_add_via_cost = []

    for relative_loads, pred_relative_tree_seq in zip(
        batch_relative_loads, batch_pred_relative_tree_seq
    ):
        pred_relative_tree = tokenizer.build_tree_structure(pred_relative_tree_seq)
        post_opt_tree_seq, cost_metrics = min_cost_connection(
            relative_loads, pred_relative_tree, tokenizer
        )

        batch_post_opt_tree_seq.append(post_opt_tree_seq)
        batch_remove_wirelength_cost.append(cost_metrics["post_remove_wirelength_cost"])
        batch_remove_via_cost.append(cost_metrics["post_remove_via_cost"])
        batch_add_wirelength_cost.append(cost_metrics["post_add_wirelength_cost"])
        batch_add_via_cost.append(cost_metrics["post_add_via_cost"])

    batch_post_opt_metrics["post_opt_tree_seq"] = batch_post_opt_tree_seq
    batch_post_opt_metrics["post_remove_wirelength_cost"] = batch_remove_wirelength_cost
    batch_post_opt_metrics["post_remove_via_cost"] = batch_remove_via_cost
    batch_post_opt_metrics["post_add_wirelength_cost"] = batch_add_wirelength_cost
    batch_post_opt_metrics["post_add_via_cost"] = batch_add_via_cost

    # Compare with GT
    batch_ground_truth = batch["relative_tree_seq"]
    metrics_acc = defaultdict(list)

    for relative_loads, post_opt_tree_seq, relative_tree_seq in zip(
        batch_relative_loads, batch_post_opt_tree_seq, batch_ground_truth
    ):
        metrics = calculate_routing_metrics(
            relative_loads, post_opt_tree_seq, relative_tree_seq, tokenizer
        )
        for k, v in metrics.items():
            metrics_acc["post_" + k].append(v)

    batch_post_opt_metrics.update(metrics_acc)

    return batch_post_opt_metrics


# ===========================================================================================
# HELPER FUNCTIONS FOR ROUTING METRICS CALCULATION
# ===========================================================================================
def calculate_routing_metrics(
    relative_loads: List[str],
    pred_tree_seq: List[str],
    gt_tree_seq: List[str],
    tokenizer: UnifiedTokenizer,
) -> Dict[str, Any]:
    """Calculate routing metrics for a single coordinate sequence pair"""
    metrics = {}

    # Perfect match check
    metrics["is_perfect_match"] = pred_tree_seq == gt_tree_seq

    # Branches topology
    BRANCH_TOKEN = tokenizer.special_token_manager.get_token_by_name("BRANCH_TOKEN")
    END_TOKEN = tokenizer.special_token_manager.get_token_by_name("END_TOKEN")

    pred_branches_ops = [
        token for token in pred_tree_seq if token in (BRANCH_TOKEN, END_TOKEN)
    ]
    true_branches_ops = [
        token for token in gt_tree_seq if token in (BRANCH_TOKEN, END_TOKEN)
    ]
    metrics["is_branch_struct_match"] = true_branches_ops == pred_branches_ops
    metrics["branch_count_pred"] = pred_branches_ops.count(BRANCH_TOKEN)
    metrics["branch_count_true"] = true_branches_ops.count(BRANCH_TOKEN)

    # Tree topology
    pred_tree = tokenizer.build_tree_structure(pred_tree_seq)
    true_tree = tokenizer.build_tree_structure(gt_tree_seq)

    pred_edges = get_edges(pred_tree)
    true_edges = get_edges(true_tree)

    pred_edges_set = set(pred_edges)
    true_edges_set = set(true_edges)

    pred_leaves = get_leaves(pred_edges, pred_tree.coord)
    true_leaves = get_leaves(true_edges, true_tree.coord)

    pred_leaves_set = set(pred_leaves)
    true_leaves_set = set(true_leaves)

    edge_intersect = len(true_edges_set.intersection(pred_edges_set))
    edge_union = len(true_edges_set.union(pred_edges_set))

    leaf_intersect = len(true_leaves_set.intersection(pred_leaves_set))
    leaf_union = len(true_leaves_set.union(pred_leaves_set))

    metrics["num_edges_pred"] = len(pred_edges)
    metrics["num_edges_true"] = len(true_edges)

    metrics["num_leaves_pred"] = len(pred_leaves_set)
    metrics["num_leaves_true"] = len(true_leaves_set)

    metrics["edge_accuracy"] = (
        edge_intersect / len(true_edges_set) if true_edges_set else 0.0
    )
    metrics["edge_precision"] = (
        edge_intersect / len(pred_edges_set) if pred_edges_set else 0.0
    )
    metrics["edge_recall"] = (
        edge_intersect / len(true_edges_set) if true_edges_set else 0.0
    )
    metrics["edge_iou"] = edge_intersect / edge_union if edge_union else 0.0

    metrics["leaf_accuracy"] = (
        leaf_intersect / len(true_leaves_set) if true_leaves_set else 0.0
    )
    metrics["leaf_precision"] = (
        leaf_intersect / len(pred_leaves_set) if pred_leaves_set else 0.0
    )
    metrics["leaf_recall"] = (
        leaf_intersect / len(true_leaves_set) if true_leaves_set else 0.0
    )
    metrics["leaf_iou"] = leaf_intersect / leaf_union if leaf_union else 0.0

    metrics["is_leaf_set_match"] = pred_leaves_set == true_leaves_set

    metrics["len_coords_pred"] = len(pred_tree_seq)
    metrics["len_coords_true"] = len(gt_tree_seq)

    # Check connection
    pred_coords = set(get_all_coords(pred_tree))
    relative_load_coords = {
        tokenizer.parse_coord(relative_load) for relative_load in relative_loads
    }
    all_loads_connected = relative_load_coords.issubset(pred_coords)
    all_leaves_is_useful = pred_leaves_set.issubset(relative_load_coords)
    metrics["is_connected_all_loads"] = all_loads_connected
    metrics["is_graceful"] = all_loads_connected and all_leaves_is_useful

    # Via (only point.m)
    metrics["num_vias_pred"] = sum(
        [abs(edge[0].m - edge[1].m) // 2 for edge in pred_edges]
    )
    metrics["num_vias_true"] = sum(
        [abs(edge[0].m - edge[1].m) // 2 for edge in true_edges]
    )
    metrics["via_ratio"] = (
        metrics["num_vias_pred"] / metrics["num_vias_true"]
        if metrics["num_vias_true"] > 0
        else (1.0 if metrics["num_vias_pred"] == 0 else 0.0)
    )

    # Wirelength (only point.x, point.y)
    metrics["wirelength_pred"] = sum(
        abs(edge[0].x - edge[1].x) + abs(edge[0].y - edge[1].y) for edge in pred_edges
    )
    metrics["wirelength_true"] = sum(
        abs(edge[0].x - edge[1].x) + abs(edge[0].y - edge[1].y) for edge in true_edges
    )
    metrics["wirelength_ratio"] = (
        metrics["wirelength_pred"] / metrics["wirelength_true"]
        if metrics["wirelength_true"] > 0
        else (1.0 if metrics["wirelength_pred"] == 0 else 0.0)
    )

    # Calculate RED metrics
    red_metrics = calculate_red_score(pred_tree, true_tree)

    metrics.update(red_metrics)

    return metrics


def get_edges(tree: Node) -> List[Tuple[CoordinatePoint, CoordinatePoint]]:
    """Get edges (consecutive point pairs) from point sequence"""
    edges = []

    # dfs
    def dfs(node: Node):
        if not node.children:
            return
        for child in node.children:
            # Add edge from current node to child
            edges.append((node.coord, child.coord))
            dfs(child)

    dfs(tree)
    return edges


def get_leaves(
    edges: List[Tuple[CoordinatePoint, CoordinatePoint]], root_coord: CoordinatePoint
) -> List[CoordinatePoint]:
    """Get leaf nodes from edges"""
    if not edges:
        return []

    # Count occurrences of each point
    point_counts = Counter(chain.from_iterable(edges))
    # Leaf points are those that appear only once
    return [
        point
        for point, count in point_counts.items()
        if point != root_coord and count == 1
    ]


# ===========================================================================================
# RED (ROUTING EDIT DISTANCE) CALCULATION FUNCTIONS
# ===========================================================================================
def calculate_red_score(pred_tree: Node, true_tree: Node) -> Dict[str, float]:
    """Calculate Routing EDIT DISTANCE"""
    scaled_pred, scaled_true = scale_trees_uniformly(pred_tree, true_tree)
    cost_info = calculate_simple_alignment_cost(scaled_pred, scaled_true)

    # Calculate normalization factor (cost of building GT from scratch)
    total_wirelength_pred = calculate_total_wirelength(scaled_pred)
    total_wirelength_true = calculate_total_wirelength(scaled_true)
    normalization_factor = total_wirelength_true + total_wirelength_pred

    # Handle edge case where GT has no wire length
    if normalization_factor == 0:
        if cost_info["red_total_cost"] == 0:
            similarity_score = 1.0
        else:
            similarity_score = 0.0
    else:
        # Calculate similarity score
        similarity_score = max(
            0.0, 1.0 - (cost_info["red_total_cost"] / normalization_factor)
        )

    result = {
        **cost_info,
        "red_total_wirelength_pred": total_wirelength_pred,
        "red_total_wirelength_true": total_wirelength_true,
        "red_normalization_factor": normalization_factor,
        "red_similarity_score": similarity_score,
    }

    return result


def manhattan_distance(p1: CoordinatePoint, p2: CoordinatePoint) -> float:
    """Calculate Manhattan distance between two points"""
    return abs(p1.x - p2.x) + abs(p1.y - p2.y) + abs(p1.m - p2.m)


def add_child(parent: Node, child: Node):
    """Add child node to parent and set parent reference"""
    parent.children.append(child)
    child.parent = parent


def cost_substitution(pred_node: Node, gt_node: Node) -> float:
    """Cost of substituting a predicted node with a ground truth node."""
    pred_coord = pred_node.coord
    gt_coord = gt_node.coord
    return manhattan_distance(pred_coord, gt_coord)


def cost_deletion(pred_node: Node) -> float:
    """Cost of deleting a predicted node."""
    return (
        manhattan_distance(pred_node.coord, pred_node.parent.coord)
        if pred_node.parent
        else 0.0
    )


def cost_insertion(gt_node: Node) -> float:
    """Cost of inserting a ground truth node."""
    return (
        manhattan_distance(gt_node.coord, gt_node.parent.coord)
        if gt_node.parent
        else 0.0
    )


def get_all_nodes(root: Node) -> List[Node]:
    """Get all nodes in the tree using DFS."""
    nodes = []

    def dfs(node: Node):
        nodes.append(node)
        for child in node.children:
            dfs(child)

    dfs(root)

    return nodes


def get_all_coords(root: Node) -> List[CoordinatePoint]:
    """Get all coordinates in the tree using DFS."""
    nodes = get_all_nodes(root)
    return [node.coord for node in nodes if node.coord]


def calculate_total_wirelength(root: Node) -> float:
    """Calculate total wirelength of the tree."""
    total_length = 0.0

    def dfs(node: Node):
        nonlocal total_length
        total_length += (
            manhattan_distance(node.coord, node.parent.coord) if node.parent else 0.0
        )
        for child in node.children:
            dfs(child)

    dfs(root)
    return total_length


def scale_trees_uniformly(pred_tree: Node, gt_tree: Node) -> Tuple[Node, Node]:
    """
    Scale both trees using a unified scaling factor to ensure fairness.
    """
    # Collect all coordinates from both trees
    pred_coords = get_all_coords(pred_tree)
    gt_coords = get_all_coords(gt_tree)
    all_coords = pred_coords + gt_coords
    # Calculate global range
    max_x = max(c.x for c in all_coords)
    min_x = min(c.x for c in all_coords)
    max_y = max(c.y for c in all_coords)
    min_y = min(c.y for c in all_coords)
    max_m = max(c.m for c in all_coords)
    min_m = min(c.m for c in all_coords)

    delta_x = max_x - min_x
    delta_y = max_y - min_y
    delta_m = max_m - min_m

    # Use unified scaling factor
    scale_factor = (delta_x + delta_y) / 2 / delta_m if delta_m != 0 else 1.0

    # Scale both trees with the same factor
    def scale_tree_with_factor(tree: Node, factor: float) -> Node:
        def copy_and_scale(node: Node) -> Node:
            if not node.coord:
                return Node()
            scaled_m = int(node.coord.m * factor)
            scaled_node = Node()
            scaled_node.coord = CoordinatePoint(node.coord.x, node.coord.y, scaled_m)
            for child in node.children:
                add_child(scaled_node, copy_and_scale(child))
            return scaled_node

        return copy_and_scale(tree)

    scaled_pred = scale_tree_with_factor(pred_tree, scale_factor)
    scaled_gt = scale_tree_with_factor(gt_tree, scale_factor)

    return scaled_pred, scaled_gt


def calculate_simple_alignment_cost(pred_tree: Node, gt_tree: Node) -> Dict[str, float]:
    """
    Calculate a simplified alignment cost using greedy matching.

    This is a simplified version of tree edit distance that uses greedy
    nearest-neighbor matching for demonstration purposes.
    """
    pred_nodes = get_all_nodes(pred_tree)
    gt_nodes = get_all_nodes(gt_tree)

    # Separate costs for analysis
    substitution_cost = 0.0
    deletion_cost = 0.0
    insertion_cost = 0.0

    # Create distance matrix
    distances = {}
    for i, pred_node in enumerate(pred_nodes):
        for j, gt_node in enumerate(gt_nodes):
            distances[(i, j)] = cost_substitution(pred_node, gt_node)

    # Greedy matching: find best matches for predicted nodes
    used_gt_indices = set()
    matched_pred_indices = set()

    # Sort by distance for greedy matching
    sorted_pairs = sorted(distances.items(), key=lambda x: x[1])

    for (pred_idx, gt_idx), dist in sorted_pairs:
        if pred_idx not in matched_pred_indices and gt_idx not in used_gt_indices:
            # Match this pair
            substitution_cost += dist
            matched_pred_indices.add(pred_idx)
            used_gt_indices.add(gt_idx)

    # Calculate deletion cost for unmatched predicted nodes
    for i, pred_node in enumerate(pred_nodes):
        if i not in matched_pred_indices:
            deletion_cost += cost_deletion(pred_node)

    # Calculate insertion cost for unmatched ground truth nodes
    for j, gt_node in enumerate(gt_nodes):
        if j not in used_gt_indices:
            insertion_cost += cost_insertion(gt_node)

    total_cost = substitution_cost + deletion_cost + insertion_cost

    return {
        "red_total_cost": total_cost,
        "red_substitution_cost": substitution_cost,
        "red_deletion_cost": deletion_cost,
        "red_insertion_cost": insertion_cost,
        "red_num_pred_nodes": len(pred_nodes),
        "red_num_gt_nodes": len(gt_nodes),
        "red_num_matches": len(matched_pred_indices),
    }


# ===========================================================================================
# EDGE DISCRETIZATION AND SPATIAL PROCESSING FUNCTIONS
# ===========================================================================================
def point_lies_on_edge(
    point: CoordinatePoint, edge_start: CoordinatePoint, edge_end: CoordinatePoint
) -> bool:
    """Check if a point lies on an edge (not at endpoints)"""
    # Check if point is collinear with edge endpoints
    if point == edge_start or point == edge_end:
        return False  # Point is at endpoint, not within edge

    # Check if point is on the axis-aligned segment
    if edge_start.x == edge_end.x == point.x:  # Vertical line in x
        return min(edge_start.y, edge_end.y) < point.y < max(
            edge_start.y, edge_end.y
        ) and min(edge_start.m, edge_end.m) <= point.m <= max(edge_start.m, edge_end.m)
    elif edge_start.y == edge_end.y == point.y:  # Horizontal line in y
        return min(edge_start.x, edge_end.x) < point.x < max(
            edge_start.x, edge_end.x
        ) and min(edge_start.m, edge_end.m) <= point.m <= max(edge_start.m, edge_end.m)
    elif edge_start.m == edge_end.m == point.m:  # Layer line in m
        return min(edge_start.x, edge_end.x) <= point.x <= max(
            edge_start.x, edge_end.x
        ) and min(edge_start.y, edge_end.y) <= point.y <= max(edge_start.y, edge_end.y)
    return False


def split_edge_at_point(
    edge_start: CoordinatePoint, edge_end: CoordinatePoint, split_point: CoordinatePoint
) -> List[Tuple[CoordinatePoint, CoordinatePoint]]:
    """Split an edge at a given point, returning two segments"""
    return [(edge_start, split_point), (split_point, edge_end)]


def initialize_discrete_tree_edges(
    pred_relative_tree: Node, relative_loads: List[str], tokenizer: UnifiedTokenizer
) -> List[Tuple[CoordinatePoint, CoordinatePoint]]:
    """Initialize discrete_tree_edges with edge splitting and non-overlapping processing"""

    # Step 1: Extract edges from pred_relative_tree
    tree_edges = get_edges(pred_relative_tree)
    if not tree_edges:
        return []

    # Step 2: Parse relative load coordinates
    load_coords = [tokenizer.parse_coord(load) for load in relative_loads]

    # Step 3: Split edges where relative_loads lie within (not at endpoints)
    discrete_edges = []
    for edge_start, edge_end in tree_edges:
        edge_segments = [(edge_start, edge_end)]

        # Check each load to see if it lies within this edge
        for load_coord in load_coords:
            new_segments = []
            for segment_start, segment_end in edge_segments:
                if point_lies_on_edge(load_coord, segment_start, segment_end):
                    # Split segment at load position
                    new_segments.extend(
                        split_edge_at_point(segment_start, segment_end, load_coord)
                    )
                else:
                    # Keep segment as is
                    new_segments.append((segment_start, segment_end))
            edge_segments = new_segments

        discrete_edges.extend(edge_segments)

    # return discrete_edges
    # Steps 4-5: Optimized sweep algorithm for intersection detection and edge splitting
    return sweep_algorithm_3d_intersection(discrete_edges)


def sweep_algorithm_3d_intersection(
    edges: List[Tuple[CoordinatePoint, CoordinatePoint]],
) -> List[Tuple[CoordinatePoint, CoordinatePoint]]:
    """
    Optimized O(E log E) sweep algorithm for 3D rectilinear edge intersection detection.

    Groups edges by axis orientation (X, Y, M) and uses plane sweep to find all
    intersection points, then splits edges at intersection points.

    Args:
        edges: List of edge tuples (start_point, end_point)

    Returns:
        List of non-overlapping edge segments after splitting at intersections
    """
    if not edges:
        return []

    # Group edges by their axis orientation
    x_aligned_edges = []  # Edges parallel to X axis (y,m constant)
    y_aligned_edges = []  # Edges parallel to Y axis (x,m constant)
    m_aligned_edges = []  # Edges parallel to M axis (x,y constant)

    for edge_start, edge_end in edges:
        if edge_start.x == edge_end.x and edge_start.m == edge_end.m:
            # Y-aligned edge: constant x,m - varies in y
            y_min, y_max = min(edge_start.y, edge_end.y), max(edge_start.y, edge_end.y)
            y_aligned_edges.append((edge_start.x, edge_start.m, y_min, y_max))
        elif edge_start.y == edge_end.y and edge_start.m == edge_end.m:
            # X-aligned edge: constant y,m - varies in x
            x_min, x_max = min(edge_start.x, edge_end.x), max(edge_start.x, edge_end.x)
            x_aligned_edges.append((edge_start.y, edge_start.m, x_min, x_max))
        elif edge_start.x == edge_end.x and edge_start.y == edge_end.y:
            # M-aligned edge: constant x,y - varies in m
            m_min, m_max = min(edge_start.m, edge_end.m), max(edge_start.m, edge_end.m)
            m_aligned_edges.append((edge_start.x, edge_start.y, m_min, m_max))

    # Find all intersection points using cross-axis intersection detection
    all_intersection_points = set()

    # Cross-axis intersections: X-Y, X-M, Y-M
    for x_edge in x_aligned_edges:
        x_y, x_m, x_min, x_max = x_edge
        for y_edge in y_aligned_edges:
            y_x, y_m, y_min, y_max = y_edge
            # X-edge (varies in x, constant y,m) intersects Y-edge (varies in y, constant x,m)
            if x_m == y_m and x_min <= y_x <= x_max and y_min <= x_y <= y_max:
                intersection_point = CoordinatePoint(y_x, x_y, x_m)
                all_intersection_points.add(intersection_point)

    # Similar logic for X-M and Y-M intersections...
    for x_edge in x_aligned_edges:
        x_y, x_m, x_min, x_max = x_edge
        for m_edge in m_aligned_edges:
            m_x, m_y, m_min, m_max = m_edge
            if x_y == m_y and x_min <= m_x <= x_max and m_min <= x_m <= m_max:
                intersection_point = CoordinatePoint(m_x, x_y, x_m)
                all_intersection_points.add(intersection_point)

    for y_edge in y_aligned_edges:
        y_x, y_m, y_min, y_max = y_edge
        for m_edge in m_aligned_edges:
            m_x, m_y, m_min, m_max = m_edge
            if y_x == m_x and y_min <= m_y <= y_max and m_min <= y_m <= m_max:
                intersection_point = CoordinatePoint(y_x, m_y, y_m)
                all_intersection_points.add(intersection_point)

    # Split all original edges at their intersection points
    result_edges = []

    for edge_start, edge_end in edges:
        # Find intersection points that lie on this edge
        edge_intersections = []
        for point in all_intersection_points:
            if point_lies_on_edge(point, edge_start, edge_end):
                edge_intersections.append(point)

        if not edge_intersections:
            # No intersections - keep edge as is
            result_edges.append((edge_start, edge_end))
        else:
            # Split edge at intersection points
            # Sort intersection points by distance from start
            edge_intersections.sort(key=lambda p: manhattan_distance(edge_start, p))

            # Create segments between consecutive points
            current_point = edge_start
            for intersection_point in edge_intersections:
                if current_point != intersection_point:
                    result_edges.append((current_point, intersection_point))
                current_point = intersection_point

            # Add final segment to end
            if current_point != edge_end:
                result_edges.append((current_point, edge_end))

    return result_edges


def is_rectilinear_compliant(start: CoordinatePoint, end: CoordinatePoint) -> bool:
    """Check if an edge moves along exactly one axis (3D rectilinear constraint)"""
    x_diff = abs(start.x - end.x)
    y_diff = abs(start.y - end.y)
    m_diff = abs(start.m - end.m)

    # Count how many axes are different
    non_zero_diffs = sum([x_diff > 0, y_diff > 0, m_diff > 0])
    return non_zero_diffs <= 1


def construct_rectilinear_edges(
    discrete_tree_edges: List[Tuple[CoordinatePoint, CoordinatePoint]],
    relative_loads: List[str],
    tokenizer: UnifiedTokenizer,
    pred_relative_tree: Node,
) -> List[Tuple[CoordinatePoint, CoordinatePoint]]:
    """Construct rectilinear_edges by connecting unconnected loads"""

    # Step 1: Compute unconnected_loads
    # Convert to set for efficient lookup
    all_coords = [coord for edge in discrete_tree_edges for coord in edge]
    coord_set = set(all_coords)

    # Filter out loads
    load_coords = [tokenizer.parse_coord(load) for load in relative_loads]
    unconnected_loads = [coord for coord in load_coords if coord not in coord_set]

    if not unconnected_loads:
        # All loads are already connected
        return []

    # Calculate prohibited layers (Don't connect on these layers)
    prohibited_layers = set(coord.m for coord in load_coords)

    # Step 2: Extract tree_nodes from discrete_tree_edges
    # Get leaf nodes from discrete edges
    tree_nodes = [
        tree_node
        for edge in discrete_tree_edges
        for tree_node in edge
        if tree_node != pred_relative_tree.coord
    ]

    tree_node_layers = set(node.m for node in tree_nodes)

    # Re-check prohibited layers, if all tree node in prohibited layer, make prohibited layer is empty
    if tree_node_layers.issubset(prohibited_layers):
        prohibited_layers = set()

    # Combine leaf nodes with all endpoints (remove duplicates)
    tree_nodes = set(tree_nodes)

    # Step 3: Generate connection_edges
    connection_edges = []

    for unconnected_load in unconnected_loads:
        # Find the closest leaf node for connection
        min_distance = float("inf")
        closest_node = None

        for tree_node in tree_nodes:
            distance = rectilinear_distance(tree_node, unconnected_load)
            if distance < min_distance and tree_node.m not in prohibited_layers:
                min_distance = distance
                closest_node = tree_node

        if closest_node:
            # Check if direct connection is rectilinear compliant
            if is_rectilinear_compliant(closest_node, unconnected_load):
                # Direct connection is compliant, add single edge
                connection_edges.append((closest_node, unconnected_load))
            else:
                # Not compliant, use enumerate_rectilinear_edges to create axis-aligned segments
                rectilinear_edges = enumerate_rectilinear_edges(
                    closest_node, unconnected_load, prohibited_layers
                )

                connection_edges.extend(rectilinear_edges)

    connection_edges = sweep_algorithm_3d_intersection(connection_edges)
    return connection_edges


def build_networkx_steiner_tree(
    discrete_tree_edges: List[Tuple[CoordinatePoint, CoordinatePoint]],
    connection_edges: List[Tuple[CoordinatePoint, CoordinatePoint]],
    relative_loads: List[str],
    pred_relative_tree: Node,
    tokenizer: UnifiedTokenizer,
    tree_weight: float = 1.0,
    connection_weight: float = 1.0,
) -> Node:
    """Build NetworkX graph and construct Steiner tree"""
    if not discrete_tree_edges and not connection_edges:
        # Handle edge case: no edges, create simple tree with driver
        if pred_relative_tree.coord:
            return pred_relative_tree
        # Fallback to first load as root
        load_coords = [tokenizer.parse_coord(load) for load in relative_loads]
        if load_coords:
            root_node = Node(str(load_coords[0]))
            root_node.coord = load_coords[0]
            return root_node
        return pred_relative_tree

    # Step 1: Construct NetworkX graph
    G = nx.Graph()

    # Add nodes and edges with weights
    for start, end in discrete_tree_edges:
        # Add nodes with coordinate attributes
        G.add_node(start, coord=start)
        G.add_node(end, coord=end)

        # Add weighted edges with rectilinear distance
        weight = rectilinear_distance(start, end)
        G.add_edge(start, end, weight=weight * tree_weight)

    for start, end in connection_edges:
        # Add nodes with coordinate attributes
        G.add_node(start, coord=start)
        G.add_node(end, coord=end)

        # Add weighted edges with rectilinear distance
        weight = rectilinear_distance(start, end)
        G.add_edge(start, end, weight=weight * connection_weight)

    # Step 2: Identify terminal nodes (driver + all relative_loads)
    # Get driver coordinate (root of tree)
    driver_coord = pred_relative_tree.coord

    # Parse relative loads
    load_coords = [tokenizer.parse_coord(load) for load in relative_loads]

    # Terminal nodes: driver + all loads
    terminal_nodes = [driver_coord] + load_coords

    # Filter terminal nodes to only include those present in graph
    graph_nodes = set(G.nodes())
    valid_terminal_nodes = [node for node in terminal_nodes if node in graph_nodes]

    if len(valid_terminal_nodes) < 2:
        # If we don't have enough terminal nodes in graph, add missing ones by connecting to closest nodes
        for terminal in terminal_nodes:
            if terminal not in graph_nodes and graph_nodes:
                # Find closest node in graph
                min_distance = float("inf")
                closest_node = None
                for graph_node in graph_nodes:
                    dist = rectilinear_distance(terminal, graph_node)
                    if dist < min_distance:
                        min_distance = dist
                        closest_node = graph_node

                if closest_node:
                    # Add terminal node and connect it to closest node
                    G.add_node(terminal, coord=terminal)
                    G.add_edge(terminal, closest_node, weight=min_distance)
                    valid_terminal_nodes.append(terminal)

    # Step 3: Apply Steiner tree algorithm
    if len(valid_terminal_nodes) >= 2:
        steiner_tree = nx.algorithms.approximation.steiner_tree(
            G, valid_terminal_nodes, weight="weight"
        )
    else:
        # Single or no terminal nodes - create minimal tree
        steiner_tree = G.copy()

    # Step 4: Convert Steiner tree back to Node structure
    steiner_edges = list(steiner_tree.edges())
    if not steiner_edges:
        # No edges case
        if valid_terminal_nodes:
            root_node = Node(str(valid_terminal_nodes[0]))
            root_node.coord = valid_terminal_nodes[0]
            return root_node
        else:
            return pred_relative_tree

    # Build tree from Steiner tree edges
    tree_node = build_tree_from_steiner_edges(steiner_edges, driver_coord)
    return tree_node


def build_tree_from_steiner_edges(
    edges: List[Tuple[CoordinatePoint, CoordinatePoint]],
    root_coord: CoordinatePoint,
) -> Node:
    """Build a Node tree structure from Steiner tree edges"""
    if not edges:
        root_node = Node(str(root_coord))
        root_node.coord = root_coord
        return root_node

    # Create adjacency list
    adj_list = defaultdict(list)
    for start, end in edges:
        adj_list[start].append(end)
        adj_list[end].append(start)

    # Create node mapping
    coord_to_node = {}
    for coord in adj_list.keys():
        node = Node(str(coord))
        node.coord = coord
        coord_to_node[coord] = node

    # Find root (prefer given root_coord, fallback to first coordinate)
    actual_root = (
        root_coord if root_coord in coord_to_node else list(coord_to_node.keys())[0]
    )
    root_node = coord_to_node[actual_root]

    # Build tree using DFS from root
    visited = set()

    def dfs_build_tree(
        current_coord: CoordinatePoint, parent_node: Optional[Node] = None
    ):
        if current_coord in visited:
            return

        visited.add(current_coord)
        current_node = coord_to_node[current_coord]

        # Add current node as child of parent (except for root)
        if parent_node and current_node not in parent_node.children:
            parent_node.children.append(current_node)

        # Visit all neighbors
        for neighbor_coord in adj_list[current_coord]:
            if neighbor_coord not in visited:
                dfs_build_tree(neighbor_coord, current_node)

    dfs_build_tree(actual_root)
    return root_node


def validate_steiner_tree_strict(
    steiner_tree_node: Node, relative_loads: List[str], tokenizer: UnifiedTokenizer
) -> None:
    """Perform comprehensive validation with strict error handling"""

    # Parse load coordinates for validation
    load_coords = [tokenizer.parse_coord(load) for load in relative_loads]
    load_coord_set = set(load_coords)

    # Get driver coordinate
    driver_coord = steiner_tree_node.coord
    if driver_coord is None:
        raise ValueError("Steiner tree root node has no coordinate")

    # Extract all tree information
    tree_edges = get_edges(steiner_tree_node)
    tree_coords = get_all_coords(steiner_tree_node)
    tree_coord_set = set(tree_coords)

    # 1. Cycle Detection using NetworkX
    if tree_edges:
        # Build NetworkX graph from tree edges
        G = nx.Graph()
        for start, end in tree_edges:
            G.add_edge(start, end)

        # Check for cycles
        if not nx.is_tree(G):
            # Find and report cycles
            try:
                cycle = nx.find_cycle(G)
                cycle_coords = [str(coord) for coord in cycle[0]]
                raise ValueError(
                    f"Tree contains cycles. Detected cycle involving coordinates: {cycle_coords}"
                )
            except nx.NetworkXNoCycle:
                # Multiple connected components but no cycles
                components = list(nx.connected_components(G))
                raise ValueError(
                    f"Tree has multiple disconnected components: {len(components)} components found"
                )

    # 2. Connectivity Validation - Verify all relative_loads are reachable from driver
    unreachable_loads = []

    def dfs_reachable(start_coord: CoordinatePoint) -> set:
        """DFS to find all reachable coordinates from start"""
        reachable = set()
        visited = set()
        stack = [start_coord]

        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            reachable.add(current)

            # Find neighbors in tree edges
            for edge_start, edge_end in tree_edges:
                if edge_start == current and edge_end not in visited:
                    stack.append(edge_end)
                elif edge_end == current and edge_start not in visited:
                    stack.append(edge_start)

        return reachable

    if tree_edges:
        reachable_from_driver = dfs_reachable(driver_coord)
        unreachable_loads = [
            coord for coord in load_coords if coord not in reachable_from_driver
        ]
    else:
        # No edges case - only driver is reachable
        unreachable_loads = [coord for coord in load_coords if coord != driver_coord]

    if unreachable_loads:
        unreachable_strs = [str(coord) for coord in unreachable_loads]
        raise ConnectionError(
            f"The following {len(unreachable_loads)} relative_loads are unreachable from driver {driver_coord}: {unreachable_strs}"
        )

    # 3. Leaf Node Validation
    if tree_edges:
        leaf_nodes = get_leaves(tree_edges, driver_coord)
        leaf_coord_set = set(leaf_nodes)

        # Check: leaf_nodes should be subset of relative_loads (leaf nodes should only be loads)
        non_load_leaves = leaf_coord_set - load_coord_set
        if non_load_leaves:
            non_load_strs = [str(coord) for coord in non_load_leaves]
            raise ValueError(
                f"Tree has leaf nodes that are not relative_loads: {non_load_strs}. All leaf nodes must be load coordinates."
            )

        # Check: relative_loads should be subset of tree_nodes (all loads must be in tree)
        missing_loads = load_coord_set - tree_coord_set
        if missing_loads:
            missing_strs = [str(coord) for coord in missing_loads]
            raise ValueError(
                f"The following {len(missing_loads)} relative_loads are not present in tree nodes: {missing_strs}"
            )
    else:
        # No edges case - only driver should be present and should match all loads if applicable
        if load_coords and driver_coord not in load_coords:
            load_strs = [str(coord) for coord in load_coords]
            raise ValueError(
                f"Tree has no edges but relative_loads {load_strs} are different from driver {driver_coord}"
            )

    # 4. Rectilinear Constraint Validation - Check each edge moves along exactly one axis
    non_compliant_edges = []
    for edge_start, edge_end in tree_edges:
        if not is_rectilinear_compliant(edge_start, edge_end):
            non_compliant_edges.append((edge_start, edge_end))

    if non_compliant_edges:
        edge_strs = [f"{start} -> {end}" for start, end in non_compliant_edges]
        raise ValueError(
            f"Tree contains {len(non_compliant_edges)} edges that violate 3D rectilinear constraints (edges must move along exactly one axis): {edge_strs}"
        )

    # 5. Additional validation - Verify tree structure integrity
    if tree_coords:
        # Check for duplicate coordinates (should not happen in valid tree)
        if len(tree_coords) != len(set(tree_coords)):
            coord_counts = Counter(tree_coords)
            duplicates = [
                str(coord) for coord, count in coord_counts.items() if count > 1
            ]
            raise ValueError(f"Tree contains duplicate coordinates: {duplicates}")

        # Verify driver is in tree
        if driver_coord not in tree_coord_set:
            raise ValueError(
                f"Driver coordinate {driver_coord} is not present in tree structure"
            )

    # If we get here, all validations passed
    return None


# ===========================================================================================
# TREE SCALING FUNCTIONS FOR MIN_COST_CONNECTION
# ===========================================================================================
def scale_tree_and_loads_for_optimization(
    tree: Node, relative_loads: List[str], tokenizer: UnifiedTokenizer
) -> Tuple[Node, List[str], int]:
    """
    Scale tree and relative loads to normalize m-axis asymmetry for optimization.

    Similar to scale_trees_uniformly used in RED metrics, but for single tree optimization.

    Args:
        tree: Original tree structure
        relative_loads: List of coordinate strings
        tokenizer: UnifiedTokenizer for coordinate parsing

    Returns:
        Tuple of (scaled_tree, scaled_load_strings, scale_factor)
    """
    return tree, relative_loads, 1.0
    # Parse relative loads to coordinates
    load_coords = [tokenizer.parse_coord(load) for load in relative_loads]

    # Collect all coordinates from tree and loads
    tree_coords = get_all_coords(tree)
    all_coords = tree_coords + load_coords

    if not all_coords:
        return tree, relative_loads, 1.0

    # Calculate coordinate ranges
    max_x = max(c.x for c in all_coords)
    min_x = min(c.x for c in all_coords)
    max_y = max(c.y for c in all_coords)
    min_y = min(c.y for c in all_coords)
    max_m = max(c.m for c in all_coords)
    min_m = min(c.m for c in all_coords)

    delta_x = max_x - min_x
    delta_y = max_y - min_y
    delta_m = max_m - min_m

    # Calculate unified scaling factor (same as RED metric calculation)
    scale_factor = (delta_x + delta_y) / 2 / delta_m if delta_m != 0 else 1.0
    scale_factor = int(scale_factor)

    # Scale the tree
    scaled_tree = scale_single_tree(tree, scale_factor)

    # Scale the relative loads
    scaled_load_coords = []
    for coord in load_coords:
        scaled_m = int(coord.m * scale_factor)
        scaled_coord = CoordinatePoint(coord.x, coord.y, scaled_m)
        scaled_load_coords.append(scaled_coord)

    # Convert scaled coordinates back to strings
    scaled_loads = [str(coord) for coord in scaled_load_coords]

    return scaled_tree, scaled_loads, scale_factor


def scale_single_tree(tree: Node, scale_factor: int) -> Node:
    """
    Scale a single tree with the given scale factor.

    Args:
        tree: Original tree node
        scale_factor: Scaling factor for m-axis coordinates

    Returns:
        New scaled tree node
    """

    def copy_and_scale(node: Node) -> Node:
        if not node.coord:
            return Node()
        scaled_m = int(node.coord.m * scale_factor)
        scaled_node = Node()
        scaled_node.coord = CoordinatePoint(node.coord.x, node.coord.y, scaled_m)
        scaled_node.coord_str = str(scaled_node.coord)

        # Copy other node attributes if they exist
        if hasattr(node, "coord_str"):
            scaled_node.coord_str = str(scaled_node.coord)

        for child in node.children:
            add_child(scaled_node, copy_and_scale(child))
        return scaled_node

    return copy_and_scale(tree)


def restore_edge_scale(
    edge: Tuple[CoordinatePoint, CoordinatePoint], scale_factor: int
) -> Tuple[CoordinatePoint, CoordinatePoint]:
    """
    Restore the original scale of an edge by applying the inverse scaling factor.

    Args:
        edge: Edge with scaled m-coordinates
        scale_factor: Original scaling factor applied

    Returns:
        Edge with restored original m-coordinates
    """
    if scale_factor == 0.0:
        return edge  # Avoid division by zero

    start, end = edge
    restored_start_m = int(start.m / scale_factor)
    restored_end_m = int(end.m / scale_factor)

    restored_start = CoordinatePoint(start.x, start.y, restored_start_m)
    restored_end = CoordinatePoint(end.x, end.y, restored_end_m)

    return (restored_start, restored_end)


def restore_tree_scale(scaled_tree: Node, scale_factor: int) -> Node:
    """
    Restore the original scale of a tree by applying the inverse scaling factor.

    Args:
        scaled_tree: Tree with scaled m-coordinates
        scale_factor: Original scaling factor applied

    Returns:
        Tree with restored original m-coordinates
    """
    if scale_factor == 0.0:
        return scaled_tree  # Avoid division by zero

    def copy_and_restore(node: Node) -> Node:
        if not node.coord:
            return Node()

        restored_m = int(node.coord.m / scale_factor)
        restored_node = Node()
        restored_node.coord = CoordinatePoint(node.coord.x, node.coord.y, restored_m)
        restored_node.coord_str = str(restored_node.coord)

        # Copy other node attributes if they exist
        if hasattr(node, "coord_str"):
            restored_node.coord_str = str(restored_node.coord)

        for child in node.children:
            add_child(restored_node, copy_and_restore(child))
        return restored_node

    return copy_and_restore(scaled_tree)


# ===========================================================================================
# POST OPT FUNCTIONS
# ===========================================================================================
def min_cost_connection(
    relative_loads: List[str],
    pred_relative_tree: Node,
    tokenizer: UnifiedTokenizer,
    scale_required: bool = False,
) -> Tuple[List[str], Dict[str, float]]:
    """
    Connect all loads with minimum cost using 3D rectilinear Steiner tree optimization.

    This function implements a comprehensive 3D rectilinear Steiner tree algorithm that ensures
    connectivity between the driver and all relative_loads while maintaining no cycles and
    satisfying 3D rectilinear constraints (each edge moves along exactly one axis).

    The algorithm consists of four main phases:
    0. Scale tree and loads to normalize m-axis asymmetry
    1. Edge discretization with rtree-based non-overlapping processing
    2. Rectilinear edge construction for unconnected loads
    3. NetworkX graph construction with Steiner tree optimization
    4. Restore original scale
    5. Comprehensive validation with strict error handling

    Args:
        relative_loads: List of coordinate strings that need to be connected
        pred_relative_tree: Existing tree structure with driver as root
        tokenizer: UnifiedTokenizer for coordinate parsing and tree conversion
        scale_required: Choose whether to apply scaling

    Returns:
        Tuple of (coordinate tokens representing optimized tree, cost metrics dictionary)
    """
    # Initialize cost tracking
    cost_metrics = {
        "post_remove_wirelength_cost": 0.0,
        "post_remove_via_cost": 0.0,
        "post_add_wirelength_cost": 0.0,
        "post_add_via_cost": 0.0,
    }

    if not relative_loads:
        return convert_tree_to_coord_tokens(pred_relative_tree, tokenizer), cost_metrics

    try:
        # Phase 0: Scale tree and relative loads to normalize m-axis asymmetry
        processed_tree, processed_loads, scale_factor = (
            scale_tree_and_loads_for_optimization(
                pred_relative_tree, relative_loads, tokenizer
            )
            if scale_required
            else (pred_relative_tree, relative_loads, 1.0)
        )

        # Phase 1: Initialize discrete_tree_edges with edge splitting and non-overlapping processing
        discrete_tree_edges = initialize_discrete_tree_edges(
            processed_tree, processed_loads, tokenizer
        )

        # Phase 2: Construct rectilinear_edges by connecting unconnected loads
        connection_edges = construct_rectilinear_edges(
            discrete_tree_edges, processed_loads, tokenizer, processed_tree
        )

        # Phase 3: Build NetworkX graph and construct Steiner tree
        steiner_tree_node = build_networkx_steiner_tree(
            discrete_tree_edges,
            connection_edges,
            processed_loads,
            processed_tree,
            tokenizer,
        )

        # Phase 4: Restore original scale
        discrete_tree_edges = (
            [
                restore_edge_scale(scaled_edge, scale_factor)
                for scaled_edge in discrete_tree_edges
            ]
            if scale_required
            else discrete_tree_edges
        )
        steiner_tree_node = (
            restore_tree_scale(steiner_tree_node, scale_factor)
            if scale_required
            else steiner_tree_node
        )

        # Phase 5: Comprehensive validation with strict error handling
        # validate_steiner_tree_strict(steiner_tree_node, relative_loads, tokenizer)

        # Calculate cost metrics
        original_edges = set(discrete_tree_edges)
        final_edges = set(get_edges(steiner_tree_node))

        # Add costs: new edges added
        added_edges = list(final_edges - original_edges)
        if added_edges:
            add_wirelength, add_via = calculate_modification_costs(added_edges)
            cost_metrics["post_add_wirelength_cost"] = add_wirelength
            cost_metrics["post_add_via_cost"] = add_via

        # Remove costs: edges removed from original
        removed_edges = list(original_edges - final_edges)
        if removed_edges:
            remove_wirelength, remove_via = calculate_modification_costs(removed_edges)
            cost_metrics["post_remove_wirelength_cost"] = remove_wirelength
            cost_metrics["post_remove_via_cost"] = remove_via

        return convert_tree_to_coord_tokens(steiner_tree_node, tokenizer), cost_metrics

    except (ValueError, ConnectionError) as e:
        # Validation failed - log error and return original tree
        logging.warning(f"Steiner tree validation failed: {e}")
        # Return original tree with zero cost adjustments
        return convert_tree_to_coord_tokens(pred_relative_tree, tokenizer), cost_metrics


def convert_tree_to_coord_tokens(tree: Node, tokenizer: UnifiedTokenizer) -> List[str]:
    """Convert a tree structure to a list of coordinate tokens."""
    result_tokens = []
    BRANCH_TOKEN = tokenizer.special_token_manager.get_token_by_name("BRANCH_TOKEN")
    END_TOKEN = tokenizer.special_token_manager.get_token_by_name("END_TOKEN")

    def traverse(node: Node, result_tokens: List[str]):
        if node is None:
            return
        # Recursively prune children first
        children = node.children
        if node.coord_str:
            result_tokens.append(node.coord_str)
        for child in children:
            if len(children) > 1:
                result_tokens.append(BRANCH_TOKEN)
            traverse(child, result_tokens)
            if len(children) > 1:
                result_tokens.append(END_TOKEN)

    traverse(tree, result_tokens)
    return result_tokens


def rectilinear_distance(p1: CoordinatePoint, p2: CoordinatePoint) -> float:
    """Calculate 3D Manhattan (rectilinear) distance between two points."""
    return abs(p1.x - p2.x) + abs(p1.y - p2.y) + abs(p1.m - p2.m)


def enumerate_rectilinear_edges(
    start: CoordinatePoint, end: CoordinatePoint, prohibited_layers: set[int]
) -> List[Tuple[CoordinatePoint, CoordinatePoint]]:
    """
    Enumerate all possible rectilinear edges from start to end.
    """
    if is_rectilinear_compliant(start, end):
        return [(start, end)]

    edges: List[Tuple[CoordinatePoint, CoordinatePoint]] = []

    xs = [start.x] + ([end.x] if end.x != start.x else [])
    ys = [start.y] + ([end.y] if end.y != start.y else [])
    ms = [start.m] + ([end.m] if end.m != start.m else [])
    ms = [m for m in ms if m not in prohibited_layers]

    if start.x != end.x:
        for y in ys:
            for z in ms:
                p = CoordinatePoint(start.x, y, z)
                q = CoordinatePoint(end.x, y, z)
                edges.append((p, q))

    if start.y != end.y:
        for x in xs:
            for z in ms:
                p = CoordinatePoint(x, start.y, z)
                q = CoordinatePoint(x, end.y, z)
                edges.append((p, q))

    if start.m != end.m:
        for x in xs:
            for y in ys:
                p = CoordinatePoint(x, y, start.m)
                q = CoordinatePoint(x, y, end.m)
                edges.append((p, q))

    return edges


def calculate_modification_costs(
    edges: List[Tuple[CoordinatePoint, CoordinatePoint]],
) -> Tuple[int, int]:
    """
    Calculate wirelength and via costs for a list of edges (modifications).

    Args:
        edges: List of edge tuples representing modifications

    Returns:
        Tuple of (wirelength_cost, via_cost)
    """
    if not edges:
        return 0, 0

    total_wirelength = 0
    total_via_cost = 0

    for start_coord, end_coord in edges:
        # Wirelength cost (x-y plane)
        wirelength = abs(start_coord.x - end_coord.x) + abs(start_coord.y - end_coord.y)
        total_wirelength += wirelength

        # Via cost (m-axis distance divided by 2)
        via_cost = abs(start_coord.m - end_coord.m) // 2
        total_via_cost += via_cost

    return total_wirelength, total_via_cost


# ===========================================================================================
# METRIC Plotting FUNCTIONS
# ===========================================================================================
def plot_PDF(
    data: dict[str, np.ndarray],
    alias: str,
    save_path: Path,
    clip_long_tail: bool = True,
    min_clip_percentage: float = 0.0,
    max_clip_percentage: float = 95.0,
):
    plt.figure(figsize=FIG_SIZE, dpi=300)

    colors = _palette_slice(len(data))
    for i, (label, values) in enumerate(data.items()):
        sns.histplot(
            values,
            color=colors[i],
            label=label,
            stat="density",
            bins=50,
            alpha=0.5,
        )

    if clip_long_tail:
        min_val_for_clip = min(
            np.percentile(values, min_clip_percentage) for values in data.values()
        )
        max_val_for_clip = max(
            np.percentile(values, max_clip_percentage) for values in data.values()
        )
        plt.xlim(min_val_for_clip, max_val_for_clip)

    plt.xlabel(f"{alias}", fontsize=FONT_SIZE)
    plt.ylabel("Density", fontsize=FONT_SIZE)
    plt.legend(frameon=False, fontsize=LEGEND_SIZE)

    # Set tick font size
    plt.xticks(fontsize=FONT_SIZE)
    plt.yticks(fontsize=FONT_SIZE)

    # Set scientific notation font size
    ax = plt.gca()
    ax.tick_params(axis="both", which="major", labelsize=FONT_SIZE)
    # Set offset text (scientific notation) font size
    ax.xaxis.offsetText.set_fontsize(FONT_SIZE)
    ax.yaxis.offsetText.set_fontsize(FONT_SIZE)

    # Hide zero ticks to prevent overlap
    # Check if both x and y axes have 0 ticks
    x_ticks = ax.get_xticks()
    y_ticks = ax.get_yticks()
    has_x_zero = 0 in x_ticks
    has_y_zero = 0 in y_ticks

    # Only remove y-axis 0 ticks if both axes have 0 ticks
    if has_x_zero and has_y_zero:
        # Save y-axis range before modifying ticks
        y_min, y_max = ax.get_ylim()
        ax.set_yticks([t for t in y_ticks if t != 0])
        # Restore original y-axis range
        ax.set_ylim(y_min, y_max)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_KDE(
    data_dict: dict[str, np.ndarray],
    alias: str,
    save_path: Path,
    clip_long_tail: bool = True,
    min_clip_percentage: float = 0.0,
    max_clip_percentage: float = 95.0,
):
    plt.figure(figsize=FIG_SIZE, dpi=300)

    colors = _palette_slice(len(data_dict))
    for i, (label, values) in enumerate(data_dict.items()):
        mean, std = np.mean(values), np.std(values)
        sns.kdeplot(
            values,
            color=colors[i],
            lw=LINE_WIDTH,
            fill=True,
            alpha=0.05,
            linestyle="--",
            label=rf"{label} ($\mu$={mean:.2f}, $\sigma$={std:.2f})",
        )
        # plt.axvline(mean, color=f"C{i}", linestyle="--", lw=LINE_WIDTH, alpha=0.85)

    if clip_long_tail:
        min_val_for_clip = min(
            np.percentile(values, min_clip_percentage) for values in data_dict.values()
        )
        max_val_for_clip = max(
            np.percentile(values, max_clip_percentage) for values in data_dict.values()
        )
        plt.xlim(min_val_for_clip, max_val_for_clip)

    plt.xlabel(f"{alias}", fontsize=FONT_SIZE)
    plt.ylabel("Density", fontsize=FONT_SIZE)
    legends = plt.legend(frameon=False, fontsize=LEGEND_SIZE)
    for legend in legends.legend_handles:
        legend.set_linestyle("-")

    ax = plt.gca()
    max_val = max(v.max() for v in data_dict.values())
    if max_val > 1e5:
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    else:
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # Set tick font size
    plt.xticks(fontsize=FONT_SIZE)
    plt.yticks(fontsize=FONT_SIZE)

    # Set scientific notation font size
    ax.tick_params(axis="both", which="major", labelsize=FONT_SIZE)
    # Set offset text (scientific notation) font size
    ax.xaxis.offsetText.set_fontsize(FONT_SIZE)
    ax.yaxis.offsetText.set_fontsize(FONT_SIZE)

    # Hide zero ticks to prevent overlap
    # Check if both x and y axes have 0 ticks
    x_ticks = ax.get_xticks()
    y_ticks = ax.get_yticks()
    has_x_zero = 0 in x_ticks
    has_y_zero = 0 in y_ticks

    # Only remove y-axis 0 ticks if both axes have 0 ticks
    if has_x_zero and has_y_zero:
        # Save y-axis range before modifying ticks
        y_min, y_max = ax.get_ylim()
        ax.set_yticks([t for t in y_ticks if t != 0])
        # Restore original y-axis range
        ax.set_ylim(y_min, y_max)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_KDE_cumulative(
    data_dict: dict[str, np.ndarray],
    alias: str,
    save_path: Path,
    clip_long_tail: bool = True,
    min_clip_percentage: float = 0.0,
    max_clip_percentage: float = 95.0,
):
    plt.figure(figsize=FIG_SIZE, dpi=300)

    colors = _palette_slice(len(data_dict))
    for i, (label, values) in enumerate(data_dict.items()):
        mean, std = np.mean(values), np.std(values)
        sns.kdeplot(
            values,
            cumulative=True,
            color=colors[i],
            lw=LINE_WIDTH,
            label=rf"{label} ($\mu$={mean:.2f}, $\sigma$={std:.2f})",
        )

    if clip_long_tail:
        min_val_for_clip = min(
            np.percentile(values, min_clip_percentage) for values in data_dict.values()
        )
        max_val_for_clip = max(
            np.percentile(values, max_clip_percentage) for values in data_dict.values()
        )
        plt.xlim(min_val_for_clip, max_val_for_clip)

    plt.xlabel(f"{alias}", fontsize=FONT_SIZE)
    plt.ylabel("Cumulative Density", fontsize=FONT_SIZE)
    plt.legend(frameon=False, fontsize=LEGEND_SIZE)

    ax = plt.gca()
    max_val = max(v.max() for v in data_dict.values())
    if max_val > 1e5:
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    else:
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # Set tick font size
    plt.xticks(fontsize=FONT_SIZE)
    plt.yticks(fontsize=FONT_SIZE)

    # Set scientific notation font size
    ax.tick_params(axis="both", which="major", labelsize=FONT_SIZE)
    # Set offset text (scientific notation) font size
    ax.xaxis.offsetText.set_fontsize(FONT_SIZE)
    ax.yaxis.offsetText.set_fontsize(FONT_SIZE)

    # Hide zero ticks to prevent overlap
    # Check if both x and y axes have 0 ticks
    x_ticks = ax.get_xticks()
    y_ticks = ax.get_yticks()
    has_x_zero = 0 in x_ticks
    has_y_zero = 0 in y_ticks

    # Only remove y-axis 0 ticks if both axes have 0 ticks
    if has_x_zero and has_y_zero:
        # Save y-axis range before modifying ticks
        y_min, y_max = ax.get_ylim()
        ax.set_yticks([t for t in y_ticks if t != 0])
        # Restore original y-axis range
        ax.set_ylim(y_min, y_max)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_CDF(
    data_dict: dict[str, np.ndarray],
    alias: str,
    save_path: Path,
    clip_long_tail: bool = True,
    min_clip_percentage: float = 0.0,
    max_clip_percentage: float = 95.0,
):
    plt.figure(figsize=FIG_SIZE, dpi=300)

    colors = _palette_slice(len(data_dict))
    for i, (label, values) in enumerate(data_dict.items()):
        sns.ecdfplot(
            values,
            color=colors[i],
            lw=LINE_WIDTH,
            label=label,
        )

    if clip_long_tail:
        min_val_for_clip = min(
            np.percentile(values, min_clip_percentage) for values in data_dict.values()
        )
        max_val_for_clip = max(
            np.percentile(values, max_clip_percentage) for values in data_dict.values()
        )
        plt.xlim(min_val_for_clip, max_val_for_clip)

    plt.xlabel(f"{alias}", fontsize=FONT_SIZE)
    plt.ylabel("Cumulative Density", fontsize=FONT_SIZE)
    plt.legend(frameon=False, fontsize=LEGEND_SIZE)

    # Set tick font size
    plt.xticks(fontsize=FONT_SIZE)
    plt.yticks(fontsize=FONT_SIZE)

    # Set scientific notation font size
    ax = plt.gca()
    ax.tick_params(axis="both", which="major", labelsize=FONT_SIZE)
    # Set offset text (scientific notation) font size
    ax.xaxis.offsetText.set_fontsize(FONT_SIZE)
    ax.yaxis.offsetText.set_fontsize(FONT_SIZE)

    # Hide zero ticks to prevent overlap
    # Check if both x and y axes have 0 ticks
    x_ticks = ax.get_xticks()
    y_ticks = ax.get_yticks()
    has_x_zero = 0 in x_ticks
    has_y_zero = 0 in y_ticks

    # Only remove y-axis 0 ticks if both axes have 0 ticks
    if has_x_zero and has_y_zero:
        # Save y-axis range before modifying ticks
        y_min, y_max = ax.get_ylim()
        ax.set_yticks([t for t in y_ticks if t != 0])
        # Restore original y-axis range
        ax.set_ylim(y_min, y_max)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
