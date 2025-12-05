#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   tokenizer_comparator.py
@Time    :   2025/08/15 11:17:34
@Author  :   Dawn Li
@Version :   1.0
@Contact :   dawnli619215645@gmail.com
@Desc    :   Tokenizer comparator for analyzing human language and domain-specific tokenization
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Union

import matplotlib.pyplot as plt
import pyarrow.compute as pc
import scienceplots
from datasets import Dataset
from transformers import AutoTokenizer

from flow import FlowConfig
from flow.utils.plot_utils import COLOR_PALETTE, palette_slice

if scienceplots:
    plt.style.use(["science", "no-latex"])


class TokenizerComparator:
    """Compares human language and domain-specific tokenizers"""

    def __init__(
        self,
        human_tokenizer_path: Union[str, Path],
        domain_tokenizer_path: Union[str, Path],
        output_dir: Path,
        plot_only: bool = False,
        fig_size: tuple[int, int] = (8, 8),
        text_size: int = 28,
        font_size: int = 36,
        palette: tuple = COLOR_PALETTE,
    ):
        """
        Initialize tokenizer comparator

        Args:
            human_tokenizer_path: Path or model name for human language tokenizer
            domain_tokenizer_path: Path to domain-specific tokenizer
            output_dir: Output directory for results
            plot_only: If True, skip loading tokenizers (for plot-only mode)
        """
        self.human_tokenizer_path = human_tokenizer_path
        self.domain_tokenizer_path = Path(domain_tokenizer_path)
        self.output_dir = Path(output_dir)
        self.plot_only = plot_only
        self.fig_size = fig_size
        self.text_size = text_size
        self.font_size = font_size
        self.palette = palette_slice(len(palette), palette)

        # Initialize tokenizers
        self.human_tokenizer = None
        self.domain_tokenizer = None

        # Create UnifiedTokenizer for additional processing if needed
        self.flow_config = FlowConfig()
        self.flow_config.tokenization.performance.num_workers = 48
        self.flow_config.tokenization.workflow.tokenizer_algorithm = "DecimalWordLevel"

        # Only load tokenizers if not in plot-only mode
        if not plot_only:
            self._load_tokenizers()

    def _load_tokenizers(self) -> None:
        """Load both human language and domain-specific tokenizers"""
        try:
            # Load human language tokenizer
            logging.info(
                f"Loading human language tokenizer: {self.human_tokenizer_path}"
            )
            self.human_tokenizer = AutoTokenizer.from_pretrained(
                str(self.human_tokenizer_path), trust_remote_code=True
            )
            logging.info(
                f"Human tokenizer loaded. Vocab size: {len(self.human_tokenizer.get_vocab())}"
            )

            # Load domain-specific tokenizer
            logging.info(
                f"Loading domain-specific tokenizer: {self.domain_tokenizer_path}"
            )
            self.domain_tokenizer = AutoTokenizer.from_pretrained(
                str(self.domain_tokenizer_path)
            )
            logging.info(
                f"Domain tokenizer loaded. Vocab size: {len(self.domain_tokenizer.get_vocab())}"
            )

        except Exception as e:
            logging.error(f"Failed to load tokenizers: {e}")
            raise

    def tokenize_corpus_with_human_tokenizer(self, dataset: Dataset) -> Dict[str, Any]:
        """
        Tokenize corpus using human language tokenizer, then compute stats fast in Arrow.
        """

        if self.human_tokenizer is None:
            raise ValueError("Human tokenizer not loaded")

        logging.info(f"Tokenizing {len(dataset)} samples with human tokenizer...")

        def tokenize_text(batch: Dict[str, Any]) -> Dict[str, Any]:
            """
            Tokenize text using human language tokenizer
            """
            return {
                "tokens": [
                    self.human_tokenizer.tokenize(seq) for seq in batch["tree_seq"]
                ]
            }

        tokenized_dataset = dataset.map(
            tokenize_text,
            batched=True,
            num_proc=self.flow_config.tokenization.performance.num_workers,
            desc="Tokenizing dataset with human tokenizer",
        )

        table = tokenized_dataset.data.table

        tokens_col = table["tokens"]

        flat_tokens = pc.list_flatten(tokens_col)
        flat_tokens = pc.drop_null(flat_tokens)

        vc = pc.value_counts(flat_tokens)
        unique_tokens = vc.field("values")
        counts = vc.field("counts")

        per_row_lengths = pc.list_value_length(tokens_col)
        total_tokens_processed = pc.sum(per_row_lengths).as_py()
        avg_tokens_per_sample = pc.mean(per_row_lengths).as_py()

        unique_tokens_used = len(unique_tokens)

        token_list = unique_tokens.to_pylist()
        count_list = counts.to_pylist()
        token_counts = dict(zip(token_list, count_list))

        vocab_dict = self.human_tokenizer.get_vocab()
        full_vocab = set(vocab_dict.keys())
        used_tokens = set(token_list)
        unused_tokens = list(full_vocab - used_tokens)

        total_vocab_size = len(full_vocab)
        vocab_utilization_rate = (
            (len(used_tokens) / total_vocab_size * 100.0) if total_vocab_size else 0.0
        )

        results = {
            "total_tokens_processed": total_tokens_processed,
            "unique_tokens_used": unique_tokens_used,
            "total_vocab_size": total_vocab_size,
            "vocab_utilization_rate": vocab_utilization_rate,
            "token_counts": token_counts,
            "used_tokens": list(used_tokens),
            "unused_tokens": unused_tokens,
            "avg_tokens_per_sample": avg_tokens_per_sample,
        }

        logging.info(
            f"Human tokenizer results: {unique_tokens_used}/{total_vocab_size} tokens used "
            f"({vocab_utilization_rate:.2f}%), total tokens processed={total_tokens_processed}"
        )
        return results

    def tokenize_corpus_with_domain_tokenizer(self, dataset: Dataset) -> Dict[str, Any]:
        """
        Tokenize corpus using domain-specific tokenizer

        Args:
            corpus: List of text strings to tokenize

        Returns:
            Dictionary containing tokenization results and statistics
        """
        if self.domain_tokenizer is None:
            raise ValueError("Domain tokenizer not loaded")

        logging.info(f"Tokenizing {len(dataset)} samples with domain tokenizer...")

        def tokenize_text(batch: Dict[str, Any]) -> Dict[str, Any]:
            """
            Tokenize text using human language tokenizer
            """
            return {
                "tokens": [
                    self.domain_tokenizer.tokenize(target_tokens)
                    for target_tokens in batch["target_tokens"]
                ]
            }

        tokenized_dataset = dataset.map(
            tokenize_text,
            batched=True,
            num_proc=self.flow_config.tokenization.performance.num_workers,
            desc="Tokenizing dataset with domain tokenizer",
        )

        table = tokenized_dataset.data.table

        tokens_col = table["tokens"]

        flat_tokens = pc.list_flatten(tokens_col)
        flat_tokens = pc.drop_null(flat_tokens)

        vc = pc.value_counts(flat_tokens)
        unique_tokens = vc.field("values")
        counts = vc.field("counts")

        per_row_lengths = pc.list_value_length(tokens_col)
        total_tokens_processed = pc.sum(per_row_lengths).as_py()
        avg_tokens_per_sample = pc.mean(per_row_lengths).as_py()

        unique_tokens_used = int(len(unique_tokens))

        token_list = unique_tokens.to_pylist()
        count_list = counts.to_pylist()
        token_counts = dict(zip(token_list, count_list))

        vocab_dict = self.domain_tokenizer.get_vocab()
        full_vocab = set(vocab_dict.keys())
        used_tokens = set(token_list)
        unused_tokens = list(full_vocab - used_tokens)

        total_vocab_size = len(full_vocab)
        vocab_utilization_rate = (
            (len(used_tokens) / total_vocab_size * 100.0) if total_vocab_size else 0.0
        )

        results = {
            "total_tokens_processed": total_tokens_processed,
            "unique_tokens_used": unique_tokens_used,
            "total_vocab_size": total_vocab_size,
            "vocab_utilization_rate": vocab_utilization_rate,
            "token_counts": token_counts,
            "used_tokens": list(used_tokens),
            "unused_tokens": unused_tokens,
            "avg_tokens_per_sample": avg_tokens_per_sample,
        }

        logging.info(
            f"Domain tokenizer results: {len(used_tokens)}/{len(full_vocab)} tokens used ({results['vocab_utilization_rate']:.2f}%), "
            f"total tokens processed={total_tokens_processed}"
        )
        return results

    def compare_tokenizers(self, dataset: Dataset) -> Dict[str, Any]:
        """
        Perform comprehensive comparison between tokenizers

        Args:
            tree_seq_corpus: Corpus for human language tokenization
            target_tokens_corpus: Corpus for domain-specific tokenization

        Returns:
            Comprehensive comparison results
        """
        logging.info("Starting tokenizer comparison analysis...")

        # Tokenize with human language tokenizer
        human_results = self.tokenize_corpus_with_human_tokenizer(dataset)

        # Tokenize with domain-specific tokenizer
        domain_results = self.tokenize_corpus_with_domain_tokenizer(dataset)

        # Create comparison summary in the requested format
        comparison_results = {
            "human_used_count": human_results["unique_tokens_used"],
            "human_unused_count": len(human_results["unused_tokens"]),
            "domain_specific_used_count": domain_results["unique_tokens_used"],
            "domain_specific_unused_count": len(domain_results["unused_tokens"]),
            "human_used_tokens": human_results["used_tokens"],
            "human_unused_tokens": human_results["unused_tokens"],
            "domain_specific_used_tokens": domain_results["used_tokens"],
            "domain_specific_unused_tokens": domain_results["unused_tokens"],
            "human_used_tokens_freq": human_results["token_counts"],
            "domain_specific_used_tokens_freq": domain_results["token_counts"],
        }

        # Add detailed analysis
        detailed_analysis = {
            "comparison_summary": comparison_results,
            "human_tokenizer_analysis": {
                "model_name": str(self.human_tokenizer_path),
                "total_vocab_size": human_results["total_vocab_size"],
                "vocab_utilization_rate": human_results["vocab_utilization_rate"],
                "avg_tokens_per_sample": human_results["avg_tokens_per_sample"],
                "total_tokens_processed": human_results["total_tokens_processed"],
            },
            "domain_tokenizer_analysis": {
                "tokenizer_path": str(self.domain_tokenizer_path),
                "total_vocab_size": domain_results["total_vocab_size"],
                "vocab_utilization_rate": domain_results["vocab_utilization_rate"],
                "avg_tokens_per_sample": domain_results["avg_tokens_per_sample"],
                "total_tokens_processed": domain_results["total_tokens_processed"],
            },
            "efficiency_metrics": {
                "human_efficiency": human_results["vocab_utilization_rate"],
                "domain_efficiency": domain_results["vocab_utilization_rate"],
                "efficiency_ratio": domain_results["vocab_utilization_rate"]
                / human_results["vocab_utilization_rate"]
                if human_results["vocab_utilization_rate"] > 0
                else float("inf"),
                "compression_ratio_human": human_results["avg_tokens_per_sample"],
                "compression_ratio_domain": domain_results["avg_tokens_per_sample"],
            },
        }

        return detailed_analysis

    def save_comparison_results(self, results: Dict[str, Any]) -> None:
        """
        Save comparison results to JSON file

        Args:
            results: Comparison results to save
        """
        output_dir = self.output_dir / "demo"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save the main comparison results in the requested format
        comparison_file = output_dir / "word_comparison.json"
        with open(comparison_file, "w", encoding="utf-8") as f:
            json.dump(results["comparison_summary"], f, indent=2, ensure_ascii=False)

        # Save detailed analysis
        detailed_file = output_dir / "detailed_analysis.json"
        with open(detailed_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logging.info(f"Comparison results saved to {comparison_file}")
        logging.info(f"Detailed analysis saved to {detailed_file}")

    def analyze_token_overlap(self) -> Dict[str, Any]:
        """
        Analyze token overlap between human and domain tokenizers

        Returns:
            Token overlap analysis results
        """
        if self.human_tokenizer is None or self.domain_tokenizer is None:
            raise ValueError("Both tokenizers must be loaded")

        # Get vocabularies
        if hasattr(self.human_tokenizer, "get_vocab"):
            human_vocab = set(self.human_tokenizer.get_vocab().keys())
        else:
            human_vocab = set()

        if hasattr(self.domain_tokenizer, "get_vocab"):
            domain_vocab = set(self.domain_tokenizer.get_vocab().keys())
        else:
            domain_vocab = set(self.domain_tokenizer.get_vocab().keys())

        # Calculate overlap
        intersection = human_vocab & domain_vocab
        human_only = human_vocab - domain_vocab
        domain_only = domain_vocab - human_vocab

        overlap_analysis = {
            "total_human_vocab": len(human_vocab),
            "total_domain_vocab": len(domain_vocab),
            "intersection_size": len(intersection),
            "human_only_size": len(human_only),
            "domain_only_size": len(domain_only),
            "overlap_percentage": len(intersection)
            / len(human_vocab | domain_vocab)
            * 100
            if (human_vocab | domain_vocab)
            else 0,
            "intersection_tokens": list(intersection),
            "human_only_tokens": list(human_only),
            "domain_only_tokens": list(domain_only),
        }

        return overlap_analysis

    def plot_top10_frequencies(
        self,
        token_freq: Dict[str, int],
        output_path: Path,
        cmap_name: str,
        title: str,
    ) -> None:
        """
        Plot top-10 token frequency distribution

        Args:
            token_freq: Dictionary of token frequencies
            output_path: Path to save the plot
            title: Title for the plot
        """
        import matplotlib as mpl

        mpl.rcParams["text.usetex"] = False
        # Drop tokens with '<' or '>' in them
        token_freq = {
            k: v for k, v in token_freq.items() if "<" not in k and ">" not in k
        }
        # Sort tokens by frequency and get top 10
        sorted_tokens = sorted(token_freq.items(), key=lambda x: x[1], reverse=True)[
            :10
        ]

        if not sorted_tokens:
            logging.warning(f"No tokens to plot for {title}")
            return

        tokens = [t[0] for t in sorted_tokens]
        frequencies = [t[1] for t in sorted_tokens]

        # Create figure
        fig, ax = plt.subplots(figsize=self.fig_size)

        # Create bars with shared palette for consistency
        cmap = plt.get_cmap(cmap_name)
        colors = [cmap(0.75 - i / 2 / len(tokens)) for i in range(len(tokens))]
        bars = ax.bar(
            range(len(tokens)),
            frequencies,
            color=colors,
            edgecolor="black",
            linewidth=1.2,
        )

        # Add value labels on top of bars
        for i, (bar, freq) in enumerate(zip(bars, frequencies)):
            label_text = f"{freq}" if freq < 1e6 else f"{freq / 1e6:.1f}M"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                label_text,
                ha="center",
                va="bottom",
                rotation=45,
                fontsize=self.text_size,
            )

        # Customize plot
        ax.set_yticklabels([""])
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(
            tokens,
            rotation=75,
            ha="center",
            fontfamily="DejaVu Sans",
            fontsize=self.font_size,
        )
        for spine in ax.spines.values():
            spine.set_visible(False)
        # Save plot
        plt.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        plt.style.use(["science", "no-latex"])
        logging.info(f"Saved top-10 frequency plot to {output_path}")

    def plot_frequencies_from_results(self, results: Dict[str, Any]) -> None:
        """
        Generate top-10 frequency plots from comparison results

        Args:
            results: Comparison results containing frequency data
        """
        comparison_summary = results.get("comparison_summary", {})

        # Plot human tokenizer frequencies
        human_freq = comparison_summary.get("human_used_tokens_freq", {})
        if human_freq:
            human_output = self.output_dir / "demo" / "human_top10.svg"
            self.plot_top10_frequencies(
                human_freq,
                human_output,
                "BuGn",
                "Top 10 Most Frequent Human Language Tokens",
            )

        # Plot domain-specific tokenizer frequencies
        domain_freq = comparison_summary.get("domain_specific_used_tokens_freq", {})
        if domain_freq:
            domain_output = self.output_dir / "demo" / "domain_top10.svg"
            self.plot_top10_frequencies(
                domain_freq,
                domain_output,
                "GnBu",
                "Top 10 Most Frequent Domain-Specific Tokens",
            )

    def plot_from_json(self, json_path: Path = None) -> None:
        """
        Plot frequency distributions from saved JSON file (for plot-only mode)

        Args:
            json_path: Path to word_comparison.json file. If None, uses default path.
        """
        if json_path is None:
            json_path = self.output_dir / "demo" / "word_comparison.json"

        if not json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {json_path}")

        logging.info(f"Loading comparison data from {json_path}")
        with open(json_path, "r", encoding="utf-8") as f:
            comparison_data = json.load(f)

        # Plot human tokenizer frequencies
        human_freq = comparison_data.get("human_used_tokens_freq", {})
        if human_freq:
            human_output = self.output_dir / "demo" / "human_top10.svg"
            self.plot_top10_frequencies(
                human_freq,
                human_output,
                "BuGn",
                "Top 10 Most Frequent Human Language Tokens",
            )

        # Plot domain-specific tokenizer frequencies
        domain_freq = comparison_data.get("domain_specific_used_tokens_freq", {})
        if domain_freq:
            domain_output = self.output_dir / "demo" / "domain_top10.svg"
            self.plot_top10_frequencies(
                domain_freq,
                domain_output,
                "GnBu",
                "Top 10 Most Frequent Domain-Specific Tokens",
            )

        logging.info("Frequency plots generated successfully from JSON")
