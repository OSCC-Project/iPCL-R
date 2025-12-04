#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   main.py
@Time    :   2025/08/15 11:17:12
@Author  :   Dawn Li
@Version :   1.0
@Contact :   dawnli619215645@gmail.com
@Desc    :   Main entry point for symbol analysis experiments comparing human language tokenization
            with domain-specific tokenization using FlowConfig and UnifiedTokenizer.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from flow.utils import setup_logging

from .dataset_processor import DatasetProcessor
from .embedding_visualizer import EmbeddingVisualizer
from .research_analyzer import ResearchAnalyzer
from .tokenizer_comparator import TokenizerComparator


class SymbolAnalyzer:
    """Main orchestrator for symbol analysis pipeline"""

    def __init__(
        self,
        dataset_dir: Path,
        output_dir: Path,
        human_tokenizer: str,
        domain_tokenizer: Path,
        human_model: Optional[str] = None,
        domain_model: Optional[Path] = None,
        plot_only: bool = False,
        update_reduce: bool = False,
    ):
        """
        Initialize symbol analyzer

        Args:
            dataset_dir: Path to dataset directory
            output_dir: Output directory for results
            human_tokenizer: Human language tokenizer model name or path
            domain_tokenizer: Path to domain-specific tokenizer
            plot_only: If True, only initialize components needed for plotting
            update_reduce: If True with plot_only, reload high-dimensional embeddings and re-reduce
        """
        self.dataset_dir = dataset_dir
        self.output_dir = output_dir
        self.human_tokenizer = human_tokenizer
        self.domain_tokenizer = domain_tokenizer
        self.human_model = human_model
        self.domain_model = domain_model
        self.plot_only = plot_only
        self.update_reduce = update_reduce

        # Initialize components (skip some in plot-only mode)
        if not plot_only:
            self.dataset_processor = DatasetProcessor(dataset_dir)
            self.tokenizer_comparator = TokenizerComparator(
                human_tokenizer, domain_tokenizer, output_dir
            )
            self.research_analyzer = ResearchAnalyzer(output_dir)
        else:
            # In plot-only mode, initialize tokenizer comparator for plotting
            self.tokenizer_comparator = TokenizerComparator(
                human_tokenizer, domain_tokenizer, output_dir, plot_only=True
            )

        # Always initialize embedding visualizer for plotting
        self.embedding_visualizer = EmbeddingVisualizer(output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_analysis(self) -> None:
        """Run complete symbol analysis pipeline"""
        logging.info("Starting comprehensive symbol analysis...")

        # Step 1: Load dataset and extract corpus
        logging.info("=== Step 1: Dataset Processing ===")
        self.dataset_processor.load_dataset()

        # Extract corpus for different tokenization methods
        self.dataset_processor.extract_corpus()

        # Get dataset statistics
        dataset_stats = self.dataset_processor.get_dataset_statistics()
        logging.info(f"Dataset contains {dataset_stats['total_samples']} samples")

        # Step 2: Tokenizer comparison
        logging.info("=== Step 2: Tokenizer Comparison ===")
        comparison_results = self.tokenizer_comparator.compare_tokenizers(
            self.dataset_processor.dataset
        )

        # Save basic comparison results
        self.tokenizer_comparator.save_comparison_results(comparison_results)

        # Generate frequency distribution plots
        logging.info("Generating token frequency distribution plots...")
        self.tokenizer_comparator.plot_frequencies_from_results(comparison_results)

        # Analyze token overlap
        overlap_analysis = self.tokenizer_comparator.analyze_token_overlap()

        # Optional: Embedding visualization if model paths are provided
        try:
            if self.human_model and self.domain_model:
                logging.info("=== Step 3: Embedding Visualization ===")
                summary = comparison_results.get("comparison_summary", {})
                human_used_tokens = summary.get("human_used_tokens", [])
                human_unused_tokens = summary.get("human_unused_tokens", [])
                domain_used_tokens = summary.get("domain_specific_used_tokens", [])

                self.embedding_visualizer.plot_both_separately(
                    human_model_path=self.human_model,
                    human_tokenizer=self.tokenizer_comparator.human_tokenizer,
                    human_used_tokens=human_used_tokens,
                    domain_model_path=self.domain_model,
                    domain_tokenizer=self.tokenizer_comparator.domain_tokenizer,
                    domain_used_tokens=domain_used_tokens,
                    human_prefix="human_embeddings",
                    domain_prefix="domain_embeddings",
                    limit_per_set=1000,
                    human_unused_tokens=human_unused_tokens,
                    font_size=36,
                    legend_size=48,
                    marker_size=500,
                    figsize=(24, 12),
                )
            else:
                logging.info(
                    "Embedding visualization skipped (human_model or domain_model not provided)."
                )
        except Exception as e:
            logging.warning(f"Embedding visualization failed: {e}")

        # Step 4: Advanced research analysis
        logging.info("=== Step 4: Research Analysis ===")

        # Extract individual tokenizer results for research analysis
        human_results = comparison_results["human_tokenizer_analysis"]
        domain_results = comparison_results["domain_tokenizer_analysis"]

        # Add token lists to results for research analysis
        human_results.update(
            {
                "used_tokens": comparison_results["comparison_summary"][
                    "human_used_tokens"
                ],
                "unused_tokens": comparison_results["comparison_summary"][
                    "human_unused_tokens"
                ],
                "token_counts": {},  # Will be populated by tokenizer_comparator if needed
            }
        )

        domain_results.update(
            {
                "used_tokens": comparison_results["comparison_summary"][
                    "domain_specific_used_tokens"
                ],
                "unused_tokens": comparison_results["comparison_summary"][
                    "domain_specific_unused_tokens"
                ],
                "token_counts": {},  # Will be populated by tokenizer_comparator if needed
            }
        )

        # Run comprehensive research analysis
        research_results = self.research_analyzer.run_comprehensive_analysis(
            human_results, domain_results, self.dataset_processor.dataset
        )

        # Step 5: Generate final report
        logging.info("=== Step 5: Final Report Generation ===")
        self._generate_final_report(
            {
                "dataset_statistics": dataset_stats,
                "tokenizer_comparison": comparison_results,
                "token_overlap_analysis": overlap_analysis,
                "research_analysis": research_results,
            }
        )

        logging.info("Symbol analysis completed successfully!")
        logging.info(f"Results saved to: {self.output_dir}")

    def run_plot_only(self) -> None:
        """Run plot-only mode: read existing CSV files and create visualizations"""
        # Generate tokenizer frequency plots from saved JSON
        logging.info("Generating tokenizer frequency plots from saved data...")
        try:
            self.tokenizer_comparator.plot_from_json()
        except FileNotFoundError as e:
            logging.warning(f"Could not generate tokenizer frequency plots: {e}")
        except Exception as e:
            logging.error(f"Error generating tokenizer frequency plots: {e}")

        # Generate embedding visualizations
        if self.update_reduce:
            logging.info("Starting plot-only mode with update-reduce...")
            # Load high-dimensional embeddings and re-apply dimensionality reduction
            try:
                output_paths = self.embedding_visualizer.plot_from_high_dim_files()
                if output_paths:
                    logging.info(
                        "Plot-only mode with update-reduce completed successfully!"
                    )
                    logging.info(f"Generated {len(output_paths)} plots")
                else:
                    logging.error("No plots were generated")
            except Exception as e:
                logging.error(f"Plot generation with update-reduce failed: {e}")
                raise
        else:
            logging.info("Starting plot-only mode...")
            # Use existing CSV files (original behavior)
            try:
                output_paths = self.embedding_visualizer.plot_from_csv_files()
                if output_paths:
                    logging.info("Plot-only mode completed successfully!")
                    logging.info(f"Generated {len(output_paths)} plots")
                else:
                    logging.error("No plots were generated")
            except Exception as e:
                logging.error(f"Plot generation failed: {e}")
                raise

    def _generate_final_report(self, all_results: dict) -> None:
        """
        Generate comprehensive final report

        Args:
            all_results: All analysis results combined
        """
        import json

        # Save comprehensive results
        final_report_path = self.output_dir / "final_report.json"
        with open(final_report_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        # Generate text summary
        summary_path = self.output_dir / "analysis_summary.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            self._write_text_summary(f, all_results)

        logging.info(f"Final report saved to {final_report_path}")
        logging.info(f"Analysis summary saved to {summary_path}")

    def _write_text_summary(self, file, results: dict) -> None:
        """Write human-readable analysis summary"""
        file.write("SYMBOL ANALYSIS REPORT\n")
        file.write("=" * 50 + "\n\n")

        # Dataset overview
        if "dataset_statistics" in results:
            stats = results["dataset_statistics"]
            file.write("Dataset Overview:\n")
            file.write(f"  Total samples: {stats.get('total_samples', 'N/A')}\n")
            file.write(
                f"  Average tree_seq length: {stats.get('tree_seq_avg_length', 'N/A'):.2f}\n"
            )
            file.write(
                f"  Average target_tokens length: {stats.get('target_tokens_avg_length', 'N/A'):.2f}\n\n"
            )

        # Tokenizer comparison summary
        if "tokenizer_comparison" in results:
            comp = results["tokenizer_comparison"]["comparison_summary"]
            file.write("Tokenizer Comparison Results:\n")
            file.write("  Human Language Tokenizer:\n")
            file.write(f"    Used tokens: {comp.get('human_used_count', 'N/A')}\n")
            file.write(f"    Unused tokens: {comp.get('human_unused_count', 'N/A')}\n")
            file.write("  Domain-Specific Tokenizer:\n")
            file.write(
                f"    Used tokens: {comp.get('domain_specific_used_count', 'N/A')}\n"
            )
            file.write(
                f"    Unused tokens: {comp.get('domain_specific_unused_count', 'N/A')}\n\n"
            )

        # Research insights
        if (
            "research_analysis" in results
            and "research_insights" in results["research_analysis"]
        ):
            insights = results["research_analysis"]["research_insights"]
            file.write("Key Research Findings:\n")
            for finding in insights.get("key_findings", []):
                file.write(f"  • {finding}\n")
            file.write("\n")


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser"""
    parser = argparse.ArgumentParser(
        description="Symbol Analysis for Domain-Specific vs Human Language Tokenization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default parameters
  python -m experiments.symbol_analysis.main
  
  # Custom dataset and tokenizers
  python -m experiments.symbol_analysis.main \\
    --dataset-dir /path/to/your/dataset \\
    --human-tokenizer google/t5-base \\
    --domain-specific-tokenizer /path/to/domain/tokenizer
  
  # Plot-only mode: read existing CSV files and generate plots
  python -m experiments.symbol_analysis.main \\
    --plot-only \\
    --output-dir /path/to/existing/results
        """,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/mnt/local_data1/liweiguo/experiments/symbol_analysis"),
        help="Output directory for analysis results",
    )

    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("/mnt/local_data1/liweiguo/dataset/experiments/train"),
        help="Dataset directory path",
    )

    parser.add_argument(
        "--human-tokenizer",
        type=str,
        default="google/t5gemma-2b-2b-ul2",
        help="Human language tokenizer model name or path",
    )

    parser.add_argument(
        "--domain-specific-tokenizer",
        type=Path,
        default=Path(
            "/mnt/local_data1/liweiguo/experiments/model_size_comparison/work_dir/Large-DecimalWordLevel/stage_tokenization/tokenizer"
        ),
        help="Domain-specific tokenizer path",
    )

    parser.add_argument(
        "--human-model",
        type=str,
        default="google/t5gemma-2b-2b-ul2",
        help="Local path to human tokenizer-compatible model (for embedding visualization).",
    )

    parser.add_argument(
        "--domain-model",
        type=Path,
        default="/mnt/local_data1/liweiguo/experiments/model_size_comparison/work_dir/Large-DecimalWordLevel/stage_training/model/checkpoint-200826",
        help="Local path to domain tokenizer-compatible model (for embedding visualization).",
    )

    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip all analysis steps and directly plot from existing CSV files in the output directory.",
    )

    parser.add_argument(
        "--update-reduce",
        action="store_true",
        help="Used with --plot-only. When enabled, reload high-dimensional embeddings and re-apply dimensionality reduction. When disabled, use existing reduced CSV files.",
    )

    return parser


def main():
    # Setup logging
    setup_logging()

    parser = create_argument_parser()
    args = parser.parse_args()

    # Validate inputs (skip some validations in plot-only mode)
    if not args.plot_only:
        if not args.dataset_dir.exists():
            logging.error(f"Dataset directory does not exist: {args.dataset_dir}")
            sys.exit(1)

        if not args.domain_specific_tokenizer.exists():
            logging.error(
                f"Domain-specific tokenizer path does not exist: {args.domain_specific_tokenizer}"
            )
            sys.exit(1)

    # Validate update-reduce usage
    if args.update_reduce and not args.plot_only:
        logging.error("--update-reduce can only be used with --plot-only")
        sys.exit(1)

    # Create analyzer
    analyzer = SymbolAnalyzer(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        human_tokenizer=args.human_tokenizer,
        domain_tokenizer=args.domain_specific_tokenizer,
        human_model=args.human_model,
        domain_model=args.domain_model,
        plot_only=args.plot_only,
        update_reduce=args.update_reduce,
    )

    # Run analysis or plot-only mode
    if args.plot_only:
        analyzer.run_plot_only()
    else:
        analyzer.run_analysis()


if __name__ == "__main__":
    main()
