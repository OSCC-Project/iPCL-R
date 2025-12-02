#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   research_analyzer.py
@Time    :   2025/08/15 11:17:24
@Author  :   Dawn Li
@Version :   1.0
@Contact :   dawnli619215645@gmail.com
@Desc    :   Advanced research analysis for strengthening domain-specific tokenization viewpoint
"""

import json
import logging
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pyarrow.compute as pc
from datasets import Dataset


class ResearchAnalyzer:
    """Advanced research analysis for strengthening domain-specific tokenization viewpoint"""

    def __init__(self, output_dir: Path):
        """
        Initialize research analyzer

        Args:
            output_dir: Output directory for analysis results
        """
        self.output_dir = Path(output_dir)

        (self.output_dir / "research_analysis").mkdir(parents=True, exist_ok=True)

    def analyze_compression_efficiency(
        self,
        human_results: Dict[str, Any],
        domain_results: Dict[str, Any],
        dataset: Dataset,
    ) -> Dict[str, Any]:
        """
        Analyze compression efficiency between tokenization methods

        Args:
            human_results: Human tokenizer results
            domain_results: Domain tokenizer results
            dataset: Dataset containing original tree sequences and target tokens

        Returns:
            Compression efficiency analysis
        """
        logging.info("Analyzing compression efficiency...")

        # Calculate compression metrics
        table = dataset.data.table
        tree_seq_length = pc.utf8_length(table["tree_seq"])
        target_tokens_length = pc.utf8_length(table["target_tokens"])

        original_chars_tree = pc.sum(tree_seq_length).as_py()
        original_chars_target = pc.sum(target_tokens_length).as_py()

        human_tokens_total = human_results["total_tokens_processed"]
        domain_tokens_total = domain_results["total_tokens_processed"]

        # Compression ratios (lower is better)
        human_compression_ratio = (
            human_tokens_total / original_chars_tree
            if original_chars_tree > 0
            else float("inf")
        )
        domain_compression_ratio = (
            domain_tokens_total / original_chars_target
            if original_chars_target > 0
            else float("inf")
        )

        # Information density (tokens per character)
        human_info_density = (
            original_chars_tree / human_tokens_total if human_tokens_total > 0 else 0
        )
        domain_info_density = (
            original_chars_target / domain_tokens_total
            if domain_tokens_total > 0
            else 0
        )

        # Vocabulary efficiency (used tokens / total vocab)
        human_vocab_efficiency = human_results["vocab_utilization_rate"]
        domain_vocab_efficiency = domain_results["vocab_utilization_rate"]

        compression_analysis = {
            "human_compression_ratio": human_compression_ratio,
            "domain_compression_ratio": domain_compression_ratio,
            "compression_improvement": (
                human_compression_ratio - domain_compression_ratio
            )
            / human_compression_ratio
            * 100
            if human_compression_ratio > 0
            else 0,
            "human_info_density": human_info_density,
            "domain_info_density": domain_info_density,
            "info_density_improvement": (domain_info_density - human_info_density)
            / human_info_density
            * 100
            if human_info_density > 0
            else 0,
            "human_vocab_efficiency": human_vocab_efficiency,
            "domain_vocab_efficiency": domain_vocab_efficiency,
            "vocab_efficiency_improvement": domain_vocab_efficiency
            - human_vocab_efficiency,
            "efficiency_ratio": domain_vocab_efficiency / human_vocab_efficiency
            if human_vocab_efficiency > 0
            else float("inf"),
        }

        logging.info(
            f"Compression analysis: Domain is {compression_analysis['compression_improvement']:.2f}% more efficient"
        )
        return compression_analysis

    def analyze_semantic_coherence(
        self, human_results: Dict[str, Any], domain_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze semantic coherence of tokenization results

        Args:
            human_results: Human tokenizer results
            domain_results: Domain tokenizer results

        Returns:
            Semantic coherence analysis
        """
        logging.info("Analyzing semantic coherence...")

        # Analyze token patterns for domain relevance
        human_tokens = human_results["used_tokens"]
        domain_tokens = domain_results["used_tokens"]

        # Count routing-related patterns in tokens
        routing_patterns = [
            r"[UDLRTB]",  # Direction tokens
            r"\d+",  # Numeric coordinates
            r"PUSH|POP|BRANCH|END",  # Tree structure
            r"DRIVER|LOAD",  # Pin types
            r"SRC_END|BOS|EOS",  # Special tokens
        ]

        def count_routing_relevance(tokens: List[str]) -> Dict[str, int]:
            relevance_counts = defaultdict(int)
            total_relevant = 0

            for token in tokens:
                for pattern in routing_patterns:
                    if re.search(pattern, token):
                        relevance_counts[pattern] += 1
                        total_relevant += 1
                        break

            relevance_counts["total_relevant"] = total_relevant
            relevance_counts["total_tokens"] = len(tokens)
            relevance_counts["relevance_ratio"] = (
                total_relevant / len(tokens) if tokens else 0
            )

            return dict(relevance_counts)

        human_relevance = count_routing_relevance(human_tokens)
        domain_relevance = count_routing_relevance(domain_tokens)

        # Calculate semantic coherence metrics
        coherence_analysis = {
            "human_routing_relevance": human_relevance,
            "domain_routing_relevance": domain_relevance,
            "relevance_improvement": domain_relevance["relevance_ratio"]
            - human_relevance["relevance_ratio"],
            "domain_specialization_score": domain_relevance["relevance_ratio"]
            / human_relevance["relevance_ratio"]
            if human_relevance["relevance_ratio"] > 0
            else float("inf"),
        }

        logging.info(
            f"Domain tokens are {coherence_analysis['domain_specialization_score']:.2f}x more routing-relevant"
        )
        return coherence_analysis

    def analyze_token_distribution(
        self, human_results: Dict[str, Any], domain_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze token frequency distributions for insights

        Args:
            human_results: Human tokenizer results
            domain_results: Domain tokenizer results

        Returns:
            Token distribution analysis
        """
        logging.info("Analyzing token distributions...")

        human_counts = human_results["token_counts"]
        domain_counts = domain_results["token_counts"]

        # Calculate distribution metrics
        def calculate_distribution_metrics(
            token_counts: Dict[str, int],
        ) -> Dict[str, float]:
            counts = list(token_counts.values())
            if not counts:
                return {}

            return {
                "mean_frequency": np.mean(counts),
                "std_frequency": np.std(counts),
                "entropy": -sum(
                    p * math.log2(p) for p in [c / sum(counts) for c in counts] if p > 0
                ),
                "gini_coefficient": self._calculate_gini(counts),
                "max_frequency": max(counts),
                "min_frequency": min(counts),
                "median_frequency": np.median(counts),
            }

        human_dist = calculate_distribution_metrics(human_counts)
        domain_dist = calculate_distribution_metrics(domain_counts)

        distribution_analysis = {
            "human_distribution": human_dist,
            "domain_distribution": domain_dist,
            "entropy_ratio": domain_dist.get("entropy", 0)
            / human_dist.get("entropy", 1)
            if human_dist.get("entropy", 0) > 0
            else 0,
            "efficiency_comparison": {
                "human_gini": human_dist.get("gini_coefficient", 0),
                "domain_gini": domain_dist.get("gini_coefficient", 0),
                "gini_improvement": human_dist.get("gini_coefficient", 0)
                - domain_dist.get("gini_coefficient", 0),
            },
        }

        return distribution_analysis

    def _calculate_gini(self, values: List[float]) -> float:
        """Calculate Gini coefficient for inequality measurement"""
        if not values:
            return 0.0

        values = sorted(values)
        n = len(values)
        cumsum = np.cumsum(values)

        return (
            n
            + 1
            - 2 * sum((n + 1 - i) * v for i, v in enumerate(values, 1)) / cumsum[-1]
        ) / n

    def generate_research_insights(
        self, all_analyses: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate comprehensive research insights supporting domain-specific tokenization

        Args:
            all_analyses: Combined analysis results

        Returns:
            Research insights and conclusions
        """
        logging.info("Generating research insights...")

        insights = {
            "key_findings": [],
            "quantitative_evidence": {},
        }

        # Extract key quantitative evidence
        if "compression_analysis" in all_analyses:
            comp = all_analyses["compression_analysis"]
            insights["quantitative_evidence"]["compression_improvement"] = comp.get(
                "compression_improvement", 0
            )
            insights["quantitative_evidence"]["vocab_efficiency_ratio"] = comp.get(
                "efficiency_ratio", 1
            )

            if comp.get("compression_improvement", 0) > 0:
                insights["key_findings"].append(
                    f"Domain-specific tokenization achieves {comp['compression_improvement']:.1f}% better compression efficiency"
                )

            if comp.get("efficiency_ratio", 1) > 1:
                insights["key_findings"].append(
                    f"Domain tokenizer is {comp['efficiency_ratio']:.1f}x more vocabulary-efficient"
                )

        # Semantic coherence insights
        if "semantic_analysis" in all_analyses:
            sem = all_analyses["semantic_analysis"]
            if sem.get("domain_specialization_score", 1) > 1:
                insights["key_findings"].append(
                    f"Domain tokens are {sem['domain_specialization_score']:.1f}x more routing-relevant than human language tokens"
                )

        # Pattern efficiency insights
        if "pattern_analysis" in all_analyses:
            pat = all_analyses["pattern_analysis"]
            coverage = pat.get("pattern_coverage_ratio", 0)
            insights["quantitative_evidence"]["pattern_coverage"] = coverage

            if coverage > 0.8:
                insights["key_findings"].append(
                    f"Domain tokenization achieves {coverage * 100:.1f}% pattern coverage for routing-specific tokens"
                )

        return insights

    def save_research_analysis(self, analyses: Dict[str, Any]) -> None:
        """
        Save comprehensive research analysis results

        Args:
            analyses: All analysis results to save
        """
        output_file = (
            self.output_dir / "research_analysis" / "comprehensive_analysis.json"
        )

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(analyses, f, indent=2, ensure_ascii=False)

        # Save summary insights
        if "research_insights" in analyses:
            insights_file = (
                self.output_dir / "research_analysis" / "research_insights.json"
            )
            with open(insights_file, "w", encoding="utf-8") as f:
                json.dump(
                    analyses["research_insights"], f, indent=2, ensure_ascii=False
                )

        logging.info(f"Research analysis saved to {output_file}")

    def run_comprehensive_analysis(
        self,
        human_results: Dict[str, Any],
        domain_results: Dict[str, Any],
        dataset: Dataset,
    ) -> Dict[str, Any]:
        """
        Run all analysis modules for comprehensive research insights

        Args:
            human_results: Human tokenizer results
            domain_results: Domain tokenizer results
            dataset: Dataset containing original tree sequences and target tokens

        Returns:
            Comprehensive analysis results
        """
        logging.info("Running comprehensive research analysis...")

        all_analyses = {}

        # Run individual analyses
        all_analyses["compression_analysis"] = self.analyze_compression_efficiency(
            human_results, domain_results, dataset
        )

        all_analyses["semantic_analysis"] = self.analyze_semantic_coherence(
            human_results, domain_results
        )

        all_analyses["distribution_analysis"] = self.analyze_token_distribution(
            human_results, domain_results
        )

        # Generate research insights
        all_analyses["research_insights"] = self.generate_research_insights(
            all_analyses
        )

        # Save results
        self.save_research_analysis(all_analyses)

        logging.info("Comprehensive research analysis completed")
        return all_analyses
