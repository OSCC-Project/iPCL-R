#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   embedding_visualizer.py
@Time    :   2025/09/11
@Author  :   Dawn Li
@Desc    :   Visualize token embedding distributions for two tokenizers/models.
              Given two tokenizers and their corresponding models, extract the
              embedding vectors for the actually used tokens on the dataset and
              render a 2D scatter plot (PCA by default) for comparison.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots
import umap
from adjustText import adjust_text
from matplotlib.patches import Rectangle
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from transformers import (
    AutoModel,
    AutoModelForSeq2SeqLM,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from flow.utils.plot_utils import palette_slice

if scienceplots:
    plt.style.use(["science"])


@dataclass
class EmbeddingPlotSpec:
    """Plot configuration for embedding visualization."""

    figsize: Tuple[int, int] = (10, 8)
    alpha: float = 0.7
    annotate_top_n: int = 0  # annotate top-n tokens from each set (0 to disable)
    random_state: int = 42
    enable_3d: bool = True  # Enable 3D visualization alongside 2D


class EmbeddingVisualizer:
    """Extracts embeddings and creates 2D/3D scatter plots comparing two tokenizers."""

    def __init__(
        self,
        output_dir: Union[str, Path],
        plot_spec: Optional[EmbeddingPlotSpec] = None,
    ):
        self.output_dir = Path(output_dir)
        self.plot_dir = self.output_dir / "embedding_plots"
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self.plot_spec = plot_spec or EmbeddingPlotSpec()

    # ------------------------------
    # Loading helpers
    # ------------------------------
    def _is_local_model_dir(self, path_or_id: Union[str, Path]) -> bool:
        p = Path(str(path_or_id))
        return p.exists() and p.is_dir()

    def _load_model(
        self, model_path_or_id: Union[str, Path]
    ) -> Optional[PreTrainedModel]:
        """Load a model from local directory. If not local, return None.

        We avoid remote downloads due to restricted network. If the given
        path is not a directory, we skip loading and log a warning.
        """

        model: Optional[PreTrainedModel] = None
        try:
            # Prefer seq2seq auto class for T5-like models, fallback to base AutoModel
            model = AutoModelForSeq2SeqLM.from_pretrained(str(model_path_or_id))
        except Exception as e1:
            logging.debug(f"AutoModelForSeq2SeqLM failed: {e1}")
            try:
                model = AutoModel.from_pretrained(str(model_path_or_id))
            except Exception as e2:
                logging.error(
                    f"Failed to load model from {model_path_or_id}: {e1} | {e2}"
                )
                return None

        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        return model

    # ------------------------------
    # Embedding and token utilities
    # ------------------------------
    def _tokens_to_ids(
        self, tokenizer: PreTrainedTokenizerBase, tokens: Sequence[str]
    ) -> List[int]:
        ids: List[int] = []
        for t in tokens:
            try:
                tid = tokenizer.convert_tokens_to_ids(t)
            except Exception:
                tid = None
            if tid is None or (
                hasattr(tokenizer, "unk_token_id") and tid == tokenizer.unk_token_id
            ):
                continue
            ids.append(int(tid))
        # de-duplicate while preserving order
        seen = set()
        unique_ids: List[int] = []
        for tid in ids:
            if tid not in seen:
                seen.add(tid)
                unique_ids.append(tid)
        return unique_ids

    def _select_embeddings(
        self, model: PreTrainedModel, token_ids: Sequence[int]
    ) -> Optional[np.ndarray]:
        try:
            emb = model.get_input_embeddings()
        except Exception as e:
            logging.error(f"Model does not expose input embeddings: {e}")
            return None

        weight = emb.weight.detach().cpu()
        max_id = weight.shape[0] - 1
        # Filter ids in range
        valid_ids = [i for i in token_ids if 0 <= i <= max_id]
        if not valid_ids:
            return None
        selected = weight[valid_ids].numpy()
        return selected

    # ------------------------------
    # Dimensionality Reduction
    # ------------------------------
    def _reduce_embeddings(
        self, embeddings: np.ndarray, method: str = "pca", n_components: int = 2
    ) -> np.ndarray:
        """Reduce embeddings using specified method.

        Args:
            embeddings: High-dimensional embeddings to reduce
            method: Dimensionality reduction method ("pca", "tsne", "umap")
            n_components: Number of components (2 for 2D, 3 for 3D)

        Returns:
            Reduced coordinates array
        """
        method = method.lower()

        if method == "pca":
            reducer = PCA(
                n_components=n_components, random_state=self.plot_spec.random_state
            )
        elif method == "tsne":
            reducer = TSNE(
                n_components=n_components,
                random_state=self.plot_spec.random_state,
                perplexity=min(30, len(embeddings) - 1),  # Ensure perplexity is valid
            )
        elif method == "umap":
            reducer = umap.UMAP(
                n_components=n_components,
                random_state=self.plot_spec.random_state,
                metric="cosine",
                init="pca",
                n_neighbors=30,
            )
        else:
            raise ValueError(f"Unsupported reduction method: {method}")

        return reducer.fit_transform(embeddings)

    def _generate_domain_labels(
        self, tokens: List[str], grouping_type: str
    ) -> List[str]:
        """Generate proper domain token labels based on grouping type"""
        labels = []

        if grouping_type == "6group":
            # Individual direction grouping: U*, D*, L*, R*, T*, B*, SPECIAL
            for token in tokens:
                first_char = token[0] if token else "SPECIAL"
                if first_char in {"U", "D", "L", "R", "T", "B"}:
                    labels.append(f"{first_char}*")
                else:
                    labels.append("SPECIAL")
        else:
            # Combined direction grouping: U-D, L-R, T-B, SPECIAL
            for token in tokens:
                first_char = token[0] if token else "SPECIAL"
                if first_char in {"U", "D"}:
                    labels.append("U-D")
                elif first_char in {"L", "R"}:
                    labels.append("L-R")
                elif first_char in {"T", "B"}:
                    labels.append("T-B")
                else:
                    labels.append("SPECIAL")

        return labels

    # ------------------------------
    # CSV Saving
    # ------------------------------
    def _save_coordinates_csv(
        self,
        filename: Union[str, Path],
        tokens: Sequence[str],
        coords: np.ndarray,
        label: str,
        is_3d: bool = False,
    ) -> None:
        """Save coordinates to CSV file (2D or 3D)."""
        out_path = self.plot_dir / str(filename)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "label": [label] * len(tokens),
            "token": tokens,
            "x": coords[:, 0],
            "y": coords[:, 1],
        }

        if is_3d and coords.shape[1] >= 3:
            data["z"] = coords[:, 2]

        df = pd.DataFrame(data)
        df.to_csv(out_path, index=False, encoding="utf-8", quoting=1)  # QUOTE_ALL

    def _save_single_coordinates_csv(
        self,
        filename: Union[str, Path],
        tokens: Sequence[str],
        coords: np.ndarray,
        label: str,
        is_3d: bool = False,
    ) -> None:
        """Save coordinates to CSV file for single tokenizer plots."""
        out_path = self.plot_dir / str(filename)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "label": [label] * len(tokens),
            "token": tokens,
            "x": coords[:, 0],
            "y": coords[:, 1],
        }

        if is_3d and coords.shape[1] >= 3:
            data["z"] = coords[:, 2]

        df = pd.DataFrame(data)
        df.to_csv(out_path, index=False, encoding="utf-8", quoting=1)  # QUOTE_ALL

    def _save_combined_coordinates_csv(
        self,
        filename: Union[str, Path],
        tokens: Sequence[str],
        coords: np.ndarray,
        labels: Sequence[str],
        is_3d: bool = False,
    ) -> None:
        """Save coordinates to CSV file with individual labels for each token."""
        out_path = self.plot_dir / str(filename)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "label": labels,
            "token": tokens,
            "x": coords[:, 0],
            "y": coords[:, 1],
        }

        if is_3d and coords.shape[1] >= 3:
            data["z"] = coords[:, 2]

        df = pd.DataFrame(data)
        df.to_csv(out_path, index=False, encoding="utf-8", quoting=1)  # QUOTE_ALL

    # ------------------------------
    # Plotting
    # ------------------------------
    def create_plot(
        self,
        coords: np.ndarray,
        tokens: List[str],
        labels: List[str],
        label_type: str,
        is_3d: bool = False,
        grouping_type: str = "4group",
        font_size: int = 18,
        legend_size: int = 22,
        marker_size: int = 48,
        show_box: bool = True,
        figsize: Optional[Tuple[int, int]] = None,
    ) -> plt.Figure:
        """Create a 2D or 3D plot with appropriate styling.

        Args:
            coords: Coordinate array (N, 2) for 2D or (N, 3) for 3D
            tokens: List of token strings
            labels: List of label strings for each token
            label_type: Type of tokenizer ("Human" or "Domain")
            is_3d: Whether to create 3D plot
            grouping_type: Type of grouping for domain plots ("4group" or "6group")
            font_size: Font size for axis labels and ticks
            legend_size: Font size for legend
            marker_size: Size of scatter plot markers
            show_box: Whether to show axis box and ticks
            figsize: Figure size tuple, uses default if None
        """
        # Use provided figsize or default
        figure_size = figsize if figsize is not None else self.plot_spec.figsize

        if is_3d:
            fig = plt.figure(figsize=figure_size)
            ax = fig.add_subplot(111, projection="3d")
        else:
            fig, ax = plt.subplots(figsize=figure_size)

        if label_type == "Domain":
            self._plot_domain_groups(
                ax, coords, tokens, labels, is_3d, grouping_type, marker_size
            )
        else:
            self._plot_standard_groups(ax, coords, tokens, labels, is_3d, marker_size)

        self._set_labels_and_styling(ax, is_3d, font_size)

        # Control axis box visibility
        if not show_box:
            ax.set_xticks([])
            ax.set_yticks([])
            if not is_3d:
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["bottom"].set_visible(False)
                ax.spines["left"].set_visible(False)

        if (
            self.plot_spec.annotate_top_n > 0 and not is_3d and label_type == "Domain"
        ):  # Annotations only for 2D
            self._add_annotations(ax, coords, tokens, labels, font_size)

        ax.legend(loc="best", fontsize=legend_size, frameon=False)
        fig.tight_layout()
        return fig

    def _plot_domain_groups(
        self,
        ax,
        coords: np.ndarray,
        tokens: List[str],
        _labels: List[str],
        is_3d: bool,
        grouping_type: str = "4group",
        marker_size: int = 48,
    ):
        """Plot domain tokens with direction grouping."""
        if grouping_type == "6group":
            groups_order = ["U*", "D*", "L*", "R*", "T*", "B*", "SPECIAL"]
            logic = "individual"
        else:
            groups_order = ["U-D", "L-R", "T-B", "SPECIAL"]
            logic = "combined"

        colors = palette_slice(len(groups_order))
        color_mapping = {group: colors[idx] for idx, group in enumerate(groups_order)}
        direction_coords = {k: [] for k in groups_order}

        for i, token in enumerate(tokens):
            first_char = token[0] if token else "SPECIAL"

            if logic == "individual":
                if first_char in {"U", "D", "L", "R", "T", "B"}:
                    group_key = f"{first_char}*"
                    direction_coords[group_key].append(coords[i])
                else:
                    direction_coords["SPECIAL"].append(coords[i])
            else:
                if first_char in {"U", "D"}:
                    direction_coords["U-D"].append(coords[i])
                elif first_char in {"L", "R"}:
                    direction_coords["L-R"].append(coords[i])
                elif first_char in {"T", "B"}:
                    direction_coords["T-B"].append(coords[i])
                else:
                    direction_coords["SPECIAL"].append(coords[i])

        for direction in groups_order:
            if direction_coords[direction]:
                coords_array = np.array(direction_coords[direction])
                color = color_mapping[direction]
                if is_3d:
                    ax.scatter(
                        coords_array[:, 0],
                        coords_array[:, 1],
                        coords_array[:, 2],
                        alpha=self.plot_spec.alpha,
                        label=direction,
                        color=color,
                        s=marker_size,
                    )
                else:
                    ax.scatter(
                        coords_array[:, 0],
                        coords_array[:, 1],
                        alpha=self.plot_spec.alpha,
                        label=direction,
                        color=color,
                        s=marker_size,
                    )

    def _plot_standard_groups(
        self,
        ax,
        coords: np.ndarray,
        _tokens: List[str],
        labels: List[str],
        is_3d: bool,
        marker_size: int = 48,
    ):
        """Plot standard groups (Human or other)."""
        seen = []
        for label in labels:
            if label not in seen:
                seen.append(label)
        unique_labels = seen
        colors = palette_slice(len(unique_labels))
        color_mapping = {label: colors[idx] for idx, label in enumerate(unique_labels)}

        for label in unique_labels:
            label_indices = [i for i, lab in enumerate(labels) if lab == label]
            label_coords = coords[label_indices]

            color = color_mapping[label]
            if is_3d:
                ax.scatter(
                    label_coords[:, 0],
                    label_coords[:, 1],
                    label_coords[:, 2],
                    alpha=self.plot_spec.alpha,
                    label=str(label),
                    color=color,
                    s=marker_size,
                )
            else:
                ax.scatter(
                    label_coords[:, 0],
                    label_coords[:, 1],
                    alpha=self.plot_spec.alpha,
                    label=str(label),
                    color=color,
                    s=marker_size,
                )

    def _set_labels_and_styling(self, ax, is_3d: bool, font_size: int = 18):
        """Set appropriate axis labels and styling."""
        ax.set_xlabel("X", fontsize=font_size)
        ax.set_ylabel("Y", fontsize=font_size)
        ax.tick_params(labelsize=font_size)
        if is_3d:
            ax.set_zlabel("Z", fontsize=font_size)

    def _add_annotations(
        self,
        ax,
        coords: np.ndarray,
        tokens: List[str],
        _labels: List[str],
        font_size: int = 5,
    ):
        """Add token annotations to 2D plots."""
        for i in range(min(self.plot_spec.annotate_top_n, len(tokens))):
            ax.annotate(
                tokens[i], (coords[i, 0], coords[i, 1]), fontsize=font_size, alpha=0.8
            )

    def plot_single_tokenizer(
        self,
        model_path: Union[str, Path],
        tokenizer: PreTrainedTokenizerBase,
        used_tokens: Sequence[str],
        label: str,
        filename_prefix: str,
        limit: Optional[int] = 1000,
        unused_tokens: Optional[Sequence[str]] = None,
        reduction_methods: Optional[List[str]] = None,
        font_size: int = 18,
        legend_size: int = 22,
        marker_size: int = 48,
        show_box: bool = True,
        figsize: Optional[Tuple[int, int]] = None,
    ) -> Optional[List[Path]]:
        """Plot a single model/tokenizer embedding distribution using multiple reduction methods in 2D and 3D.

        Args:
            reduction_methods: List of reduction methods to use ["pca", "tsne", "umap"]
            unused_tokens: Optional unused tokens to sample and plot for comparison.
                          For human tokenizer, unused tokens will be sampled (up to limit).

        Returns:
            List of paths to saved plots, one for each reduction method and dimension.
        """
        import random

        if reduction_methods is None:
            reduction_methods = ["pca", "tsne", "umap"]

        model = self._load_model(model_path)
        if model is None:
            logging.warning(f"Skipping single plot; model not loaded: {model_path}")
            return None

        # Process used tokens
        token_ids = self._tokens_to_ids(tokenizer, used_tokens)
        if limit is not None and limit > 0:
            token_ids = token_ids[:limit]
        emb = self._select_embeddings(model, token_ids)
        if emb is None or emb.size == 0:
            logging.warning("No valid embeddings for single plot; skip.")
            return None

        tokens_filtered = [tokenizer.convert_ids_to_tokens(i) for i in token_ids]

        # Process unused tokens if provided and this is human tokenizer
        unused_emb = None
        unused_tokens_filtered = []
        if unused_tokens and label == "Human":
            # Sample unused tokens (max limit)
            unused_sample = list(unused_tokens)
            if len(unused_sample) > limit:
                random.seed(self.plot_spec.random_state)
                unused_sample = random.sample(unused_sample, limit)

            unused_token_ids = self._tokens_to_ids(tokenizer, unused_sample)
            unused_emb = self._select_embeddings(model, unused_token_ids)

            if unused_emb is not None and unused_emb.size > 0:
                unused_tokens_filtered = [
                    tokenizer.convert_ids_to_tokens(i) for i in unused_token_ids
                ]

        # Combine embeddings for consistent dimensionality reduction
        all_embeddings = [emb]
        if unused_emb is not None:
            all_embeddings.append(unused_emb)
        combined_emb = np.concatenate(all_embeddings, axis=0)

        # Save original high-dimensional embeddings
        self._save_high_dimensional_embeddings(
            combined_emb,
            tokens_filtered,
            unused_tokens_filtered,
            filename_prefix,
            label,
        )

        output_paths = []

        # Create plots for each reduction method and dimension combination
        for method in reduction_methods:
            try:
                # Generate plots for both 2D and 3D if enabled
                dimensions = [2]
                if self.plot_spec.enable_3d:
                    dimensions.append(3)

                for n_components in dimensions:
                    is_3d = n_components == 3
                    dim_suffix = "3d" if is_3d else "2d"

                    # Apply dimensionality reduction
                    combined_coords = self._reduce_embeddings(
                        combined_emb, method, n_components
                    )

                    # Split back the coordinates
                    split_idx = len(tokens_filtered)
                    coords = combined_coords[:split_idx]
                    unused_coords = (
                        combined_coords[split_idx:] if unused_emb is not None else None
                    )

                    # For domain tokenizer, create plots with different groupings
                    if label == "Domain":
                        grouping_configs = [
                            {"suffix": "6group", "grouping_type": "6group"},
                            {"suffix": "4group", "grouping_type": "4group"},
                        ]

                        for config in grouping_configs:
                            # Generate filename with dimension suffix
                            method_filename_prefix = f"{filename_prefix}_{method}_{config['suffix']}_{dim_suffix}"

                            # Save CSV with proper domain labels
                            domain_labels = self._generate_domain_labels(
                                tokens_filtered, config["grouping_type"]
                            )
                            self._save_combined_coordinates_csv(
                                f"{method_filename_prefix}.csv",
                                tokens_filtered,
                                coords,
                                domain_labels,
                                is_3d,
                            )

                            # Create plot using EmbeddingPlotter
                            labels = [label] * len(tokens_filtered)  # Domain labels
                            fig = self.create_plot(
                                coords,
                                tokens_filtered,
                                labels,
                                label,
                                is_3d,
                                config["grouping_type"],
                                font_size,
                                legend_size,
                                marker_size,
                                show_box,
                                figsize,
                            )

                            # Save plot
                            out_path = self.plot_dir / f"{method_filename_prefix}.pdf"
                            fig.savefig(out_path, dpi=300, bbox_inches="tight")
                            plt.close(fig)
                            output_paths.append(out_path)
                            logging.info(
                                f"Saved {method.upper()} {config['suffix']} {dim_suffix.upper()} embedding plot to: {out_path}"
                            )

                            # Create and save detail plot for 2D domain plots only (6group only)
                            if not is_3d and config["grouping_type"] == "6group":
                                detail_fig = self._create_domain_detail_plot(
                                    coords,
                                    tokens_filtered,
                                    config["grouping_type"],
                                    cluster_index=0,
                                    main_w_n=16,
                                    main_h_n=12,
                                    detail_n=8,
                                    font_size=36,
                                    legend_size=48,
                                    main_marker_size=500,
                                    detail_marker_size=800,
                                    main_show_box=False,
                                )
                                detail_out_path = (
                                    self.plot_dir
                                    / f"{method_filename_prefix}_detail.pdf"
                                )
                                detail_fig.savefig(
                                    detail_out_path, dpi=300, bbox_inches="tight"
                                )
                                plt.close(detail_fig)
                                output_paths.append(detail_out_path)
                                logging.info(
                                    f"Saved {method.upper()} {config['suffix']} {dim_suffix.upper()} detail plot to: {detail_out_path}"
                                )

                    else:
                        # For human or other tokenizers
                        method_filename_prefix = (
                            f"{filename_prefix}_{method}_{dim_suffix}"
                        )

                        # Prepare all tokens and labels
                        all_tokens = tokens_filtered.copy()
                        if label == "Human":
                            all_labels = ["Used"] * len(tokens_filtered)
                        else:
                            all_labels = [label] * len(tokens_filtered)
                        all_coords = coords.copy()

                        if unused_coords is not None and label == "Human":
                            unused_count = len(unused_tokens_filtered)
                            all_tokens.extend(unused_tokens_filtered)
                            unsed_count_str = (
                                str(unused_count)
                                if unused_count < 1000
                                else str(unused_count // 1000) + "K"
                            )
                            all_labels.extend(
                                [f"Unused ({unsed_count_str})"] * unused_count
                            )
                            all_coords = np.vstack([all_coords, unused_coords])

                            # Save combined CSV with dimension data
                            self._save_combined_coordinates_csv(
                                f"{method_filename_prefix}.csv",
                                all_tokens,
                                all_coords,
                                all_labels,
                                is_3d,
                            )
                        else:
                            # Save only used tokens
                            self._save_single_coordinates_csv(
                                f"{method_filename_prefix}.csv",
                                tokens_filtered,
                                coords,
                                label,
                                is_3d,
                            )

                        # Create plot using EmbeddingPlotter
                        fig = self.create_plot(
                            all_coords,
                            all_tokens,
                            all_labels,
                            label,
                            is_3d,
                            "4group",  # default grouping for non-domain
                            font_size,
                            legend_size,
                            marker_size,
                            show_box,
                            figsize,
                        )

                        # Save plot
                        out_path = self.plot_dir / f"{method_filename_prefix}.pdf"
                        fig.savefig(out_path, dpi=300, bbox_inches="tight")
                        plt.close(fig)
                        output_paths.append(out_path)
                        logging.info(
                            f"Saved {method.upper()} {dim_suffix.upper()} embedding plot to: {out_path}"
                        )

            except Exception as e:
                logging.warning(f"Failed to create {method.upper()} plot: {e}")
                continue

        return output_paths if output_paths else None

    def plot_both_separately(
        self,
        human_model_path: Union[str, Path],
        human_tokenizer: PreTrainedTokenizerBase,
        human_used_tokens: Sequence[str],
        domain_model_path: Union[str, Path],
        domain_tokenizer: PreTrainedTokenizerBase,
        domain_used_tokens: Sequence[str],
        human_prefix: str = "human_embeddings",
        domain_prefix: str = "domain_embeddings",
        limit_per_set: Optional[int] = 1000,
        human_unused_tokens: Optional[Sequence[str]] = None,
        reduction_methods: Optional[List[str]] = None,
        font_size: int = 18,
        legend_size: int = 22,
        marker_size: int = 48,
        show_box: bool = True,
        figsize: Optional[Tuple[int, int]] = None,
    ) -> Tuple[Optional[List[Path]], Optional[List[Path]]]:
        """Plot two models separately using multiple reduction methods in 2D and 3D and return their output paths."""
        h_paths = self.plot_single_tokenizer(
            model_path=human_model_path,
            tokenizer=human_tokenizer,
            used_tokens=human_used_tokens,
            label="Human",
            filename_prefix=human_prefix,
            limit=limit_per_set,
            unused_tokens=human_unused_tokens,
            reduction_methods=reduction_methods,
            font_size=font_size,
            legend_size=legend_size,
            marker_size=marker_size,
            show_box=show_box,
            figsize=figsize,
        )
        d_paths = self.plot_single_tokenizer(
            model_path=domain_model_path,
            tokenizer=domain_tokenizer,
            used_tokens=domain_used_tokens,
            label="Domain",
            filename_prefix=domain_prefix,
            limit=limit_per_set,
            reduction_methods=reduction_methods,
            font_size=font_size,
            legend_size=legend_size,
            marker_size=marker_size,
            show_box=show_box,
            figsize=figsize,
        )
        return h_paths, d_paths

    def plot_from_csv_files(
        self, csv_directory: Optional[Union[str, Path]] = None
    ) -> Optional[List[Path]]:
        """
        Create plots directly from existing CSV files.

        Args:
            csv_directory: Directory containing CSV files. If None, uses self.plot_dir

        Returns:
            List of paths to created plot files
        """
        if csv_directory is None:
            csv_directory = self.plot_dir
        else:
            csv_directory = Path(csv_directory)

        if not csv_directory.exists():
            logging.error(f"CSV directory not found: {csv_directory}")
            return None

        # Find CSV files
        csv_files = list(csv_directory.glob("*.csv"))
        if not csv_files:
            logging.error(f"No CSV files found in {csv_directory}")
            return None

        logging.info(f"Found {len(csv_files)} CSV files to plot")

        # Group files by tokenizer type
        human_files = []
        domain_files = []

        for csv_file in csv_files:
            filename = csv_file.name
            if filename.startswith("human_"):
                human_files.append(csv_file)
            elif filename.startswith("domain_"):
                domain_files.append(csv_file)
            else:
                logging.warning(f"Unrecognized CSV file pattern: {filename}")

        logging.info(f"Found {len(human_files)} human tokenizer CSV files")
        logging.info(f"Found {len(domain_files)} domain tokenizer CSV files")

        # Create plots
        output_paths = []
        try:
            if human_files:
                human_paths = self._create_plots_from_csv_group(human_files, "Human")
                if human_paths:
                    output_paths.extend(human_paths)

            if domain_files:
                domain_paths = self._create_plots_from_csv_group(domain_files, "Domain")
                if domain_paths:
                    output_paths.extend(domain_paths)

            logging.info(
                f"Successfully created {len(output_paths)} plots from CSV files"
            )
            return output_paths

        except Exception as e:
            logging.error(f"Failed to create plots from CSV files: {e}")
            raise

    def _create_plots_from_csv_group(
        self, csv_files: List[Path], tokenizer_type: str
    ) -> Optional[List[Path]]:
        """Create plots from a group of CSV files for a specific tokenizer type"""
        import matplotlib.pyplot as plt
        import pandas as pd

        if not csv_files:
            return None

        output_paths = []

        for csv_file in csv_files:
            try:
                logging.info(f"Processing {csv_file.name}...")

                # Read CSV file
                df = pd.read_csv(csv_file)
                if df.empty:
                    logging.warning(f"Empty CSV file: {csv_file.name}")
                    continue

                # Extract metadata from filename
                filename = csv_file.stem
                parts = filename.split("_")

                # Determine dimensions and method
                is_3d = "z" in df.columns
                method = self._extract_method_from_filename(parts)
                grouping_type = self._extract_grouping_from_filename(
                    filename, tokenizer_type
                )

                # Extract data
                coords = self._extract_coordinates_from_df(df, is_3d)
                tokens = df["token"].tolist()
                labels = df["label"].tolist()

                # Create plot using existing plotter infrastructure
                fig = self._create_plot_from_data(
                    coords,
                    tokens,
                    labels,
                    method,
                    tokenizer_type,
                    is_3d,
                    grouping_type,
                    font_size=36,
                    legend_size=48,
                    marker_size=300,
                    figsize=(16, 12),
                    show_box=False,
                )

                # Title removed as requested

                # Save plot
                plot_filename = filename + ".pdf"
                plot_path = self.plot_dir / plot_filename
                fig.savefig(plot_path, dpi=300, bbox_inches="tight")
                plt.close(fig)

                output_paths.append(plot_path)
                logging.info(f"Saved plot: {plot_path}")

                # Create and save detail plot for 2D domain plots only (6group only)
                if (
                    tokenizer_type == "Domain"
                    and not is_3d
                    and grouping_type == "6group"
                ):
                    detail_fig = self._create_domain_detail_plot(
                        coords,
                        tokens,
                        grouping_type,
                        cluster_index=0,
                        main_w_n=16,
                        main_h_n=12,
                        detail_n=8,
                        font_size=36,
                        legend_size=48,
                        main_marker_size=500,
                        detail_marker_size=800,
                        main_show_box=False,
                    )
                    detail_plot_filename = filename + "_detail.pdf"
                    detail_plot_path = self.plot_dir / detail_plot_filename
                    detail_fig.savefig(detail_plot_path, dpi=300, bbox_inches="tight")
                    plt.close(detail_fig)

                    output_paths.append(detail_plot_path)
                    logging.info(f"Saved detail plot: {detail_plot_path}")

            except Exception as e:
                logging.error(f"Failed to process {csv_file.name}: {e}")
                continue

        return output_paths if output_paths else None

    def plot_from_high_dim_files(
        self, high_dim_directory: Optional[Union[str, Path]] = None
    ) -> Optional[List[Path]]:
        """Plot from high-dimensional embedding parquet files by re-applying dimensionality reduction

        Args:
            high_dim_directory: Directory containing high-dimensional embedding files.
                               Uses plot_dir if None.

        Returns:
            List of output plot paths if successful, None otherwise
        """
        if high_dim_directory is None:
            search_dir = self.plot_dir
        else:
            search_dir = Path(high_dim_directory)

        if not search_dir.exists():
            logging.error(
                f"High-dimensional embedding directory not found: {search_dir}"
            )
            return None

        # Find all high-dimensional embedding files
        high_dim_files = list(search_dir.glob("*_high_dim_embeddings.parquet"))

        if not high_dim_files:
            logging.error(f"No high-dimensional embedding files found in {search_dir}")
            return None

        logging.info(f"Found {len(high_dim_files)} high-dimensional embedding files")

        output_paths = []

        # Process each high-dimensional embedding file
        for high_dim_file in high_dim_files:
            try:
                output_paths_for_file = self._plot_from_single_high_dim_file(
                    high_dim_file
                )
                if output_paths_for_file:
                    output_paths.extend(output_paths_for_file)
            except Exception as e:
                logging.error(f"Failed to process high-dim file {high_dim_file}: {e}")
                continue

        if output_paths:
            logging.info(
                f"Generated {len(output_paths)} plots from high-dimensional embeddings"
            )
        else:
            logging.warning("No plots were generated from high-dimensional embeddings")

        return output_paths if output_paths else None

    def _plot_from_single_high_dim_file(
        self, high_dim_file: Path
    ) -> Optional[List[Path]]:
        """Process a single high-dimensional embedding file and create plots"""
        import pandas as pd

        logging.info(f"Processing high-dimensional embedding file: {high_dim_file}")

        # Parse filename to extract metadata
        filename_stem = high_dim_file.stem.replace("_high_dim_embeddings", "")

        # Load high-dimensional embeddings
        try:
            df = pd.read_parquet(high_dim_file)
        except Exception as e:
            logging.error(
                f"Failed to load high-dimensional embeddings from {high_dim_file}: {e}"
            )
            return None

        # Extract embeddings and metadata
        embedding_cols = [col for col in df.columns if col.startswith("emb_")]
        if not embedding_cols:
            logging.error(f"No embedding columns found in {high_dim_file}")
            return None

        # Get high-dimensional embeddings
        high_dim_embeddings = df[embedding_cols].values
        tokens = df["token"].tolist()
        labels = df["label"].tolist()
        tokenizer_type = df["tokenizer_type"].iloc[0]  # Should be consistent

        logging.info(
            f"Loaded {high_dim_embeddings.shape[0]} embeddings of dimension {high_dim_embeddings.shape[1]} for {tokenizer_type} tokenizer"
        )

        # Apply dimensionality reduction methods and save CSV files
        reduction_methods = ["pca", "tsne", "umap"]

        for method in reduction_methods:
            try:
                # Generate for both 2D and 3D if enabled
                dimensions = [2]
                if self.plot_spec.enable_3d:
                    dimensions.append(3)

                for n_components in dimensions:
                    is_3d = n_components == 3
                    dim_suffix = "3d" if is_3d else "2d"

                    # Apply dimensionality reduction
                    coords = self._reduce_embeddings(
                        high_dim_embeddings, method, n_components
                    )

                    # Create CSV filename
                    method_filename_prefix = f"{filename_stem}_{method}_{dim_suffix}"

                    if tokenizer_type == "Domain":
                        # For domain tokenizer, create CSV for both 6group and 4group
                        grouping_configs = [
                            {"suffix": "6group", "grouping_type": "6group"},
                            {"suffix": "4group", "grouping_type": "4group"},
                        ]

                        for config in grouping_configs:
                            # Use same naming format as original code: {prefix}_{method}_{grouping}_{dimension}
                            config_method_filename_prefix = f"{filename_stem}_{method}_{config['suffix']}_{dim_suffix}"

                            # Generate domain labels
                            domain_labels = self._generate_domain_labels(
                                tokens, config["grouping_type"]
                            )

                            # Create CSV data
                            csv_data = {
                                "x": coords[:, 0],
                                "y": coords[:, 1],
                                "token": tokens,
                                "label": domain_labels,
                                "method": [method] * len(tokens),
                                "tokenizer_type": [tokenizer_type] * len(tokens),
                            }

                            if is_3d:
                                csv_data["z"] = coords[:, 2]

                            csv_df = pd.DataFrame(csv_data)
                            csv_out_path = (
                                self.plot_dir / f"{config_method_filename_prefix}.csv"
                            )
                            csv_df.to_csv(csv_out_path, index=False)

                            logging.info(f"Saved CSV file: {csv_out_path}")

                    else:
                        # For human tokenizer, create CSV with appropriate labels
                        all_labels = []
                        for label in labels:
                            if label == "Used":
                                all_labels.append(label)
                            elif label == "Unused":
                                # Convert to format matching original interface
                                unused_count = labels.count("Unused")
                                unused_count_str = (
                                    str(unused_count)
                                    if unused_count < 1000
                                    else str(unused_count // 1000) + "K"
                                )
                                all_labels.append(f"Unused ({unused_count_str})")
                            else:
                                all_labels.append(label)

                        # Create and save CSV data
                        csv_data = {
                            "x": coords[:, 0],
                            "y": coords[:, 1],
                            "token": tokens,
                            "label": all_labels,
                            "method": [method] * len(tokens),
                            "tokenizer_type": [tokenizer_type] * len(tokens),
                        }

                        if is_3d:
                            csv_data["z"] = coords[:, 2]

                        csv_df = pd.DataFrame(csv_data)
                        csv_out_path = self.plot_dir / f"{method_filename_prefix}.csv"
                        csv_df.to_csv(csv_out_path, index=False)

                        logging.info(f"Saved CSV file: {csv_out_path}")

            except Exception as e:
                logging.error(f"Failed to process {method} for {high_dim_file}: {e}")
                continue

        # After generating all CSV files, call plot_from_csv_files to create plots
        logging.info(
            "All CSV files generated, now creating plots using plot_from_csv_files..."
        )
        plot_output_paths = self.plot_from_csv_files()

        return plot_output_paths if plot_output_paths else None

    def _extract_method_from_filename(self, parts: List[str]) -> str:
        """Extract dimensionality reduction method from filename parts"""
        for part in parts:
            if part.lower() in ["pca", "tsne", "umap"]:
                return part.lower()
        return "pca"  # default

    def _extract_grouping_from_filename(
        self, filename: str, tokenizer_type: str
    ) -> str:
        """Extract grouping type from filename for domain tokenizer"""
        if tokenizer_type == "Domain":
            return "6group" if "6group" in filename else "4group"
        return "standard"

    def _extract_coordinates_from_df(self, df: pd.DataFrame, is_3d: bool) -> np.ndarray:
        """Extract coordinate data from DataFrame"""
        if is_3d and "z" in df.columns:
            return df[["x", "y", "z"]].values
        else:
            return df[["x", "y"]].values

    def _create_plot_from_data(
        self,
        coords: np.ndarray,
        tokens: List[str],
        labels: List[str],
        method: str,
        tokenizer_type: str,
        is_3d: bool,
        grouping_type: str,
        font_size: int = 18,
        legend_size: int = 22,
        marker_size: int = 48,
        show_box: bool = True,
        figsize: Optional[Tuple[int, int]] = None,
    ) -> plt.Figure:
        """Create a plot from coordinate data using existing plotting infrastructure"""
        import matplotlib.pyplot as plt

        # Create figure
        figure_size = figsize if figsize is not None else self.plot_spec.figsize
        if is_3d:
            fig = plt.figure(figsize=figure_size)
            ax = fig.add_subplot(111, projection="3d")
        else:
            fig, ax = plt.subplots(figsize=figure_size)

        # Plot data based on tokenizer type
        if tokenizer_type == "Domain":
            self._plot_domain_data_from_tokens(
                ax, coords, tokens, is_3d, grouping_type, marker_size=marker_size
            )
        else:
            self._plot_human_data_from_labels(
                ax, coords, labels, is_3d, marker_size=marker_size
            )

        # Control axis box visibility
        if not show_box:
            ax.set_xticks([])
            ax.set_yticks([])
            if not is_3d:
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["bottom"].set_visible(False)
                ax.spines["left"].set_visible(False)
        else:
            # Set axis labels
            self._set_axis_labels_for_method(ax, method, is_3d, font_size)

        # Add legend and layout
        ax.legend(
            loc="best",
            fontsize=legend_size,
            frameon=True,
            fancybox=True,
            shadow=True,
            facecolor="white",
            framealpha=0.9,
        )
        fig.tight_layout()

        return fig

    def _get_domain_color_mapping(self, grouping_type: str) -> dict:
        """Get consistent color mapping for domain groups"""
        if grouping_type == "6group":
            groups_order = ["U*", "D*", "L*", "R*", "T*", "B*", "SPECIAL"]
        else:
            groups_order = ["U-D", "L-R", "T-B", "SPECIAL"]

        colors = palette_slice(len(groups_order))
        return {group: colors[idx] for idx, group in enumerate(groups_order)}

    def _plot_domain_data_from_tokens(
        self,
        ax,
        coords: np.ndarray,
        tokens: List[str],
        is_3d: bool,
        grouping_type: str,
        filter_groups: List[str] = None,
        marker_size: int = 48,
    ) -> None:
        """Plot domain tokenizer data with direction grouping from token names

        Args:
            filter_groups: If provided, only plot these groups (e.g., ['U*', 'D*'])
        """
        # Determine grouping configuration
        if grouping_type == "6group":
            groups_order = ["U*", "D*", "L*", "R*", "T*", "B*", "SPECIAL"]
            logic = "individual"
        else:
            groups_order = ["U-D", "L-R", "T-B", "SPECIAL"]
            logic = "combined"

        # Get color mapping
        color_mapping = self._get_domain_color_mapping(grouping_type)

        # Group coordinates by direction
        direction_coords = {k: [] for k in groups_order}

        for i, token in enumerate(tokens):
            first_char = token[0] if token else "SPECIAL"

            if logic == "individual":
                if first_char in {"U", "D", "L", "R", "T", "B"}:
                    group_key = f"{first_char}*"
                    direction_coords[group_key].append(coords[i])
                else:
                    direction_coords["SPECIAL"].append(coords[i])
            else:  # combined logic
                if first_char in {"U", "D"}:
                    direction_coords["U-D"].append(coords[i])
                elif first_char in {"L", "R"}:
                    direction_coords["L-R"].append(coords[i])
                elif first_char in {"T", "B"}:
                    direction_coords["T-B"].append(coords[i])
                else:
                    direction_coords["SPECIAL"].append(coords[i])

        # Filter groups if specified
        if filter_groups:
            groups_to_plot = [g for g in groups_order if g in filter_groups]
        else:
            groups_to_plot = groups_order

        # Plot each group
        for direction in groups_to_plot:
            if direction_coords[direction]:
                coords_array = np.array(direction_coords[direction])
                color = color_mapping[direction]
                if is_3d:
                    ax.scatter(
                        coords_array[:, 0],
                        coords_array[:, 1],
                        coords_array[:, 2],
                        alpha=self.plot_spec.alpha,
                        label=direction,
                        color=color,
                        s=marker_size,
                    )
                else:
                    ax.scatter(
                        coords_array[:, 0],
                        coords_array[:, 1],
                        alpha=self.plot_spec.alpha,
                        label=direction,
                        color=color,
                        s=marker_size,
                    )

    def _find_densest_cluster_with_dbscan(
        self, coords: np.ndarray, eps: float = 1.5, min_samples: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Use DBSCAN to find the densest cluster in 2D coordinates"""
        # Apply DBSCAN clustering
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(coords)

        # Find the largest cluster (excluding noise points labeled as -1)
        unique_labels = np.unique(cluster_labels)
        cluster_sizes = {}

        for label in unique_labels:
            if label != -1:  # Exclude noise points
                cluster_sizes[label] = np.sum(cluster_labels == label)

        if not cluster_sizes:
            # If no valid clusters found, return the center region
            center_x, center_y = np.mean(coords, axis=0)
            std_x, std_y = np.std(coords, axis=0)

            # Select points within 1 standard deviation from center
            center_mask = (np.abs(coords[:, 0] - center_x) <= std_x) & (
                np.abs(coords[:, 1] - center_y) <= std_y
            )
            return np.where(center_mask)[0], coords[center_mask]

        # Get the densest cluster
        densest_label = max(cluster_sizes, key=cluster_sizes.get)
        densest_indices = np.where(cluster_labels == densest_label)[0]

        return densest_indices, coords[densest_indices]

    def _find_group_clusters(
        self,
        coords: np.ndarray,
        tokens: List[str],
        max_points: int = 10,
        cluster_index: int = 0,
    ) -> dict:
        """Find clusters for each group pair using KMeans: U*/D*, L*/R*, T*/B*

        Args:
            max_points: Maximum points per cluster
            cluster_index: Which cluster to select (0=densest, 1=second densest, etc.)

        Returns:
            Dictionary with group names as keys and (bbox_coords, bbox_tokens, target_coords, target_tokens, filter_groups) as values
            - bbox_coords/tokens: All points in the bounding box (including non-target)
            - target_coords/tokens: Only target group points for annotation
        """
        # Define the three group pairs
        group_pairs = {
            "U*/D*": (["U*", "D*"], {"U", "D"}),
            "L*/R*": (["L*", "R*"], {"L", "R"}),
            "T*/B*": (["T*", "B*"], {"T", "B"}),
        }

        results = {}

        # Process each group pair
        for group_name, (filter_groups, first_chars) in group_pairs.items():
            # Find tokens belonging to this group
            group_indices = []
            group_tokens = []
            group_first_chars = []

            for i, token in enumerate(tokens):
                first_char = token[0] if token else "SPECIAL"
                if first_char in first_chars:
                    group_indices.append(i)
                    group_tokens.append(token)
                    group_first_chars.append(first_char)

            if (
                len(group_indices) < 4
            ):  # Need at least 4 points for meaningful clustering
                # Not enough tokens for this group
                results[group_name] = (None, None, None, None, filter_groups)
                continue

            # Check if both categories have at least 2 points
            char_counts = {char: group_first_chars.count(char) for char in first_chars}
            if any(count < 2 for count in char_counts.values()):
                # Not enough points in each category, use most central approach
                center = np.mean(coords[group_indices], axis=0)
                distances = np.linalg.norm(coords[group_indices] - center, axis=1)
                closest_indices = np.argsort(distances)[
                    : min(max_points, len(group_indices))
                ]
                selected_global_indices = [group_indices[i] for i in closest_indices]
                selected_coords = coords[selected_global_indices]
                selected_tokens = [group_tokens[i] for i in closest_indices]

                # Calculate bounding box for all points in the selection area
                x_min, x_max = (
                    np.min(selected_coords[:, 0]),
                    np.max(selected_coords[:, 0]),
                )
                y_min, y_max = (
                    np.min(selected_coords[:, 1]),
                    np.max(selected_coords[:, 1]),
                )
                x_range = x_max - x_min if x_max != x_min else 1.0
                y_range = y_max - y_min if y_max != y_min else 1.0
                x_padding = max(0.3 * x_range, 0.5)
                y_padding = max(0.3 * y_range, 0.5)

                # Find all points within bounding box
                bbox_mask = (
                    (coords[:, 0] >= x_min - x_padding)
                    & (coords[:, 0] <= x_max + x_padding)
                    & (coords[:, 1] >= y_min - y_padding)
                    & (coords[:, 1] <= y_max + y_padding)
                )
                bbox_indices = np.where(bbox_mask)[0]
                bbox_coords = coords[bbox_indices]
                bbox_tokens = [tokens[i] for i in bbox_indices]

                results[group_name] = (
                    bbox_coords,
                    bbox_tokens,
                    selected_coords,
                    selected_tokens,
                    filter_groups,
                )
                continue

            group_coords = coords[group_indices]

            # Calculate k for KMeans: num_points / max_points
            num_points = len(group_coords)
            k = max(
                1, min(num_points // max_points, num_points - 1)
            )  # Ensure k is valid

            # Apply KMeans clustering
            from sklearn.cluster import KMeans

            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(group_coords)

            # Calculate cluster densities and sort by density
            clusters_with_density = []
            for cluster_id in range(k):
                cluster_mask = cluster_labels == cluster_id
                cluster_coords_subset = group_coords[cluster_mask]
                cluster_tokens_subset = [
                    group_tokens[i] for i in range(len(group_tokens)) if cluster_mask[i]
                ]
                cluster_first_chars = [
                    token[0] if token else "SPECIAL" for token in cluster_tokens_subset
                ]

                # Check if this cluster has at least 2 points from each required category
                cluster_char_counts = {
                    char: cluster_first_chars.count(char) for char in first_chars
                }
                if not all(count >= 2 for count in cluster_char_counts.values()):
                    continue  # Skip clusters that don't have enough of each category

                # Calculate cluster density (points per unit area)
                if len(cluster_coords_subset) >= 2:
                    # Calculate convex hull area or use bounding box area
                    x_min_c, x_max_c = (
                        np.min(cluster_coords_subset[:, 0]),
                        np.max(cluster_coords_subset[:, 0]),
                    )
                    y_min_c, y_max_c = (
                        np.min(cluster_coords_subset[:, 1]),
                        np.max(cluster_coords_subset[:, 1]),
                    )
                    area = max(
                        (x_max_c - x_min_c) * (y_max_c - y_min_c), 1e-6
                    )  # Avoid division by zero
                    density = len(cluster_coords_subset) / area
                else:
                    density = 0

                clusters_with_density.append(
                    (cluster_id, density, len(cluster_coords_subset))
                )

            if not clusters_with_density:
                # No valid clusters found, fall back to central selection
                center = np.mean(group_coords, axis=0)
                distances = np.linalg.norm(group_coords - center, axis=1)
                closest_indices = np.argsort(distances)[
                    : min(max_points, len(group_coords))
                ]
                selected_coords = group_coords[closest_indices]
                selected_tokens = [group_tokens[i] for i in closest_indices]

                # Calculate bounding box
                x_min, x_max = (
                    np.min(selected_coords[:, 0]),
                    np.max(selected_coords[:, 0]),
                )
                y_min, y_max = (
                    np.min(selected_coords[:, 1]),
                    np.max(selected_coords[:, 1]),
                )
                x_range = x_max - x_min if x_max != x_min else 1.0
                y_range = y_max - y_min if y_max != y_min else 1.0
                x_padding = max(0.3 * x_range, 0.5)
                y_padding = max(0.3 * y_range, 0.5)

                bbox_mask = (
                    (coords[:, 0] >= x_min - x_padding)
                    & (coords[:, 0] <= x_max + x_padding)
                    & (coords[:, 1] >= y_min - y_padding)
                    & (coords[:, 1] <= y_max + y_padding)
                )
                bbox_indices = np.where(bbox_mask)[0]
                bbox_coords = coords[bbox_indices]
                bbox_tokens = [tokens[i] for i in bbox_indices]

                results[group_name] = (
                    bbox_coords,
                    bbox_tokens,
                    selected_coords,
                    selected_tokens,
                    filter_groups,
                )
            else:
                # Sort clusters by density (descending) and select the specified index
                clusters_with_density.sort(key=lambda x: x[1], reverse=True)

                # Select cluster by index (default to 0 = densest)
                if cluster_index < len(clusters_with_density):
                    selected_cluster_id = clusters_with_density[cluster_index][0]
                else:
                    selected_cluster_id = clusters_with_density[0][
                        0
                    ]  # Fallback to densest

                selected_mask = cluster_labels == selected_cluster_id
                cluster_coords = group_coords[selected_mask]
                cluster_tokens_list = [
                    group_tokens[i]
                    for i in range(len(group_tokens))
                    if selected_mask[i]
                ]

                # Limit to max_points if cluster is too large
                if len(cluster_coords) > max_points:
                    cluster_center = np.mean(cluster_coords, axis=0)
                    distances = np.linalg.norm(cluster_coords - cluster_center, axis=1)
                    closest_indices = np.argsort(distances)[:max_points]
                    cluster_coords = cluster_coords[closest_indices]
                    cluster_tokens_list = [
                        cluster_tokens_list[i] for i in closest_indices
                    ]

                # Calculate bounding box for all points in the cluster area
                x_min, x_max = (
                    np.min(cluster_coords[:, 0]),
                    np.max(cluster_coords[:, 0]),
                )
                y_min, y_max = (
                    np.min(cluster_coords[:, 1]),
                    np.max(cluster_coords[:, 1]),
                )
                x_range = x_max - x_min if x_max != x_min else 1.0
                y_range = y_max - y_min if y_max != y_min else 1.0
                x_padding = max(0.3 * x_range, 0.5)
                y_padding = max(0.3 * y_range, 0.5)

                # Find all points within bounding box (including non-target groups)
                bbox_mask = (
                    (coords[:, 0] >= x_min - x_padding)
                    & (coords[:, 0] <= x_max + x_padding)
                    & (coords[:, 1] >= y_min - y_padding)
                    & (coords[:, 1] <= y_max + y_padding)
                )
                bbox_indices = np.where(bbox_mask)[0]
                bbox_coords = coords[bbox_indices]
                bbox_tokens = [tokens[i] for i in bbox_indices]

                results[group_name] = (
                    bbox_coords,
                    bbox_tokens,
                    cluster_coords,
                    cluster_tokens_list,
                    filter_groups,
                )

        return results

    def _create_domain_detail_plot(
        self,
        coords: np.ndarray,
        tokens: List[str],
        grouping_type: str,
        cluster_index: int = 0,
        main_w_n: int = 16,
        main_h_n: int = 16,
        detail_n: int = 6,
        font_size: int = 18,
        legend_size: int = 22,
        main_marker_size: int = 48,
        detail_marker_size: int = 72,
        main_show_box: bool = True,
    ) -> plt.Figure:
        """Create a domain plot with main plot on top and 3 local detail zoom-ins on bottom for 6group only"""
        if grouping_type != "6group":
            raise ValueError("Detail plot is only supported for 6group")

        # Check layout validity: main_w_n and detail_n must have same parity (both odd or both even)
        if main_w_n % 2 != detail_n % 2:
            logging.error(
                f"Invalid layout parameters: detail_n={detail_n}, main_w_n={main_w_n}. "
                f"main_w_n and detail_n must have same parity (both odd or both even)."
            )
            raise ValueError(
                "Layout parameters invalid: main_w_n and detail_n must have same parity"
            )

        # Find the clusters for all three group pairs using KMeans
        group_clusters = self._find_group_clusters(
            coords, tokens, cluster_index=cluster_index
        )

        # Calculate dynamic layout based on parameters
        # Total width: detail_n*3 + 2 (gaps between detail plots)
        total_width = detail_n * 3 + 2
        # Total height: main_h_n + detail_n
        total_height = main_h_n + detail_n

        # Calculate main plot centering
        main_margin = (total_width - main_w_n) // 2

        # Create figure with dynamic sizing
        fig = plt.figure(figsize=(total_width, total_height))

        # Use dynamic grid layout:
        # Top: main plot main_w_n x main_h_n (rows 0 to main_h_n-1, centered)
        # Gap: row main_h_n (1 row spacing)
        # Bottom: 3 detail plots detail_n x detail_n each (with 1 unit gaps between plots)
        ax_main = plt.subplot2grid(
            (total_height, total_width),
            (0, main_margin + main_margin // 2),
            rowspan=main_h_n,
            colspan=main_w_n,
            fig=fig,
        )

        # Plot complete 6group data on main plot
        self._plot_domain_data_from_tokens(
            ax_main, coords, tokens, False, grouping_type, marker_size=main_marker_size
        )

        # Set consistent font sizes for main plot
        ax_main.set_xlabel("X", fontsize=font_size)
        ax_main.set_ylabel("Y", fontsize=font_size)
        ax_main.tick_params(labelsize=font_size)

        # Add legend inside main plot with automatic best positioning
        ax_main.legend(
            bbox_to_anchor=(0, 1.0),
            loc="upper right",
            fontsize=legend_size,
            frameon=True,
            fancybox=True,
            shadow=True,
        )
        ax_main_legend = ax_main.get_legend()
        ax_main_legend.get_frame().set_facecolor("white")
        ax_main_legend.get_frame().set_alpha(0.9)

        # Control main plot axis box visibility
        if not main_show_box:
            ax_main.set_xlabel("")
            ax_main.set_ylabel("")
            ax_main.set_xticks([])
            ax_main.set_yticks([])
            ax_main.spines["top"].set_visible(False)
            ax_main.spines["right"].set_visible(False)
            ax_main.spines["bottom"].set_visible(False)
            ax_main.spines["left"].set_visible(False)

        # Create 3 detail plots on the bottom (each detail_n x detail_n) with 1-unit gaps
        detail_start_row = main_h_n  # After main plot and gap
        detail_axes = [
            plt.subplot2grid(
                (total_height, total_width),
                (detail_start_row, 0),
                rowspan=detail_n,
                colspan=detail_n,
                fig=fig,
            ),  # Bottom left
            plt.subplot2grid(
                (total_height, total_width),
                (detail_start_row, detail_n + 1),
                rowspan=detail_n,
                colspan=detail_n,
                fig=fig,
            ),  # Bottom center
            plt.subplot2grid(
                (total_height, total_width),
                (detail_start_row, detail_n * 2 + 2),
                rowspan=detail_n,
                colspan=detail_n,
                fig=fig,
            ),  # Bottom right
        ]

        # Set spines linewidth for detail plots
        for ax in detail_axes:
            for spine in ax.spines.values():
                spine.set_linewidth(5)
                spine.set_color("darkgray")

        # First, collect bounding boxes for all clusters and sort by x coordinate
        cluster_data_with_boxes = []
        group_names = ["U*/D*", "L*/R*", "T*/B*"]

        for group_name in group_names:
            cluster_data = group_clusters.get(
                group_name, (None, None, None, None, None)
            )
            bbox_coords, bbox_tokens, target_coords, target_tokens, filter_groups = (
                cluster_data
            )

            if target_coords is not None and len(target_coords) > 0:
                # Calculate bounding box for this group's cluster (use target coords for box calculation)
                x_min, x_max = np.min(target_coords[:, 0]), np.max(target_coords[:, 0])
                y_min, y_max = np.min(target_coords[:, 1]), np.max(target_coords[:, 1])

                # Calculate better padding based on data spread
                x_range = x_max - x_min
                y_range = y_max - y_min

                # Use larger padding for small clusters
                if x_range < 0.1:  # Very tight cluster
                    x_padding = 0.5
                else:
                    x_padding = x_range * 0.3

                if y_range < 0.1:  # Very tight cluster
                    y_padding = 0.5
                else:
                    y_padding = y_range * 0.3

                x_min_padded = x_min - x_padding
                x_max_padded = x_max + x_padding
                y_min_padded = y_min - y_padding
                y_max_padded = y_max + y_padding

                box = (x_min_padded, x_max_padded, y_min_padded, y_max_padded)
                # Use center x coordinate for sorting (left to right)
                center_x = (x_min + x_max) / 2
                cluster_data_with_boxes.append(
                    (group_name, cluster_data, box, center_x)
                )
            else:
                # Still add to list with None values for consistent indexing
                cluster_data_with_boxes.append(
                    (group_name, cluster_data, None, float("inf"))
                )

        # Sort by center_x coordinate (left to right: lowest x to highest x)
        cluster_data_with_boxes.sort(
            key=lambda x: x[3] if x[3] != float("inf") else float("inf")
        )

        # Extract sorted data
        sorted_group_names = [item[0] for item in cluster_data_with_boxes]
        sorted_cluster_data = [item[1] for item in cluster_data_with_boxes]
        cluster_boxes = [item[2] for item in cluster_data_with_boxes]

        # Draw bounding boxes on main plot
        box_colors = ["darkgray", "darkgray", "darkgray"]  # All boxes use dark gray

        for i, (box, color) in enumerate(zip(cluster_boxes, box_colors)):
            if box is not None:
                x_min, x_max, y_min, y_max = box
                width = x_max - x_min
                height = y_max - y_min
                rect = Rectangle(
                    (x_min, y_min),
                    width,
                    height,
                    linewidth=5,
                    edgecolor=color,
                    facecolor="none",
                    alpha=0.8,
                )
                ax_main.add_patch(rect)

        # Add connection arrows from boxes to detail plots (before text processing)
        self._add_connection_arrows(fig, ax_main, detail_axes, cluster_boxes)

        # Now process each group's detail plot using sorted order
        for i, (group_name, ax_detail) in enumerate(
            zip(sorted_group_names, detail_axes)
        ):
            cluster_data = sorted_cluster_data[i]
            bbox_coords, bbox_tokens, target_coords, target_tokens, filter_groups = (
                cluster_data
            )
            box = cluster_boxes[i]

            if bbox_coords is not None and len(bbox_coords) > 0 and box is not None:
                x_min, x_max, y_min, y_max = box

                # Plot ALL points in the bounding box
                self._plot_domain_detail_with_transparency(
                    ax_detail,
                    bbox_coords,
                    bbox_tokens,
                    target_tokens,
                    grouping_type,
                    filter_groups,
                    marker_size=detail_marker_size,
                )

                # Remove ticks, labels, and title as requested
                ax_detail.set_xticks([])
                ax_detail.set_yticks([])
                ax_detail.set_xlabel("")
                ax_detail.set_ylabel("")

                # Set axis limits with better margins for aesthetics and make square
                x_center = (x_min + x_max) / 2
                y_center = (y_min + y_max) / 2
                x_range = x_max - x_min
                y_range = y_max - y_min
                max_range = max(x_range, y_range)

                # Make square by using max range for both axes
                ax_detail.set_xlim(x_center - max_range / 2, x_center + max_range / 2)
                ax_detail.set_ylim(y_center - max_range / 2, y_center + max_range / 2)
                ax_detail.set_aspect("equal")

                # First: Add symmetric token connections (before text)
                if target_coords is not None and target_tokens is not None:
                    self._add_symmetric_token_connections(
                        ax_detail, target_coords, target_tokens
                    )

                # Second: Add token annotations ONLY for target tokens (with larger font)
                if target_coords is not None and target_tokens is not None:
                    # Create text objects for adjustText to process
                    texts = []
                    for j, token in enumerate(target_tokens):
                        text = ax_detail.text(
                            target_coords[j, 0],
                            target_coords[j, 1],
                            token,
                            fontsize=font_size,
                            alpha=0.9,
                            ha="center",
                            va="center",
                            fontweight="bold",
                        )
                        texts.append(text)

                    # Third: Use adjustText to automatically prevent text overlap
                    if len(texts) > 1:  # Only apply if there are multiple texts
                        adjust_text(
                            texts,
                            ax=ax_detail,
                            expand=(
                                1.5,
                                1.5,
                            ),
                            arrowprops=dict(arrowstyle="->", color="gray", lw=1),
                        )
            else:
                # No cluster found for this group
                ax_detail.text(
                    0.5,
                    0.5,
                    f"No {group_name} cluster",
                    transform=ax_detail.transAxes,
                    ha="center",
                    va="center",
                    fontsize=font_size,
                    fontweight="bold",
                )
                ax_detail.set_xticks([])
                ax_detail.set_yticks([])
                ax_detail.set_xlabel("")
                ax_detail.set_ylabel("")
                ax_detail.set_aspect("equal")

        return fig

    def _save_high_dimensional_embeddings(
        self,
        combined_emb: np.ndarray,
        tokens_filtered: List[str],
        unused_tokens_filtered: List[str],
        filename_prefix: str,
        label: str,
    ) -> None:
        """Save original high-dimensional embeddings to file for later processing"""
        import pandas as pd

        # Create dataframe with embeddings and metadata
        all_tokens = tokens_filtered + unused_tokens_filtered
        n_used = len(tokens_filtered)

        # Create labels: "Used" for used tokens, "Unused" for unused tokens
        token_labels = ["Used"] * n_used + ["Unused"] * len(unused_tokens_filtered)

        # Create embedding dataframe
        embedding_dim = combined_emb.shape[1]
        data = {
            "token": all_tokens,
            "label": token_labels,
            "tokenizer_type": [label] * len(all_tokens),
        }

        # Add embedding dimensions as columns
        for i in range(embedding_dim):
            data[f"emb_{i}"] = combined_emb[:, i]

        df = pd.DataFrame(data)

        # Save to parquet file for efficient storage and loading
        output_path = self.plot_dir / f"{filename_prefix}_high_dim_embeddings.parquet"
        df.to_parquet(output_path, index=False)

        logging.info(
            f"Saved high-dimensional embeddings ({embedding_dim}D) to: {output_path}"
        )

    def _plot_domain_detail_with_transparency(
        self,
        ax,
        bbox_coords: np.ndarray,
        bbox_tokens: List[str],
        target_tokens: List[str],
        grouping_type: str,
        filter_groups: List[str],
        marker_size: int = 72,
    ) -> None:
        """Plot domain detail with target groups normal transparency and others high transparency"""
        # Get color mapping
        color_mapping = self._get_domain_color_mapping(grouping_type)

        # Create set of target tokens for quick lookup
        target_token_set = set(target_tokens) if target_tokens else set()

        # Group tokens by category
        token_groups = {}
        if grouping_type == "6group":
            groups_order = ["U*", "D*", "L*", "R*", "T*", "B*", "SPECIAL"]
            for i, token in enumerate(bbox_tokens):
                first_char = token[0] if token else "SPECIAL"
                if first_char in "UDLRTB":
                    group_key = f"{first_char}*"
                else:
                    group_key = "SPECIAL"

                if group_key not in token_groups:
                    token_groups[group_key] = []
                token_groups[group_key].append((i, token))
        else:
            # 4group logic would go here if needed
            return

        # Plot each group with appropriate transparency
        for group_key in groups_order:
            if group_key not in token_groups:
                continue

            indices, tokens_in_group = zip(*token_groups[group_key])
            group_coords = bbox_coords[list(indices)]

            # Check if this group is a target group
            is_target_group = group_key in filter_groups if filter_groups else False

            # Get color for this group
            color = color_mapping.get(group_key, "gray")

            if is_target_group:
                # For target groups, separate points with and without text
                points_with_text = []
                points_without_text = []

                for i, (idx, token) in enumerate(token_groups[group_key]):
                    if token in target_token_set:
                        points_with_text.append(i)
                    else:
                        points_without_text.append(i)

                # Plot points with text (normal transparency)
                if points_with_text:
                    text_coords = group_coords[points_with_text]
                    ax.scatter(
                        text_coords[:, 0],
                        text_coords[:, 1],
                        alpha=self.plot_spec.alpha,
                        label=group_key,
                        color=color,
                        s=marker_size,
                    )

                # Plot points without text (higher transparency)
                if points_without_text:
                    no_text_coords = group_coords[points_without_text]
                    ax.scatter(
                        no_text_coords[:, 0],
                        no_text_coords[:, 1],
                        alpha=0.3,
                        color=color,
                        s=marker_size,
                    )
            else:
                # Non-target groups: all points with high transparency
                ax.scatter(
                    group_coords[:, 0],
                    group_coords[:, 1],
                    alpha=0.2,
                    color=color,
                    s=marker_size,
                )

    def _add_connection_arrows(self, fig, ax_main, detail_axes, cluster_boxes):
        """Add dashed arrows connecting main plot box bottom edges to detail plot top edges"""
        from matplotlib.patches import ConnectionPatch

        for i, (ax_detail, box) in enumerate(zip(detail_axes, cluster_boxes)):
            if box is not None:
                x_min, x_max, y_min, y_max = box

                # Calculate box bottom edge center point
                box_center_x = (x_min + x_max) / 2
                box_bottom_y = y_min

                # Calculate detail plot top edge center point
                # All detail plots use top center for connection
                detail_edge_x = 0.5  # Center of detail plot
                detail_edge_y = 1.0  # Top edge

                # Create connection patch (dashed arrow) from box bottom to detail top
                con = ConnectionPatch(
                    xyA=(box_center_x, box_bottom_y),
                    coordsA=ax_main.transData,
                    xyB=(detail_edge_x, detail_edge_y),
                    coordsB=ax_detail.transAxes,
                    arrowstyle="->",
                    shrinkA=2,
                    shrinkB=2,
                    linestyle="--",
                    linewidth=5,
                    color="darkgray",
                    alpha=0.8,
                )
                fig.add_artist(con)

    def _add_symmetric_token_connections(self, ax, target_coords, target_tokens):
        """Add gray dashed lines between symmetric tokens (e.g., U40 and D40)"""
        if (
            target_coords is None
            or target_tokens is None
            or len(target_coords) == 0
            or len(target_tokens) == 0
        ):
            return

        # Find symmetric pairs
        token_groups = {}
        for i, token in enumerate(target_tokens):
            if len(token) > 1:
                prefix = token[0]  # U, D, L, R, T, B
                suffix = token[1:]  # numeric part
                if suffix not in token_groups:
                    token_groups[suffix] = {}
                token_groups[suffix][prefix] = (i, target_coords[i])

        # Draw connections for symmetric pairs
        symmetric_pairs = [("U", "D"), ("L", "R"), ("T", "B")]

        for suffix, group in token_groups.items():
            for prefix1, prefix2 in symmetric_pairs:
                if prefix1 in group and prefix2 in group:
                    i1, pos1 = group[prefix1]
                    i2, pos2 = group[prefix2]

                    # Draw dashed line
                    ax.plot(
                        [pos1[0], pos2[0]],
                        [pos1[1], pos2[1]],
                        linestyle="--",
                        color="gray",
                        alpha=0.6,
                        linewidth=5,
                    )

    def _plot_human_data_from_labels(
        self,
        ax,
        coords: np.ndarray,
        labels: List[str],
        is_3d: bool,
        marker_size: int = 48,
    ) -> None:
        """Plot human tokenizer data without using different markers"""
        seen: List[str] = []
        for label in labels:
            if label not in seen:
                seen.append(label)
        unique_labels = seen
        colors = palette_slice(len(unique_labels))
        color_mapping = {label: colors[idx] for idx, label in enumerate(unique_labels)}

        for label in unique_labels:
            label_indices = [i for i, lab in enumerate(labels) if lab == label]
            label_coords = coords[label_indices]
            color = color_mapping[label]

            if is_3d:
                ax.scatter(
                    label_coords[:, 0],
                    label_coords[:, 1],
                    label_coords[:, 2],
                    alpha=self.plot_spec.alpha,
                    label=str(label),
                    color=color,
                    s=marker_size,
                )
            else:
                ax.scatter(
                    label_coords[:, 0],
                    label_coords[:, 1],
                    alpha=self.plot_spec.alpha,
                    label=str(label),
                    color=color,
                    s=marker_size,
                )

    def _set_axis_labels_for_method(
        self, ax, method: str, is_3d: bool, font_size: int = 18
    ) -> None:
        """Set appropriate axis labels using unified X, Y, Z format"""
        ax.set_xlabel("X", fontsize=font_size)
        ax.set_ylabel("Y", fontsize=font_size)
        ax.tick_params(labelsize=font_size)
        if is_3d:
            ax.set_zlabel("Z", fontsize=font_size)

    def _generate_plot_title(
        self, tokenizer_type: str, method: str, is_3d: bool, grouping_type: str
    ) -> str:
        """Generate an appropriate title for the plot"""
        dim_str = "3D" if is_3d else "2D"
        title = f"{tokenizer_type} {method.upper()} {dim_str}"

        if tokenizer_type == "Domain" and grouping_type in ["4group", "6group"]:
            title += f" ({grouping_type})"

        return title
