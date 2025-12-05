#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   treeization.py
@Time    :   2025/08/14 16:38:56
@Author  :   Dawn Li
@Version :   1.0
@Contact :   dawnli619215645@gmail.com
@Desc    :   Treeization demo for Net Routing Tokenization
"""

import argparse
import logging
from pathlib import Path

import cv2
import matplotlib
import networkx as nx
import numpy as np
from networkx.drawing.nx_agraph import graphviz_layout
from tqdm import tqdm

from flow.config import TokenizationStageConfig
from flow.tokenization import UnifiedTokenizer
from flow.utils import setup_logging

if matplotlib:
    matplotlib.use("Agg")

    matplotlib.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 120,
            "path.simplify": True,
            "path.simplify_threshold": 0.5,
            "agg.path.chunksize": 10000,
            "figure.autolayout": False,
            "font.family": "DejaVu Sans Mono",
            "axes.linewidth": 0.5,
            "xtick.minor.visible": False,
            "ytick.minor.visible": False,
            "lines.antialiased": False,
        }
    )
    matplotlib.interactive(False)
    import matplotlib.pyplot as plt

COLORS = {
    "node_default": "#4FC3F7",
    "node_visited": "#DE8322",
    "node_current": "#6A6967",
    "edge_default": "#90A4AE",
    "edge_active": "#1394A4",
    "background": "#262626",
    "text_default": "#FFFFFF",
    "text_branch": "#B61E1E",
    "text_end": "#188651",
    "highlight_branch": "#B61E1E",
    "highlight_end": "#188651",
}

input_tokens = [
    "(0, 0, 0)",
    "(0, 0, 2)",
    "[BRANCH]",
    "(780, 0, 2)",
    "(780, 0, 4)",
    "[BRANCH]",
    "(780, 3000, 4)",
    "(780, 3000, 2)",
    "[END]",
    "[BRANCH]",
    "(780, 0, 6)",
    "[BRANCH]",
    "(2580, 0, 6)",
    "(2580, 0, 4)",
    "[BRANCH]",
    "(2580, 1800, 4)",
    "(2580, 1800, 6)",
    "(4380, 1800, 6)",
    "(4380, 1800, 4)",
    "(4380, 2000, 4)",
    "[BRANCH]",
    "(4380, 3000, 4)",
    "(4380, 3000, 2)",
    "[END]",
    "[BRANCH]",
    "(4780, 2000, 4)",
    "(4780, 1400, 4)",
    "(4980, 1400, 4)",
    "(4980, -100, 4)",
    "[BRANCH]",
    "(4980, -400, 4)",
    "(4980, -400, 2)",
    "(4760, -400, 2)",
    "(4760, -600, 2)",
    "(4580, -600, 2)",
    "[END]",
    "[BRANCH]",
    "(5380, -100, 4)",
    "[BRANCH]",
    "(5380, -2800, 4)",
    "(5580, -2800, 4)",
    "(5580, -4600, 4)",
    "[BRANCH]",
    "(5580, -4600, 2)",
    "(6160, -4600, 2)",
    "(6160, -4400, 2)",
    "(6470, -4400, 2)",
    "(6470, -4400, 0)",
    "[END]",
    "[BRANCH]",
    "(5580, -5600, 4)",
    "[BRANCH]",
    "(5580, -6400, 4)",
    "(5380, -6400, 4)",
    "(5380, -8000, 4)",
    "(5380, -8000, 2)",
    "(4230, -8000, 2)",
    "(4230, -8000, 0)",
    "[END]",
    "[BRANCH]",
    "(5580, -5600, 2)",
    "(5350, -5600, 2)",
    "(5350, -5600, 0)",
    "[END]",
    "[END]",
    "[END]",
    "[BRANCH]",
    "(5380, 200, 4)",
    "(5580, 200, 4)",
    "(5580, 1800, 4)",
    "(5580, 1800, 2)",
    "(7000, 1800, 2)",
    "(7000, 1400, 2)",
    "(6780, 1400, 2)",
    "[END]",
    "[END]",
    "[END]",
    "[END]",
    "[BRANCH]",
    "(2580, -600, 4)",
    "(2580, -600, 2)",
    "[END]",
    "[END]",
    "[BRANCH]",
    "(380, 0, 6)",
    "(380, 0, 8)",
    "(380, -4800, 8)",
    "(380, -4800, 6)",
    "(380, -4800, 4)",
    "(380, -8000, 4)",
    "(380, -8000, 2)",
    "(310, -8000, 2)",
    "(310, -8000, 0)",
    "[END]",
    "[END]",
    "[END]",
    "[BRANCH]",
    "(-620, 0, 2)",
    "(-620, 0, 4)",
    "(-620, -600, 4)",
    "[BRANCH]",
    "(-620, -600, 6)",
    "(-820, -600, 6)",
    "(-820, -800, 6)",
    "(-5620, -800, 6)",
    "[BRANCH]",
    "(-7220, -800, 6)",
    "[BRANCH]",
    "(-9420, -800, 6)",
    "(-9420, -800, 4)",
    "[BRANCH]",
    "(-9420, 1400, 4)",
    "(-9420, 1400, 2)",
    "[END]",
    "[BRANCH]",
    "(-9420, -2600, 4)",
    "(-9420, -2600, 6)",
    "(-12820, -2600, 6)",
    "[BRANCH]",
    "(-12820, -2600, 4)",
    "(-12820, -2000, 4)",
    "(-12820, -2000, 2)",
    "(-12070, -2000, 2)",
    "(-12070, -2000, 0)",
    "[END]",
    "[BRANCH]",
    "(-13620, -2600, 6)",
    "(-13620, -2600, 8)",
    "(-13620, -5800, 8)",
    "(-13620, -5800, 6)",
    "(-14820, -5800, 6)",
    "(-14820, -5800, 4)",
    "[BRANCH]",
    "(-14820, -8000, 4)",
    "(-14820, -8000, 2)",
    "(-13470, -8000, 2)",
    "(-13470, -8000, 0)",
    "[END]",
    "[BRANCH]",
    "(-14820, -4600, 4)",
    "(-14620, -4600, 4)",
    "(-14620, -4200, 4)",
    "(-14620, -4200, 2)",
    "[END]",
    "[END]",
    "[END]",
    "[END]",
    "[BRANCH]",
    "(-7220, -800, 4)",
    "(-7220, -600, 4)",
    "(-7220, -600, 2)",
    "(-7220, -600, 0)",
    "[END]",
    "[END]",
    "[BRANCH]",
    "(-5620, -800, 4)",
    "(-5620, -2200, 4)",
    "(-5820, -2200, 4)",
    "(-5820, -4600, 4)",
    "(-5820, -4600, 2)",
    "(-5600, -4600, 2)",
    "(-5600, -4400, 2)",
    "(-5290, -4400, 2)",
    "(-5290, -4400, 0)",
    "[END]",
    "[END]",
    "[BRANCH]",
    "(-620, -600, 2)",
    "[END]",
    "[END]",
]


class TreeVisualizer:
    def __init__(self):
        self.config = TokenizationStageConfig()
        self.config.workflow.tokenizer_algorithm = "None"
        self.unified_tokenizer = UnifiedTokenizer(self.config)
        self._tree_structure_cache = None
        self._graph_cache = None

    def build_tree_structure_from_tokens(self, tokens: list[str]):
        if self._tree_structure_cache is None:
            self._tree_structure_cache = self.unified_tokenizer.build_tree_structure(
                tokens
            )
        return self._tree_structure_cache

    def build_networkx_graph_from_tree(self, root_node) -> nx.DiGraph:
        if self._graph_cache is None:
            G = nx.DiGraph()

            def traverse_and_build_graph(node, node_id=0):
                if node is None or node.coord_str is None:
                    return node_id
                node_name = f"n{node_id}"
                G.add_node(node_name, label=node.coord_str, coord=node.coord)
                current_id = node_id + 1
                for child in node.children:
                    child_id = current_id
                    child_name = f"n{child_id}"
                    current_id = traverse_and_build_graph(child, child_id)
                    if child.coord_str is not None:
                        G.add_edge(node_name, child_name)
                return current_id

            traverse_and_build_graph(root_node)
            self._graph_cache = G
        return self._graph_cache

    def create_dfs_traversal_order(self, graph: nx.DiGraph) -> tuple[list, list]:
        adj = {u: list(graph.successors(u)) for u in graph.nodes()}
        dfs_nodes, dfs_edges = [], []

        def dfs(u):
            dfs_nodes.append(u)
            for v in adj.get(u, []):
                dfs_edges.append((u, v))
                dfs(v)

        if "n0" in graph.nodes():
            dfs("n0")
        return dfs_nodes, dfs_edges

    def create_token_sequence_for_animation(self, tokens: list[str]) -> list:
        sequence = []
        node_id = 0
        branch_stack = []
        last_node = None

        BRANCH_TOKEN = self.unified_tokenizer.get_special_token("BRANCH_TOKEN")
        END_TOKEN = self.unified_tokenizer.get_special_token("END_TOKEN")

        for tok in tokens:
            if tok == BRANCH_TOKEN:
                sequence.append(("BRANCH", last_node))
                branch_stack.append(last_node)
            elif tok == END_TOKEN:
                if branch_stack:
                    branch_node = branch_stack.pop()
                    sequence.append(("END", branch_node))
                    last_node = branch_node
            else:
                if self.unified_tokenizer.is_coordinate_string(tok):
                    node_name = f"n{node_id}"
                    sequence.append(("NODE", node_name, tok))
                    last_node = node_name
                    node_id += 1
        return sequence


def create_initial_frame():
    fig, ax = plt.subplots(figsize=(16, 18))
    fig.patch.set_facecolor(COLORS["background"])
    ax.axis("off")
    ax.set_title(
        "Net Routing Generation",
        fontsize=48,
        fontweight="bold",
        color=COLORS["text_default"],
        family="DejaVu Sans Mono",
        y=1.05,
    )
    return fig


def _wrap_routing_text(tokens: list[str], upto: int, width: int = 144) -> list[str]:
    current_input_str = " ".join(tokens[:upto]) if upto > 0 else ""
    full_text = "Routing: " + current_input_str
    lines = [full_text[i : i + width] for i in range(0, len(full_text), width)]
    return [line.ljust(width) for line in lines]


def _figure_to_rgb_array(fig) -> np.ndarray:
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    argb = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape(h, w, 4)
    rgb = argb[:, :, 1:4].copy()
    return rgb


def _create_axes(fig):
    gs = fig.add_gridspec(
        nrows=4,
        ncols=1,
        height_ratios=[0.1, 0.6, 0.25, 0.05],
        left=0.04,
        right=0.98,
        top=0.98,
        bottom=0.04,
        hspace=0.0,
    )
    ax_title = fig.add_subplot(gs[0, 0])
    ax_tree = fig.add_subplot(gs[1, 0])
    ax_routing = fig.add_subplot(gs[2, 0])
    ax_token = fig.add_subplot(gs[3, 0])

    for ax in (ax_title, ax_tree, ax_routing, ax_token):
        ax.set_facecolor(COLORS["background"])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)

    return {
        "title": ax_title,
        "tree": ax_tree,
        "routing": ax_routing,
        "token": ax_token,
    }


def _draw_frame_on_axes(
    axes,
    graph,
    pos,
    nodes_drawn,
    edges_drawn,
    current_token_info,
    original_tokens,
    xlim,
    ylim,
    blink_alpha=1.0,
    token_index=0,
):
    ax_title, ax_tree, ax_routing, ax_token = (
        axes["title"],
        axes["tree"],
        axes["routing"],
        axes["token"],
    )

    for ax in (ax_title, ax_tree, ax_routing, ax_token):
        ax.cla()
        ax.set_facecolor(COLORS["background"])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)

    ax_title.text(
        0.5,
        0.5,
        "Net Routing Generation",
        ha="center",
        va="center",
        fontsize=48,
        fontweight="bold",
        color=COLORS["text_default"],
        family="DejaVu Sans Mono",
        transform=ax_title.transAxes,
    )
    ax_title.set_xlim(0, 1)
    ax_title.set_ylim(0, 1)

    current_token = current_token_info[0] if current_token_info else None
    current_node = (
        current_token_info[1]
        if current_token_info and len(current_token_info) > 1
        else None
    )

    node_colors, node_edgecolors, linewidths = [], [], []
    for node in nodes_drawn:
        if node == current_node and current_token in ["BRANCH", "END"]:
            if current_token == "BRANCH":
                node_colors.append(COLORS["highlight_branch"])
                node_edgecolors.append(COLORS["highlight_branch"])
                linewidths.append(4 * blink_alpha)
            else:
                node_colors.append(COLORS["highlight_end"])
                node_edgecolors.append(COLORS["highlight_end"])
                linewidths.append(4 * blink_alpha)
        elif node == nodes_drawn[-1] and current_token == "NODE":
            node_colors.append(COLORS["node_current"])
            node_edgecolors.append(COLORS["node_current"])
            linewidths.append(2)
        else:
            node_colors.append(COLORS["node_visited"])
            node_edgecolors.append(COLORS["node_visited"])
            linewidths.append(1)

    if nodes_drawn:
        nx.draw_networkx_nodes(
            graph,
            pos,
            nodelist=nodes_drawn,
            node_color=node_colors,
            edgecolors=node_edgecolors,
            linewidths=linewidths,
            node_size=300,
            ax=ax_tree,
        )
        nx.draw_networkx_labels(
            graph,
            pos,
            labels={n: graph.nodes[n]["label"] for n in nodes_drawn},
            font_size=8,
            font_family="DejaVu Sans Mono",
            font_color=COLORS["text_default"],
            ax=ax_tree,
            clip_on=False,
        )
    if edges_drawn:
        nx.draw_networkx_edges(
            graph,
            pos,
            edgelist=edges_drawn,
            edge_color=COLORS["edge_active"],
            arrowstyle="-|>",
            arrowsize=10,
            width=2,
            ax=ax_tree,
        )

    ax_tree.set_xlim(*xlim)
    ax_tree.set_ylim(*ylim)
    ax_tree.set_anchor("C")
    ax_tree.margins(0)

    lines = _wrap_routing_text(original_tokens, token_index, width=144)
    y = 0.95
    for line in lines:
        ax_routing.text(
            0.02,
            y,
            line,
            ha="left",
            va="top",
            fontsize=12,
            fontweight="bold",
            color=COLORS["text_default"],
            family="DejaVu Sans Mono",
            transform=ax_routing.transAxes,
        )
        y -= 0.055
        if y < 0.02:
            break
    ax_routing.text(
        0.5,
        0.98,
        "-" * 144,
        ha="center",
        va="top",
        fontsize=12,
        fontweight="bold",
        color=COLORS["text_default"],
        family="DejaVu Sans Mono",
        transform=ax_routing.transAxes,
    )
    ax_routing.set_xlim(0, 1)
    ax_routing.set_ylim(0, 1)

    if current_token == "BRANCH":
        token_display, token_color = "BRANCH", COLORS["text_branch"]
    elif current_token == "END":
        token_display, token_color = "END", COLORS["text_end"]
    elif current_token == "NODE" and len(current_token_info) > 2:
        token_display, token_color = current_token_info[2], COLORS["text_default"]
    else:
        token_display, token_color = "NODE", COLORS["text_default"]

    ax_token.text(
        0.48,
        0.5,
        "Generate Token:",
        ha="right",
        va="center",
        fontsize=24,
        fontweight="bold",
        color=COLORS["text_default"],
        family="DejaVu Sans Mono",
        transform=ax_token.transAxes,
    )
    ax_token.text(
        0.52,
        0.5,
        token_display,
        ha="left",
        va="center",
        fontsize=24,
        fontweight="bold",
        color=token_color,
        alpha=blink_alpha if current_token in ["BRANCH", "END"] else 1.0,
        family="DejaVu Sans Mono",
        transform=ax_token.transAxes,
    )
    ax_token.set_xlim(0, 1)
    ax_token.set_ylim(0, 1)


def create_frame(
    graph,
    pos,
    nodes_drawn,
    edges_drawn,
    current_token_info,
    original_tokens,
    blink_alpha=1.0,
    token_index=0,
):
    fig = plt.figure(figsize=(16, 18), constrained_layout=False)
    fig.set_layout_engine(None)
    fig.patch.set_facecolor(COLORS["background"])
    axes = _create_axes(fig)
    _draw_frame_on_axes(
        axes,
        graph,
        pos,
        nodes_drawn,
        edges_drawn,
        current_token_info,
        original_tokens,
        xlim=(0, 1),
        ylim=(0, 1),
        blink_alpha=blink_alpha,
        token_index=token_index,
    )
    return fig


def create_animation(output_dir: Path):
    visualizer = TreeVisualizer()

    root_node = visualizer.build_tree_structure_from_tokens(input_tokens)
    graph = visualizer.build_networkx_graph_from_tree(root_node)
    pos = graphviz_layout(graph, prog="dot")

    dfs_nodes, dfs_edges = visualizer.create_dfs_traversal_order(graph)
    token_sequence = visualizer.create_token_sequence_for_animation(input_tokens)

    token_mapping = []
    tokens_index = 0
    for token_info in token_sequence:
        token_type = token_info[0]
        if token_type == "NODE":
            while tokens_index < len(input_tokens) and input_tokens[tokens_index] in [
                "[BRANCH]",
                "[END]",
            ]:
                tokens_index += 1
            token_mapping.append(
                tokens_index if tokens_index < len(input_tokens) else -1
            )
            if tokens_index < len(input_tokens):
                tokens_index += 1
        elif token_type in ["BRANCH", "END"]:
            target = f"[{token_type}]"
            while (
                tokens_index < len(input_tokens)
                and input_tokens[tokens_index] != target
            ):
                tokens_index += 1
            token_mapping.append(
                tokens_index if tokens_index < len(input_tokens) else -1
            )
            if tokens_index < len(input_tokens):
                tokens_index += 1

    if pos:
        x_coords = [x for x, y in pos.values()]
        y_coords = [y for x, y in pos.values()]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        x_span = max(1.0, x_max - x_min)
        y_span = max(1.0, y_max - y_min)
        pad_x = x_span * 0.06
        pad_y = y_span * 0.10
        xlim = (x_min - pad_x, x_max + pad_x)
        ylim = (y_min - pad_y, y_max + pad_y)
    else:
        xlim, ylim = (0, 1), (0, 1)

    fig = plt.figure(figsize=(16, 18))
    fig.patch.set_facecolor(COLORS["background"])
    axes = _create_axes(fig)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "dfs_tree.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 10

    _draw_frame_on_axes(
        axes,
        graph,
        pos,
        nodes_drawn=[],
        edges_drawn=[],
        current_token_info=("NODE", None, None),
        original_tokens=input_tokens,
        xlim=xlim,
        ylim=ylim,
        blink_alpha=1.0,
        token_index=0,
    )
    init_frame = _figure_to_rgb_array(fig)
    h, w = init_frame.shape[:2]
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    writer.write(cv2.cvtColor(init_frame, cv2.COLOR_RGB2BGR))

    for i, token_info in tqdm(
        enumerate(token_sequence), desc="Generating frames", total=len(token_sequence)
    ):
        token_type = token_info[0]
        current_token_index = token_mapping[i] if i < len(token_mapping) else 0

        if token_type == "NODE":
            node_name, node_coord = token_info[1], token_info[2]
            try:
                node_idx = dfs_nodes.index(node_name)
                nodes_to_draw = dfs_nodes[: node_idx + 1]
                edges_to_draw = dfs_edges[:node_idx] if node_idx > 0 else []
            except ValueError:
                continue
            current_token_info = ("NODE", node_name, node_coord)

        else:
            if i > 0:
                last_node_idx = -1
                for j in range(i - 1, -1, -1):
                    if token_sequence[j][0] == "NODE":
                        try:
                            last_node_idx = dfs_nodes.index(token_sequence[j][1])
                            break
                        except ValueError:
                            continue
                if last_node_idx >= 0:
                    nodes_to_draw = dfs_nodes[: last_node_idx + 1]
                    edges_to_draw = (
                        dfs_edges[:last_node_idx] if last_node_idx > 0 else []
                    )
                else:
                    nodes_to_draw, edges_to_draw = [], []
            else:
                nodes_to_draw, edges_to_draw = [], []
            current_token_info = (
                token_type,
                token_info[1] if len(token_info) > 1 else None,
            )

        _draw_frame_on_axes(
            axes,
            graph,
            pos,
            nodes_drawn=nodes_to_draw,
            edges_drawn=edges_to_draw,
            current_token_info=current_token_info,
            original_tokens=input_tokens,
            xlim=xlim,
            ylim=ylim,
            blink_alpha=1.0,
            token_index=current_token_index + 1,
        )
        frame = _figure_to_rgb_array(fig)
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        if token_type in ["BRANCH", "END"]:
            for _ in range(3):
                for alpha in (0.3, 1.0):
                    _draw_frame_on_axes(
                        axes,
                        graph,
                        pos,
                        nodes_drawn=nodes_to_draw,
                        edges_drawn=edges_to_draw,
                        current_token_info=current_token_info,
                        original_tokens=input_tokens,
                        xlim=xlim,
                        ylim=ylim,
                        blink_alpha=alpha,
                        token_index=current_token_index + 1,
                    )
                    frame = _figure_to_rgb_array(fig)
                    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    logging.info("Finalizing video...")
    writer.release()
    plt.close(fig)
    logging.info(f"Animation saved as {output_path}")


def main():
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Generate tree visualization animation"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/mnt/local_data1/liweiguo/experiments/demo/treeization"),
    )
    args = parser.parse_args()
    create_animation(args.output_dir)


if __name__ == "__main__":
    main()
