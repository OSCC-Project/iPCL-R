#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   rewards.py
@Time    :   2025/10/15 11:42:05
@Author  :   Dawn Li
@Version :   1.0
@Contact :   dawnli619215645@gmail.com
@Desc    :   Reward function implementations for GRPO fine-tuning experiments.
"""

import logging
import math
from abc import ABC, abstractmethod
from collections import Counter
from itertools import chain
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

from flow.evaluation.pipeline import compute_uniform_scale_factor, scale_trees_uniformly
from flow.tokenization import UnifiedTokenizer
from flow.tokenization.tokenizer import Node
from flow.utils import CoordinatePoint


class RewardCalculator(ABC):
    """Base interface for reward computation."""

    @abstractmethod
    def __call__(self, completions: Iterable[str], **kwargs) -> List[float]:
        """Compute rewards for a batch of completions."""
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement __call__ for"
            f" completions type {type(completions).__name__}"
            f" with kwargs {list(kwargs.keys())}"
        )

    @property
    def __name__(self) -> str:
        """Provide a readable name so GRPOTrainer can log callable identifiers."""
        return getattr(self, "_reward_name", self.__class__.__name__)


def _collect_edges(root: Node) -> List[Tuple[CoordinatePoint, CoordinatePoint]]:
    """Collect parent-child coordinate pairs from a routing tree."""
    edges = []
    stack = [root]
    while stack:
        node = stack.pop()
        for child in node.children:
            if node.coord and child.coord:
                edges.append((node.coord, child.coord))
            stack.append(child)
    return edges


def _get_all_coords(root: Node) -> List[CoordinatePoint]:
    """Get all coordinates in the tree using DFS."""
    coords = []
    stack = [root]
    while stack:
        node = stack.pop()
        if node.coord:
            coords.append(node.coord)
        for child in node.children:
            stack.append(child)
    return coords


def _get_leaves(
    edges: List[Tuple[CoordinatePoint, CoordinatePoint]], root_coord: CoordinatePoint
) -> List[CoordinatePoint]:
    """Get leaf nodes from edges.

    Args:
        edges: List of edges (parent, child) in the tree
        root_coord: Coordinate of the root node

    Returns:
        List of leaf node coordinates (points that appear only once and are not root)
    """
    if not edges:
        return []

    # Count occurrences of each point
    point_counts = Counter(chain.from_iterable(edges))
    # Leaf points are those that appear only once and are not root
    return [
        point
        for point, count in point_counts.items()
        if count == 1 and point != root_coord
    ]


def _manhattan_2d(src: CoordinatePoint, dst: CoordinatePoint) -> int:
    """2D Manhattan length between two coordinates (ignores metal layer delta)."""
    return abs(src.x - dst.x) + abs(src.y - dst.y)


class WirelengthReward(RewardCalculator):
    """Normalized wirelength improvement relative to ground truth (wirelength only)."""

    def __init__(
        self,
        tokenizer: UnifiedTokenizer,
        failure_penalty: float = -0.2,
        improvement_clip: float = 1.0,
    ):
        """
        Args:
            tokenizer: UnifiedTokenizer instance
            failure_penalty: Penalty for parsing failures (default: -0.2)
            improvement_clip: Clamp for normalized improvement value
        """
        self.tokenizer = tokenizer
        self.failure_penalty = failure_penalty
        self.improvement_clip = improvement_clip

    def _parse_wirelength(self, completion: str) -> float:
        """
        Parse completion and compute wirelength only.

        Args:
            completion: Token string to parse

        Returns:
            Wirelength value
        """
        if not completion or not completion.strip():
            raise ValueError("Empty completion")

        routing_sequence = self.tokenizer.convert_tokens_to_routing(completion)
        tree = self.tokenizer.build_tree_structure(routing_sequence)

        if tree is None or tree.coord is None:
            raise ValueError("Failed to build tree structure")

        edges = _collect_edges(tree)
        if not edges:
            return 0.0

        wirelength = sum(abs(src.x - dst.x) + abs(src.y - dst.y) for src, dst in edges)
        return wirelength

    def _score_single(
        self,
        completion: str,
        target_tokens: str,
    ) -> float:
        """
        Compute reward for a single completion against ground truth.

        Args:
            completion: Predicted token string
            target_tokens: Ground truth token string

        Returns:
            Reward score
        """
        if not completion or not completion.strip():
            return self.failure_penalty

        try:
            pred_wire = self._parse_wirelength(completion)
            gt_wire = self._parse_wirelength(target_tokens)
        except Exception as exc:  # pragma: no cover - defensive logging
            logging.debug("Failed to evaluate completion '%s': %s", completion, exc)
            return self.failure_penalty

        if gt_wire <= 0.0 or pred_wire <= 0.0:
            logging.debug(
                "Invalid wirelength (gt=%s, pred=%s), returning failure_penalty",
                gt_wire,
                pred_wire,
            )
            return self.failure_penalty

        denom = max(gt_wire, pred_wire, 1e-6)
        wire_improvement = (gt_wire - pred_wire) / denom
        wire_improvement = max(
            -self.improvement_clip, min(self.improvement_clip, wire_improvement)
        )
        return wire_improvement

    def __call__(self, completions: Iterable[str], **kwargs) -> List[float]:
        """
        Compute rewards for a batch of completions.

        Args:
            completions: Iterable of completion strings
            **kwargs: Must contain 'target_tokens' (List[str])

        Returns:
            List of reward scores
        """
        target_tokens = kwargs.get("target_tokens")
        if target_tokens is None:
            raise ValueError("Missing required argument 'target_tokens' in kwargs")

        completions_list = list(completions)
        if len(completions_list) != len(target_tokens):
            raise ValueError(
                f"Number of completions ({len(completions_list)}) must match "
                f"number of target_tokens ({len(target_tokens)})"
            )

        return [
            self._score_single(comp, target)
            for comp, target in zip(completions_list, target_tokens)
        ]


class AdaptiveWirelengthViaReward(RewardCalculator):
    """Normalized wirelength improvement relative to ground truth."""

    def __init__(
        self,
        tokenizer: UnifiedTokenizer,
        via_weight: float = 0.3,
        failure_penalty: float = -0.2,
        improvement_clip: float = 1.0,
    ):
        """
        Args:
            tokenizer: UnifiedTokenizer instance
            via_weight: Weight applied to the via-count improvement term
            failure_penalty: Penalty for parsing failures (default: -0.2)
            improvement_clip: Clamp for normalized improvement value
        """
        self.tokenizer = tokenizer
        self.via_weight = via_weight
        self.failure_penalty = failure_penalty
        self.improvement_clip = improvement_clip

    def _parse_metrics(self, completion: str) -> Tuple[float, float]:
        """
        Parse completion and compute wirelength and vias.

        Args:
            completion: Token string to parse

        Returns:
            Tuple of (wirelength, vias)
        """
        if not completion or not completion.strip():
            raise ValueError("Empty completion")

        routing_sequence = self.tokenizer.convert_tokens_to_routing(completion)
        tree = self.tokenizer.build_tree_structure(routing_sequence)

        if tree is None or tree.coord is None:
            raise ValueError("Failed to build tree structure")

        edges = _collect_edges(tree)
        if not edges:
            return 0.0, 0.0

        wirelength = sum(abs(src.x - dst.x) + abs(src.y - dst.y) for src, dst in edges)
        vias = sum(abs(src.m - dst.m) / 2.0 for src, dst in edges)

        return wirelength, vias

    def _score_single(
        self,
        completion: str,
        target_tokens: str,
    ) -> float:
        """
        Compute reward for a single completion against ground truth.

        Args:
            completion: Predicted token string
            target_tokens: Ground truth token string

        Returns:
            Reward score
        """
        if not completion or not completion.strip():
            return self.failure_penalty

        try:
            pred_wire, pred_via = self._parse_metrics(completion)
            gt_wire, gt_via = self._parse_metrics(target_tokens)
        except Exception as exc:  # pragma: no cover - defensive logging
            logging.debug("Failed to evaluate completion '%s': %s", completion, exc)
            return self.failure_penalty

        if gt_wire <= 0.0 or pred_wire <= 0.0:
            logging.debug(
                "Invalid wirelength (gt=%s, pred=%s), returning failure_penalty",
                gt_wire,
                pred_wire,
            )
            return self.failure_penalty

        denom = max(gt_wire, pred_wire, 1e-6)
        wire_improvement = (gt_wire - pred_wire) / denom

        if gt_via <= 0.0 and pred_via <= 0.0:
            via_improvement = 0.0
        else:
            via_denom = max(gt_via, pred_via, 1e-6)
            via_improvement = (gt_via - pred_via) / via_denom

        improvement = wire_improvement + self.via_weight * via_improvement
        improvement = max(
            -self.improvement_clip, min(self.improvement_clip, improvement)
        )
        return improvement

    def __call__(self, completions: Iterable[str], **kwargs) -> List[float]:
        """
        Compute rewards for a batch of completions.

        Args:
            completions: Iterable of completion strings
            **kwargs: Must contain 'target_tokens' (List[str])

        Returns:
            List of reward scores
        """
        target_tokens = kwargs.get("target_tokens")
        if target_tokens is None:
            raise ValueError("Missing required argument 'target_tokens' in kwargs")

        completions_list = list(completions)
        if len(completions_list) != len(target_tokens):
            raise ValueError(
                f"Number of completions ({len(completions_list)}) must match "
                f"number of target_tokens ({len(target_tokens)})"
            )

        return [
            self._score_single(comp, target)
            for comp, target in zip(completions_list, target_tokens)
        ]


class ConnectivityReward(RewardCalculator):
    """Reward for checking if all loads are connected."""

    def __init__(
        self,
        tokenizer: UnifiedTokenizer,
        penalty: float = -1.0,
        use_continuous: bool = False,
        failure_penalty: float = -1.0,
    ):
        """
        Args:
            tokenizer: UnifiedTokenizer instance
            penalty: Penalty when not all loads connected (default: -1.0)
            use_continuous: Use continuous reward (connected_ratio) instead of binary (default: False)
            failure_penalty: Penalty for parsing failures (default: -1.0)
        """
        self.tokenizer = tokenizer
        self.penalty = penalty
        self.use_continuous = use_continuous
        self.failure_penalty = failure_penalty

    def _check_connectivity(
        self, pred_coords: List[CoordinatePoint], relative_loads: List[str]
    ) -> Tuple[bool, float]:
        """
        Check connectivity between predicted tree and relative loads.

        Args:
            pred_coords: List of coordinates in the predicted tree
            relative_loads: List of relative load coordinate strings

        Returns:
            Tuple of (all_loads_connected, connected_ratio)
        """
        # Convert to set for efficient lookup
        pred_coords_set = set(pred_coords)

        # Parse relative load coordinates
        relative_load_coords = set()
        for relative_load in relative_loads:
            try:
                coord = self.tokenizer.parse_coord(relative_load)
                relative_load_coords.add(coord)
            except Exception as exc:
                logging.debug(
                    "Failed to parse relative load '%s': %s", relative_load, exc
                )
                continue

        # Check connectivity
        if not relative_load_coords:
            # If no valid relative loads, consider as fully connected
            return True, 1.0

        all_loads_connected = relative_load_coords.issubset(pred_coords_set)
        connected_count = len(relative_load_coords.intersection(pred_coords_set))
        connected_ratio = connected_count / len(relative_load_coords)

        return all_loads_connected, connected_ratio

    def _score_single(self, completion: str, relative_loads: List[str]) -> float:
        """
        Compute connectivity reward for a single completion.

        Args:
            completion: The generated completion string
            relative_loads: List of relative load coordinate strings

        Returns:
            Reward value
        """
        # Handle empty relative loads
        if not relative_loads:
            return 1.0

        # Handle empty completion
        if not completion or not completion.strip():
            return self.failure_penalty

        try:
            # Parse completion to tree
            routing_sequence = self.tokenizer.convert_tokens_to_routing(completion)
            tree = self.tokenizer.build_tree_structure(routing_sequence)

            if tree is None or tree.coord is None:
                logging.debug("Failed to build tree from completion")
                return self.failure_penalty

            # Get all coordinates from tree
            pred_coords = _get_all_coords(tree)

            # Check connectivity
            all_loads_connected, connected_ratio = self._check_connectivity(
                pred_coords, relative_loads
            )

            # Compute reward
            if self.use_continuous:
                reward = connected_ratio
            else:
                reward = 1.0 if all_loads_connected else self.penalty

            logging.debug(
                "Connectivity check: all_connected=%s, ratio=%.3f, reward=%.3f",
                all_loads_connected,
                connected_ratio,
                reward,
            )

            return reward

        except Exception as exc:
            logging.debug("Failed to evaluate connectivity for completion: %s", exc)
            return self.failure_penalty

    def __call__(self, completions: Iterable[str], **kwargs) -> List[float]:
        """
        Compute rewards for a batch of completions.

        Args:
            completions: Batch of completion strings
            **kwargs: Must contain 'relative_loads' (List[List[str]])

        Returns:
            List of reward values
        """
        relative_loads_batch = kwargs.get("relative_loads", [])

        if not relative_loads_batch:
            logging.warning("No relative_loads provided, returning default rewards")
            return [1.0] * len(list(completions))

        completions_list = list(completions)

        if len(completions_list) != len(relative_loads_batch):
            raise ValueError(
                f"Number of completions ({len(completions_list)}) must match "
                f"number of relative_loads ({len(relative_loads_batch)})"
            )

        return [
            self._score_single(completion, relative_loads)
            for completion, relative_loads in zip(
                completions_list, relative_loads_batch
            )
        ]


class GracefulReward(RewardCalculator):
    """Reward for checking if the routing is graceful (all loads connected AND all leaves are useful)."""

    def __init__(
        self,
        tokenizer: UnifiedTokenizer,
        penalty: float = -1.0,
        failure_penalty: float = -1.0,
        use_continuous: bool = False,
    ):
        """
        Args:
            tokenizer: UnifiedTokenizer instance
            penalty: Penalty when not graceful (default: -1.0)
            failure_penalty: Penalty for parsing failures (default: -1.0)
            use_continuous: Return a continuous score instead of binary pass/fail.
        """
        self.tokenizer = tokenizer
        self.penalty = penalty
        self.failure_penalty = failure_penalty
        self.use_continuous = use_continuous

    def _check_graceful(self, completion: str, relative_loads: List[str]) -> bool:
        """Check if routing is graceful.

        A routing is graceful if:
        1. All loads are connected (relative_loads are subset of pred_coords)
        2. All leaves are useful (pred_leaves are subset of relative_loads)

        Args:
            completion: Routing sequence string
            relative_loads: List of relative load coordinate strings

        Returns:
            True if graceful, False otherwise
        """
        # Parse routing sequence and build tree
        routing_sequence = self.tokenizer.convert_tokens_to_routing(completion)
        pred_tree = self.tokenizer.build_tree_structure(routing_sequence)

        if pred_tree is None or pred_tree.coord is None:
            return False

        # Collect edges and leaves
        pred_edges = _collect_edges(pred_tree)
        pred_leaves = _get_leaves(pred_edges, pred_tree.coord)
        pred_leaves_set = set(pred_leaves)

        # Get all coordinates in predicted tree
        pred_coords = set(_get_all_coords(pred_tree))

        # Parse relative load coordinates
        relative_load_coords = {
            self.tokenizer.parse_coord(relative_load)
            for relative_load in relative_loads
        }

        # Check graceful conditions
        all_loads_connected = relative_load_coords.issubset(pred_coords)
        all_leaves_is_useful = pred_leaves_set.issubset(relative_load_coords)
        is_graceful = all_loads_connected and all_leaves_is_useful

        return is_graceful

    def _continuous_score(
        self, completion: str, relative_loads: List[str]
    ) -> float:
        """Return a continuous graceful score in [0, 1]."""
        routing_sequence = self.tokenizer.convert_tokens_to_routing(completion)
        pred_tree = self.tokenizer.build_tree_structure(routing_sequence)

        if pred_tree is None or pred_tree.coord is None:
            return self.failure_penalty

        pred_edges = _collect_edges(pred_tree)
        pred_leaves = _get_leaves(pred_edges, pred_tree.coord)
        pred_leaves_set = set(pred_leaves)

        pred_coords = set(_get_all_coords(pred_tree))

        relative_load_coords = set()
        for relative_load in relative_loads:
            try:
                relative_load_coords.add(self.tokenizer.parse_coord(relative_load))
            except Exception as exc:
                logging.debug(
                    "Failed to parse relative load '%s': %s", relative_load, exc
                )

        if not relative_load_coords:
            return 1.0 if not pred_leaves_set else 0.0

        connected_count = len(relative_load_coords.intersection(pred_coords))
        connected_ratio = connected_count / len(relative_load_coords)

        if pred_leaves_set:
            useful_leaf_ratio = len(
                pred_leaves_set.intersection(relative_load_coords)
            ) / len(pred_leaves_set)
        else:
            useful_leaf_ratio = 0.0

        graceful_ratio = 0.5 * connected_ratio + 0.5 * useful_leaf_ratio
        return graceful_ratio

    def _score_single(self, completion: str, relative_loads: List[str]) -> float:
        """Score a single completion.

        Args:
            completion: Routing sequence string
            relative_loads: List of relative load coordinate strings

        Returns:
            Reward value (1.0 if graceful, penalty otherwise, failure_penalty if error)
        """
        if not completion or not completion.strip():
            return self.failure_penalty

        if not relative_loads:
            logging.debug("Empty relative_loads, returning failure_penalty")
            return self.failure_penalty

        try:
            if self.use_continuous:
                reward = self._continuous_score(completion, relative_loads)
                logging.debug(
                    "Completion graceful_ratio=%.3f, reward=%.3f", reward, reward
                )
                return reward

            is_graceful = self._check_graceful(completion, relative_loads)
            reward = 1.0 if is_graceful else self.penalty
            logging.debug("Completion graceful=%s, reward=%.2f", is_graceful, reward)
            return reward
        except Exception as exc:  # pragma: no cover - defensive logging
            logging.debug("Failed to evaluate completion '%s': %s", completion, exc)
            return self.failure_penalty

    def __call__(self, completions: Iterable[str], **kwargs) -> List[float]:
        """Compute rewards for a batch of completions.

        Args:
            completions: Iterable of routing sequence strings
            **kwargs: Must contain 'relative_loads' (List[List[str]])

        Returns:
            List of reward values
        """
        relative_loads_batch = kwargs.get("relative_loads", None)
        if relative_loads_batch is None:
            logging.warning(
                "Missing 'relative_loads' in kwargs, returning failure penalties"
            )
            return [self.failure_penalty] * len(list(completions))

        completions_list = list(completions)
        if len(relative_loads_batch) != len(completions_list):
            logging.warning(
                "Mismatch between completions (%d) and relative_loads (%d), returning failure penalties",
                len(completions_list),
                len(relative_loads_batch),
            )
            return [self.failure_penalty] * len(completions_list)

        return [
            self._score_single(text, loads)
            for text, loads in zip(completions_list, relative_loads_batch)
        ]


class ElmoreDelayReward(RewardCalculator):
    """Normalized Elmore delay improvement using max sink delay."""

    def __init__(
        self,
        tokenizer: UnifiedTokenizer,
        unit_resistance: float = 1.0,
        unit_capacitance: float = 1.0,
        load_capacitance: float = 1.0,
        db_unit: float = 2000.0,
        failure_penalty: float = -0.2,
        improvement_clip: float = 1.0,
        improvement_scale: float = 1.0,
    ):
        """
        Args:
            tokenizer: UnifiedTokenizer instance.
            unit_resistance: Resistance per unit Manhattan length (ohms, normalized).
            unit_capacitance: Capacitance per unit Manhattan length (F, normalized).
            load_capacitance: Lumped load capacitance placed on each sink (F, normalized).
            db_unit: DEF/DBU per micron (or grid-to-physical scaling). Manhattan length
                will be divided by this value before applying R/C; defaults to 2000.
            failure_penalty: Penalty when parsing fails or delay is invalid.
            improvement_clip: Clamp for normalized improvement value.
            improvement_scale: Scale factor applied to normalized improvement.
        """
        self.tokenizer = tokenizer
        self.unit_resistance = unit_resistance
        self.unit_capacitance = unit_capacitance
        self.load_capacitance = load_capacitance
        self.db_unit = db_unit if db_unit and db_unit > 0 else 2000.0
        self.failure_penalty = failure_penalty
        self.improvement_clip = improvement_clip
        self.improvement_scale = improvement_scale

    def _parse_tree(self, completion: str) -> Node:
        """Parse a completion into a routing tree."""
        routing_sequence = self.tokenizer.convert_tokens_to_routing(completion)
        tree = self.tokenizer.build_tree_structure(routing_sequence)
        if tree is None or tree.coord is None:
            raise ValueError("Failed to build tree structure")
        return tree

    def _collect_load_coordinates(
        self,
        relative_loads: Optional[List[str]],
        tree: Node,
        scale_factor: float = 1.0,
    ) -> Set[CoordinatePoint]:
        """Prefer user-provided loads; fall back to tree leaves."""
        load_coords: Set[CoordinatePoint] = set()
        if relative_loads:
            for relative_load in relative_loads:
                try:
                    coord = self.tokenizer.parse_coord(relative_load)
                    if scale_factor != 1.0:
                        coord = CoordinatePoint(
                            coord.x, coord.y, int(coord.m * scale_factor)
                        )
                    load_coords.add(coord)
                except Exception as exc:
                    logging.debug(
                        "Failed to parse relative load '%s': %s", relative_load, exc
                    )

        if not load_coords:
            edges = _collect_edges(tree)
            leaves = _get_leaves(edges, tree.coord) if edges else []
            if leaves:
                load_coords.update(leaves)
            elif tree.coord:
                load_coords.add(tree.coord)

        return load_coords

    def _compute_delay_from_tree(
        self, tree: Node, load_coords: Set[CoordinatePoint]
    ) -> float:
        """Compute max Elmore delay to any sink in a prepared tree."""
        subtree_caps: Dict[Node, float] = {}

        def accumulate_capacitance(node: Node) -> float:
            if node.coord is None:
                return 0.0
            total_cap = self.load_capacitance if node.coord in load_coords else 0.0
            for child in node.children:
                if child.coord is None:
                    continue
                edge_length = _manhattan_2d(node.coord, child.coord)
                physical_len = edge_length / self.db_unit
                edge_cap = physical_len * self.unit_capacitance
                total_cap += edge_cap
                total_cap += accumulate_capacitance(child)
            subtree_caps[node] = total_cap
            return total_cap

        accumulate_capacitance(tree)

        delays: List[float] = []

        def accumulate_delay(node: Node, upstream_delay: float) -> None:
            if node.coord is None:
                return
            for child in node.children:
                if child.coord is None:
                    continue
                edge_length = _manhattan_2d(node.coord, child.coord)
                physical_len = edge_length / self.db_unit
                edge_resistance = physical_len * self.unit_resistance
                edge_cap = physical_len * self.unit_capacitance
                downstream_cap = subtree_caps.get(child, 0.0) + edge_cap
                edge_delay = edge_resistance * downstream_cap
                total_delay = upstream_delay + edge_delay
                if child.coord in load_coords or not child.children:
                    delays.append(total_delay)
                accumulate_delay(child, total_delay)

        accumulate_delay(tree, 0.0)

        if not delays:
            return 0.0
        return max(delays)

    def _compute_max_delay(
        self, completion: str, relative_loads: Optional[List[str]]
    ) -> float:
        """Compute max Elmore delay to any sink in the tree."""
        if not completion or not completion.strip():
            raise ValueError("Empty completion")

        tree = self._parse_tree(completion)
        load_coords = self._collect_load_coordinates(relative_loads, tree)
        return self._compute_delay_from_tree(tree, load_coords)

    def _score_single(
        self,
        completion: str,
        target_tokens: str,
        relative_loads: Optional[List[str]],
    ) -> float:
        """Reward predicted delay improvement relative to ground truth."""
        if not completion or not completion.strip():
            return self.failure_penalty

        try:
            pred_tree = self._parse_tree(completion)
            gt_tree = self._parse_tree(target_tokens)

            scale_factor = compute_uniform_scale_factor(pred_tree, gt_tree)
            scaled_pred_tree, scaled_gt_tree = scale_trees_uniformly(
                pred_tree, gt_tree, scale_factor
            )

            pred_load_coords = self._collect_load_coordinates(
                relative_loads, scaled_pred_tree, scale_factor
            )
            gt_load_coords = self._collect_load_coordinates(
                relative_loads, scaled_gt_tree, scale_factor
            )

            pred_delay = self._compute_delay_from_tree(
                scaled_pred_tree, pred_load_coords
            )
            gt_delay = self._compute_delay_from_tree(scaled_gt_tree, gt_load_coords)
        except Exception as exc:  # pragma: no cover - defensive logging
            logging.debug("Failed to evaluate completion '%s': %s", completion, exc)
            return self.failure_penalty

        if gt_delay <= 0.0 or pred_delay <= 0.0:
            logging.debug(
                "Invalid delay (gt=%s, pred=%s), returning failure_penalty",
                gt_delay,
                pred_delay,
            )
            return self.failure_penalty

        denom = max(gt_delay, pred_delay, 1e-6)
        improvement = (gt_delay - pred_delay) / denom
        improvement = max(-self.improvement_clip, min(self.improvement_clip, improvement))
        return improvement * self.improvement_scale

    def __call__(self, completions: Iterable[str], **kwargs) -> List[float]:
        target_tokens = kwargs.get("target_tokens")
        if target_tokens is None:
            raise ValueError("Missing required argument 'target_tokens' in kwargs")

        relative_loads_batch = kwargs.get("relative_loads")
        completions_list = list(completions)

        if len(completions_list) != len(target_tokens):
            raise ValueError(
                f"Number of completions ({len(completions_list)}) must match "
                f"number of target_tokens ({len(target_tokens)})"
            )

        if relative_loads_batch is None:
            relative_loads_batch = [None] * len(completions_list)
        elif len(relative_loads_batch) != len(completions_list):
            raise ValueError(
                f"Number of completions ({len(completions_list)}) must match "
                f"number of relative_loads ({len(relative_loads_batch)})"
            )

        return [
            self._score_single(comp, target, relative_loads_batch[idx])
            for idx, (comp, target) in enumerate(
                zip(completions_list, target_tokens)
            )
        ]


class GatedCompositeReward(RewardCalculator):
    """Composite reward that gates an improvement metric by structural success."""

    def __init__(
        self,
        improvement_reward: RewardCalculator,
        gate_rewards: Dict[str, RewardCalculator],
        direction_reward: Optional[RewardCalculator] = None,
        reward_name: Optional[str] = None,
        improvement_label: str = "improvement",
        direction_label: str = "direction",
    ):
        """Initialize the gated composite reward.

        Args:
            improvement_reward: Primary improvement reward instance (positive is better).
            gate_rewards: Mapping of gate names to reward instances used for masking.
            direction_reward: Optional reward that determines sign of the final mask.
            reward_name: Friendly display name for logging.
            improvement_label: Label used when logging the improvement component.
            direction_label: Label used when logging the direction_reward component.
        """
        self.improvement_reward = improvement_reward
        self.gate_rewards = gate_rewards
        self.direction_reward = direction_reward
        self.improvement_label = improvement_label or "improvement"
        self.direction_label = direction_label
        self.last_components: List[Dict[str, float]] = []
        self._reward_name = reward_name or "GatedCompositeReward"

    @staticmethod
    def _clamp_positive(value: float) -> float:
        """Clamp reward component to [0, 1] for masking."""
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return 0.0
        if math.isnan(numeric) or math.isinf(numeric):
            return 0.0
        return max(0.0, min(1.0, numeric))

    def __call__(self, completions: Iterable[str], **kwargs) -> List[float]:
        completions_list = list(completions)
        if not completions_list:
            self.last_components = []
            return []

        improvement_scores = self.improvement_reward(completions_list, **kwargs)
        gate_scores = {
            name: reward(completions_list, **kwargs)
            for name, reward in self.gate_rewards.items()
        }
        direction_scores = (
            self.direction_reward(completions_list, **kwargs)
            if self.direction_reward
            else None
        )

        rewards: List[float] = []
        components: List[Dict[str, float]] = []

        for idx, improvement in enumerate(improvement_scores):
            improvement_value = float(improvement)
            mask = 1.0
            component_values: Dict[str, float] = {
                self.improvement_label: improvement_value,
                "improvement": improvement_value,
            }

            for gate_name, scores in gate_scores.items():
                gate_value = float(scores[idx])
                gate_mask = self._clamp_positive(gate_value)
                component_values[gate_name] = gate_value
                component_values[f"{gate_name}_mask"] = gate_mask
                mask *= gate_mask

            if direction_scores is not None:
                direction_value = float(direction_scores[idx])
                direction_mask = 1.0 if direction_value > 0.0 else -1.0
                component_values[self.direction_label] = direction_value
                component_values[f"{self.direction_label}_mask"] = direction_mask
                mask *= direction_mask

            final_reward = (
                mask * max(improvement_value, 0.0)
                if mask > 0.0
                else min(improvement_value, 0.0)
            )

            rewards.append(final_reward)
            component_values["mask"] = mask
            component_values["final"] = final_reward
            components.append(component_values)

        self.last_components = components
        return rewards


class WeightedCompositeReward(RewardCalculator):
    """Composite reward that combines components via weighted sum."""

    def __init__(
        self,
        improvement_reward: RewardCalculator,
        gate_rewards: Dict[str, RewardCalculator],
        improvement_weight: float = 1.0,
        gate_weights: Optional[Dict[str, float]] = None,
        direction_reward: Optional[RewardCalculator] = None,
        direction_weight: float = 0.0,
        reward_name: Optional[str] = None,
        improvement_label: str = "improvement",
        direction_label: str = "direction",
    ):
        self.improvement_reward = improvement_reward
        self.gate_rewards = gate_rewards
        self.improvement_weight = improvement_weight
        self.gate_weights = gate_weights or {}
        self.direction_reward = direction_reward
        self.direction_weight = direction_weight
        self.improvement_label = improvement_label or "improvement"
        self.direction_label = direction_label
        self.last_components: List[Dict[str, float]] = []
        self._reward_name = reward_name or "WeightedCompositeReward"

    def __call__(self, completions: Iterable[str], **kwargs) -> List[float]:
        completions_list = list(completions)
        if not completions_list:
            self.last_components = []
            return []

        improvement_scores = self.improvement_reward(completions_list, **kwargs)
        gate_scores = {
            name: reward(completions_list, **kwargs)
            for name, reward in self.gate_rewards.items()
        }
        direction_scores = (
            self.direction_reward(completions_list, **kwargs)
            if self.direction_reward
            else None
        )

        rewards: List[float] = []
        components: List[Dict[str, float]] = []

        for idx, improvement in enumerate(improvement_scores):
            improvement_value = float(improvement)
            total = self.improvement_weight * improvement_value
            component_values: Dict[str, float] = {
                self.improvement_label: improvement_value,
                "improvement": improvement_value,
                f"{self.improvement_label}_weighted": total,
            }

            for gate_name, scores in gate_scores.items():
                gate_value = float(scores[idx])
                weight = float(self.gate_weights.get(gate_name, 0.0))
                total += weight * gate_value
                component_values[gate_name] = gate_value
                component_values[f"{gate_name}_weighted"] = weight * gate_value

            if direction_scores is not None:
                direction_value = float(direction_scores[idx])
                total += self.direction_weight * direction_value
                component_values[self.direction_label] = direction_value
                component_values[f"{self.direction_label}_weighted"] = (
                    self.direction_weight * direction_value
                )

            component_values["final"] = total
            rewards.append(total)
            components.append(component_values)

        self.last_components = components
        return rewards


REWARD_BUILDERS: Dict[str, Callable[..., RewardCalculator]] = {
    "wirelength": WirelengthReward,
    "adaptive_wl_via": AdaptiveWirelengthViaReward,
    "connectivity": ConnectivityReward,
    "graceful": GracefulReward,
    "elmore_delay": ElmoreDelayReward,
}


def create_reward(
    name: str,
    tokenizer: UnifiedTokenizer,
    **kwargs,
) -> RewardCalculator:
    """Instantiate a reward calculator by name."""
    key = (name or "wirelength").lower()

    if key == "gated_wl_composite":
        adaptive_kwargs = kwargs.pop("adaptive_kwargs", {})
        connectivity_kwargs = kwargs.pop("connectivity_kwargs", {})
        graceful_kwargs = kwargs.pop("graceful_kwargs", {})
        wirelength_kwargs = kwargs.pop("wirelength_kwargs", {})

        adaptive = AdaptiveWirelengthViaReward(tokenizer=tokenizer, **adaptive_kwargs)
        connectivity = ConnectivityReward(tokenizer=tokenizer, **connectivity_kwargs)
        graceful = GracefulReward(tokenizer=tokenizer, **graceful_kwargs)
        wirelength = WirelengthReward(tokenizer=tokenizer, **wirelength_kwargs)

        return GatedCompositeReward(
            improvement_reward=adaptive,
            gate_rewards={
                "connectivity": connectivity,
                "graceful": graceful,
            },
            direction_reward=wirelength,
            reward_name="GatedWLCompositeReward",
            improvement_label="adaptive",
            direction_label="wirelength",
        )

    if key == "gated_timing_composite":
        elmore_kwargs = kwargs.pop("elmore_kwargs", {})
        connectivity_kwargs = kwargs.pop("connectivity_kwargs", {})
        graceful_kwargs = kwargs.pop("graceful_kwargs", {})

        elmore = ElmoreDelayReward(tokenizer=tokenizer, **elmore_kwargs)
        connectivity = ConnectivityReward(tokenizer=tokenizer, **connectivity_kwargs)
        graceful = GracefulReward(tokenizer=tokenizer, **graceful_kwargs)

        return GatedCompositeReward(
            improvement_reward=elmore,
            gate_rewards={
                "connectivity": connectivity,
                "graceful": graceful,
            },
            direction_reward=None,
            reward_name="GatedTimingCompositeReward",
            improvement_label="elmore_delay",
        )

    if key == "weighted_timing_composite":
        elmore_kwargs = kwargs.pop("elmore_kwargs", {})
        connectivity_kwargs = kwargs.pop("connectivity_kwargs", {})
        graceful_kwargs = kwargs.pop("graceful_kwargs", {})
        composite_weights = kwargs.pop("composite_weights", None)

        if composite_weights is None:
            composite_weights = (0.8, 0.1, 0.1)
        if len(composite_weights) != 3:
            raise ValueError(
                "composite_weights must be a tuple of three floats: improvement, connectivity, graceful."
            )

        elmore = ElmoreDelayReward(tokenizer=tokenizer, **elmore_kwargs)
        connectivity = ConnectivityReward(tokenizer=tokenizer, **connectivity_kwargs)
        graceful = GracefulReward(tokenizer=tokenizer, **graceful_kwargs)

        return WeightedCompositeReward(
            improvement_reward=elmore,
            gate_rewards={
                "connectivity": connectivity,
                "graceful": graceful,
            },
            improvement_weight=float(composite_weights[0]),
            gate_weights={
                "connectivity": float(composite_weights[1]),
                "graceful": float(composite_weights[2]),
            },
            direction_reward=None,
            direction_weight=0.0,
            reward_name="WeightedTimingCompositeReward",
            improvement_label="elmore_delay",
        )

    if key not in REWARD_BUILDERS:
        raise ValueError(f"Unsupported reward type: {name}")
    return REWARD_BUILDERS[key](tokenizer=tokenizer, **kwargs)
