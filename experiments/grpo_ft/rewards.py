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
from typing import Callable, Dict, Iterable, List, Tuple

from flow.tokenization import UnifiedTokenizer
from flow.tokenization.tokenizer import Node
from flow.utils import CoordinatePoint


class RewardCalculator(ABC):
    """Base interface for reward computation."""

    @abstractmethod
    def __call__(self, completions: Iterable[str], **kwargs) -> List[float]:
        """Compute rewards for a batch of completions."""

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
    ):
        """
        Args:
            tokenizer: UnifiedTokenizer instance
            penalty: Penalty when not graceful (default: -1.0)
            failure_penalty: Penalty for parsing failures (default: -1.0)
        """
        self.tokenizer = tokenizer
        self.penalty = penalty
        self.failure_penalty = failure_penalty

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


class GatedCompositeReward(RewardCalculator):
    """Composite reward that gates wirelength improvement by structural success."""

    def __init__(
        self,
        adaptive_reward: AdaptiveWirelengthViaReward,
        connectivity_reward: ConnectivityReward,
        graceful_reward: GracefulReward,
        wirelength_reward: WirelengthReward,
    ):
        """Initialize the gated composite reward.

        Args:
            adaptive_reward: Wirelength+via improvement reward instance.
            connectivity_reward: Connectivity checker (returns 1 or penalty).
            graceful_reward: Gracefulness checker (returns 1 or penalty).
            wirelength_reward: Wirelength-only reward instance for gating.
        """
        self.adaptive_reward = adaptive_reward
        self.connectivity_reward = connectivity_reward
        self.graceful_reward = graceful_reward
        self.wirelength_reward = wirelength_reward
        self.last_components: List[Dict[str, float]] = []
        self._reward_name = "GatedCompositeReward"

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

        adaptive_scores = self.adaptive_reward(completions_list, **kwargs)
        connectivity_scores = self.connectivity_reward(completions_list, **kwargs)
        graceful_scores = self.graceful_reward(completions_list, **kwargs)
        wirelength_scores = self.wirelength_reward(completions_list, **kwargs)

        rewards: List[float] = []
        components: List[Dict[str, float]] = []

        for adaptive, connectivity, graceful, wirelength in zip(
            adaptive_scores, connectivity_scores, graceful_scores, wirelength_scores
        ):
            adaptive_value = float(adaptive)
            wirelength_value = float(wirelength)
            improvement_value = adaptive_value
            connectivity_mask = self._clamp_positive(connectivity)
            graceful_mask = self._clamp_positive(graceful)
            wirelength_mask = 1.0 if wirelength_value > 0.0 else -1.0
            mask = connectivity_mask * graceful_mask * wirelength_mask

            final_reward = (
                mask * max(adaptive_value, 0.0)
                if mask > 0.0
                else min(adaptive_value, 0.0)
            )

            rewards.append(final_reward)
            components.append(
                {
                    "adaptive": float(adaptive),
                    "connectivity": float(connectivity),
                    "graceful": float(graceful),
                    "wirelength": float(wirelength),
                    "connectivity_mask": connectivity_mask,
                    "graceful_mask": graceful_mask,
                    "wirelength_mask": wirelength_mask,
                    "mask": mask,
                    "improvement": improvement_value,
                    "final": final_reward,
                }
            )

        self.last_components = components
        return rewards


REWARD_BUILDERS: Dict[str, Callable[..., RewardCalculator]] = {
    "wirelength": WirelengthReward,
    "adaptive_wl_via": AdaptiveWirelengthViaReward,
    "connectivity": ConnectivityReward,
    "graceful": GracefulReward,
}


def create_reward(
    name: str,
    tokenizer: UnifiedTokenizer,
    **kwargs,
) -> RewardCalculator:
    """Instantiate a reward calculator by name."""
    key = (name or "wirelength").lower()
    if key == "gated_composite":
        adaptive_kwargs = kwargs.pop("adaptive_kwargs", {})
        connectivity_kwargs = kwargs.pop("connectivity_kwargs", {})
        graceful_kwargs = kwargs.pop("graceful_kwargs", {})
        wirelength_kwargs = kwargs.pop("wirelength_kwargs", {})

        adaptive = AdaptiveWirelengthViaReward(tokenizer=tokenizer, **adaptive_kwargs)
        connectivity = ConnectivityReward(tokenizer=tokenizer, **connectivity_kwargs)
        graceful = GracefulReward(tokenizer=tokenizer, **graceful_kwargs)
        wirelength = WirelengthReward(tokenizer=tokenizer, **wirelength_kwargs)

        return GatedCompositeReward(
            adaptive_reward=adaptive,
            connectivity_reward=connectivity,
            graceful_reward=graceful,
            wirelength_reward=wirelength,
        )

    if key not in REWARD_BUILDERS:
        raise ValueError(f"Unsupported reward type: {name}")
    return REWARD_BUILDERS[key](tokenizer=tokenizer, **kwargs)
