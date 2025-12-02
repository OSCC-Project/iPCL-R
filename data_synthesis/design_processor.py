#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   design_processor.py
@Time    :   2025/08/02 12:04:13
@Author  :   Dawn Li
@Version :   1.0
@Contact :   dawnli619215645@gmail.com
@Desc    :   Self-contained design data processor for data synthesis
"""

import logging
from collections import defaultdict
from enum import Enum
from itertools import islice
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import pandas as pd
import torch
from rtree import index
from tqdm import tqdm

from .base import ConfigurationManager, DataGenerator
from .feature_gen import LmT28FeatureGenerate

try:
    from .feature_gen import setup_aieda

    if setup_aieda():
        from aieda.data.database.vectors import (
            VectorNet,
            VectorNetRoutingGraph,
            VectorNetRoutingPoint,
        )
except Exception as e:
    print(f"Failed to set up AIEDA: {e}")


class LmWirePatternDirection(Enum):
    UP = "U"
    DOWN = "D"
    LEFT = "L"
    RIGHT = "R"
    VIA_TOP = "T"
    VIA_BOTTOM = "B"


class LmWirePatternUnit:
    def __init__(self, direction: LmWirePatternDirection, length: int):
        self.direction = direction
        self.length = length


class LmWirePatternSequence:
    def __init__(
        self,
        name: str,
        units: List[LmWirePatternUnit],
        points: List[VectorNetRoutingPoint],
    ):
        self.name = name
        self.units = units
        self.points = points


class NetPatternProcessor:
    """Handles pattern calculation and manipulation"""

    def __init__(self, max_turn_num: int = 4):
        self._max_turn_num = max_turn_num

    def build_pattern(self, path: List[VectorNetRoutingPoint]) -> LmWirePatternSequence:
        """Build pattern from LmPath list"""
        return self._calc_pattern(path)

    def unravel_pattern(
        self, start: VectorNetRoutingPoint, pattern: LmWirePatternSequence
    ) -> List[LmWirePatternSequence]:
        """Unravel long patterns into smaller chunks"""
        locs = self._correct_pattern_direction(start, pattern)
        corrected_pattern = self._calc_pattern(locs)

        turn_num = len(corrected_pattern.points) - 1
        if turn_num <= self._max_turn_num + 1:
            return [corrected_pattern]

        return self._split_pattern_recursively(corrected_pattern)

    def _correct_pattern_direction(
        self, start: VectorNetRoutingPoint, pattern: LmWirePatternSequence
    ) -> List[VectorNetRoutingPoint]:
        """Correct the direction of pattern points"""
        locs = pattern.points
        if locs[0] != start:
            locs = list(reversed(locs))
            if locs[0] != start:
                raise ValueError("Invalid pattern")
        return locs

    def _split_pattern_recursively(
        self, pattern: LmWirePatternSequence
    ) -> List[LmWirePatternSequence]:
        """Recursively split pattern into smaller chunks"""
        patterns = []

        def split_pattern(pattern: LmWirePatternSequence):
            points = pattern.points
            if len(points) <= self._max_turn_num + 1:
                patterns.append(pattern)
                return
            mid = len(points) // 2
            split_pattern(self._calc_pattern(points[: mid + 1]))
            split_pattern(self._calc_pattern(points[mid:]))

        split_pattern(pattern)
        return patterns

    def _calc_pattern(
        self, points: List[VectorNetRoutingPoint]
    ) -> LmWirePatternSequence:
        """Calculate pattern from points"""
        units = []
        for start, end in zip(points, islice(points, 1, None)):
            if start == end:
                continue

            unit = self._create_pattern_unit(start, end)
            units.append(unit)

        if not units:
            return LmWirePatternSequence("", [], points)

        pattern_name = "".join(f"{unit.direction.value}{unit.length}" for unit in units)
        return LmWirePatternSequence(pattern_name, units, points)

    def _create_pattern_unit(
        self, start: VectorNetRoutingPoint, end: VectorNetRoutingPoint
    ) -> LmWirePatternUnit:
        """Create a single pattern unit from two points"""
        diff_num = sum(
            1
            for i in range(3)
            if getattr(start, ["x", "y", "layer_id"][i])
            != getattr(end, ["x", "y", "layer_id"][i])
        )
        if diff_num > 1:
            logging.error(f"Invalid points: {start}, {end}")

        if start.x == end.x and start.y == end.y:
            direction = (
                LmWirePatternDirection.VIA_TOP
                if start.layer_id < end.layer_id
                else LmWirePatternDirection.VIA_BOTTOM
            )
            length = abs(start.layer_id - end.layer_id)
        elif start.x == end.x:
            direction = (
                LmWirePatternDirection.UP
                if start.y < end.y
                else LmWirePatternDirection.DOWN
            )
            length = abs(start.y - end.y)
        else:
            direction = (
                LmWirePatternDirection.RIGHT
                if start.x < end.x
                else LmWirePatternDirection.LEFT
            )
            length = abs(start.x - end.x)

        return LmWirePatternUnit(direction, length)


class NetGraphProcessor:
    """Handles graph construction and validation"""

    def __init__(self, pattern_processor: NetPatternProcessor):
        self.pattern_processor = pattern_processor

    def process_net_graph(self, lm_net: VectorNet) -> Tuple[nx.Graph, bool]:
        """Process VectorNet into networkx graph"""
        routing_graph = lm_net.routing_graph
        if not routing_graph or not routing_graph.vertices or not routing_graph.edges:
            logging.error(f"Invalid routing graph for net {lm_net.name}")
            return nx.Graph(), False
        nx_graph = self._build_networkx_graph(routing_graph)
        return self._validate_and_orient_graph(nx_graph, lm_net.name)

    def _build_networkx_graph(self, routing_graph: VectorNetRoutingGraph) -> nx.Graph:
        """Build networkx graph from VectorNetRoutingGraph data"""
        vertices = routing_graph.vertices
        edges = routing_graph.edges

        nx_graph = nx.Graph()
        # Keep Driver Pin only one
        first_driver_pin_id = next(
            (vertex.id for vertex in vertices if vertex.is_driver_pin), None
        )
        if first_driver_pin_id is None:
            logging.error("No driver pin found in the routing graph")
            return nx_graph
        # Create Nodes
        for vertex in vertices:
            nx_graph.add_node(
                vertex.id,
                Location=vertex.point,
                IsDriver=(vertex.id == first_driver_pin_id),
                IsLoad=vertex.is_pin and not vertex.is_driver_pin,
            )
        # Create Edges
        for edge in edges:
            path = edge.path
            pattern = self.pattern_processor.build_pattern(path)
            source_id = edge.source_id
            target_id = edge.target_id
            nx_graph.add_edge(source_id, target_id, Pattern=pattern)

        return nx_graph

    def _validate_and_orient_graph(
        self, graph: nx.Graph, net_name: str
    ) -> Tuple[nx.Graph, bool]:
        """Validate graph structure and orient patterns"""
        if not self._is_valid_tree_structure(graph, net_name):
            return graph, False

        self._orient_patterns_from_driver(graph)
        return graph, True

    def _is_valid_tree_structure(self, graph: nx.Graph, net_name: str) -> bool:
        """Check if graph is a valid tree structure"""
        connected_components = list(nx.connected_components(graph))
        if len(connected_components) > 1:
            logging.warning(f"Net {net_name} has multiple connected components")
            return False

        if not nx.is_tree(graph):
            logging.warning(f"Net {net_name} has circle")
            return False

        return True

    def _orient_patterns_from_driver(self, graph: nx.Graph):
        """Orient all patterns starting from driver pin using DFS"""
        start_node = next(
            (node for node in graph.nodes if graph.nodes[node]["IsDriver"]), None
        )
        if start_node is None:
            return

        visited = set()
        stack = [start_node]

        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)

            for neighbor in graph.neighbors(node):
                if neighbor not in visited:
                    self._orient_edge_pattern(graph, node, neighbor)
                    stack.append(neighbor)

    def _orient_edge_pattern(self, graph: nx.Graph, node: int, neighbor: int):
        """Orient pattern for a specific edge"""
        pattern: LmWirePatternSequence = graph.edges[node, neighbor]["Pattern"]
        start = graph.nodes[node]["Location"]
        locs = pattern.points

        if locs[0] != start:
            locs = list(reversed(locs))
            if locs[0] != start:
                raise ValueError("Invalid pattern")

        oriented_pattern = self.pattern_processor._calc_pattern(locs)
        graph.edges[node, neighbor]["Pattern"] = oriented_pattern


class NetSequenceExtractor:
    """Extracts sequences from processed graphs"""

    def __init__(self, pattern_processor: NetPatternProcessor):
        self.pattern_processor = pattern_processor

    def extract_pin2pin_sequences(
        self, graph: nx.Graph
    ) -> Tuple[List[List[str]], List[List[dict]]]:
        """Extract pin-to-pin pattern and location sequences"""
        if not graph:
            return [], []

        pin2pin_patterns, pin2pin_locations = self._traverse_for_sequences(graph)

        return pin2pin_patterns, pin2pin_locations

    def extract_net_sequence(
        self, net_name: str, graph: nx.Graph, loc_seqs: List[List[dict]]
    ) -> Dict:
        """Extract complete network sequence representation"""
        point_seqs = self._convert_to_point_sequences(loc_seqs)
        tree_builder = NetSteinerTreeBuilder()

        net_loads, net_steiner_seq = tree_builder.build_steiner_sequence(point_seqs)

        driver_node = next(
            (node for node in graph.nodes if graph.nodes[node]["IsDriver"]), None
        )
        abs_driver_pin_loc = graph.nodes[driver_node]["Location"]

        return {
            "net_name": net_name,
            "driver": str(abs_driver_pin_loc),
            "loads": net_loads,
            "tree_seq": net_steiner_seq,
        }

    def _traverse_for_sequences(
        self, graph: nx.Graph
    ) -> Tuple[List[List[str]], List[List[dict]]]:
        """Traverse graph to extract sequences using iterative DFS"""
        pin2pin_patterns = []
        pin2pin_locations = []

        # traverse all path from 'IsDriver' node to 'IsLoad' nodes
        driver_node = next(
            (node for node in graph.nodes if graph.nodes[node]["IsDriver"]), None
        )
        load_nodes = [node for node in graph.nodes if graph.nodes[node]["IsLoad"]]
        if driver_node is None or not load_nodes:
            logging.warning("No valid driver or load nodes found in the graph")
            return pin2pin_patterns, pin2pin_locations
        # find the path from driver to each load node (by nx.shortest_paths)
        paths = nx.single_source_shortest_path(graph, driver_node)
        for load_node in load_nodes:
            path = paths.get(load_node)
            if path is None:
                logging.warning(
                    f"No path found from driver {driver_node} to load {load_node}"
                )
                continue
            pattern_seq, loc_seq = self._build_path_sequence(graph, path)

            pin2pin_patterns.append(pattern_seq)
            pin2pin_locations.append(loc_seq)

        # convert location sequences to relative coordinates
        # driver_loc = graph.nodes[driver_node]['Location']
        # self._convert_to_relative_coordinates(pin2pin_locations, driver_loc)

        return pin2pin_patterns, pin2pin_locations

    def _build_path_sequence(
        self, graph: nx.Graph, path: List[int]
    ) -> Tuple[List[str], List[Dict]]:
        """Build pattern and location sequences for a given path"""
        pattern_seq = []
        loc_seq = []

        for i in range(len(path) - 1):
            source = path[i]
            target = path[i + 1]
            edge = graph.edges[source, target]

            source_loc = graph.nodes[source]["Location"]

            pattern = edge["Pattern"]
            unraveled_patterns = self.pattern_processor.unravel_pattern(
                source_loc, pattern
            )

            pattern_seq.extend([p.name for p in unraveled_patterns])
            loc_seq.extend([p.points for p in unraveled_patterns])

        # Flatten the location sequence
        loc_seq = self._flatten_location_sequence(loc_seq)

        return pattern_seq, loc_seq

    def _flatten_location_sequence(
        self, loc_seq: List[List[VectorNetRoutingPoint]]
    ) -> List[dict]:
        """Flatten location sequence removing duplicate points"""
        flattened = []
        for i, locs in enumerate(loc_seq):
            if i < len(loc_seq) - 1 and locs[-1] == loc_seq[i + 1][0]:
                locs = locs[:-1]
            flattened.extend(
                [{"x": loc.x, "y": loc.y, "z": loc.layer_id} for loc in locs]
            )
        return flattened

    def _convert_to_relative_coordinates(
        self,
        pin2pin_locations: List[List[dict]],
        abs_driver_pin_loc: VectorNetRoutingPoint,
    ):
        """Convert absolute coordinates to relative coordinates"""
        for loc_seq in pin2pin_locations:
            for loc in loc_seq:
                loc["x"] -= abs_driver_pin_loc.x
                loc["y"] -= abs_driver_pin_loc.y
                loc["z"] -= abs_driver_pin_loc.layer_id

    def _convert_to_point_sequences(
        self, loc_seqs: List[List[dict]]
    ) -> List[List[VectorNetRoutingPoint]]:
        """Convert location sequences to point sequences"""
        return [
            [VectorNetRoutingPoint(p["x"], p["y"], p["z"]) for p in loc_seq]
            for loc_seq in loc_seqs
        ]

    def _map_loads_to_patterns(
        self,
        net_loads: List[str],
        point_seqs: List[List[VectorNetRoutingPoint]],
        pattern_seqs: List[List[str]],
    ) -> List[List[str]]:
        """Map loads to their corresponding patterns"""
        load_to_pattern = {
            str(path[-1]): pattern for path, pattern in zip(point_seqs, pattern_seqs)
        }
        return [load_to_pattern[load] for load in net_loads]


class NetSteinerTreeNode:
    def __init__(self, point: Optional[VectorNetRoutingPoint] = None):
        self.point = point
        self.children = dict()


class NetSteinerTreeBuilder:
    """Builds Steiner tree representation from point sequences"""

    def __init__(
        self, branch_start_tok: str = "[BRANCH]", branch_end_tok: str = "[END]"
    ):
        self.branch_start_tok = branch_start_tok
        self.branch_end_tok = branch_end_tok

    def build_steiner_sequence(
        self, point_seqs: List[List[VectorNetRoutingPoint]]
    ) -> Tuple[List[str], List[str]]:
        """Build Steiner tree sequence from point sequences"""
        root = self._build_prefix_tree(point_seqs)
        return self._traverse_tree_iteratively(root)

    def _build_prefix_tree(self, point_seqs: List[List[VectorNetRoutingPoint]]):
        """Build prefix tree from point sequences"""
        root = NetSteinerTreeNode()
        for path in point_seqs:
            node = root
            for pt in path:
                if pt not in node.children:
                    node.children[pt] = NetSteinerTreeNode(point=pt)
                node = node.children[pt]
        return root

    def _traverse_tree_iteratively(
        self, root: NetSteinerTreeNode
    ) -> Tuple[List[str], List[str]]:
        """Traverse tree iteratively to build sequences"""
        net_loads = []
        net_steiner_seq = []

        stack = [("visit", root)]
        while stack:
            entry, node = stack.pop()

            if entry == "branch_start":
                net_steiner_seq.append(self.branch_start_tok)
            elif entry == "branch_end":
                net_steiner_seq.append(self.branch_end_tok)
            else:  # entry == 'visit'
                if node.point is not None:
                    net_steiner_seq.append(str(node.point))

                children = list(node.children.values())
                is_branch = len(children) > 1

                if children:
                    for child in reversed(children):
                        if is_branch:
                            stack.append(("branch_end", None))
                        stack.append(("visit", child))
                        if is_branch:
                            stack.append(("branch_start", None))
                else:
                    if node.point is not None:
                        net_loads.append(str(node.point))

        return net_loads, net_steiner_seq


class DesignGraphProcessor:
    """Processes design graph with nets as nodes"""

    def process_design_graph(self, lm_nets: List[VectorNet]) -> nx.Graph:
        """Process design graph with nets as nodes, edges represent shared instance relations"""

        graph = nx.Graph()
        net_to_driver = {}
        inst_to_load_nets = {}
        net_bboxes = {}

        for lm_net in lm_nets:
            net_name = lm_net.name
            graph.add_node(net_name)

            driver_inst_name = None
            load_inst_names = []

            for pin in lm_net.pins:
                if pin.is_driver:
                    driver_inst_name = pin.instance
                else:
                    load_inst_names.append(pin.instance)

            if driver_inst_name:
                net_to_driver[net_name] = driver_inst_name

            for inst in load_inst_names:
                if inst not in inst_to_load_nets:
                    inst_to_load_nets[inst] = set()
                inst_to_load_nets[inst].add(net_name)

            routing_graph = lm_net.routing_graph
            if routing_graph and routing_graph.vertices:
                pin_vertices = [v for v in routing_graph.vertices if v.is_pin]
                if pin_vertices:
                    x_coords = [v.point.x for v in pin_vertices]
                    y_coords = [v.point.y for v in pin_vertices]
                    z_coords = [v.point.layer_id for v in pin_vertices]

                    if x_coords and y_coords and z_coords:
                        x_min, x_max = min(x_coords), max(x_coords)
                        y_min, y_max = min(y_coords), max(y_coords)
                        z_min, z_max = min(z_coords), max(z_coords)

                        bbox = (x_min, y_min, z_min, x_max, y_max, z_max)
                        net_bboxes[net_name] = bbox

                        graph.nodes[net_name]["features"] = {
                            "x_min": x_min,
                            "y_min": y_min,
                            "x_max": x_max,
                            "y_max": y_max,
                            "z_min": z_min,
                            "z_max": z_max,
                            "pin_num": lm_net.pin_num,
                        }

        for net, driver_inst in net_to_driver.items():
            load_nets = inst_to_load_nets.get(driver_inst, set())
            for load_net in load_nets:
                if load_net != net:
                    bbox_a = net_bboxes.get(net, None)
                    bbox_b = net_bboxes.get(load_net, None)
                    overlap = self.calculate_overlap(bbox_a, bbox_b)
                    graph.add_edge(
                        net, load_net, relation="connects_to", overlap=overlap
                    )

        if len(net_bboxes) > 100_000:
            # GPU brute-force path
            overlap_edges = self.gpu_brute_force_overlap(net_bboxes)
            for net_a, net_b in overlap_edges:
                bbox_a = net_bboxes[net_a]
                bbox_b = net_bboxes[net_b]
                overlap = self.calculate_overlap(bbox_a, bbox_b)
                graph.add_edge(net_a, net_b, relation="nearby_to", overlap=overlap)
        else:
            # Original R-Tree path
            p = index.Property()
            p.dimension = 3
            rtree_index = index.Index(properties=p)

            net_to_id = {}
            for idx, (net_name, bbox) in enumerate(net_bboxes.items()):
                net_to_id[net_name] = idx
                rtree_index.insert(idx, bbox, obj=net_name)

            added_pairs = set()
            for net_name, bbox in net_bboxes.items():
                for other_net_name in rtree_index.intersection(bbox, objects=True):
                    other_net_name = other_net_name.object
                    if other_net_name != net_name:
                        edge_pair = tuple(sorted([net_name, other_net_name]))
                        if edge_pair not in added_pairs:
                            bbox_a = net_bboxes[net_name]
                            bbox_b = net_bboxes[other_net_name]
                            overlap = self.calculate_overlap(bbox_a, bbox_b)
                            graph.add_edge(
                                net_name,
                                other_net_name,
                                relation="nearby_to",
                                overlap=overlap,
                            )
                            added_pairs.add(edge_pair)

        return graph

    def gpu_brute_force_overlap(self, net_bboxes, device="cuda"):
        """
        Detect overlapping bbox pairs using GPU brute-force.
        Args:
            net_bboxes: dict of {net_name: (x_min, y_min, z_min, x_max, y_max, z_max)}
        Returns:
            Set of tuple(net_a, net_b) where the bboxes overlap
        """
        net_names = list(net_bboxes.keys())
        bboxes = torch.tensor(
            [net_bboxes[name] for name in net_names], dtype=torch.float32, device=device
        )  # (N, 6)
        N = bboxes.shape[0]

        chunk_size = 10000  # Adjustable for memory fit
        edges = set()

        for i in range(0, N, chunk_size):
            b1 = bboxes[i : i + chunk_size]  # (B1, 6)
            n1 = net_names[i : i + chunk_size]

            for j in range(0, N, chunk_size):
                b2 = bboxes[j : j + chunk_size]  # (B2, 6)
                n2 = net_names[j : j + chunk_size]

                max_xyz = torch.min(b1[:, None, 3:], b2[None, :, 3:])  # (B1, B2, 3)
                min_xyz = torch.max(b1[:, None, :3], b2[None, :, :3])
                inter = (max_xyz - min_xyz).clamp(min=0)  # intersection dims
                inter_volume = inter.prod(dim=2)

                overlap_mask = inter_volume > 0
                idx1, idx2 = overlap_mask.nonzero(as_tuple=True)

                for a, b in zip(idx1.tolist(), idx2.tolist()):
                    if i + a >= j + b:  # avoid duplicate pairs and self
                        continue
                    edges.add((n1[a], n2[b]))

        return edges

    def calculate_overlap(
        self,
        net_bbox_a: Tuple[int, int, int, int, int, int],
        net_bbox_b: Tuple[int, int, int, int, int, int],
    ) -> float:
        """Calculate overlap volume between two bounding boxes"""
        if net_bbox_a is None or net_bbox_b is None:
            return 0.0
        x_min_a, y_min_a, z_min_a, x_max_a, y_max_a, z_max_a = net_bbox_a
        x_min_b, y_min_b, z_min_b, x_max_b, y_max_b, z_max_b = net_bbox_b
        x_overlap = max(0, min(x_max_a, x_max_b) - max(x_min_a, x_min_b))
        y_overlap = max(0, min(y_max_a, y_max_b) - max(y_min_a, y_min_b))
        z_overlap = max(0, min(z_max_a, z_max_b) - max(z_min_a, z_min_b))
        area = x_overlap * y_overlap
        if z_overlap > 0:
            return area * z_overlap  # Volume
        return area  # Area if no z overlap


class DesignDataProcessor(DataGenerator):
    """Processes individual design data and generates Parquet files"""

    def __init__(
        self,
        design_name: str,
        workspace: Path,
        output_dir: Path,
        config: Optional[ConfigurationManager] = None,
    ):
        """
        Initialize design processor

        Args:
            design_name: Name of the design
            workspace: Path to design workspace
            output_dir: Output directory for Parquet files
            config: Configuration manager instance
        """
        super().__init__(design_name, output_dir)
        self.workspace = workspace
        self.config = config or ConfigurationManager()

        # Initialize processing components
        max_turn_num = self.config.get("max_turn_num", 4)
        self.net_pattern_processor = NetPatternProcessor(max_turn_num)
        self.net_graph_processor = NetGraphProcessor(self.net_pattern_processor)
        self.net_sequence_extractor = NetSequenceExtractor(self.net_pattern_processor)
        self.design_graph_processor = DesignGraphProcessor()

        # Pattern statistics
        self._pattern_count = defaultdict(int)

    def generate_data_types(self) -> Dict[str, pd.DataFrame]:
        """Generate all data types as Parquet-ready DataFrames"""
        # Check if rebuild is needed
        if not self.config.get("rebuild", False):
            existing_data = self._check_existing_data()
            if existing_data:
                logging.info(f"Using existing data for {self.design_name}")
                return existing_data

        # Load network data
        lm_nets = self._load_net_data()
        if not lm_nets:
            logging.error(f"No nets found for design {self.design_name}")
            return {}

        # Process all nets
        processed_data = self._process_all_nets(lm_nets)

        # Convert to DataFrame format
        data_frames = self._convert_to_dataframes(processed_data)

        logging.info(f"Generated {len(data_frames)} data types for {self.design_name}")
        return data_frames

    def _check_existing_data(self) -> Optional[Dict[str, pd.DataFrame]]:
        """Check for existing processed data"""
        from .base import ParquetDataLoader

        if not ParquetDataLoader.validate_parquet_structure(self.output_dir):
            return None

        # Load existing data
        data_types = self.config.get(
            "data_types",
            ["net_seqs", "pin2pin_pattern_seqs", "pin2pin_loc_seqs", "design_graph"],
        )
        return ParquetDataLoader.load_design_data(self.output_dir, data_types)

    def _load_net_data(self):
        """Load network data from workspace"""
        try:
            gen = LmT28FeatureGenerate(
                dir_workspace=self.workspace, design=self.design_name
            )
            return gen.get_lm_graph()
        except Exception as e:
            logging.error(f"Error loading net data: {e}")
            return []

    def _process_all_nets(self, lm_nets) -> Dict[str, List]:
        """Process all nets and aggregate results"""
        pin2pin_pattern_seqs = []
        pin2pin_loc_seqs = []
        net_seqs = []
        skip_count = 0

        logging.info(f"Processing {len(lm_nets)} nets...")

        for lm_net in tqdm(lm_nets, desc=f"Processing {self.design_name} nets"):
            try:
                net_data = self._process_single_net(lm_net)
                if net_data:
                    pin2pin_pattern_seqs.extend(net_data["pin2pin_patterns"])
                    pin2pin_loc_seqs.extend(net_data["pin2pin_locations"])
                    net_seqs.append(net_data["net_sequence"])
                else:
                    skip_count += 1
            except Exception as e:
                logging.error(f"Error processing net {lm_net.name}: {e}")
                skip_count += 1

        if skip_count > 0:
            logging.warning(f"Skipped {skip_count} nets in {self.design_name}")

        enable_logs = self.config.get("enable_dataset_logs", True)

        # Sort net sequences
        if enable_logs:
            import time

            start_time = time.time()
        sorted_net_seqs = self._sort_net_seqs(net_seqs)
        if enable_logs:
            sort_duration = time.time() - start_time
            logging.info(
                f"Sort net sequences: {len(net_seqs)} nets sorted in {sort_duration:.3f}s"
            )

        # Process design graph
        if enable_logs:
            start_time = time.time()
        design_graph = self._process_design_graph(lm_nets)
        if enable_logs:
            graph_duration = time.time() - start_time
            logging.info(
                f"Process design graph: {len(lm_nets)} nets processed in {graph_duration:.3f}s"
            )

        # Add connected and overlap information
        if enable_logs:
            start_time = time.time()
        self._process_top_k_connected_info(sorted_net_seqs, design_graph)
        self._process_top_k_overlap_info(sorted_net_seqs, design_graph)
        if enable_logs:
            info_duration = time.time() - start_time
            logging.info(
                f"Add connected and overlap information: processed in {info_duration:.3f}s"
            )

        return {
            "pin2pin_pattern_seqs": pin2pin_pattern_seqs,
            "pin2pin_loc_seqs": pin2pin_loc_seqs,
            "net_seqs": sorted_net_seqs,
            "design_graph": design_graph,
        }

    def _process_single_net(self, lm_net) -> Optional[Dict]:
        """Process a single network"""
        graph, is_feasible = self.net_graph_processor.process_net_graph(lm_net)
        if not is_feasible:
            return None

        pattern_seqs, loc_seqs = self.net_sequence_extractor.extract_pin2pin_sequences(
            graph
        )
        net_seq = self.net_sequence_extractor.extract_net_sequence(
            lm_net.name, graph, loc_seqs
        )

        # Update pattern statistics
        for seq in pattern_seqs:
            for pattern in seq:
                self._pattern_count[pattern] += 1

        return {
            "pin2pin_patterns": pattern_seqs,
            "pin2pin_locations": loc_seqs,
            "net_sequence": net_seq,
        }

    def _process_design_graph(self, lm_nets) -> nx.Graph:
        """Process design graph (nets as nodes)"""
        design_graph = self.design_graph_processor.process_design_graph(lm_nets)
        return design_graph

    def _process_top_k_connected_info(
        self, net_seqs: List[dict], design_graph: nx.Graph, top_k: int = 5
    ):
        """Process top-k connected information for design graph"""
        connects_to_edges = [
            (u, v, d)
            for u, v, d in design_graph.edges(data=True)
            if d.get("relation") == "connects_to"
        ]
        connects_only_graph = nx.Graph(connects_to_edges)
        net_datas = {net["net_name"]: net for net in net_seqs}

        for net_name in connects_only_graph.nodes:
            neighbors = list(connects_only_graph.neighbors(net_name))
            if not neighbors:
                continue

            sorted_neighbors = sorted(
                neighbors,
                key=lambda n: connects_only_graph.nodes[n].get("overlap", 0),
                reverse=True,
            )

            net_data = net_datas.get(net_name, {})
            for i in range(min(top_k, len(sorted_neighbors))):
                neighbor = sorted_neighbors[i]
                overlap_volume = connects_only_graph.nodes[neighbor].get("overlap", 0)
                neighbor_data = net_datas.get(neighbor, {})
                net_data.setdefault("connected_info", []).append(
                    {
                        "net_name": neighbor,
                        "overlap_volume": overlap_volume,
                        "driver": neighbor_data.get("driver", ""),
                        "loads": neighbor_data.get("loads", []),
                    }
                )

    def _process_top_k_overlap_info(
        self, net_seqs: List[dict], design_graph: nx.Graph, top_k: int = 5
    ):
        """Process top-k overlap information for design graph"""
        nearby_to_edges = [
            (u, v, d)
            for u, v, d in design_graph.edges(data=True)
            if d.get("relation") == "nearby_to"
        ]
        nearby_only_graph = nx.Graph(nearby_to_edges)
        net_datas = {net["net_name"]: net for net in net_seqs}

        for net_name in nearby_only_graph.nodes:
            neighbors = list(nearby_only_graph.neighbors(net_name))
            if not neighbors:
                continue

            sorted_neighbors = sorted(
                neighbors,
                key=lambda n: nearby_only_graph[net_name][n].get("overlap", 0),
                reverse=True,
            )

            net_data = net_datas.get(net_name, {})
            for i in range(min(top_k, len(sorted_neighbors))):
                neighbor = sorted_neighbors[i]
                overlap_volume = nearby_only_graph[net_name][neighbor].get("overlap", 0)
                neighbor_data = net_datas.get(neighbor, {})
                net_data.setdefault("overlap_info", []).append(
                    {
                        "net_name": neighbor,
                        "overlap_volume": overlap_volume,
                        "driver": neighbor_data.get("driver", ""),
                        "loads": neighbor_data.get("loads", []),
                    }
                )

    def _sort_net_seqs(self, net_seqs: List[dict]) -> List[dict]:
        """Sort network sequences by load count and bounding box area"""

        def get_bounding_box_area(loads):
            if not loads:
                return 0
            x_coords, y_coords = [], []
            for load in loads:
                coords = load[1:-1].split(", ")
                x_coords.append(int(coords[0]))
                y_coords.append(int(coords[1]))

            if not x_coords or not y_coords:
                return 0

            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            return (x_max - x_min) * (y_max - y_min)

        net_seqs.sort(
            key=lambda x: (
                -len(x.get("loads", [])),
                -get_bounding_box_area(x.get("loads", [])),
            )
        )
        return net_seqs

    def _convert_to_dataframes(
        self, processed_data: Dict[str, List]
    ) -> Dict[str, pd.DataFrame]:
        """Convert processed data to Parquet-ready DataFrames with optimized performance and timing logs"""
        import time

        data_frames = {}
        enable_logs = self.config.get("enable_dataset_logs", True)

        # Only process requested data types
        requested_types = self.config.get(
            "data_types",
            ["net_seqs", "pin2pin_pattern_seqs", "pin2pin_loc_seqs", "design_graph"],
        )

        # Convert pin2pin pattern sequences
        if (
            "pin2pin_pattern_seqs" in requested_types
            and "pin2pin_pattern_seqs" in processed_data
            and processed_data["pin2pin_pattern_seqs"]
        ):
            start_time = time.time()
            pattern_data = [
                {
                    "sequence_id": i,
                    "patterns": seq,
                }
                for i, seq in enumerate(processed_data["pin2pin_pattern_seqs"])
            ]
            data_frames["pin2pin_pattern_seqs"] = pd.DataFrame(pattern_data)
            conversion_time = time.time() - start_time
            if enable_logs:
                logging.info(
                    f"Converted pin2pin_pattern_seqs: {len(pattern_data)} records in {conversion_time:.2f}s"
                )

        # Convert pin2pin location sequences
        if (
            "pin2pin_loc_seqs" in requested_types
            and "pin2pin_loc_seqs" in processed_data
            and processed_data["pin2pin_loc_seqs"]
        ):
            start_time = time.time()
            location_data = [
                {
                    "sequence_id": i,
                    "locations": seq,
                }
                for i, seq in enumerate(processed_data["pin2pin_loc_seqs"])
            ]
            data_frames["pin2pin_loc_seqs"] = pd.DataFrame(location_data)
            conversion_time = time.time() - start_time
            if enable_logs:
                logging.info(
                    f"Converted pin2pin_loc_seqs: {len(location_data)} records in {conversion_time:.2f}s"
                )

        # Convert net sequences
        if (
            "net_seqs" in requested_types
            and "net_seqs" in processed_data
            and processed_data["net_seqs"]
        ):
            start_time = time.time()
            # Pre-allocate and batch process for better performance
            net_data = processed_data["net_seqs"].copy()
            data_frames["net_seqs"] = pd.DataFrame(net_data)
            conversion_time = time.time() - start_time
            if enable_logs:
                logging.info(
                    f"Converted net_seqs: {len(net_data)} records in {conversion_time:.2f}s"
                )

        # Convert design graph
        if "design_graph" in requested_types and "design_graph" in processed_data:
            start_time = time.time()
            graph = processed_data["design_graph"]
            # convert to DataFrame
            data_frames["design_graph"] = nx.to_pandas_edgelist(graph)
            conversion_time = time.time() - start_time
            if enable_logs:
                logging.info(
                    f"Converted design_graph: 1 record in {conversion_time:.2f}s"
                )

        return data_frames

    def get_pattern_statistics(self) -> pd.DataFrame:
        """Get pattern frequency statistics"""
        if not self._pattern_count:
            return pd.DataFrame(columns=["Pattern", "Count"])

        df = pd.DataFrame(self._pattern_count.items(), columns=["Pattern", "Count"])
        return df.sort_values(by="Count", ascending=False)

    def save_pattern_statistics(self, csv_path: Path = None):
        """Save pattern statistics to CSV"""
        if csv_path is None:
            csv_path = self.output_dir / "pattern_statistics.csv"

        stats_df = self.get_pattern_statistics()
        stats_df.to_csv(csv_path, index=False)
        logging.info(f"Pattern statistics saved to {csv_path}")


class BatchDesignProcessor:
    """Processes multiple designs in batch"""

    def __init__(
        self,
        designs_config: Dict[str, Dict[str, Any]],
        base_output_dir: Path,
        config: Optional[ConfigurationManager] = None,
    ):
        """
        Initialize batch processor

        Args:
            designs_config: Dict mapping design_name -> {'workspace': path, 'output_dir': path}
            base_output_dir: Base output directory
            config: Configuration manager
        """
        self.designs_config = designs_config
        self.base_output_dir = base_output_dir
        self.config = config or ConfigurationManager()

    def process_all_designs(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Process all designs and return results"""
        results = {}

        for design_name, design_config in self.designs_config.items():
            logging.info(f"Processing design: {design_name}")

            try:
                processor = DesignDataProcessor(
                    design_name=design_name,
                    workspace=design_config["workspace"],
                    output_dir=design_config.get(
                        "output_dir", self.base_output_dir / design_name
                    ),
                    config=self.config,
                )

                # Generate data
                data_frames = processor.generate_data_types()

                # Save to Parquet files with detailed logging based on config
                processor.save_parquet_files(
                    data_frames, self.config.get("enable_dataset_logs", True)
                )

                # Save pattern statistics
                processor.save_pattern_statistics()

                results[design_name] = data_frames
                logging.info(f"Successfully processed design: {design_name}")

            except Exception as e:
                logging.error(f"Error processing design {design_name}: {e}")
                results[design_name] = {}

        return results
