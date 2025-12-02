#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   tokenizer.py
@Time    :   2025/08/01 11:15:13
@Author  :   Dawn Li
@Version :   1.0
@Contact :   dawnli619215645@gmail.com
@Desc    :   UnifiedTokenizer implementation supporting different tokenization algorithms
             (DecimalWordLevel, Seg-BPE, Concat-BPE, Seg-BBPE, Concat-BBPE) with
             coordinate parsing, direction token processing, and routing sequence conversion
"""

import json
import logging
import random
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

from flow.config import TokenizationAlgorithm, TokenizationStageConfig
from flow.utils import (
    CoordinatePoint,
    UnifiedTokenPreprocessor,
    create_unified_special_token_manager,
)


class Node:
    def __init__(self, coord_str: str = None):
        self.coord_str: str = coord_str
        self.coord: CoordinatePoint = None
        self.parent: Node = None
        self.children: List[Node] = []


class UnifiedTokenizer:
    """
    Unified tokenizer supporting both preprocessing and traditional tokenization.

    This class migrates preprocessing methods from DatasetPreparator to support
    end-to-end processing from raw coordinate data to token IDs and back.
    """

    def __init__(
        self, config: Optional[Union[TokenizationStageConfig, str, Path]] = None
    ):
        """
        Initialize UnifiedTokenizer with config object or config file path.

        Args:
            config: TokenizerConfig object, path to config file, or None to use defaults
        """
        # Load config from various sources
        if isinstance(config, (str, Path)):
            self.config = self._load_config_from_file(config)
        elif isinstance(config, TokenizationStageConfig):
            self.config = config
        elif config is None:
            # Use default config
            self.config = TokenizationStageConfig()
        else:
            raise ValueError(
                f"Invalid config type: {type(config)}. Expected TokenizerConfig, str, Path, or None."
            )

        self.advanced_config = self.config.advanced
        self.workflow_config = self.config.workflow
        self.performance_config = self.config.performance

        # Initialize unified special token manager
        self.special_token_manager = create_unified_special_token_manager(
            overlap_info_require=self.advanced_config.overlap_info_require,
            connected_info_require=self.advanced_config.connected_info_require,
        )

        # Initialize unified token preprocessor (replaces coordinate_parser, direction_processor, text_preprocessor)
        self.token_preprocessor = UnifiedTokenPreprocessor()

        # Traditional tokenizer will be initialized later (composition, not inheritance)
        self.tokenizer = None

        # Algrithm setting
        self.use_decimal_decomposition = (
            self.workflow_config.tokenizer_algorithm
            == TokenizationAlgorithm.DECIMAL_WORD_LEVEL.value
        )
        self.use_concatenation = (
            self.workflow_config.tokenizer_algorithm
            == TokenizationAlgorithm.CONCAT_BPE.value
            or self.workflow_config.tokenizer_algorithm
            == TokenizationAlgorithm.CONCAT_BBPE.value
        )
        self.use_segmentation = (
            self.workflow_config.tokenizer_algorithm
            == TokenizationAlgorithm.SEG_BPE.value
            or self.workflow_config.tokenizer_algorithm
            == TokenizationAlgorithm.SEG_BBPE.value
        )

    def _load_config_from_file(
        self, config_path: Union[str, Path]
    ) -> TokenizationStageConfig:
        """Load TokenizerConfig from JSON file."""
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_dict = json.load(f)

            # Convert dict to TokenizerConfig
            return TokenizationStageConfig.from_dict(config_dict)

        except Exception as e:
            raise ValueError(f"Failed to load config from {config_path}: {e}")

    def _save_config_to_file(self, save_path: Union[str, Path]):
        """Save TokenizerConfig to JSON file."""
        save_path = Path(save_path)

        try:
            # Convert config to dict for JSON serialization
            config_dict = self.config.to_dict()

            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)

            logging.info(f"Config saved to {save_path}")

        except Exception as e:
            logging.error(f"Failed to save config to {save_path}: {e}")
            raise

    def train(self, training_texts: List[str]) -> PreTrainedTokenizerFast:
        """
        Train tokenizer on provided texts.

        Args:
            training_texts: List of text strings for training

        Returns:
            Trained PreTrainedTokenizerFast tokenizer
        """
        logging.info(
            f"Training tokenizer with algorithm: {self.workflow_config.tokenizer_algorithm}"
        )

        if (
            self.workflow_config.tokenizer_algorithm
            == TokenizationAlgorithm.DECIMAL_WORD_LEVEL.value
        ):
            self.tokenizer = self.build_word_level_tokenizer(training_texts)
        elif (
            self.workflow_config.tokenizer_algorithm
            == TokenizationAlgorithm.CONCAT_BPE.value
            or self.workflow_config.tokenizer_algorithm
            == TokenizationAlgorithm.SEG_BPE.value
        ):
            self.tokenizer = self.build_bpe_tokenizer(
                training_texts, self.workflow_config.target_vocab_size
            )
        elif (
            self.workflow_config.tokenizer_algorithm
            == TokenizationAlgorithm.CONCAT_BBPE.value
            or self.workflow_config.tokenizer_algorithm
            == TokenizationAlgorithm.SEG_BBPE.value
        ):
            self.tokenizer = self.build_byte_level_bpe_tokenizer(
                training_texts, self.workflow_config.target_vocab_size
            )
        else:
            raise ValueError(
                f"Unsupported tokenization algorithm: {self.workflow_config.tokenizer_algorithm}"
            )

        return self.tokenizer

    # === Source/Target Conversion Methods (sample <-> tokens <-> ids) ===
    def convert_loads_to_relative_loads(
        self, driver: str, loads: List[str]
    ) -> List[str]:
        """
        Convert loads from net sample to relative coordinate tokens.

        Args:
            driver: Driver coordinate as a string
            loads: List of load coordinates as strings

        Returns:
            List of relative coordinate tokens as strings
        """
        driver_coord = self.parse_coord(driver)
        relative_loads = []
        for load in loads:
            load_coord = self.parse_coord(load)
            relative_coord = load_coord - driver_coord
            relative_loads.append(str(relative_coord))

        return relative_loads

    def convert_tree_seq_to_relative_tree_seq(
        self, driver: str, tree_seq: List[str]
    ) -> List[str]:
        """
        Convert loads from net sample to relative tree sequence tokens.

        Args:
            driver: Driver coordinate as a string
            tree_seq: List of tree sequence tokens as strings

        Returns:
            List of relative tree sequence tokens as strings
        """
        driver_coord = self.parse_coord(driver)

        relative_tree_seq = []
        for token in tree_seq:
            if self.is_coordinate_string(token):
                coord = self.parse_coord(token)
                relative_coord = coord - driver_coord
                relative_tree_seq.append(str(relative_coord))
            else:
                relative_tree_seq.append(token)

        return relative_tree_seq

    def convert_source_to_directional_token(
        self,
        driver: str,
        loads: List[str],
        overlap_info: List[Dict] = None,
        connected_info: List[Dict] = None,
    ) -> str:
        """
        Build source sequence from net sample

        Args:
            driver: Driver coordinate as a string
            loads: List of load coordinates as strings
            overlap_info: List of overlap information dictionaries
            connected_info: List of connected information dictionaries

        Returns:
            Source sequence as a string

        Example input:
            "driver": "(x, y, m)",
            "loads": ["(x1, y1, m1)", "(x2, y2, m2)", ...],
            "overlap_info": [{"driver": "(x, y, m)", "loads": ["(x1, y1, m1)", ...]}, ...],
            "connected_info": [{"driver": "(x, y, m)", "loads": ["(x1, y1, m1)", ...]}, ...],
            ... (else fields as needed)

        Example output:
            "DRIVER_TOKEN R100 B2 LOAD_TOKEN R50 ..."
        """
        # Add BOS token to start the sequence
        converted_tokens = [self.get_special_token("BOS_TOKEN")]

        # Add driver information
        converted_tokens.extend(
            self.build_driver_tokens(driver, self.get_special_token("DRIVER_TOKEN"))
        )

        # Add load information as sequence of direction tokens
        converted_tokens.extend(
            self.build_relative_loads_tokens(
                driver, loads, self.get_special_token("LOAD_TOKEN")
            )
        )

        # Add overlap information if required
        if (
            self.advanced_config.overlap_info_require
            and self.advanced_config.overlap_top_k > 0
        ):
            if overlap_info:
                converted_tokens.extend(self.build_overlap_tokens(overlap_info))

        # Add connected information if required
        if (
            self.advanced_config.connected_info_require
            and self.advanced_config.connected_top_k > 0
        ):
            if connected_info:
                converted_tokens.extend(self.build_connected_tokens(connected_info))

        # Add SRC_END_TOKEN to end the sequence
        converted_tokens.append(self.get_special_token("SRC_END_TOKEN"))

        # Apply text preprocessing if configured
        converted_tokens = self.apply_token_preprocessing(converted_tokens)
        return " ".join(converted_tokens)

    def convert_relative_target_to_directional_token(
        self, relative_tree_seq: List[str]
    ) -> str:
        """
        Build target sequence from net sample

        Args:
            relative_tree_seq: List of relative tree sequence tokens as strings

        Returns:
            Target sequence as a string

        Example input:
            "relative_tree_seq": ["(0, 0, 0)", "(100, 50, 2)", "[BRANCH]", "(100, 50, 0)", "[END]", ...]
            ... (else fields as needed)

        Example output:
            "R100 U50 T2 <PUSH> B2 <POP> ..."
        """
        # Check list str
        if not isinstance(relative_tree_seq, list):
            logging.error(
                f"Expected relative_tree_seq to be a list, got {type(relative_tree_seq)}: {relative_tree_seq}"
            )
            return ""

        converted_tokens, stack = [], []
        last_coord = CoordinatePoint(0, 0, 0)

        BRANCH_TOKEN = self.get_special_token("BRANCH_TOKEN")
        END_TOKEN = self.get_special_token("END_TOKEN")
        PUSH_TOKEN = self.get_special_token("PUSH_TOKEN")
        POP_TOKEN = self.get_special_token("POP_TOKEN")
        EOS_TOKEN = self.get_special_token("EOS_TOKEN")
        for i, token in enumerate(relative_tree_seq):
            if token == BRANCH_TOKEN:
                converted_tokens.append(PUSH_TOKEN)
                stack.append(last_coord)
                continue
            if token == END_TOKEN:
                converted_tokens.append(POP_TOKEN)
                last_coord = stack.pop() if stack else last_coord
                continue

            if self.is_coordinate_string(token):
                cur = self.parse_coord(token)
                direction_tokens = self.relative_coordinate_to_direction_tokens(
                    last_coord, cur
                )
                converted_tokens.extend(direction_tokens)
                last_coord = cur
            else:
                logging.debug(f"Unexpected token in 'relative_tree_seq': {token}")

        while stack:
            converted_tokens.append(POP_TOKEN)
            stack.pop()

        # Add EOS token to end the sequence
        converted_tokens.append(EOS_TOKEN)

        # Apply text preprocessing if configured
        converted_tokens = self.apply_token_preprocessing(converted_tokens)
        return " ".join(converted_tokens)

    def remove_special_token(self, text: str) -> str:
        """
        Remove all special tokens from a given text.

        Args:
            text: Input text containing special tokens

        Returns:
            Text with all special tokens removed
        """
        # Get all special tokens
        special_tokens = self.special_token_manager.get_all_special_tokens()

        # Remove each special token from the text
        for token in special_tokens:
            text = text.replace(token, "")

        # Clean up extra spaces
        return " ".join(text.split())

    def seg_tokens(self, tokens: Union[str, List[str]]) -> Union[str, List[str]]:
        return self.token_preprocessor.segment_concatenated_tokens(tokens=tokens)

    def apply_token_preprocessing(self, tokens: Union[str, List[str]]) -> List[str]:
        """Apply configured token preprocessing (e.g., decimal decomposition, concatenation, segmentation)"""
        return self.token_preprocessor.apply_preprocessing_pipeline(
            tokens=tokens,
            use_decimal_decomposition=self.use_decimal_decomposition,
            use_concatenation=self.use_concatenation,
            use_segmentation=self.use_segmentation,
        )

    def convert_tokens_to_routing(self, tokens: Union[str, List[str]]) -> List[str]:
        """
        Convert sequence of movement tokens to absolute coordinates (target/routing).

        This method reconstructs coordinate sequences from direction tokens, handling:
        - Direction tokens (R100, L50, U200, D75, T1, B2)
        - Tree structure tokens (PUSH/POP → BRANCH/END)
        - Coordinate sequence simplification

        Args:
            tokens: Either a decoded sentence (str) that needs to be split by spaces,
                   or a list of tokens (List[str]) already split externally

        Returns:
            List of coordinate strings and tree structure tokens

        Example input:
            "R100 U50 T2 <PUSH> B2 <POP> R50 ..."
        or
            ["R100", "U50", "T2", "<PUSH>", "B2", "<POP>", "R50", ...]

        Example output:
            ["(0, 0, 0)", "(100, 50, 2)", "<BRANCH>", "(100, 50, 0)", "<END>", "(150, 50, 2)", ...]
        """
        # Handle input format - convert string to token list if needed
        if isinstance(tokens, str):
            token_list = tokens.split()
        elif isinstance(tokens, list):
            token_list = tokens
        else:
            raise ValueError(f"Expected str or List[str], got {type(tokens)}")

        # convert to segmentation tokens
        token_list = self.seg_tokens(token_list)

        # Get special tokens from token manager
        EOS_TOKEN = self.get_special_token("EOS_TOKEN")
        PAD_TOKEN = self.get_special_token("PAD_TOKEN")
        BOS_TOKEN = self.get_special_token("BOS_TOKEN")
        PUSH_TOKEN = self.get_special_token("PUSH_TOKEN")
        POP_TOKEN = self.get_special_token("POP_TOKEN")
        BRANCH_TOKEN = self.get_special_token("BRANCH_TOKEN")
        END_TOKEN = self.get_special_token("END_TOKEN")

        # Filter special tokens
        processed_tokens = []
        for token in token_list:
            if token == EOS_TOKEN:
                break
            if token in [PAD_TOKEN, BOS_TOKEN]:
                continue
            processed_tokens.append(token)

        # Convert to absolute coordinates using utility functions
        current_coord = CoordinatePoint(0, 0, 0)
        output_sequence = [str(current_coord)]
        coordinate_stack = []

        for token in processed_tokens:
            if token == PUSH_TOKEN:
                coordinate_stack.append(current_coord)
                output_sequence.append(BRANCH_TOKEN)
                continue

            if token == POP_TOKEN:
                if coordinate_stack:
                    current_coord = coordinate_stack.pop()
                output_sequence.append(END_TOKEN)
                continue

            # Parse and apply movement tokens
            if self.is_direction_token(token):
                delta_coord = self.direction_token_to_coordinate(token)
                new_coord = current_coord + delta_coord
                if new_coord != current_coord:
                    current_coord = new_coord
                    output_sequence.append(str(current_coord))
            else:
                logging.warning(f"Invalid direction token: {token}. Skipping.")

        output_sequence = self.simplify_coordinate_sequence(output_sequence)
        return output_sequence

    def build_tree_structure(self, coord_sequence: List[str]) -> Node:
        """
        Build tree structure from coordinate sequence with branch/end tokens.

        Args:
            coord_sequence: List of coordinate strings and tree structure tokens

        Returns:
            Root node of the tree structure
        """
        if not coord_sequence:
            return Node()

        # Dummy root node
        root = Node()
        stack_parent: List[Node] = []
        current_parent: Node = root
        BRANCH_TOKEN = self.get_special_token("BRANCH_TOKEN")
        END_TOKEN = self.get_special_token("END_TOKEN")

        for token in coord_sequence:
            if self.is_coordinate_string(token):
                node = Node(token)
                node.coord = self.parse_coord(token)
                node.parent = current_parent
                current_parent.children.append(node)
                current_parent = node

            elif token == BRANCH_TOKEN:
                stack_parent.append(current_parent)

            elif token == END_TOKEN:
                if stack_parent:
                    current_parent = stack_parent.pop()

        # Remove dummy root node
        if root.children:
            # If root has children, return the first child as the actual root
            root = root.children[0]
            root.parent = None

        return root

    def simplify_coordinate_sequence(self, coord_sequence: List[str]) -> List[str]:
        """
        Simplify coordinate sequence by pruning collinear nodes and redundant coordinates.
        Supports tree structure with branch/end tokens.

        Args:
            coord_sequence: List of coordinate strings and tree structure tokens

        Returns:
            Simplified coordinate sequence as a list of strings

        Example input:
            ["(100, 100, 2)", "[BRANCH]", "(100, 50, 0)", "(100, 0, 0)", "[END]", "(150, 100, 2)", ...]
                                          --------------
                                          Redundant here

        Example output:
            ["(100, 100, 2)", "[BRANCH]", "(100, 0, 0)", "[END]", "(150, 50, 2)", ...]
        """

        # --- Helpers ---
        def get_direction(
            a: CoordinatePoint, b: CoordinatePoint
        ) -> Tuple[int, int, int]:
            delta = b - a
            dx, dy, dm = delta.x, delta.y, delta.m
            return ((dx > 0) - (dx < 0), (dy > 0) - (dy < 0), (dm > 0) - (dm < 0))

        # --- 1. Build tree structure ---
        BRANCH_TOKEN = self.get_special_token("BRANCH_TOKEN")
        END_TOKEN = self.get_special_token("END_TOKEN")
        root = self.build_tree_structure(coord_sequence)

        # --- 2. Prune collinear single-in/single-out nodes ---
        def is_redundant(node: Node) -> bool:
            # A node is redundant if it has one child and is collinear with its parent
            if len(node.children) == 1 and node.parent:
                cur_coord = node.coord
                parent_coord = node.parent.coord
                child_coord = node.children[0].coord
                if cur_coord and parent_coord and child_coord:
                    direction_parent = get_direction(parent_coord, cur_coord)
                    direction_child = get_direction(cur_coord, child_coord)
                    if direction_parent == direction_child:
                        return True
                    # redundant coordinate
                    if cur_coord == parent_coord or cur_coord == child_coord:
                        return True

            return False

        def prune(node: Node, result: List[str] = None):
            # Recursively prune children first
            children = node.children
            if not is_redundant(node) and node.coord_str:
                result.append(node.coord_str)
            for child in children:
                if len(children) > 1:
                    result.append(BRANCH_TOKEN)
                prune(child, result)
                if len(children) > 1:
                    result.append(END_TOKEN)

        result = []
        prune(root, result)

        return result

    # === PreTrainedTokenizerFast-Compatible Methods ===
    def save_pretrained(self, save_directory: str):
        """
        Save trained tokenizer to directory and bind unified_tokenizer.json config file.

        Args:
            save_directory: Directory to save tokenizer and config
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained. Call train() first.")

        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save the PreTrainedTokenizerFast
        self.tokenizer.save_pretrained(save_directory)

        # Save unified tokenizer config
        config_file = save_path / "unified_tokenizer.json"
        self._save_config_to_file(config_file)

        logging.info(f"Tokenizer and config saved to {save_directory}")

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        config: Optional[Union[TokenizationStageConfig, str, Path]] = None,
    ) -> "UnifiedTokenizer":
        """
        Load tokenizer from pretrained model with optional config loading.

        Args:
            model_path: Path to pretrained tokenizer
            config: TokenizerConfig instance, config file path, or None to auto-load

        Returns:
            UnifiedTokenizer with loaded tokenizer
        """
        model_path = Path(model_path)

        # Handle config loading with fallback priority
        if config is not None:
            # Use provided config
            if isinstance(config, (str, Path)):
                instance = cls(config)
            elif isinstance(config, TokenizationStageConfig):
                instance = cls(config)
            else:
                raise ValueError(f"Invalid config type: {type(config)}")
        else:
            # Try to load config from model_path
            config_file = model_path / "unified_tokenizer.json"
            if config_file.exists():
                instance = cls(config_file)
            else:
                # Raise warning and exception if no config found

                warnings.warn(
                    f"No config provided and unified_tokenizer.json not found in {model_path}. "
                    f"Cannot load UnifiedTokenizer without configuration.",
                    UserWarning,
                )
                raise FileNotFoundError(
                    f"Cannot load UnifiedTokenizer: no config provided and "
                    f"unified_tokenizer.json not found in {model_path}"
                )

        # Load the PreTrainedTokenizerFast
        try:
            instance.tokenizer = PreTrainedTokenizerFast.from_pretrained(
                str(model_path)
            )
            return instance
        except Exception as e:
            raise ValueError(f"Failed to load tokenizer from {model_path}: {e}")

    # === Helper Methods ===
    def get_special_token(self, token_name: str) -> str:
        """Get special token by name from the unified special token manager"""
        return self.special_token_manager.get_token_by_name(token_name)

    def convert_tree_token(self, token: str) -> str:
        """Convert tree tokens from raw format to special token format"""
        return self.special_token_manager.convert_tree_token(token)

    def is_coordinate_string(self, coord_str: str) -> bool:
        """Check if token is a valid coordinate string"""
        return self.token_preprocessor.validate_coordinate_format(coord_str)

    def is_direction_token(self, token: str) -> bool:
        """Check if token is a valid direction token"""
        return self.token_preprocessor.validate_direction_token(token)

    def extract_direction_tokens(self, token: str) -> List[str]:
        """Extract direction tokens from a list of tokens"""
        return self.token_preprocessor.extract_direction_tokens(token)

    def parse_coord(self, coord_str: str) -> CoordinatePoint:
        """Parse coordinate string using coordinate utilities"""
        coord = self.token_preprocessor.parse_coordinate_string(coord_str)
        return coord

    def direction_token_to_coordinate(self, token: str) -> CoordinatePoint:
        """Convert direction token to coordinate"""
        return self.token_preprocessor.direction_token_to_coordinate(token)

    def coordinate_str_to_direction_tokens(self, coord: CoordinatePoint) -> List[str]:
        """Convert coordinate string to tokens using text preprocessor"""
        start_coord = CoordinatePoint(0, 0, 0)
        return self.token_preprocessor.relative_coordinate_to_direction_tokens(
            start_coord, coord
        )

    def relative_coordinate_to_direction_tokens(
        self, from_coord: CoordinatePoint, to_coord: CoordinatePoint
    ) -> List[str]:
        """Convert relative coordinates to direction tokens"""
        return self.token_preprocessor.relative_coordinate_to_direction_tokens(
            from_coord, to_coord
        )

    def sort_coordinates_lexicographic(self, coord_strings: List[str]) -> List[str]:
        """Sort coordinate strings using coordinate processor"""
        return self.token_preprocessor.sort_coordinate_strings_lexicographic(
            coord_strings
        )

    def sort_coordinates_clockwise(
        self, driver: str, coord_strings: List[str]
    ) -> List[str]:
        """Sort coordinate strings in clockwise order using coordinate processor"""
        return self.token_preprocessor.sort_coordinate_strings_clockwise(
            driver, coord_strings
        )

    def build_driver_tokens(self, driver: str, driver_token: str) -> List[str]:
        """Build driver tokens from driver coordinate"""
        if not driver or not driver.strip():
            return []
        if not isinstance(driver, str):
            raise ValueError(f"Expected driver to be a string, got {type(driver)}")

        tokens = [driver_token]
        if driver:
            # Convert driver coordinate to tokens
            driver_coord = self.parse_coord(driver)
            direction_tokens = self.coordinate_str_to_direction_tokens(driver_coord)
            tokens.extend(direction_tokens)
        return tokens

    def build_relative_loads_tokens(
        self, driver: str, loads: List[str], load_token: str
    ) -> List[str]:
        """Build relative loads tokens from load coordinates"""
        if not loads:
            return []
        if not isinstance(loads, list):
            raise ValueError(f"Expected list of loads, got {type(loads)}")

        processed_loads = loads.copy()
        if self.advanced_config.use_coord_sorted_input:
            processed_loads = self.sort_coordinates_clockwise(driver, processed_loads)
        else:
            random.shuffle(processed_loads)

        tokens = []
        driver_coord = self.parse_coord(driver)
        for load_str in processed_loads:
            # Add LOAD_TOKEN for each load
            tokens.append(load_token)
            # Convert load coordinate to tokens
            curr_coord = self.parse_coord(load_str)
            # Convert to direction tokens
            direction_tokens = self.relative_coordinate_to_direction_tokens(
                driver_coord, curr_coord
            )
            tokens.extend(direction_tokens)
        return tokens

    def build_overlap_tokens(self, overlap_info: List[Dict]) -> List[str]:
        """Build overlap information tokens"""
        if not overlap_info:
            return []
        if not isinstance(overlap_info, list):
            raise ValueError(f"Expected list of overlap info, got {type(overlap_info)}")
        if not all(isinstance(item, dict) for item in overlap_info):
            raise ValueError("All items in overlap_info must be dictionaries")

        tokens = [self.get_special_token("OVERLAP_START_TOKEN")]

        for i, overlap in enumerate(overlap_info[: self.advanced_config.overlap_top_k]):
            if i > 0:
                tokens.append(self.get_special_token("OVERLAP_SEP_TOKEN"))

            # Add overlap driver and loads
            driver = overlap.get("driver", "")
            tokens.extend(
                self.build_driver_tokens(
                    driver, self.get_special_token("OVERLAP_DRIVER_TOKEN")
                )
            )

            loads = overlap.get("loads", [])
            tokens.extend(
                self.build_relative_loads_tokens(
                    driver, loads, self.get_special_token("OVERLAP_LOAD_TOKEN")
                )
            )

        tokens.append(self.get_special_token("OVERLAP_END_TOKEN"))
        return tokens

    def build_connected_tokens(self, connected_info: List[Dict]) -> List[str]:
        """Build connected information tokens"""
        if not connected_info:
            return []
        if not isinstance(connected_info, list):
            raise ValueError(
                f"Expected list of connected info, got {type(connected_info)}"
            )
        if not all(isinstance(item, dict) for item in connected_info):
            raise ValueError("All items in connected_info must be dictionaries")

        tokens = [self.get_special_token("CONNECTED_START_TOKEN")]

        for i, connected in enumerate(
            connected_info[: self.advanced_config.connected_top_k]
        ):
            if i > 0:
                tokens.append(self.get_special_token("CONNECTED_SEP_TOKEN"))

            # Add connected driver and loads
            driver = connected.get("driver", "")
            tokens.extend(
                self.build_driver_tokens(
                    driver, self.get_special_token("CONNECTED_DRIVER_TOKEN")
                )
            )

            loads = connected.get("loads", [])
            tokens.extend(
                self.build_relative_loads_tokens(
                    driver, loads, self.get_special_token("CONNECTED_LOAD_TOKEN")
                )
            )

        tokens.append(self.get_special_token("CONNECTED_END_TOKEN"))
        return tokens

    # === Tokenizer Building Methods ===
    def build_word_level_tokenizer(
        self, training_texts: List[str]
    ) -> PreTrainedTokenizerFast:
        """Build word-level tokenizer."""
        logging.info(
            f"Building word-level tokenizer (decimal decomposition: {self.use_decimal_decomposition})"
        )

        special_tokens_dict = self.special_token_manager.get_special_tokens_dict()
        special_tokens = self.special_token_manager.get_all_special_tokens()
        special_tokens_set = set(special_tokens)

        all_tokens = set()
        for text in training_texts:
            all_tokens.update(text.split())

        data_tokens = all_tokens - special_tokens_set

        sorted_data_tokens = sorted(data_tokens)

        final_vocab_map = {}

        for idx, token in enumerate(sorted_data_tokens):
            final_vocab_map[token] = idx

        start_idx = len(sorted_data_tokens)
        for idx, token in enumerate(special_tokens):
            final_vocab_map[token] = start_idx + idx

        # Build a WordLevel tokenizer
        tokenizer = Tokenizer(
            models.WordLevel(
                vocab=final_vocab_map, unk_token=special_tokens_dict["unk_token"]
            )
        )
        tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()

        # Wrap in Fast tokenizer
        hf_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            bos_token=special_tokens_dict["bos_token"],
            eos_token=special_tokens_dict["eos_token"],
            pad_token=special_tokens_dict["pad_token"],
            unk_token=special_tokens_dict["unk_token"],
            model_max_length=self.workflow_config.max_sequence_length,
        )

        core_special_tokens = self.special_token_manager.get_core_special_tokens()
        additional_special_tokens = (
            self.special_token_manager.get_additional_special_tokens()
        )
        hf_tokenizer.add_tokens(core_special_tokens, special_tokens=True)
        hf_tokenizer.add_tokens(additional_special_tokens, special_tokens=False)

        return hf_tokenizer

    def build_bpe_tokenizer(
        self, training_texts: List[str], vocab_size: int
    ) -> PreTrainedTokenizerFast:
        """Build BPE tokenizer."""
        logging.info(f"Building BPE tokenizer with vocab size: {vocab_size}")

        # Get special tokens from unified manager
        special_tokens_dict = self.special_token_manager.get_special_tokens_dict()

        # Create tokenizer with BPE model
        tokenizer = Tokenizer(models.BPE(unk_token=special_tokens_dict["unk_token"]))

        # Add pre-tokenizer (WhitespaceSplit preserves special tokens like <DRIVER>)
        tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()

        # Add decoder to convert BPE tokens back to text
        tokenizer.decoder = decoders.BPEDecoder()

        # Train tokenizer
        trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=2)

        tokenizer.train_from_iterator(training_texts, trainer=trainer)

        # Convert to HuggingFace tokenizer
        hf_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            bos_token=special_tokens_dict["bos_token"],
            eos_token=special_tokens_dict["eos_token"],
            pad_token=special_tokens_dict["pad_token"],
            unk_token=special_tokens_dict["unk_token"],
            model_max_length=self.workflow_config.max_sequence_length,
        )

        core_special_tokens = self.special_token_manager.get_core_special_tokens()
        additional_special_tokens = (
            self.special_token_manager.get_additional_special_tokens()
        )
        hf_tokenizer.add_tokens(core_special_tokens, special_tokens=True)
        hf_tokenizer.add_tokens(additional_special_tokens, special_tokens=False)

        return hf_tokenizer

    def build_byte_level_bpe_tokenizer(
        self, training_texts: List[str], vocab_size: int
    ) -> PreTrainedTokenizerFast:
        """Build Byte-level BPE tokenizer."""
        logging.info(f"Building Byte-level BPE tokenizer with vocab size: {vocab_size}")

        # Get special tokens from unified manager
        special_tokens_dict = self.special_token_manager.get_special_tokens_dict()

        # Create tokenizer with Byte-level BPE model
        tokenizer = Tokenizer(models.BPE(unk_token=special_tokens_dict["unk_token"]))

        # Add byte-level pre-tokenizer
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

        # Add decoder for byte-level BPE
        tokenizer.decoder = decoders.ByteLevel()

        # Train tokenizer
        trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=2)

        tokenizer.train_from_iterator(training_texts, trainer=trainer)

        # Convert to HuggingFace tokenizer
        hf_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            bos_token=special_tokens_dict["bos_token"],
            eos_token=special_tokens_dict["eos_token"],
            pad_token=special_tokens_dict["pad_token"],
            unk_token=special_tokens_dict["unk_token"],
            model_max_length=self.workflow_config.max_sequence_length,
        )

        core_special_tokens = self.special_token_manager.get_core_special_tokens()
        additional_special_tokens = (
            self.special_token_manager.get_additional_special_tokens()
        )
        hf_tokenizer.add_tokens(core_special_tokens, special_tokens=True)
        hf_tokenizer.add_tokens(additional_special_tokens, special_tokens=False)

        return hf_tokenizer

    # === End of UnifiedTokenizer Class ===
