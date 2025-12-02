#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   special_tokens.py
@Time    :   2025/08/01 11:15:56
@Author  :   Dawn Li
@Version :   1.0
@Contact :   dawnli619215645@gmail.com
@Desc    :   Special token management for routing patterns including core tokens
             (PAD, BOS, EOS), tree structure tokens (PUSH, POP, BRANCH, END),
             source tokens (DRIVER, LOAD), and conditional overlap/connected tokens
"""

from dataclasses import dataclass
from typing import Dict, List, Set


@dataclass
class SpecialTokenConfig:
    """Configuration for which special token features to include"""

    overlap_info_required: bool = False
    connected_info_required: bool = False
    use_tree_structure: bool = True  # Always use PUSH/POP and BRANCH/END
    use_source_info: bool = True  # Always use DRIVER/LOAD


class SpecialTokenManager:
    """Unified manager for all special tokens following original net_pattern_gen design"""

    # Core special tokens (always included)
    CORE_TOKENS = {
        "BOS_TOKEN": "<BOS>",
        "EOS_TOKEN": "<EOS>",
        "UNKNOWN_TOKEN": "<UNK_LEN>",
        "PAD_TOKEN": "<PAD>",
        "SRC_END_TOKEN": "<SRC_END>",
    }

    # Source information tokens (always included)
    SOURCE_TOKENS = {"DRIVER_TOKEN": "<DRIVER>", "LOAD_TOKEN": "<LOAD>"}

    # Tree structure tokens (always included)
    TREE_TOKENS = {
        "PUSH_TOKEN": "<PUSH>",
        "POP_TOKEN": "<POP>",
        "BRANCH_TOKEN": "[BRANCH]",
        "END_TOKEN": "[END]",
    }

    # Overlap information tokens (conditional)
    OVERLAP_TOKENS = {
        "OVERLAP_START_TOKEN": "<OVERLAP_START>",
        "OVERLAP_END_TOKEN": "<OVERLAP_END>",
        "OVERLAP_SEP_TOKEN": "<OVERLAP_SEP>",
        "OVERLAP_DRIVER_TOKEN": "<OVERLAP_DRIVER>",
        "OVERLAP_LOAD_TOKEN": "<OVERLAP_LOAD>",
    }

    # Connected information tokens (conditional)
    CONNECTED_TOKENS = {
        "CONNECTED_START_TOKEN": "<CONNECTED_START>",
        "CONNECTED_END_TOKEN": "<CONNECTED_END>",
        "CONNECTED_SEP_TOKEN": "<CONNECTED_SEP>",
        "CONNECTED_DRIVER_TOKEN": "<CONNECTED_DRIVER>",
        "CONNECTED_LOAD_TOKEN": "<CONNECTED_LOAD>",
    }

    def __init__(self, config: SpecialTokenConfig):
        self.config = config
        self._special_tokens_cache = None
        self._token_to_name_cache = None

    def get_core_special_tokens(self) -> List[str]:
        """Get core special tokens"""
        return list(self.CORE_TOKENS.values())

    def get_additional_special_tokens(self) -> List[str]:
        """Get additional special tokens based on configuration"""
        tokens = []
        if self.config.use_source_info:
            tokens.extend(self.SOURCE_TOKENS.values())
        if self.config.use_tree_structure:
            tokens.extend(self.TREE_TOKENS.values())
        if self.config.overlap_info_required:
            tokens.extend(self.OVERLAP_TOKENS.values())
        if self.config.connected_info_required:
            tokens.extend(self.CONNECTED_TOKENS.values())
        return tokens

    def get_all_special_tokens(self) -> List[str]:
        """Get all special tokens based on configuration"""
        if self._special_tokens_cache is not None:
            return self._special_tokens_cache

        tokens = []

        # Always include core tokens
        tokens.extend(self.CORE_TOKENS.values())

        # Always include source tokens if enabled
        if self.config.use_source_info:
            tokens.extend(self.SOURCE_TOKENS.values())

        # Always include tree tokens if enabled
        if self.config.use_tree_structure:
            tokens.extend(self.TREE_TOKENS.values())

        # Conditionally include overlap tokens
        if self.config.overlap_info_required:
            tokens.extend(self.OVERLAP_TOKENS.values())

        # Conditionally include connected tokens
        if self.config.connected_info_required:
            tokens.extend(self.CONNECTED_TOKENS.values())

        # Cache result
        self._special_tokens_cache = tokens
        return tokens

    def get_special_token_set(self) -> Set[str]:
        """Get special tokens as a set for fast lookup"""
        return set(self.get_all_special_tokens())

    def get_token_to_name_mapping(self) -> Dict[str, str]:
        """Get mapping from token string to token name"""
        if self._token_to_name_cache is not None:
            return self._token_to_name_cache

        mapping = {}

        # Add all token categories
        for name, token in self.CORE_TOKENS.items():
            mapping[token] = name

        if self.config.use_source_info:
            for name, token in self.SOURCE_TOKENS.items():
                mapping[token] = name

        if self.config.use_tree_structure:
            for name, token in self.TREE_TOKENS.items():
                mapping[token] = name

        if self.config.overlap_info_required:
            for name, token in self.OVERLAP_TOKENS.items():
                mapping[token] = name

        if self.config.connected_info_required:
            for name, token in self.CONNECTED_TOKENS.items():
                mapping[token] = name

        self._token_to_name_cache = mapping
        return mapping

    def is_special_token(self, token: str) -> bool:
        """Check if a token is a special token"""
        return token in self.get_special_token_set()

    def get_special_tokens_dict(self) -> Dict[str, str]:
        """Get special tokens in the format expected by transformers tokenizers"""
        tokens = self.get_all_special_tokens()
        return {
            "bos_token": self.CORE_TOKENS["BOS_TOKEN"],
            "eos_token": self.CORE_TOKENS["EOS_TOKEN"],
            "unk_token": self.CORE_TOKENS["UNKNOWN_TOKEN"],
            "pad_token": self.CORE_TOKENS["PAD_TOKEN"],
            "additional_special_tokens": [
                token
                for token in tokens
                if token
                not in [
                    self.CORE_TOKENS["PAD_TOKEN"],
                    self.CORE_TOKENS["BOS_TOKEN"],
                    self.CORE_TOKENS["EOS_TOKEN"],
                    self.CORE_TOKENS["UNKNOWN_TOKEN"],
                ]
            ],
        }

    def convert_tree_token(self, token: str) -> str:
        """Convert tree tokens from raw format to special token format"""
        if token == "[BRANCH]":
            return self.TREE_TOKENS["PUSH_TOKEN"]
        elif token == "[END]":
            return self.TREE_TOKENS["POP_TOKEN"]
        else:
            return token

    def get_token_by_name(self, name: str) -> str:
        """Get token string by token name"""
        all_tokens = {
            **self.CORE_TOKENS,
            **self.SOURCE_TOKENS,
            **self.TREE_TOKENS,
            **self.OVERLAP_TOKENS,
            **self.CONNECTED_TOKENS,
        }
        return all_tokens.get(name, f"<UNKNOWN:{name}>")

    def get_all_special_tokens_dict(self) -> Dict[str, str]:
        """Get all special tokens as a dictionary mapping names to tokens"""
        result = {**self.CORE_TOKENS}

        if self.config.use_source_info:
            result.update(self.SOURCE_TOKENS)

        if self.config.use_tree_structure:
            result.update(self.TREE_TOKENS)

        if self.config.overlap_info_required:
            result.update(self.OVERLAP_TOKENS)

        if self.config.connected_info_required:
            result.update(self.CONNECTED_TOKENS)

        return result

    @classmethod
    def from_original_config(
        cls, overlap_info_require: bool = False, connected_info_require: bool = False
    ) -> "SpecialTokenManager":
        """Create manager from original net_pattern_gen config flags"""
        config = SpecialTokenConfig(
            overlap_info_required=overlap_info_require,
            connected_info_required=connected_info_require,
            use_tree_structure=True,
            use_source_info=True,
        )
        return cls(config)


def create_unified_special_token_manager(
    overlap_info_require: bool = False, connected_info_require: bool = False
) -> SpecialTokenManager:
    """Factory function to create unified special token manager"""
    return SpecialTokenManager.from_original_config(
        overlap_info_require=overlap_info_require,
        connected_info_require=connected_info_require,
    )
