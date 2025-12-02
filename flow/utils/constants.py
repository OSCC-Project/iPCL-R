#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   constants.py
@Time    :   2025/08/01 11:15:35
@Author  :   Dawn Li
@Version :   1.0
@Contact :   dawnli619215645@gmail.com
@Desc    :   Constants and patterns for routing coordinate system including direction
             mappings (R/L/U/D/T/B), coordinate parsing regex, direction token
             validation, and layer movement definitions
"""

import re
from typing import Dict, Tuple

# =============================================================================
# Direction and Coordinate Constants
# =============================================================================

# Direction mapping for coordinate movements (from both modules)
DIRECTION_MAP: Dict[str, Tuple[int, int, int]] = {
    "R": (1, 0, 0),  # Right (+X)
    "L": (-1, 0, 0),  # Left (-X)
    "U": (0, 1, 0),  # Up (+Y)
    "D": (0, -1, 0),  # Down (-Y)
    "T": (0, 0, 1),  # Top (+Metal layer)
    "B": (0, 0, -1),  # Bottom (-Metal layer)
}

# Direction characters for validation
DIRECTION_CHARS = set(DIRECTION_MAP.keys())

# Same-layer movement directions (no metal layer change)
FLATTEN_DIRECTIONS = {"R", "L", "U", "D"}

# Layer change directions
VIA_DIRECTIONS = {"T", "B"}


# =============================================================================
# Regular Expression Patterns
# =============================================================================
# Coordinate pattern: matches (x, y, m) format
COORDINATE_PATTERN = re.compile(r"\((-?\d+),\s*(-?\d+),\s*(-?\d+)\)")

# Direction token pattern: matches R123, L45, etc.
DIRECTION_TOKEN_PATTERN = re.compile(r"[RLUDTB]\d+")
