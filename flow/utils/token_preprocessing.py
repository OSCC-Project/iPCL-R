#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   token_preprocessing.py
@Time    :   2025/08/01 11:16:03
@Author  :   Dawn Li
@Version :   1.0
@Contact :   dawnli619215645@gmail.com
@Desc    :   Unified token preprocessing for routing patterns including coordinate
             parsing, direction token processing, decimal decomposition,
             concatenation/segmentation, and sequence transformation utilities
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

from .constants import (
    COORDINATE_PATTERN,
    DIRECTION_MAP,
    DIRECTION_TOKEN_PATTERN,
    FLATTEN_DIRECTIONS,
    VIA_DIRECTIONS,
)
from .exceptions import (
    CoordinateParsingError,
    DirectionTokenError,
    TokenPreprocessingError,
)

# =============================================================================
# Core Data Structures
# =============================================================================


@dataclass(order=True, frozen=True)
class CoordinatePoint:
    """
    Immutable coordinate point representation with arithmetic operations.

    Represents a 3D point in the routing space with x, y coordinates
    and metal layer m.
    """

    x: int
    y: int
    m: int  # metal layer

    def __post_init__(self):
        """Validate coordinate values"""
        if (
            not isinstance(self.x, int)
            or not isinstance(self.y, int)
            or not isinstance(self.m, int)
        ):
            raise TypeError(
                "CoordinatePoint values must be integers: "
                f"x={self.x}, y={self.y}, m={self.m}"
            )

    def __add__(self, other: "CoordinatePoint") -> "CoordinatePoint":
        """Add two coordinate points"""
        if not isinstance(other, CoordinatePoint):
            raise TypeError("Can only add CoordinatePoint to CoordinatePoint")
        return CoordinatePoint(self.x + other.x, self.y + other.y, self.m + other.m)

    def __sub__(self, other: "CoordinatePoint") -> "CoordinatePoint":
        """Subtract two coordinate points"""
        if not isinstance(other, CoordinatePoint):
            raise TypeError("Can only subtract CoordinatePoint from CoordinatePoint")
        return CoordinatePoint(self.x - other.x, self.y - other.y, self.m - other.m)

    def __mul__(self, other) -> "CoordinatePoint":
        """Multiply coordinate point by scalar or another point"""
        if isinstance(other, CoordinatePoint):
            # Element-wise multiplication with another point
            return CoordinatePoint(self.x * other.x, self.y * other.y, self.m * other.m)
        elif isinstance(other, (int, float)):
            # Scalar multiplication
            result_x = int(self.x * other)
            result_y = int(self.y * other)
            result_m = int(self.m * other)
            return CoordinatePoint(result_x, result_y, result_m)
        else:
            raise TypeError(
                f"Can only multiply CoordinatePoint by int, float, or CoordinatePoint, "
                f"got {type(other)}"
            )

    def __rmul__(self, other) -> "CoordinatePoint":
        """Right multiplication (scalar * point)"""
        return self.__mul__(other)

    def __truediv__(self, other) -> "CoordinatePoint":
        """Divide coordinate point by scalar or another point"""
        if isinstance(other, CoordinatePoint):
            # Element-wise division with another point
            if other.x == 0 or other.y == 0 or other.m == 0:
                raise ZeroDivisionError(
                    "Cannot divide by zero in coordinate components"
                )
            return CoordinatePoint(
                int(self.x / other.x), int(self.y / other.y), int(self.m / other.m)
            )
        elif isinstance(other, (int, float)):
            # Scalar division
            if other == 0:
                raise ZeroDivisionError("Cannot divide by zero")
            result_x = int(self.x / other)
            result_y = int(self.y / other)
            result_m = int(self.m / other)
            return CoordinatePoint(result_x, result_y, result_m)
        else:
            raise TypeError(
                f"Can only divide CoordinatePoint by int, float, or CoordinatePoint, "
                f"got {type(other)}"
            )

    def __floordiv__(self, other) -> "CoordinatePoint":
        """Floor division of coordinate point"""
        if isinstance(other, CoordinatePoint):
            # Element-wise floor division with another point
            if other.x == 0 or other.y == 0 or other.m == 0:
                raise ZeroDivisionError(
                    "Cannot divide by zero in coordinate components"
                )
            return CoordinatePoint(
                self.x // other.x, self.y // other.y, self.m // other.m
            )
        elif isinstance(other, (int, float)):
            # Scalar floor division
            if other == 0:
                raise ZeroDivisionError("Cannot divide by zero")
            return CoordinatePoint(self.x // other, self.y // other, self.m // other)
        else:
            raise TypeError(
                f"Can only floor divide CoordinatePoint by int, float, or CoordinatePoint, "
                f"got {type(other)}"
            )

    def __str__(self) -> str:
        """String representation in standard format"""
        return f"({self.x}, {self.y}, {self.m})"

    def to_tuple(self) -> Tuple[int, int, int]:
        """Convert to tuple representation"""
        return (self.x, self.y, self.m)

    def distance_to(self, other: "CoordinatePoint") -> float:
        """Calculate Euclidean distance to another point"""
        return math.sqrt(
            (self.x - other.x) ** 2 + (self.y - other.y) ** 2 + (self.m - other.m) ** 2
        )

    def manhattan_distance_to(self, other: "CoordinatePoint") -> int:
        """Calculate Manhattan distance to another point"""
        return abs(self.x - other.x) + abs(self.y - other.y) + abs(self.m - other.m)

    def is_same_layer(self, other: "CoordinatePoint") -> bool:
        """Check if two points are on the same metal layer"""
        return self.m == other.m


# =============================================================================
# Unified Token Preprocessor
# =============================================================================


class UnifiedTokenPreprocessor:
    """
    Unified token preprocessing for routing pattern sequences.

    Combines all preprocessing operations (decimal decomposition, concatenation,
    segmentation, coordinate parsing, sorting, etc.) into a single, coherent
    interface that consolidates functionality from coordinate_utils, token_processing,
    and text_preprocessing modules.
    """

    def __init__(self):
        """Initialize preprocessor with embedded patterns and utilities."""
        # Compile regex patterns for efficiency
        self.coord_pattern = COORDINATE_PATTERN
        self.direction_token_pattern = DIRECTION_TOKEN_PATTERN

    # =========================================================================
    # Direction Token Processing Methods
    # =========================================================================
    def direction_token(self, dx: int, dy: int, dm: int = 0) -> List[str]:
        """Convert coordinate differences to direction tokens."""
        tokens = []

        # Handle X movement
        if dx != 0:
            direction = "R" if dx > 0 else "L"
            distance = abs(dx)
            tokens.append(f"{direction}{distance}")

        # Handle Y movement
        if dy != 0:
            direction = "U" if dy > 0 else "D"
            distance = abs(dy)
            tokens.append(f"{direction}{distance}")

        # Handle metal layer movement
        if dm != 0:
            direction = "T" if dm > 0 else "B"
            distance = abs(dm)
            tokens.append(f"{direction}{distance}")

        return tokens

    def relative_coordinate_to_direction_tokens(
        self, from_coord: CoordinatePoint, to_coord: CoordinatePoint
    ) -> List[str]:
        """Convert coordinate movement to direction tokens."""
        diff = to_coord - from_coord
        dx, dy, dm = diff.x, diff.y, diff.m
        return self.direction_token(dx, dy, dm)

    def direction_token_to_coordinate(
        self, token: str, start_coord: Optional[CoordinatePoint] = None
    ) -> CoordinatePoint:
        """Convert a single direction token to coordinate movement."""
        if start_coord is None:
            start_coord = CoordinatePoint(0, 0, 0)

        current_coord = start_coord
        if not self.validate_direction_token(token):
            raise DirectionTokenError(f"Invalid direction token: {token}")
        movement = self.parse_direction_token(token)
        if movement:
            dx, dy, dm = movement
            current_coord = CoordinatePoint(
                current_coord.x + dx, current_coord.y + dy, current_coord.m + dm
            )
        return current_coord

    def parse_direction_token(self, token: str) -> Optional[Tuple[int, int, int]]:
        """Parse direction token into coordinate movement."""
        if not token or len(token) < 2:
            return None

        direction_char = token[0].upper()
        distance_str = token[1:]

        # Validate direction character
        if direction_char not in DIRECTION_MAP:
            return None

        # Parse distance
        try:
            distance = int(distance_str)
        except ValueError:
            return None

        # Get direction vector and apply distance
        dx, dy, dm = DIRECTION_MAP[direction_char]
        return (dx * distance, dy * distance, dm * distance)

    def validate_direction_token(self, token: str) -> bool:
        """Validate direction token format."""
        if not isinstance(token, str) or not token:
            return False
        return bool(self.direction_token_pattern.match(token))

    def extract_direction_tokens(self, text: str) -> List[str]:
        """Extract all direction tokens from text."""
        return self.direction_token_pattern.findall(text)

    # =========================================================================
    # Coordinate Processing Methods
    # =========================================================================
    def sort_coordinate_strings_lexicographic(
        self, coord_strings: List[str]
    ) -> List[str]:
        """Sort coordinate strings using lexicographic ordering."""
        # Parse coordinates with their original strings
        coord_pairs = []
        for coord_str in coord_strings:
            coord = self.parse_coordinate_string(coord_str)
            if coord is not None:
                coord_pairs.append((coord, coord_str))
        # Sort by coordinate values
        coord_pairs.sort(key=lambda pair: (pair[0].x, pair[0].y, pair[0].m))
        # Return sorted strings
        return [pair[1] for pair in coord_pairs]

    def sort_coordinate_strings_clockwise(
        self, center: str, coord_strings: List[str]
    ) -> List[str]:
        """Sort coordinate strings in clockwise order around the center point."""
        coords = [
            self.parse_coordinate_string(coord_string) for coord_string in coord_strings
        ]
        if not coords:
            return []
        # Find center point
        center_coord = self.parse_coordinate_string(center)
        if center_coord is None:
            return coord_strings
        # Sort by angle from center

        def angle_from_center(coord: CoordinatePoint) -> float:
            dx = coord.x - center_coord.x
            dy = coord.y - center_coord.y
            return math.atan2(dy, dx)

        sorted_coords = sorted(coords, key=angle_from_center)
        return [str(coord) for coord in sorted_coords]

    def parse_coordinate_string(self, coord_str: str) -> Optional[CoordinatePoint]:
        """Parse coordinate string with error handling."""
        if not coord_str or not isinstance(coord_str, str):
            return None

        coord_str = coord_str.strip()
        if not coord_str:
            return None

        try:
            match = self.coord_pattern.match(coord_str)
            if not match:
                return None
            x, y, m = map(int, match.groups())
            return CoordinatePoint(x, y, m)
        except (ValueError, TypeError) as e:
            raise CoordinateParsingError(
                coord_str, f"Failed to convert coordinate components to integers: {e}"
            )

    def validate_coordinate_format(self, coord_str: str) -> bool:
        """Validate coordinate string format without parsing."""
        if not coord_str or not isinstance(coord_str, str):
            return False
        return bool(self.coord_pattern.match(coord_str.strip()))

    # =========================================================================
    # Token Decimal Decomposition, Concatenation and Segmentation Methods
    # =========================================================================
    def apply_preprocessing_pipeline(
        self,
        tokens: Union[str, List[str]],
        use_decimal_decomposition: bool = False,
        use_concatenation: bool = False,
        use_segmentation: bool = False,
        remove_tokens: Optional[List[str]] = None,
    ) -> Union[str, List[str]]:
        """
        Apply complete preprocessing pipeline with flexible input/output types.

        Chains multiple preprocessing operations in the correct order while maintaining
        input/output type consistency (str→str, List[str]→List[str]).

        Args:
            tokens: Input tokens as string (space-separated) or pre-split list
            use_decimal_decomposition: Apply decimal decomposition (splits large numbers)
            use_concatenation: Apply concatenation (merges same-layer direction tokens)
            use_segmentation: Apply segmentation (splits concatenated tokens)
            remove_tokens: List of specific tokens to remove from output

        Returns:
            Preprocessed tokens in the same format as input:
            - str input → str output (space-separated)
            - List[str] input → List[str] output (token list)

        Processing Order:
            1. Decimal decomposition (if enabled)
            2. Concatenation OR segmentation (mutually exclusive)
            3. Token removal (if specified)

        Examples:
            String pipeline: 'R2200 U50 B1' → 'R2000 R200 U50 B1' → 'R2000R200U50 B1'
            List pipeline: ['R2200', 'U50', 'B1'] → ['R2000', 'R200', 'U50', 'B1'] → ['R2000R200U50', 'B1']

        Raises:
            TokenPreprocessingError: If incompatible options are specified
            ValueError: If input type is not str or List[str]
        """
        # Validate compatibility
        self.validate_preprocessing_compatibility(
            use_decimal_decomposition, use_concatenation, use_segmentation
        )

        result = tokens

        # Apply decimal decomposition first
        if use_decimal_decomposition:
            result = self.apply_segmentation(result)
            result = self.apply_decimal_decomposition(result)
        else:
            # Apply concatenation or segmentation (mutually exclusive)
            if use_concatenation:
                result = self.apply_concatenation(result)
            elif use_segmentation:
                result = self.apply_segmentation(result)

        # Remove special tokens if specified
        if remove_tokens:
            # Convert to string temporarily for remove_special_tokens
            if isinstance(result, list):
                temp_str = " ".join(result)
                temp_str = self.remove_special_tokens(temp_str, remove_tokens)
                result = temp_str.split()
            else:
                result = self.remove_special_tokens(result, remove_tokens)

        return result

    def apply_decimal_decomposition(
        self, tokens: Union[str, List[str]]
    ) -> Union[str, List[str]]:
        """
        Apply decimal decomposition preprocessing to direction tokens.

        Finds direction tokens (R/L/U/D followed by numbers) and decomposes them (except T/B tokens).
        into individual decimal components for better tokenization granularity.

        Supports both string input (space-separated tokens) and pre-split token lists.

        Args:
            tokens: Input tokens as string (space-separated tokens) or pre-split list of tokens

        Returns:
            Processed tokens in the same format as input:
            - str input → str output (space-separated)
            - List[str] input → List[str] output (token list)

        Examples:
            String input: 'R2200 B2 D300' → 'R2000 R200 B2 D300'
            List input: ['R2200', 'B2', 'D300'] → ['R2000', 'R200', 'B2', 'D300']

        Note:
            - Only decomposes tokens with values ≥ 10 (single digits remain unchanged)
            - Preserves non-direction tokens unchanged
            - Uses powers of 10 decomposition (2200 → 2000 + 200)
        """

        def decompose_token(token: str) -> List[str]:
            """Decompose a single direction token into decimal components"""
            if not token or len(token) < 2:
                return [token]

            prefix = token[0].upper()
            # Only decompose same-layer directions (R/L/U/D), preserve T/B tokens
            if prefix not in FLATTEN_DIRECTIONS or not token[1:].isdigit():
                return [token]

            value = int(token[1:])
            toks = self.split_decimal_token(prefix, value)
            return toks

        # Handle input format
        if isinstance(tokens, str):
            tokens = tokens.split()
            result = []
            for token in tokens:
                result.extend(decompose_token(token))
            return " ".join(result)
        elif isinstance(tokens, list):
            result = []
            for token in tokens:
                result.extend(decompose_token(token))
            return result
        else:
            raise ValueError(f"Expected str or List[str], got {type(tokens)}")

    def apply_segmentation(
        self, tokens: Union[str, List[str]]
    ) -> Union[str, List[str]]:
        """
        Apply segmentation preprocessing to separate concatenated direction tokens.

        Segments concatenated direction tokens (like R200U2000R200) that were previously
        concatenated to reduce token count. This is the reverse operation of concatenation.

        Supports both string input (space-separated tokens) and pre-split token lists.

        Args:
            tokens: Input tokens as string (space-separated tokens) or pre-split list of tokens

        Returns:
            Processed tokens in the same format as input:
            - str input → str output (space-separated)
            - List[str] input → List[str] output (token list)

        Examples:
            String input: 'R200U2000R200 B2 D300' → 'R200 U2000 R200 B2 D300'
            List input: ['R200U2000R200', 'B2', 'D300'] → ['R200', 'U2000', 'R200', 'B2', 'D300']

        Note:
            - Only segments tokens that match concatenated direction pattern
            - Preserves non-direction tokens and special tokens unchanged
            - Works with all direction prefixes: R/L (horizontal), U/D (vertical), T/B (layer)
        """
        # Handle input format
        if isinstance(tokens, str):
            # String input - split, process, and rejoin
            tokens = tokens.split()
            segmented_tokens = self.segment_concatenated_tokens(tokens)
            return " ".join(segmented_tokens)
        elif isinstance(tokens, list):
            # List input - process token list directly
            return self.segment_concatenated_tokens(tokens)
        else:
            raise ValueError(f"Expected str or List[str], got {type(tokens)}")

    def apply_concatenation(
        self, tokens: Union[str, List[str]]
    ) -> Union[str, List[str]]:
        """
        Apply concatenation preprocessing to merge consecutive same-layer direction tokens.

        Concatenates consecutive direction tokens that operate on the same metal layer
        (R/L/U/D tokens) without layer change tokens (T/B) in between. This reduces
        token count while preserving routing semantics.

        Supports both string input (space-separated tokens) and pre-split token lists.

        Args:
            tokens: Input tokens as string (space-separated tokens) or pre-split list of tokens

        Returns:
            Processed tokens in the same format as input:
            - str input → str output (space-separated)
            - List[str] input → List[str] output (token list)

        Examples:
            String input: 'R200 U2000 R200 B2 D300' → 'R200U2000R200 B2 D300'
            List input: ['R200', 'U2000', 'R200', 'B2', 'D300'] → ['R200U2000R200', 'B2', 'D300']

        Note:
            - Only concatenates horizontal/vertical movement tokens (R/L/U/D)
            - Layer change tokens (T/B) and special tokens break concatenation
            - Preserves routing semantics while reducing vocabulary size
            - Inverse operation of segmentation
        """
        # Handle input format
        if isinstance(tokens, str):
            # String input - split, process, and rejoin
            tokens = tokens.split()
            concatenated_tokens = self.concatenate_same_layer_tokens(tokens)
            return " ".join(concatenated_tokens)
        elif isinstance(tokens, list):
            # List input - process token list directly
            return self.concatenate_same_layer_tokens(tokens)
        else:
            raise ValueError(f"Expected str or List[str], got {type(tokens)}")

    def validate_preprocessing_compatibility(
        self,
        use_decimal_decomposition: bool = False,
        use_concatenation: bool = False,
        use_segmentation: bool = False,
    ) -> None:
        """
        Validate preprocessing option compatibility.

        Args:
            use_decimal_decomposition: Whether decimal decomposition is enabled
            use_concatenation: Whether concatenation is enabled
            use_segmentation: Whether segmentation is enabled

        Raises:
            TokenPreprocessingError: If incompatible options are enabled
        """
        if use_decimal_decomposition and use_concatenation:
            raise TokenPreprocessingError(
                tokens="N/A",
                operation="validation",
                reason="Cannot use decimal decomposition with concatenation (use_segmentation instead)",
            )
        if use_concatenation and use_segmentation:
            raise TokenPreprocessingError(
                tokens="N/A",
                operation="validation",
                reason="Cannot use both concatenation and segmentation simultaneously",
            )

    def remove_special_tokens(self, text: str, special_tokens: List[str]) -> str:
        """
        Remove special tokens from text (from tokenization/pipeline.py).

        Args:
            text: Text containing special tokens
            special_tokens: List of special tokens to remove

        Returns:
            Text with special tokens removed
        """
        result = text
        for token in special_tokens:
            result = result.replace(token, " ")
        return " ".join(result.split())  # Clean up extra spaces

    def split_decimal_token(self, prefix: str, value: int) -> List[str]:
        """
        Split decimal number into components (from dataset_generation/preparator.py).

        Args:
            prefix: Direction prefix (R, L, U, D, T, B)
            value: Distance value to decompose

        Returns:
            List of decimal component tokens
        """
        if value <= 10:
            return [f"{prefix}{value}"]

        s = str(value)
        n = len(s)
        toks = []
        for i, ch in enumerate(s):
            d = int(ch)
            if d == 0:
                continue
            place = 10 ** (n - i - 1)
            toks.append(f"{prefix}{d * place}")
        return toks

    def segment_concatenated_tokens(self, tokens: List[str]) -> List[str]:
        """Segment concatenated direction tokens into individual tokens."""
        if not tokens:
            return []

        result = []
        for token in tokens:
            # Check if token contains multiple direction tokens
            direction_matches = self.direction_token_pattern.findall(token)
            if len(direction_matches) > 1:
                # Split concatenated token into individual direction tokens
                result.extend(direction_matches)
            else:
                # Single token or non-direction token
                result.append(token)
        return result

    def concatenate_same_layer_tokens(self, tokens: List[str]) -> List[str]:
        """Concatenate consecutive direction tokens on same layer."""
        if not tokens:
            return []

        result = []
        current_group = []

        for token in tokens:
            if self.is_flatten_token(token):
                current_group.append(token)
            else:
                # Flush current group
                if current_group:
                    if len(current_group) == 1:
                        result.append(current_group[0])
                    else:
                        result.append("".join(current_group))
                    current_group = []
                # Add the non-same-layer token
                result.append(token)

        # Handle remaining group
        if current_group:
            if len(current_group) == 1:
                result.append(current_group[0])
            else:
                result.append("".join(current_group))

        return result

    def is_flatten_token(self, token: str) -> bool:
        """Check if token represents same-layer movement (R/L/U/D)."""
        if not isinstance(token, str) or len(token) < 2:
            return False
        # Check if it's a direction token and if it's same-layer
        direction_char = token[0].upper()
        return (
            direction_char in FLATTEN_DIRECTIONS
            and token[1:].isdigit()
            and not any(char in token for char in VIA_DIRECTIONS)
        )
