#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   plot_utils.py
@Time    :   2025/12/04 17:07:43
@Author  :   Dawn Li
@Version :   1.0
@Contact :   dawnli619215645@gmail.com
@Desc    :   Plotting utilities shared across evaluation components.
"""

from typing import List, Sequence, Tuple

COLOR_PALETTE: List[Tuple[float, float, float]] = [
    (249 / 255, 199 / 255, 79 / 255),
    (196 / 255, 194 / 255, 94 / 255),
    (144 / 255, 190 / 255, 109 / 255),
    (106 / 255, 180 / 255, 124 / 255),
    (67 / 255, 170 / 255, 139 / 255),
    (77 / 255, 144 / 255, 142 / 255),
    (87 / 255, 117 / 255, 144 / 255),
]


def palette_slice(
    count: int, palette: Sequence[Tuple[float, float, float]] = COLOR_PALETTE
) -> List[Tuple[float, float, float]]:
    """
    Return a slice of distinct colors, evenly spaced across the palette.

    Args:
        count: Number of colors requested.
        palette: Source palette to sample from.

    Returns:
        List of RGB tuples sized to ``count``.
    """
    if count <= 0:
        return []

    palette_len = len(palette)
    if count >= palette_len:
        return [palette[i % palette_len] for i in range(count)]
    if count == 1:
        return [palette[0]]

    indices = [int(round(i * (palette_len - 1) / (count - 1))) for i in range(count)]
    return [palette[idx] for idx in indices]
