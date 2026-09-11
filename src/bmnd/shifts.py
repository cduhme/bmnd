from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
import numpy as np

from .enums import ReferenceScheduleMode, coerce_enum


@dataclass(frozen=True)
class ReferenceScheduleItem:
    shift: tuple[int, ...]
    reference_abs: tuple[int, ...]
    reference_shifted: tuple[int, ...]


def get_shift_params(
    block_size: tuple[int, ...], step_size: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """These shift matrices are taken from the official bm4d implementation (https://pypi.org/project/bm4d/)."""
    block_size_1 = min(block_size[0], block_size[1])
    block_size_3 = block_size[2]

    step_size_1 = min(step_size[0], step_size[1])
    step_size_3 = step_size[2]

    shift_list = [[0], [0, 1], [0, 1, 2, 1], [0, 2, 1], [0, 2, 0, 1], [0, 2], [0, 1, 2], [0, 3, 2, 1], [0, 3],
                  [0, 2, 3, 1], [0, 1, 1, 0], [0, 3, 1], [0, 2, 1, 3], [0, 4, 3, 1], [0, 3, 4, 1], [0, 4, 2],
                  [0, 4], [0, 2, 4, 2], [0, 2, 4], [0, 4, 1, 3], [0, 1, 2, 3], [0, 3, 1, 2], [0, 3, 0, 2],
                  [0, 3, 1, 4], [0, 5, 1, 4], [0, 5], [0, 1, 4, 5], [0, 5, 1], [0, 1, 3], [0, 3, 5, 2],
                  [0, 5, 2, 3], [0, 1, 3, 4], [0, 5, 4, 1], [0, 2, 4, 1], [0, 3, 1, 5], [0, 6, 2, 4],
                  [0, 1, 4, 2], [0, 4, 6, 2], [0, 6, 3], [0, 2, 3, 5], [0, 1, 2, 4], [0, 1, 3, 5],
                  [0, 2, 4, 6], [0, 5, 2], [0, 4, 0, 1], [0, 5, 0, 2], [0, 6, 4, 2], [0, 7, 2, 5],
                  [0, 6], [0, 1, 4], [0, 5, 3, 2], [0, 2, 5], [0, 5, 7, 2], [0, 7, 2], [0, 7, 1, 6],
                  [0, 4, 5, 1], [0, 1, 5], [0, 3, 2, 5], [0, 4, 2, 6], [0, 3, 6, 1], [0, 6, 1],
                  [0, 1, 2, 6], [0, 6, 2], [0, 3, 6], [0, 5, 2, 7], [0, 1, 6, 7], [0, 6, 7, 1],
                  [0, 2, 7], [0, 7, 6, 1], [0, 1, 7], [0, 3, 6, 3]]

    shifts_1 = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 6, 6, 6, 0, 0, 0, 0, 0, 7, 7, 7, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 1, 1, 10, 1, 0, 0, 0, 0, 6, 6, 6, 6, 0, 0, 0, 0, 7, 7, 7, 7, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 6, 6, 6, 0, 0, 0, 0, 0, 5, 5, 5, 0, 0, 0, 0, 0, 5,
                5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 1, 1, 10, 10, 0, 0, 0, 0, 6, 6, 6, 6, 0, 0, 0, 0, 5, 5, 5, 5, 0, 0, 0, 0, 5,
                5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 1, 1, 10, 10, 10, 0, 0, 0, 6, 6, 6, 6, 6, 0, 0, 0, 5, 5, 5, 5, 5, 0, 0, 0, 5,
                5, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 1, 10, 1, 0, 0, 0, 0, 0, 3, 3, 3, 0, 0, 0, 0, 0, 20, 20, 20, 0, 0, 0, 0, 0, 22,
                5, 5, 0, 0, 0, 0, 0, 15, 15, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 3, 3, 3, 3, 0, 0, 0, 0, 20, 20, 20, 20, 0, 0, 0, 0, 22,
                22, 5, 5, 0, 0, 0, 0, 15, 15, 15, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 1, 1, 10, 1, 1, 0, 0, 0, 3, 3, 3, 3, 3, 0, 0, 0, 20, 20, 20, 20, 20, 0, 0, 0, 22,
                22, 5, 5, 5, 0, 0, 0, 15, 15, 15, 15, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 3, 3, 3, 3, 3, 3, 0, 0, 20, 20, 20, 20, 20, 20, 0, 0, 22,
                22, 22, 5, 5, 5, 0, 0, 15, 15, 15, 15, 15, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 6, 6, 6, 0, 0, 0, 0, 0, 5, 5, 5, 0, 0, 0, 0, 0, 22,
                5, 5, 0, 0, 0, 0, 0, 15, 8, 8, 0, 0, 0, 0, 0, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 1, 1, 10, 10, 0, 0, 0, 0, 6, 6, 6, 6, 0, 0, 0, 0, 5, 5, 5, 5, 0, 0, 0, 0, 22,
                22, 5, 5, 0, 0, 0, 0, 8, 8, 8, 8, 0, 0, 0, 0, 8, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 1, 1, 10, 10, 10, 0, 0, 0, 6, 6, 6, 6, 6, 0, 0, 0, 5, 5, 5, 5, 5, 0, 0, 0, 22,
                22, 5, 5, 5, 0, 0, 0, 8, 8, 8, 8, 8, 0, 0, 0, 8, 8, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 10, 1, 10, 10, 10, 10, 0, 0, 6, 6, 6, 6, 6, 6, 0, 0, 5, 5, 5, 5, 5, 5, 0, 0, 22,
                22, 22, 5, 5, 5, 0, 0, 15, 15, 8, 8, 8, 8, 0, 0, 8, 8, 8, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 1, 1, 10, 10, 10, 10, 10, 0, 6, 6, 6, 6, 6, 6, 6, 0, 5, 5, 5, 5, 5, 5, 5, 0, 22,
                22, 28, 5, 5, 5, 5, 0, 15, 15, 8, 8, 8, 8, 8, 0, 8, 8, 8, 8, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 10, 10, 10, 0, 0, 0, 0, 0, 6, 6, 6, 0, 0, 0, 0, 0, 9, 9, 9, 0, 0, 0, 0, 0, 44,
                44, 44, 0, 0, 0, 0, 0, 8, 8, 8, 0, 0, 0, 0, 0, 45, 5, 5, 0, 0, 0, 0, 0, 46, 46, 46, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 1, 10, 10, 10, 0, 0, 0, 0, 6, 6, 6, 6, 0, 0, 0, 0, 9, 9, 9, 9, 0, 0, 0, 0, 44,
                44, 44, 44, 0, 0, 0, 0, 8, 8, 8, 8, 0, 0, 0, 0, 45, 5, 5, 5, 0, 0, 0, 0, 46, 46, 46, 46, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 1, 1, 10, 10, 10, 0, 0, 0, 6, 6, 6, 6, 6, 0, 0, 0, 9, 9, 9, 9, 9, 0, 0, 0, 44,
                44, 1, 44, 44, 0, 0, 0, 8, 8, 8, 8, 8, 0, 0, 0, 5, 5, 5, 5, 5, 0, 0, 0, 46, 46, 46, 46, 46, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 10, 10, 10, 10, 10, 10, 0, 0, 3, 3, 6, 3, 3, 6, 0, 0, 9, 9, 9, 9, 9, 9, 0, 0, 44,
                44, 44, 44, 44, 44, 0, 0, 8, 8, 8, 8, 8, 8, 0, 0, 45, 45, 5, 5, 5, 5, 0, 0, 46, 46, 46, 46, 46, 46, 0,
                0, 0,
                0, 0, 0, 0, 0, 0, 0, 10, 10, 10, 10, 10, 10, 10, 0, 3, 3, 3, 3, 3, 3, 6, 0, 9, 9, 9, 9, 9, 9, 9, 0, 44,
                44, 44, 6, 44, 44, 44, 0, 8, 8, 8, 8, 8, 8, 8, 0, 45, 45, 5, 5, 5, 5, 5, 0, 46, 46, 46, 46, 46, 46, 46,
                0, 0,
                0, 0, 0, 0, 0, 0, 0, 1, 10, 10, 10, 10, 10, 10, 10, 3, 6, 3, 6, 3, 3, 3, 6, 9, 9, 9, 9, 9, 9, 9, 9, 44,
                44, 44, 44, 44, 44, 44, 44, 8, 8, 8, 8, 8, 8, 8, 8, 45, 45, 5, 5, 5, 5, 5, 5, 46, 46, 46, 46, 46, 46,
                46, 46, 71]

    shifts_2 = [0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 1, 0, 0, 0, 0, 0, 4, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 2, 1, 1, 0, 0, 0, 0, 0, 4, 5, 5, 0, 0, 0, 0, 0, 7, 8, 8, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 9, 7, 8, 7, 0, 0, 0, 0, 6, 6, 5, 5, 0, 0, 0, 0, 9, 11, 8, 8, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 2, 1, 1, 0, 0, 0, 0, 0, 3, 5, 5, 0, 0, 0, 0, 0, 12, 5, 5, 0, 0, 0, 0, 0, 13,
                8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 9, 7, 8, 8, 0, 0, 0, 0, 6, 6, 5, 5, 0, 0, 0, 0, 7, 5, 5, 5, 0, 0, 0, 0, 14,
                15, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 7, 7, 8, 8, 8, 0, 0, 0, 15, 15, 16, 16, 16, 0, 0, 0, 17, 18, 5, 5, 5, 0, 0, 0, 19,
                15, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 2, 1, 1, 0, 0, 0, 0, 0, 4, 5, 5, 0, 0, 0, 0, 0, 21, 8, 8, 0, 0, 0, 0, 0, 23,
                8, 8, 0, 0, 0, 0, 0, 24, 16, 25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 9, 20, 20, 20, 0, 0, 0, 0, 3, 3, 5, 5, 0, 0, 0, 0, 12, 20, 8, 8, 0, 0, 0, 0, 13,
                18, 8, 8, 0, 0, 0, 0, 26, 27, 16, 25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 20, 9, 1, 20, 20, 0, 0, 0, 18, 18, 5, 5, 5, 0, 0, 0, 20, 28, 8, 8, 8, 0, 0, 0, 18,
                18, 8, 8, 8, 0, 0, 0, 24, 15, 16, 16, 25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 9, 20, 20, 20, 20, 20, 0, 0, 15, 18, 18, 5, 5, 5, 0, 0, 29, 30, 30, 25, 25, 25, 0,
                0, 31,
                23, 18, 8, 8, 8, 0, 0, 32, 24, 27, 16, 16, 25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 2, 1, 1, 0, 0, 0, 0, 0, 2, 1, 1, 0, 0, 0, 0, 0, 12, 5, 5, 0, 0, 0, 0, 0, 33,
                5, 5, 0, 0, 0, 0, 0, 34, 16, 16, 0, 0, 0, 0, 0, 35, 25, 25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 9, 7, 8, 8, 0, 0, 0, 0, 6, 6, 1, 1, 0, 0, 0, 0, 7, 5, 5, 5, 0, 0, 0, 0, 36,
                18, 5, 5, 0, 0, 0, 0, 18, 18, 16, 16, 0, 0, 0, 0, 37, 38, 25, 25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 7, 7, 8, 8, 8, 0, 0, 0, 6, 3, 1, 1, 1, 0, 0, 0, 12, 12, 5, 5, 5, 0, 0, 0, 18,
                18, 5, 5, 5, 0, 0, 0, 15, 15, 16, 16, 16, 0, 0, 0, 35, 38, 25, 25, 25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0,
                0, 0, 0, 0, 0, 0, 0, 32, 7, 32, 8, 8, 8, 0, 0, 3, 6, 6, 1, 1, 1, 0, 0, 39, 29, 17, 5, 5, 5, 0, 0, 40,
                33, 18, 5, 5, 5, 0, 0, 41, 34, 18, 16, 16, 16, 0, 0, 37, 35, 38, 25, 25, 25, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0,
                0, 0, 0, 0, 0, 0, 0, 7, 9, 32, 8, 8, 8, 8, 0, 6, 9, 3, 1, 1, 1, 1, 0, 29, 39, 42, 42, 42, 42, 42, 0, 42,
                42, 38, 5, 5, 5, 5, 0, 37, 37, 15, 16, 16, 16, 16, 0, 35, 35, 43, 25, 25, 25, 25, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 2, 1, 1, 0, 0, 0, 0, 0, 3, 6, 6, 0, 0, 0, 0, 0, 7, 5, 5, 0, 0, 0, 0, 0, 19,
                16, 16, 0, 0, 0, 0, 0, 29, 8, 8, 0, 0, 0, 0, 0, 37, 25, 25, 0, 0, 0, 0, 0, 47, 48, 48, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 9, 7, 8, 8, 0, 0, 0, 0, 6, 6, 6, 6, 0, 0, 0, 0, 20, 11, 5, 5, 0, 0, 0, 0, 49,
                49, 16, 16, 0, 0, 0, 0, 50, 51, 8, 8, 0, 0, 0, 0, 35, 51, 25, 25, 0, 0, 0, 0, 52, 53, 48, 48, 0, 0, 0,
                0, 0,
                0, 0, 0, 0, 0, 0, 0, 7, 9, 8, 8, 8, 0, 0, 0, 15, 15, 15, 15, 15, 0, 0, 0, 12, 11, 5, 5, 5, 0, 0, 0, 49,
                6, 1, 16, 16, 0, 0, 0, 29, 39, 8, 8, 8, 0, 0, 0, 51, 43, 25, 25, 25, 0, 0, 0, 54, 53, 48, 48, 48, 0, 0,
                0, 0,
                0, 0, 0, 0, 0, 0, 0, 55, 26, 32, 8, 8, 8, 0, 0, 27, 27, 15, 56, 56, 15, 0, 0, 39, 29, 51, 5, 5, 5, 0, 0,
                26,
                55, 55, 16, 16, 16, 0, 0, 39, 57, 8, 8, 8, 8, 0, 0, 58, 42, 51, 25, 25, 25, 0, 0, 52, 47, 54, 48, 48,
                48, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 26, 26, 32, 8, 8, 8, 8, 0, 56, 27, 56, 56, 56, 56, 15, 0, 59, 59, 60, 48, 48, 48,
                48, 0, 61,
                61, 62, 48, 48, 48, 48, 0, 29, 29, 63, 8, 8, 8, 8, 0, 37, 37, 51, 25, 25, 25, 25, 0, 47, 64, 53, 48, 48,
                48, 48, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 20, 55, 32, 32, 8, 8, 8, 8, 56, 15, 27, 15, 56, 56, 56, 15, 65, 66, 67, 54, 48, 48,
                48, 48, 66,
                68, 69, 66, 48, 48, 48, 48, 29, 50, 38, 70, 8, 8, 8, 8, 37, 46, 51, 51, 25, 25, 25, 25, 64, 52, 67, 54,
                48, 48, 48, 48, 71]

    if block_size_3 == 1 and step_size_3 == 1 and 2 < block_size_1 < 9 and step_size_1 < 8:
        shift_list = [[0], [0, 1], [0, 2, 1], [0, 2], [0, 1, 2], [0, 3, 2, 1], [0, 3], [0, 1, 1, 0], [0, 1, 2, 3],
                      [0, 5], [0, 4], [0, 2, 3, 1], [0, 4, 0, 1], [0, 6]]
        shifts_1 = [0, 1, 3, 0, 0, 0, 0, 0, 0, 1, 3, 6, 0, 0, 0, 0, 0, 1, 3, 3, 6, 0, 0, 0, 0, 1, 3, 6, 6, 9, 0, 0, 0,
                    1, 1, 3, 3, 10, 9, 0, 0, 1, 4, 3, 10, 6, 9, 13, 14]
        block_size_1 -= 1
        step_size_1 -= 1
        ix = (block_size_1 - 2) * 8 + step_size_1
        sh_1 = shift_list[shifts_1[ix]]
        return np.array([0], dtype=int), np.array(sh_1, dtype=int), np.array([0], dtype=int)

    if block_size_1 > 8 or block_size_3 > 8 or step_size_1 > block_size_1 or step_size_3 > block_size_3 or \
        block_size_3 > block_size_1 or min(block_size_1, block_size_3) < 3:
        return np.array([0], dtype=int), np.array([0], dtype=int), np.array([0], dtype=int)

    block_size_1 -= 1
    block_size_3 -= 1
    step_size_1 -= 1
    step_size_3 -= 1

    ix = (block_size_1 - 2) * 6 * 8 * 8 + (block_size_3 - 2) * 8 * 8 + step_size_1 * 8 + step_size_3
    sh_1 = shift_list[shifts_1[ix]]
    sh_2 = shift_list[shifts_2[ix]]

    return np.array(sh_2, dtype=int), np.array(sh_1, dtype=int), np.array([0], dtype=int)



def _generate_axis_shifts(
    block: int,
    step: int,
    *,
    allow_multiple: bool,
    density: float,
) -> np.ndarray:
    if not allow_multiple or block < 3 or step < 1 or step > block:
        return np.array([0], dtype=int)

    safe_density = max(float(density), 0.0)
    shift_count = max(1, math.ceil(safe_density * (block - 1) / step))
    shift_count = min(shift_count, block)
    if shift_count == 1:
        return np.array([0], dtype=int)

    positions = np.linspace(0.0, float(block - 1), num=shift_count)
    shifts = sorted({int(round(position)) for position in positions} | {0})
    return np.array(shifts, dtype=int)


def get_generated_reference_shifts(
    block_size: tuple[int, ...],
    step_size: tuple[int, ...],
    shift_density: float = 1.0,
) -> tuple[np.ndarray, ...]:
    shifts: list[np.ndarray] = []
    ndim = len(block_size)
    for axis, (block, step) in enumerate(zip(block_size, step_size)):
        shifts.append(
            _generate_axis_shifts(
                int(block),
                int(step),
                allow_multiple=axis < max(ndim - 1, 0),
                density=shift_density,
            )
        )
    return tuple(shifts)


def get_balanced_reference_shifts(
    block_size: tuple[int, ...],
    step_size: tuple[int, ...],
    shift_density: float = 1.0,
) -> tuple[np.ndarray, ...]:
    return tuple(
        _generate_axis_shifts(
            int(block),
            int(step),
            allow_multiple=True,
            density=shift_density,
        )
        for block, step in zip(block_size, step_size)
    )


def get_sparse_reference_shifts(
    block_size: tuple[int, ...],
    step_size: tuple[int, ...],
    shift_density: float = 1.0,
    *,
    active_axes: tuple[int, ...] | None = None,
) -> tuple[np.ndarray, ...]:
    if active_axes is None:
        active_axes = tuple(
            axis for axis, block in enumerate(block_size) if int(block) > 1
        )
    shifted_axes = set(active_axes[:-1])
    return tuple(
        _generate_axis_shifts(
            int(block),
            int(step),
            allow_multiple=axis in shifted_axes,
            density=shift_density,
        )
        for axis, (block, step) in enumerate(zip(block_size, step_size))
    )


def get_reference_shifts(
    block_size: tuple[int, ...],
    step_size: tuple[int, ...],
    mode: ReferenceScheduleMode = ReferenceScheduleMode.OFFICIAL,
    shift_density: float = 1.0,
) -> tuple[np.ndarray, ...]:
    mode = coerce_enum(mode, ReferenceScheduleMode, "reference_schedule_mode")
    if mode is ReferenceScheduleMode.OFF:
        return tuple(np.array([0], dtype=int) for _ in block_size)
    if mode is ReferenceScheduleMode.GENERATED:
        return get_generated_reference_shifts(
            block_size,
            step_size,
            shift_density=shift_density,
        )
    if mode is ReferenceScheduleMode.BALANCED:
        return get_balanced_reference_shifts(
            block_size,
            step_size,
            shift_density=shift_density,
        )
    if mode is ReferenceScheduleMode.SPARSE:
        return get_sparse_reference_shifts(
            block_size,
            step_size,
            shift_density=shift_density,
        )

    if mode is not ReferenceScheduleMode.OFFICIAL:
        raise ValueError(
            f"Unknown reference shift mode: {mode!r} "
            "(expected 'official', 'generated', 'balanced', 'sparse', or 'off')."
        )

    if len(block_size) == 2:
        shifts_x, shifts_y, _ = get_shift_params(
            (block_size[0], block_size[1], 1),
            (step_size[0], step_size[1], 1),
        )
        return shifts_y, shifts_x

    if len(block_size) == 3:
        shifts_x, shifts_y, shifts_z = get_shift_params(block_size, step_size)
        return shifts_y, shifts_x, shifts_z

    return tuple(np.array([0], dtype=int) for _ in block_size)


def get_reference_starts(
    counts: tuple[int, ...],
    block_size: tuple[int, ...],
    step_size: tuple[int, ...],
    mode: ReferenceScheduleMode = ReferenceScheduleMode.OFFICIAL,
    shift_density: float = 1.0,
) -> list[list[int]]:
    mode = coerce_enum(mode, ReferenceScheduleMode, "reference_schedule_mode")
    if mode is ReferenceScheduleMode.OFF:
        return _get_unshifted_reference_starts(counts, step_size)

    shifts = get_reference_shifts(
        block_size,
        step_size,
        mode=mode,
        shift_density=shift_density,
    )
    starts_per_dim: list[list[int]] = []
    for dim, dim_count in enumerate(counts):
        dim_shifts = shifts[dim] if dim < len(shifts) else np.array([0], dtype=int)
        starts = {
            start
            for shift in dim_shifts.tolist()
            for start in range(int(shift), dim_count, step_size[dim])
            if start < dim_count
        }
        if not starts:
            starts = {0}
        starts_per_dim.append(sorted(starts))
    return starts_per_dim


def get_reference_positions(
    counts: tuple[int, ...],
    block_size: tuple[int, ...],
    step_size: tuple[int, ...],
    mode: ReferenceScheduleMode = ReferenceScheduleMode.OFFICIAL,
    shift_density: float = 1.0,
) -> list[tuple[int, ...]]:
    starts_per_dim = get_reference_starts(
        counts,
        block_size,
        step_size,
        mode=mode,
        shift_density=shift_density,
    )
    return list(itertools.product(*starts_per_dim))


def get_reference_schedule(
    counts: tuple[int, ...],
    block_size: tuple[int, ...],
    step_size: tuple[int, ...],
    mode: ReferenceScheduleMode = ReferenceScheduleMode.OFFICIAL,
    shift_density: float = 1.0,
    schedule_density: int = 1,
) -> list[ReferenceScheduleItem]:
    mode = coerce_enum(mode, ReferenceScheduleMode, "reference_schedule_mode")
    if mode is ReferenceScheduleMode.OFF:
        return _get_reference_schedule_off(counts, step_size)
    if mode is ReferenceScheduleMode.GENERATED:
        return _get_reference_schedule_generated(
            counts,
            block_size,
            step_size,
            shift_density=shift_density,
            schedule_density=schedule_density,
        )
    if mode is ReferenceScheduleMode.BALANCED:
        return _get_reference_schedule_balanced(
            counts,
            block_size,
            step_size,
            shift_density=shift_density,
            schedule_density=schedule_density,
        )
    if mode is ReferenceScheduleMode.SPARSE:
        return _get_reference_schedule_sparse(
            counts,
            block_size,
            step_size,
            shift_density=shift_density,
            schedule_density=schedule_density,
        )

    if mode is not ReferenceScheduleMode.OFFICIAL:
        raise ValueError(
            f"Unknown reference schedule mode: {mode!r} "
            "(expected 'official', 'generated', 'balanced', 'sparse', or 'off')."
        )

    if len(block_size) == 3:
        return _get_reference_schedule_3d(counts, block_size, step_size)

    shifts = get_reference_shifts(
        block_size,
        step_size,
        mode=mode,
        shift_density=shift_density,
    )
    shift_options: list[list[int]] = []
    for dim in range(len(counts)):
        dim_shifts = shifts[dim] if dim < len(shifts) else np.array([0], dtype=int)
        shift_options.append([int(shift) for shift in dim_shifts.tolist()] or [0])

    schedule: list[ReferenceScheduleItem] = []
    for shift_tuple in itertools.product(*shift_options):
        starts_per_dim: list[list[int]] = []
        for dim, dim_count in enumerate(counts):
            starts = list(range(shift_tuple[dim], dim_count, step_size[dim]))
            if not starts:
                starts = [0]
            starts_per_dim.append(starts)

        for reference_abs in itertools.product(*starts_per_dim):
            reference_abs_tuple = tuple(int(x) for x in reference_abs)
            reference_shifted = tuple(
                int(reference_abs_tuple[dim] - shift_tuple[dim]) for dim in range(len(counts))
            )
            schedule.append(
                ReferenceScheduleItem(
                    shift=tuple(int(x) for x in shift_tuple),
                    reference_abs=reference_abs_tuple,
                    reference_shifted=reference_shifted,
                )
            )

    return schedule


def _get_reference_schedule_off(
    counts: tuple[int, ...],
    step_size: tuple[int, ...],
) -> list[ReferenceScheduleItem]:
    starts_per_dim = _get_unshifted_reference_starts(counts, step_size)
    if not starts_per_dim:
        return []

    zero_shift = (0,) * len(counts)
    return [
        ReferenceScheduleItem(
            shift=zero_shift,
            reference_abs=tuple(int(position) for position in reference_abs),
            reference_shifted=tuple(int(position) for position in reference_abs),
        )
        for reference_abs in itertools.product(*starts_per_dim)
    ]


def _get_unshifted_reference_starts(
    counts: tuple[int, ...],
    step_size: tuple[int, ...],
) -> list[list[int]]:
    starts_per_dim: list[list[int]] = []
    for count, step in zip(counts, step_size):
        last_valid = max(int(count) - 1, 0)
        starts = list(range(0, last_valid + 1, int(step)))
        if starts[-1] != last_valid:
            starts.append(last_valid)
        starts_per_dim.append(starts)
    return starts_per_dim


def _get_reference_schedule_generated(
    counts: tuple[int, ...],
    block_size: tuple[int, ...],
    step_size: tuple[int, ...],
    shift_density: float = 1.0,
    schedule_density: int = 1,
) -> list[ReferenceScheduleItem]:
    ndim = len(counts)
    if ndim == 0:
        return []

    shifts = get_generated_reference_shifts(
        block_size,
        step_size,
        shift_density=shift_density,
    )
    last_valid = [max(int(count) - 1, 0) for count in counts]
    slot_counts = [
        (last_valid[dim] + step_size[dim] - 1) // step_size[dim] + 1 for dim in range(ndim)
    ]
    phase_variants = max(1, int(schedule_density))

    schedule: list[ReferenceScheduleItem] = []
    seen: set[tuple[int, ...]] = set()

    for slot_tuple in itertools.product(*(range(count) for count in slot_counts)):
        for phase in range(phase_variants):
            reference_shifted = []
            reference_abs = []
            shift = []
            for dim in range(ndim):
                base = min(slot_tuple[dim] * step_size[dim], last_valid[dim])
                if dim < ndim - 1:
                    dim_shifts = shifts[dim].tolist() or [0]
                    offset_index = (slot_tuple[dim + 1] + phase) % len(dim_shifts)
                    offset = int(dim_shifts[offset_index])
                else:
                    offset = 0
                abs_pos = min(max(base - offset, 0), last_valid[dim])
                if slot_tuple[dim] == slot_counts[dim] - 1:
                    abs_pos = last_valid[dim]
                reference_shifted.append(int(base))
                reference_abs.append(int(abs_pos))
                shift.append(int(base - abs_pos))

            reference_abs_tuple = tuple(reference_abs)
            if reference_abs_tuple in seen:
                continue
            seen.add(reference_abs_tuple)
            schedule.append(
                ReferenceScheduleItem(
                    shift=tuple(shift),
                    reference_abs=reference_abs_tuple,
                    reference_shifted=tuple(reference_shifted),
                )
            )

    return schedule


def _get_reference_schedule_balanced(
    counts: tuple[int, ...],
    block_size: tuple[int, ...],
    step_size: tuple[int, ...],
    shift_density: float = 1.0,
    schedule_density: int = 1,
) -> list[ReferenceScheduleItem]:
    ndim = len(counts)
    if ndim == 0:
        return []

    shifts = get_balanced_reference_shifts(
        block_size,
        step_size,
        shift_density=shift_density,
    )
    last_valid = [max(int(count) - 1, 0) for count in counts]
    slot_counts = [
        (last_valid[dim] + step_size[dim] - 1) // step_size[dim] + 1
        for dim in range(ndim)
    ]
    phase_variants = max(1, int(schedule_density))

    schedule: list[ReferenceScheduleItem] = []
    seen: set[tuple[int, ...]] = set()

    for slot_tuple in itertools.product(*(range(count) for count in slot_counts)):
        slot_sum = sum(slot_tuple)
        for phase in range(phase_variants):
            reference_shifted = []
            reference_abs = []
            shift = []
            for dim in range(ndim):
                base = min(slot_tuple[dim] * step_size[dim], last_valid[dim])
                dim_shifts = shifts[dim]
                offset_index = (phase + slot_sum - slot_tuple[dim]) % len(dim_shifts)
                offset = int(dim_shifts[offset_index])
                abs_pos = min(max(base - offset, 0), last_valid[dim])
                if slot_tuple[dim] == slot_counts[dim] - 1:
                    abs_pos = last_valid[dim]
                reference_shifted.append(int(base))
                reference_abs.append(int(abs_pos))
                shift.append(int(base - abs_pos))

            reference_abs_tuple = tuple(reference_abs)
            if reference_abs_tuple in seen:
                continue
            seen.add(reference_abs_tuple)
            schedule.append(
                ReferenceScheduleItem(
                    shift=tuple(shift),
                    reference_abs=reference_abs_tuple,
                    reference_shifted=tuple(reference_shifted),
                )
            )

    return schedule


def _get_reference_schedule_sparse(
    counts: tuple[int, ...],
    block_size: tuple[int, ...],
    step_size: tuple[int, ...],
    shift_density: float = 1.0,
    schedule_density: int = 1,
) -> list[ReferenceScheduleItem]:
    ndim = len(counts)
    if ndim == 0:
        return []

    active_axes = tuple(
        dim
        for dim in range(ndim)
        if int(counts[dim]) > 1 or int(block_size[dim]) > 1
    )
    next_active_axis = {
        active_axes[index]: active_axes[index + 1]
        for index in range(max(len(active_axes) - 1, 0))
    }
    shifts = get_sparse_reference_shifts(
        block_size,
        step_size,
        shift_density=shift_density,
        active_axes=active_axes,
    )
    last_valid = [max(int(count) - 1, 0) for count in counts]
    slot_counts = [
        (last_valid[dim] + step_size[dim] - 1) // step_size[dim] + 1
        for dim in range(ndim)
    ]
    phase_variants = max(1, int(schedule_density))

    schedule: list[ReferenceScheduleItem] = []
    seen: set[tuple[int, ...]] = set()

    for slot_tuple in itertools.product(*(range(count) for count in slot_counts)):
        for phase in range(phase_variants):
            reference_shifted = []
            reference_abs = []
            shift = []
            for dim in range(ndim):
                base = min(slot_tuple[dim] * step_size[dim], last_valid[dim])
                controller_axis = next_active_axis.get(dim)
                if controller_axis is None:
                    offset = 0
                else:
                    dim_shifts = shifts[dim]
                    offset_index = (slot_tuple[controller_axis] + phase) % len(dim_shifts)
                    offset = int(dim_shifts[offset_index])
                abs_pos = min(max(base - offset, 0), last_valid[dim])
                if slot_tuple[dim] == slot_counts[dim] - 1:
                    abs_pos = last_valid[dim]
                reference_shifted.append(int(base))
                reference_abs.append(int(abs_pos))
                shift.append(int(base - abs_pos))

            reference_abs_tuple = tuple(reference_abs)
            if reference_abs_tuple in seen:
                continue
            seen.add(reference_abs_tuple)
            schedule.append(
                ReferenceScheduleItem(
                    shift=tuple(shift),
                    reference_abs=reference_abs_tuple,
                    reference_shifted=tuple(reference_shifted),
                )
            )

    return schedule


def _get_reference_schedule_3d(
    counts: tuple[int, ...],
    block_size: tuple[int, ...],
    step_size: tuple[int, ...],
) -> list[ReferenceScheduleItem]:
    shifts = get_reference_shifts(block_size, step_size)
    last_valid = [max(int(count) - 1, 0) for count in counts]
    slot_counts = [
        (last_valid[dim] + step_size[dim] - 1) // step_size[dim] + 1 for dim in range(3)
    ]

    row_shifts = [int(shift) for shift in shifts[0].tolist()] or [0]
    col_shifts = [int(shift) for shift in shifts[1].tolist()] or [0]

    schedule: list[ReferenceScheduleItem] = []
    seen: set[tuple[int, int, int]] = set()

    for z_slot in range(slot_counts[2]):
        z_abs = min(z_slot * step_size[2], last_valid[2])
        col_offset = -col_shifts[z_slot % len(col_shifts)]
        for row_slot in range(slot_counts[0]):
            row_base = min(row_slot * step_size[0], last_valid[0])
            for col_slot in range(slot_counts[1]):
                col_base = min(col_slot * step_size[1], last_valid[1])
                col_abs = min(max(col_slot * step_size[1] + col_offset, 0), last_valid[1])
                if col_slot == slot_counts[1] - 1:
                    col_abs = last_valid[1]
                row_offset = -row_shifts[col_slot % len(row_shifts)]
                row_abs = min(max(row_slot * step_size[0] + row_offset, 0), last_valid[0])
                if row_slot == slot_counts[0] - 1:
                    row_abs = last_valid[0]
                reference_abs = (row_abs, col_abs, z_abs)
                if reference_abs in seen:
                    continue
                seen.add(reference_abs)
                schedule.append(
                    ReferenceScheduleItem(
                        shift=(row_base - row_abs, col_base - col_abs, 0),
                        reference_abs=reference_abs,
                        reference_shifted=(row_base, col_base, z_abs),
                    )
                )

    return schedule
