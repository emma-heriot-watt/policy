from typing import Union

import numpy as np
import torch
from numpy import typing


def decompress_simbot_mask(
    compressed_mask: list[list[int]],
    image_width: int = 300,
    image_height: int = 300,
    return_tensor: bool = False,
) -> Union[torch.Tensor, typing.NDArray[np.float64]]:
    """Decompress a compressed mask array.

    Adopted from
    https://us-east-1.console.aws.amazon.com/codesuite/codecommit/repositories/AlexaSimbotMLToolbox/browse/refs/heads/main/--/AlexaSimbotToolbox/arena_wrapper/util/__init__.py?region=us-east-1
    """
    mask = np.zeros((image_width, image_height))
    for start_idx, run_len in compressed_mask:
        for idx in range(start_idx, start_idx + run_len):
            mask[idx // image_width, idx % image_height] = 1
    if return_tensor:
        return torch.tensor(mask)
    return mask


def compress_simbot_mask(
    segmentation_mask: Union[typing.NDArray[np.float64], list[list[int]]]
) -> list[list[int]]:
    """Compress a binary 2D array mask for the simbot arena.

    Adopted from
    https://us-east-1.console.aws.amazon.com/codesuite/codecommit/repositories/AlexaSimbotMLToolbox/browse/refs/heads/main/--/AlexaSimbotToolbox/arena_wrapper/util/__init__.py?region=us-east-1
    """
    # list of lists of run lengths for 1s, which are assumed to be less frequent.
    run_len_compressed: list[list[int]] = []
    idx = 0
    curr_run = False
    run_len = 0
    for x_idx, _ in enumerate(segmentation_mask):
        for y_idx, _ in enumerate(segmentation_mask[x_idx]):
            (curr_run, run_len, run_len_compressed) = get_compressed_mask_values(
                seg_xy=segmentation_mask[x_idx][y_idx],
                idx=idx,
                curr_run=curr_run,
                run_len=run_len,
                run_len_compressed=run_len_compressed,
            )
            idx += 1
    if curr_run:
        run_len_compressed[-1][1] = run_len
    return run_len_compressed


def get_compressed_mask_values(
    seg_xy: int,
    idx: int,
    curr_run: bool,
    run_len: int,
    run_len_compressed: list[list[int]],
) -> tuple[bool, int, list[list[int]]]:
    """Get values for the compressed version of the mask."""
    if seg_xy == 1 and not curr_run:
        curr_run = True
        run_len_compressed.append([idx, None])  # type: ignore[list-item]
    if seg_xy == 0 and curr_run:
        curr_run = False
        run_len_compressed[-1][1] = run_len
        run_len = 0
    if curr_run:
        run_len += 1
    return (curr_run, run_len, run_len_compressed)
