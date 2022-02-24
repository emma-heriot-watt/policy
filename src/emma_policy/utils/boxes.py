# Copyright (c) Facebook, Inc. and its affiliates.
import math
from enum import IntEnum, unique
from typing import Generator, Union

import numpy as np
import torch


RawBoxType = Union[list[float], tuple[float, ...], torch.Tensor, np.ndarray]


@unique
class BoxMode(IntEnum):
    """Different ways to represent a box.

    `XYXY_ABS`:     (x0, y0, x1, y1) in absolute floating points coordinates. The coordinates in
                    range [0, width or height].
    `XYWH_ABS`:     (x0, y0, w, h) in absolute floating points coordinates.
    `XYWHA_ABS`:    (xc, yc, w, h, a) in absolute floating points coordinates. (xc, yc) is the
                    center of the rotated box, and the angle a is in degrees CCW.
    """

    XYXY_ABS = 0  # noqa: WPS115
    XYWH_ABS = 1  # noqa: WPS115
    XYXY_REL = 2  # noqa: WPS115
    XYWH_REL = 3  # noqa: WPS115
    XYWHA_ABS = 4  # noqa: WPS115

    def convert(self, box: RawBoxType, from_mode: "BoxMode", to_mode: "BoxMode") -> RawBoxType:
        """Convert box to a different mode, returning in the same type as provided.

        Args:
            box: can be a k-tuple, k-list or an Nxk array/tensor, where k = 4 or 5.
            from_mode (BoxMode): Mode to convert from.
            to_mode (BoxMode): Mode to convert to.

        Returns:
            The converted box of the same type.

        Raises:
            AssertionError: Relative mode is not supported
        """
        if from_mode == to_mode:
            return box

        if to_mode in self._unsupported_modes or from_mode in self._unsupported_modes:
            raise AssertionError("Relative mode is not supported.")

        original_type = type(box)
        box_as_tensor = self._convert_to_torch(box)

        converted_box = self._convert(box_as_tensor, from_mode, to_mode)

        if isinstance(box, (list, tuple)):
            return original_type(converted_box.flatten().tolist())
        if isinstance(box, np.ndarray):
            return converted_box.numpy()

        return converted_box

    @property
    def _unsupported_modes(self) -> list["BoxMode"]:
        """Get a list of the unsupported modes."""
        return [BoxMode.XYXY_REL, BoxMode.XYWH_REL]

    def _convert(
        self, box: torch.Tensor, from_mode: "BoxMode", to_mode: "BoxMode"
    ) -> torch.Tensor:
        """Convert box to the desired mode if it's supported."""
        convert_functions = {
            BoxMode.XYWHA_ABS: {
                BoxMode.XYXY_ABS: self._convert_xywha_abs_to_xyxy_abs,
            },
            BoxMode.XYWH_ABS: {
                BoxMode.XYWHA_ABS: self._convert_xywh_abs_to_xywha_abs,
                BoxMode.XYXY_ABS: self._convert_xywh_abs_to_xyxy_abs,
            },
            BoxMode.XYXY_ABS: {
                BoxMode.XYWHA_ABS: self._convert_xyxy_abs_to_xywh_abs,
            },
        }

        try:
            converted_box = convert_functions[from_mode][to_mode](box)
        except KeyError:
            raise NotImplementedError(
                f"Conversion from BoxMode {from_mode} to {to_mode} is not supported."
            )

        return converted_box

    def _convert_xywha_abs_to_xyxy_abs(self, box: torch.Tensor) -> torch.Tensor:
        """Convert XYWHA_ABS to XYXY_ABS."""
        if box.shape[-1] != 5:
            raise AssertionError("The last dimension of input shape must be 5 for XYWHA format")

        original_dtype = box.dtype
        box = box.double()

        width = box[:, 2]
        height = box[:, 3]
        angle = box[:, 4]
        cos = torch.abs(torch.cos(angle * math.pi / 180))  # noqa: WPS432
        sin = torch.abs(torch.sin(angle * math.pi / 180))  # noqa: WPS432

        # Compute the horizontal bounding rectangle of the rotated box
        new_width = cos * width + sin * height
        new_height = cos * height + sin * width

        # Convert center to top-left corner
        box[:, 0] -= new_width / 2
        box[:, 1] -= new_height / 2

        # Bottom-right corner
        box[:, 2] = box[:, 0] + new_width
        box[:, 3] = box[:, 1] + new_height

        box = box[:, :4].to(dtype=original_dtype)

        return box

    def _convert_xywh_abs_to_xywha_abs(self, box: torch.Tensor) -> torch.Tensor:
        """Convert XYWH_ABS to XYWHA_ABS."""
        original_dtype = box.dtype
        box = box.double()

        box[:, 0] += box[:, 2] / 2
        box[:, 1] += box[:, 3] / 2

        angles = torch.zeros((box.size(0), 1), dtype=box.dtype)

        box = torch.cat((box, angles), dim=1).to(dtype=original_dtype)

        return box

    def _convert_xyxy_abs_to_xywh_abs(self, box: torch.Tensor) -> torch.Tensor:
        """Convert XYXY_ABS to XYWH_ABS."""
        box[:, 2] += box[:, 0]
        box[:, 3] += box[:, 1]

        return box

    def _convert_xywh_abs_to_xyxy_abs(self, box: torch.Tensor) -> torch.Tensor:
        """Convert XYWH_ABS to XYXY_ABS."""
        box[:, 2] -= box[:, 0]
        box[:, 3] -= box[:, 1]

        return box

    def _convert_to_torch(self, box: RawBoxType) -> torch.Tensor:
        """Convert the box from whatever type it is to a torch Tensor."""
        if isinstance(box, torch.Tensor):
            return box.clone()

        if isinstance(box, (list, tuple)):
            if len(box) < 4 or len(box) > 5:
                raise AssertionError(
                    "`BoxMode.convert` takes either a k-tuple/list or an Nxk array/tensor where k == 4 or 5."
                )

            return torch.tensor(box)[None, :]

        if isinstance(box, np.ndarray):
            return torch.from_numpy(np.asarray(box)).clone()

        raise NotImplementedError


class Boxes:
    """This structure stores a list of boxes as a Nx4 torch.Tensor.

    It supports some common methods about boxes
    (`area`, `clip`, `nonempty`, etc),
    and also behaves like a Tensor
    (support indexing, `to(device)`, `.device`, and iteration over all boxes)

    Attributes:
        tensor (torch.Tensor): float matrix of Nx4. Each row is (x1, y1, x2, y2).
    """

    def __init__(self, tensor: torch.Tensor) -> None:
        device = tensor.device if isinstance(tensor, torch.Tensor) else torch.device("cpu")
        tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device)

        if tensor.numel() == 0:
            # Use reshape, so we don't end up creating a new tensor that does not depend on
            # the inputs (and consequently confuses jit)
            tensor = tensor.reshape((-1, 4)).to(dtype=torch.float32, device=device)

        if tensor.dim() != 2 and tensor.size(-1) != 4:
            raise AssertionError(f"Tensor shape is incorrect. Current shape is {tensor.size()}")

        self.tensor = tensor

    @property
    def device(self) -> torch.device:
        """Get the device."""
        return self.tensor.device

    def __getitem__(self, index: Union[int, slice, torch.BoolTensor]) -> "Boxes":
        """Get a new `Boxes` by indexing.

        The following usage are allowed:
            1. `new_boxes = boxes[3]`: return a `Boxes` which contains only one box.
            2. `new_boxes = boxes[2:10]`: return a slice of boxes.
            3. `new_boxes = boxes[vector]`, where vector is a torch.BoolTensor
                with `length = len(boxes)`. Nonzero elements in the vector will be selected.

        Note that the returned Boxes might share storage with this Boxes, subject to Pytorch's
        indexing semantics.
        """
        if isinstance(index, int):
            return Boxes(self.tensor[index].view(1, -1))

        box = self.tensor[index]

        if box.dim() != 2:
            raise AssertionError(f"Indexing on Boxes with {index} failed to return a matrix!")

        return Boxes(box)

    def __len__(self) -> int:
        """Get number of boxes."""
        return self.tensor.shape[0]

    def __repr__(self) -> str:
        """Get string representation of this Boxes."""
        return f"Boxes({str(self.tensor)})"

    def clone(self) -> "Boxes":
        """Create a clone."""
        return Boxes(self.tensor.clone())

    def to(self, device: torch.device) -> "Boxes":
        """Move to another device."""
        # Boxes are assumed float32 and does not support to(dtype)
        return Boxes(self.tensor.to(device=device))

    def area(self) -> torch.Tensor:
        """Computes the area of all the boxes."""
        box = self.tensor
        width_difference = box[:, 2] - box[:, 0]
        height_difference = box[:, 3] - box[:, 1]
        area = width_difference * height_difference
        return area

    def clip(self, box_size: tuple[int, int]) -> None:
        """Clip (in place) the boxes.

        This is done by limiting x coordinates to the range [0, width] and y coordinates to the
        range [0, height].
        """
        if not torch.isfinite(self.tensor).all():
            raise AssertionError("Box tensor contains infinite or NaN!")

        h, w = box_size
        x1 = self.tensor[:, 0].clamp(min=0, max=w)
        y1 = self.tensor[:, 1].clamp(min=0, max=h)
        x2 = self.tensor[:, 2].clamp(min=0, max=w)
        y2 = self.tensor[:, 3].clamp(min=0, max=h)

        self.tensor = torch.stack((x1, y1, x2, y2), dim=-1)

    def nonempty(self, threshold: float = 0) -> torch.Tensor:
        """Find boxes that are non-empty.

        A box is considered empty, if either of its side is no larger than threshold.

        Args:
            threshold (float): Boxes larger than this threshold are considered empty.
                Defaults to 0.

        Returns:
            A torch Tensor that is a binary vector which represents whether each box is empty
            (False) or non-empty (True).
        """
        box = self.tensor
        widths = box[:, 2] - box[:, 0]
        heights = box[:, 3] - box[:, 1]
        keep = (widths > threshold) & (heights > threshold)  # noqa: WPS465
        return keep

    def inside_box(self, box_size: tuple[int, int], boundary_threshold: int = 0) -> torch.Tensor:
        """Get the inside of the box.

        Args:
            box_size (height, width): Size of the reference box.
            boundary_threshold (int): Boxes that extend beyond the reference box
                boundary by more than boundary_threshold are considered "outside".

        Returns:
            a binary vector, indicating whether each box is inside the reference box.
        """
        height, width = box_size
        inds_inside = (
            (self.tensor[..., 0] >= -boundary_threshold)  # noqa: WPS465
            & (self.tensor[..., 1] >= -boundary_threshold)
            & (self.tensor[..., 2] < width + boundary_threshold)
            & (self.tensor[..., 3] < height + boundary_threshold)
        )
        return inds_inside

    def get_centers(self) -> torch.Tensor:
        """Get the center of the box as Nx2 array of (x, y)."""
        return (self.tensor[:, :2] + self.tensor[:, 2:]) / 2

    def scale(self, scale_x: float, scale_y: float) -> None:
        """Scale the box with horizontal and vertical scaling factors."""
        self.tensor[:, 0::2] *= scale_x
        self.tensor[:, 1::2] *= scale_y

    @classmethod
    def cat(cls, boxes_list: list["Boxes"]) -> "Boxes":
        """Concatenates a list of Boxes into a single Boxes."""
        if not isinstance(boxes_list, (list, tuple)):
            raise AssertionError("Boxes list must be a list or a tuple.")

        if boxes_list:
            return cls(torch.empty(0))

        if not all([isinstance(box, Boxes) for box in boxes_list]):
            raise AssertionError("Every box in the list must be an instance of `Boxes`")

        # use torch.cat (v.s. layers.cat) so the returned boxes never share storage with input
        cat_boxes = cls(torch.cat([b.tensor for b in boxes_list], dim=0))
        return cat_boxes

    @torch.jit.unused
    def __iter__(self) -> Generator[torch.Tensor, None, None]:
        """Yield a box as a Tensor of shape (4,) at a time."""
        yield from self.tensor  # https://github.com/pytorch/pytorch/issues/18627


def pairwise_intersection(boxes1: Boxes, boxes2: Boxes) -> torch.Tensor:
    """Compute intersection area between all NxM pairs of boxes.

    The box order must be (xmin, ymin, xmax, ymax)

    Args:
        boxes1 (Boxes): Contains N boxes.
        boxes2 (Boxes): Contains M boxes.

    Returns:
        torch.Tensor: intersection of the boxes, sized [N,M].
    """
    box1_tensor = boxes1.tensor
    box2_tensor = boxes2.tensor

    width_height_min = torch.min(box1_tensor[:, None, 2:], box2_tensor[:, 2:])
    width_height_max = torch.max(box1_tensor[:, None, :2], box2_tensor[:, :2])

    width_height = width_height_min - width_height_max  # [N,M,2]
    width_height.clamp_(min=0)  # [N,M,2]

    intersection = width_height.prod(dim=2)  # [N,M]
    return intersection


def pairwise_iou(boxes1: Boxes, boxes2: Boxes) -> torch.Tensor:
    """Compute the IoU (intersection over union) between **all** N x M pairs of boxes.

    Implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
    with slight modifications

    The box order must be (xmin, ymin, xmax, ymax).

    Args:
        boxes1 (Boxes): Contains N boxes.
        boxes2 (Boxes): Contains M boxes.

    Returns:
        torch.Tensor: IoU, sized [N,M].
    """
    area1 = boxes1.area()  # [N]
    area2 = boxes2.area()  # [M]
    inter = pairwise_intersection(boxes1, boxes2)

    # handle empty boxes
    iou = torch.where(
        inter > 0,
        inter / (area1[:, None] + area2 - inter),
        torch.zeros(1, dtype=inter.dtype, device=inter.device),
    )
    return iou


def pairwise_ioa(boxes1: Boxes, boxes2: Boxes) -> torch.Tensor:
    """Compute the IoA (intersection over boxes2 area).

    Args:
        boxes1 (Boxes): Contains N boxes.
        boxes2 (Boxes): Contains M boxes.

    Returns:
        torch.Tensor: IoA, sized [N,M].
    """
    area2 = boxes2.area()  # [M]
    inter = pairwise_intersection(boxes1, boxes2)

    # handle empty boxes
    ioa = torch.where(
        inter > 0, inter / area2, torch.zeros(1, dtype=inter.dtype, device=inter.device)
    )
    return ioa


def pairwise_point_box_distance(points: torch.Tensor, boxes: Boxes) -> torch.Tensor:
    """Pairwise distance between N points and M boxes.

    The distance between a point and a box is represented by the distance from the point to 4 edges
    of the box. Distances are all positive when the point is inside the box.

    Args:
        points: Nx2 coordinates. Each row is (x, y)
        boxes: M boxes

    Returns:
        Tensor: distances of size (N, M, 4). The 4 values are distances from
            the point to the left, top, right, bottom of the box.
    """
    x, y = points.unsqueeze(dim=2).unbind(dim=1)  # (N, 1)
    x0, y0, x1, y1 = boxes.tensor.unsqueeze(dim=0).unbind(dim=2)  # (1, M)
    return torch.stack([x - x0, y - y0, x1 - x, y1 - y], dim=2)  # noqa: WPS221


def matched_pairwise_iou(boxes1: Boxes, boxes2: Boxes) -> torch.Tensor:
    """Compute pairwise intersection over union (IoU) of two sets of matched boxes.

    Both boxes must have the same number of boxes.

    Similar to `pairwise_iou`, but computes only diagonal elements of the matrix.

    Args:
        boxes1 (Boxes): bounding boxes, sized [N,4].
        boxes2 (Boxes): same length as boxes1

    Returns:
        Tensor: iou, sized [N].

    Raises:
        AssertionError: If Boxes do not contain same number of entries.
    """
    if len(boxes1) != len(boxes2):
        raise AssertionError(
            f"boxlists should have the same number of entries, got {len(boxes1)} and {len(boxes2)}"
        )

    area1 = boxes1.area()  # [N]
    area2 = boxes2.area()  # [N]
    box1, box2 = boxes1.tensor, boxes2.tensor
    lt = torch.max(box1[:, :2], box2[:, :2])  # [N,2]
    rb = torch.min(box1[:, 2:], box2[:, 2:])  # [N,2]
    wh = (rb - lt).clamp(min=0)  # [N,2]
    inter = wh[:, 0] * wh[:, 1]  # [N]
    iou = inter / (area1 + area2 - inter)  # [N]
    return iou
