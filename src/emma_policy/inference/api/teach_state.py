from pathlib import Path
from typing import Literal, TypedDict

from emma_policy.inference.model_wrapper import PolicyModelWrapper


TeachDatasetSplit = Literal["train", "valid_seen", "valid_unseen", "test_seen", "test_unseen"]


class ApiStore(TypedDict, total=False):
    """Common state for the API."""

    data_dir: Path
    images_dir: Path
    split: TeachDatasetSplit
    model: PolicyModelWrapper
