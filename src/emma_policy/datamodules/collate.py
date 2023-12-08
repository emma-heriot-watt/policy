import dataclasses
from typing import Any, Optional

import torch
from torch.nn.utils.rnn import pad_sequence

from emma_policy.datamodules.batch_attention_masks import make_batch_attention_masks
from emma_policy.datamodules.emma_dataclasses import (
    EmmaDatasetBatch,
    EmmaDatasetItem,
    EmmaDatasetPadding,
)


RAW_FIELDS = ("raw_target",)


def _pad_sequence(seq: list[torch.Tensor], padding_value: int) -> torch.Tensor:
    if not seq:
        return torch.empty(0)
    return pad_sequence(seq, batch_first=True, padding_value=padding_value)


def collate_fn(batch: list[Optional[EmmaDatasetItem]]) -> EmmaDatasetBatch:
    """Collate lists of samples into batches after padding."""
    fields = dataclasses.fields(EmmaDatasetItem)
    padding = EmmaDatasetPadding()

    raw_batch: dict[Any, Any] = {}
    for field in fields:
        if field.name in RAW_FIELDS:
            raw_batch[field.name] = [
                getattr(sample, field.name)
                for sample in batch
                if sample is not None and getattr(sample, field.name) is not None
            ]
        else:
            raw_batch[field.name] = _pad_sequence(
                [
                    getattr(sample, field.name)
                    for sample in batch
                    if sample is not None and getattr(sample, field.name) is not None
                ],
                padding_value=getattr(padding, field.name),
            )
    make_batch_attention_masks(raw_batch, padding_value=padding.attention_mask)
    # For the new version decoder_input_ids are not needed for image-based tasks
    # In fact they must be removed since if they are present they are assumed by the model.
    if (  # noqa: WPS337
        raw_batch["decoder_input_ids"] is None
        or raw_batch["decoder_input_ids"].shape[0] != raw_batch["target_token_ids"].shape[0]
    ):
        raw_batch.pop("decoder_input_ids", None)
    return EmmaDatasetBatch(**raw_batch)
