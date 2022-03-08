import dataclasses
from typing import Optional

import torch
from torch.nn.utils.rnn import pad_sequence

from emma_policy.datamodules.emma_dataclasses import (
    EmmaDatasetBatch,
    EmmaDatasetItem,
    EmmaDatasetPadding,
)


def collate_fn(batch: list[Optional[EmmaDatasetItem]]) -> EmmaDatasetBatch:
    """Collate lists of samples into batches after padding."""
    fields = dataclasses.fields(EmmaDatasetItem)
    padding = EmmaDatasetPadding()

    raw_batch = {
        field.name: pad_sequence(
            [getattr(sample, field.name) for sample in batch if sample is not None],
            batch_first=True,
            padding_value=getattr(padding, field.name),
        )
        for field in fields
    }
    raw_batch["attention_mask"] = torch.cat(
        [
            raw_batch["scene_attention_mask"],
            raw_batch["object_attention_mask"],
            raw_batch["text_attention_mask"],
        ],
        dim=-1,
    )

    # TODO: define global attention pattern
    raw_batch["global_attention_mask"] = torch.zeros_like(raw_batch["attention_mask"])

    return EmmaDatasetBatch(**raw_batch)
