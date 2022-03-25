import dataclasses
from typing import Optional

import torch
from torch.nn.utils.rnn import pad_sequence

from emma_policy.datamodules.emma_dataclasses import (
    EmmaDatasetBatch,
    EmmaDatasetItem,
    EmmaDatasetPadding,
)


def make_text_history_global_pattern(
    total_seq_len: int,
    text_temporal_ids: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Create 1D global attention mask that treats input text as global tokens."""
    bsz = text_temporal_ids.shape[0]
    global_attention_mask = torch.zeros((bsz, total_seq_len))

    # text tokens are at the end of the total sequence
    # use abs because text_temporal_ids will be -1 for text and 0 for padding tokens.
    global_attention_mask[:, -text_temporal_ids.shape[1] :] = torch.abs(text_temporal_ids)
    return global_attention_mask.to(dtype)


def make_encoder_causal_mask_batch(
    scene_temporal_ids: torch.Tensor,
    object_temporal_ids: torch.Tensor,
    text_temporal_ids: torch.Tensor,
    dtype: torch.dtype,
    padding_value: int = 0,
) -> torch.Tensor:
    """Create a 2D attention mask that masks future and padding tokens."""
    pos_mask_value = 1
    neg_mask_value = 0
    temp_ids = torch.cat([scene_temporal_ids, object_temporal_ids, text_temporal_ids], dim=1)
    total_seq_len = temp_ids.shape[1]
    bsz = temp_ids.shape[0]
    attention_mask = torch.full((bsz, total_seq_len, total_seq_len), neg_mask_value)

    expanded_temp_ids_cols = (
        temp_ids[:, :, None].expand(bsz, total_seq_len, total_seq_len).to(dtype)
    )
    expanded_temp_ids_rows = (
        temp_ids[:, None, :].expand(bsz, total_seq_len, total_seq_len).to(dtype)
    )
    # Mask future tokens
    attention_mask.masked_fill_(expanded_temp_ids_rows <= expanded_temp_ids_cols, pos_mask_value)
    # Mask attention to and from padding tokens
    attention_mask.masked_fill_(expanded_temp_ids_cols == padding_value, neg_mask_value)
    attention_mask.masked_fill_(expanded_temp_ids_rows == padding_value, neg_mask_value)
    return attention_mask.to(dtype)


def make_batch_attention_masks(raw_batch: dict[str, torch.Tensor]) -> None:
    """Make local and global attention masks."""
    raw_batch["attention_mask"] = torch.cat(
        [
            raw_batch["scene_attention_mask"],
            raw_batch["object_attention_mask"],
            raw_batch["text_attention_mask"],
        ],
        dim=-1,
    )
    got_scene_ids = (
        raw_batch["scene_temporal_ids"].shape[-1] == raw_batch["scene_attention_mask"].shape[-1]
    )
    got_object_ids = (
        raw_batch["object_temporal_ids"].shape[-1] == raw_batch["object_attention_mask"].shape[-1]
    )
    if got_scene_ids and got_object_ids:
        text_temporal_ids = torch.zeros_like(raw_batch["text_attention_mask"])
        text_temporal_ids.masked_fill_(raw_batch["text_attention_mask"] == 1, -1)
        # TODO: Use the causal attention mask once the model can handle 2D attention masks
        # raw_batch["attention_mask"] = make_encoder_causal_mask_batch(
        #     scene_temporal_ids=raw_batch["scene_temporal_ids"],
        #     object_temporal_ids=raw_batch["object_temporal_ids"],
        #     text_temporal_ids=text_temporal_ids,
        #     dtype=raw_batch["scene_temporal_ids"].dtype,
        #     padding_value=padding.attention_mask,
        # )
        raw_batch["global_attention_mask"] = make_text_history_global_pattern(
            total_seq_len=raw_batch["attention_mask"].shape[-1],
            text_temporal_ids=text_temporal_ids,
            dtype=raw_batch["scene_temporal_ids"].dtype,
        )

    else:
        # TODO: define global attention pattern
        raw_batch["global_attention_mask"] = torch.zeros_like(raw_batch["attention_mask"])

    raw_batch.pop("scene_temporal_ids", None)
    raw_batch.pop("object_temporal_ids", None)


def _pad_sequence(seq: list[torch.Tensor], padding_value: int) -> torch.Tensor:
    if not seq:
        return torch.empty(0)
    return pad_sequence(seq, batch_first=True, padding_value=padding_value)


def collate_fn(batch: list[Optional[EmmaDatasetItem]]) -> EmmaDatasetBatch:
    """Collate lists of samples into batches after padding."""
    fields = dataclasses.fields(EmmaDatasetItem)
    padding = EmmaDatasetPadding()

    raw_batch = {
        field.name: _pad_sequence(
            [
                getattr(sample, field.name)
                for sample in batch
                if sample is not None and getattr(sample, field.name) is not None
            ],
            padding_value=getattr(padding, field.name),
        )
        for field in fields
    }
    make_batch_attention_masks(raw_batch)

    return EmmaDatasetBatch(**raw_batch)
