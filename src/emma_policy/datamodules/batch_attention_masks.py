import torch


def make_text_history_global_pattern(
    total_seq_len: int,
    text_attention_mask: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Create 1D global attention mask that treats input text as global tokens."""
    bsz = text_attention_mask.shape[0]
    global_attention_mask = torch.zeros((bsz, total_seq_len))

    # text tokens are at the end of the total sequence
    global_attention_mask[:, -text_attention_mask.shape[1] :] = text_attention_mask
    return global_attention_mask.to(dtype)


def make_mask_from_temporal_ids(
    source_temporal_ids: torch.Tensor,
    target_temporal_ids: torch.Tensor,
    dtype: torch.dtype,
    pos_mask_value: int = 1,
    neg_mask_value: int = 0,
    padding_value: int = 0,
) -> torch.Tensor:
    """Constructs a mask from temporal ids with shape [bsz, tgt_seq_len, src_seq_len]."""
    bsz, input_len = source_temporal_ids.shape[0], source_temporal_ids.shape[1]
    target_len = target_temporal_ids.shape[1]
    mask = torch.full((source_temporal_ids.shape[0], target_len, input_len), neg_mask_value)

    expanded_mask_cond_cols = (
        target_temporal_ids[:, :, None].expand(bsz, target_len, input_len).to(dtype)
    )
    expanded_mask_cond_rows = (
        source_temporal_ids[:, None, :].expand(bsz, target_len, input_len).to(dtype)
    )
    mask.masked_fill_(expanded_mask_cond_rows <= expanded_mask_cond_cols, pos_mask_value)
    mask.masked_fill_(expanded_mask_cond_cols == padding_value, neg_mask_value)
    mask.masked_fill_(expanded_mask_cond_rows == padding_value, neg_mask_value)
    return mask.to(dtype)


def make_batch_attention_masks(raw_batch: dict[str, torch.Tensor], padding_value: int) -> None:
    """Make local and global attention masks."""
    got_scene_ids = (
        raw_batch["scene_temporal_ids"].shape[-1] == raw_batch["scene_attention_mask"].shape[-1]
    )
    got_object_ids = (
        raw_batch["object_temporal_ids"].shape[-1] == raw_batch["object_attention_mask"].shape[-1]
    )
    if got_scene_ids and got_object_ids:
        text_temporal_ids = torch.zeros_like(raw_batch["text_attention_mask"])
        text_temporal_ids.masked_fill_(raw_batch["text_attention_mask"] == 1, -1)

        input_temporal_ids = torch.cat(
            [raw_batch["scene_temporal_ids"], raw_batch["object_temporal_ids"], text_temporal_ids],
            dim=1,
        )
        # Create a 2D attention mask that masks future and padding tokens
        raw_batch["attention_mask"] = make_mask_from_temporal_ids(
            source_temporal_ids=input_temporal_ids,
            target_temporal_ids=input_temporal_ids,
            dtype=raw_batch["text_attention_mask"].dtype,
            padding_value=padding_value,
        )

        # Masks the attention from decoder to future tokens
        raw_batch["decoder_encoder_attention_mask"] = make_mask_from_temporal_ids(
            source_temporal_ids=input_temporal_ids,
            target_temporal_ids=raw_batch["target_temporal_ids"],
            dtype=raw_batch["text_attention_mask"].dtype,
            padding_value=padding_value,
        )
    else:
        raw_batch["attention_mask"] = torch.cat(
            [
                raw_batch["scene_attention_mask"],
                raw_batch["object_attention_mask"],
                raw_batch["text_attention_mask"],
            ],
            dim=-1,
        )

    raw_batch["global_attention_mask"] = make_text_history_global_pattern(
        total_seq_len=raw_batch["attention_mask"].shape[-1],
        text_attention_mask=raw_batch["text_attention_mask"],
        dtype=raw_batch["attention_mask"].dtype,
    )

    raw_batch.pop("scene_temporal_ids", None)
    raw_batch.pop("object_temporal_ids", None)
    raw_batch.pop("target_temporal_ids", None)
