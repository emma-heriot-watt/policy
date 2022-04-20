from typing import Optional

import torch
from transformers.models.led.modeling_led import _expand_mask  # noqa: WPS450


def expand_mask_and_transform_values(
    attention_mask: torch.Tensor,
    dtype: torch.dtype,
    tgt_len: Optional[int] = None,
    expand: Optional[bool] = True,
) -> torch.Tensor:
    """Expands a 1D or 2D attention mask per batch.

    The attention mask has binary values where 0=dont attend, 1=attend, and transforms these values
    to -inf, 0. The expanded mask has the shape of (bsz, 1, tgt_seq_len, src_seq_len).
    Example:
        tgt_len = 8
        attention_mask = torch.tensor([[0, 0, 0, 1, 0, 1, 1, 1]], dtype=torch.float32)
        expanded_attention_mask = expand_mask_and_transform_values(
            attention_mask=attention_mask,
            dtype=attention_mask.dtype,
            tgt_len=tgt_len
        )

        attention_mask = torch.tensor([[0, 0, 0, 1, 0, 1, 1, 1]], dtype=torch.float32).expand(
            1, 8, 8
        )
        expanded_attention_mask = expand_mask_and_transform_values(
            attention_mask=attention_mask,
            dtype=attention_mask.dtype,
        )

    Args:
        attention_mask (torch.Tensor): The provided attention mask. Should have shape
            (bsz, src_seq_len) or (bsz, tgt_seq_len, src_seq_len).
        dtype (torch.dtype): The dtype for the expanded attention mask, e.g torch.float32
        tgt_len (Optional[int]): Optionally provide the target length of the sequence. This is used
            only when the attention mask is 1D.
        expand (Optional[bool]): Do not expand the attention mask. This will only change the binary
            values to -inf, 0.

    Raises:
        ValueError: The provided mask should either have a 1D shape, e.g. (bsz, src_seq_len) or,
            a 2D shape, e.g. (bsz, tgt_seq_len, src_seq_len).

    Returns:
        torch.Tensor: The expanded mask with its binary values transformed to -inf, 0.
    """
    if len(attention_mask.shape) == 2:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        expanded_attention_mask = _expand_mask(attention_mask, dtype=dtype, tgt_len=tgt_len)
        # If the target length is None the attention mask is for the encoder's self attention;
        # Keep [bsz, seq_len] and keep only the value transformation.
        if not expand:
            expanded_attention_mask = expanded_attention_mask[:, 0, 0, :]
    elif len(attention_mask.shape) == 3:
        # [bsz, tgt_seq_len, src_seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        if expand:
            expanded_attention_mask = attention_mask[:, None, :, :]
        else:
            expanded_attention_mask = attention_mask
        inverted_mask = 1.0 - expanded_attention_mask
        expanded_attention_mask = inverted_mask.masked_fill(
            inverted_mask.bool(), torch.finfo(inverted_mask.dtype).min
        )
        expanded_attention_mask *= inverted_mask
    else:
        sz = tuple(attention_mask.size())
        raise ValueError(
            "Expected encoder_attention_mask to have shape either (bsz, src_seq_len) or "
            + f"(bsz, tgt_seq_len, src_seq_len). Found {sz}"
        )
    return expanded_attention_mask
