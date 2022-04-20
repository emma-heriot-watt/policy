import torch
from pytest_cases import parametrize

from emma_policy.models.expand_mask import expand_mask_and_transform_values


@parametrize(
    "one_d_attention_mask,two_d_attention_mask,tgt_len,dtype",
    [
        (
            torch.tensor([[0, 0, 0, 1, 0, 1, 1, 1]]),
            torch.tensor([[0, 0, 0, 1, 0, 1, 1, 1]]).expand(1, 8, 8),
            8,
            torch.float32,
        ),
    ],
)
def test_decoder_expand_one_d_matches_two_d_attention(
    one_d_attention_mask: torch.Tensor,
    two_d_attention_mask: torch.Tensor,
    tgt_len: int,
    dtype: torch.dtype,
) -> None:
    """Make sure that the expansion of the attention mask used in the encoder has the output for 1D
    and 2D attention masks."""
    if one_d_attention_mask.dtype != dtype:
        one_d_attention_mask = one_d_attention_mask.to(dtype)
    if two_d_attention_mask.dtype != dtype:
        two_d_attention_mask = two_d_attention_mask.to(dtype)
    one_d_expanded = expand_mask_and_transform_values(
        attention_mask=one_d_attention_mask,
        dtype=one_d_attention_mask.dtype,
        tgt_len=tgt_len,
        expand=True,
    )
    two_d_expanded = expand_mask_and_transform_values(
        attention_mask=two_d_attention_mask,
        dtype=two_d_attention_mask.dtype,
        expand=True,
    )
    assert torch.equal(one_d_expanded, two_d_expanded)
