import torch
from pytest_cases import parametrize

from emma_policy.models.configuration_emma import EmmaConfig
from emma_policy.models.encoder_layers import EmmaEncoderSelfAttention
from emma_policy.models.expand_mask import expand_mask_and_transform_values


NO_ATTN = -10000


@parametrize(
    "attention_mask,target_output,window_overlap",
    [
        (
            torch.tensor(
                [
                    [
                        [1, 2, 3, 4, 5, 6, 7, 8],
                        [9, 10, 11, 12, 13, 14, 15, 16],
                        [17, 18, 19, 20, 21, 22, 23, 24],
                        [25, 26, 27, 28, 29, 30, 31, 32],
                        [33, 34, 35, 36, 37, 38, 39, 40],
                        [41, 42, 43, 44, 45, 46, 47, 48],
                        [49, 50, 51, 52, 53, 54, 55, 56],
                        [57, 58, 59, 60, 61, 62, 63, 64],
                    ],
                    [
                        [1, 2, 3, 4, 5, 6, 7, 8],
                        [9, 10, 11, 12, 13, 14, 15, 16],
                        [17, 18, 19, 20, 21, 22, 23, 24],
                        [25, 26, 27, 28, 29, 30, 31, 32],
                        [33, 34, 35, 36, 37, 38, 39, 40],
                        [41, 42, 43, 44, 45, 46, 47, 48],
                        [49, 50, 51, 52, 53, 54, 55, 56],
                        [57, 58, 59, 60, 61, 62, 63, 64],
                    ],
                ]
            ),
            torch.tensor(
                [
                    [
                        [-float("inf"), -float("inf"), 1, 2, 3],
                        [-float("inf"), 9, 10, 11, 12],
                        [17, 18, 19, 20, 21],
                        [26, 27, 28, 29, 30],
                        [35, 36, 37, 38, 39],
                        [44, 45, 46, 47, 48],
                        [53, 54, 55, 56, -float("inf")],
                        [62, 63, 64, -float("inf"), -float("inf")],
                    ],
                    [
                        [-float("inf"), -float("inf"), 1, 2, 3],
                        [-float("inf"), 9, 10, 11, 12],
                        [17, 18, 19, 20, 21],
                        [26, 27, 28, 29, 30],
                        [35, 36, 37, 38, 39],
                        [44, 45, 46, 47, 48],
                        [53, 54, 55, 56, -float("inf")],
                        [62, 63, 64, -float("inf"), -float("inf")],
                    ],
                ]
            ),
            2,
        ),
    ],
)
def test_encoder_diagonal_attention(
    attention_mask: torch.Tensor,
    target_output: torch.Tensor,
    window_overlap: int,
) -> None:
    config = EmmaConfig()
    config.attention_window = [window_overlap]
    self_attention = EmmaEncoderSelfAttention(config=config, layer_id=0)
    output = self_attention._diagonal_from_full_mask(
        mask=attention_mask,
        window_overlap=window_overlap,
    )
    assert torch.equal(output.squeeze(2), target_output)


@parametrize(
    "one_d_attention_mask,two_d_attention_mask,window_overlap,dtype",
    [
        (
            torch.tensor([0, 0, 0, NO_ATTN, 0, NO_ATTN, NO_ATTN, NO_ATTN])[None, :, None, None],
            torch.tensor([0, 0, 0, NO_ATTN, 0, NO_ATTN, NO_ATTN, NO_ATTN]).repeat(1, 8, 1),
            2,
            torch.float32,
        ),
    ],
)
def test_encoder_one_d_matches_two_d_diagonal_attention(
    one_d_attention_mask: torch.Tensor,
    two_d_attention_mask: torch.Tensor,
    window_overlap: int,
    dtype: torch.dtype,
) -> None:
    if one_d_attention_mask.dtype != dtype:
        one_d_attention_mask = one_d_attention_mask.to(dtype)
    if two_d_attention_mask.dtype != dtype:
        two_d_attention_mask = two_d_attention_mask.to(dtype)
    config = EmmaConfig()
    config.attention_window = [window_overlap]
    self_attention = EmmaEncoderSelfAttention(config=config, layer_id=0)
    diagonal_from_one_d = self_attention._sliding_chunks_query_key_matmul(
        one_d_attention_mask.new_ones(size=one_d_attention_mask.size()),
        one_d_attention_mask,
        window_overlap,
    )
    diagonal_from_two_d = self_attention._diagonal_from_full_mask(
        mask=two_d_attention_mask,
        window_overlap=window_overlap,
    )
    assert torch.equal(diagonal_from_one_d, diagonal_from_two_d)


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
def test_encoder_expand_one_d_matches_two_d_attention(
    one_d_attention_mask: torch.Tensor,
    two_d_attention_mask: torch.Tensor,
    tgt_len: int,
    dtype: torch.dtype,
) -> None:
    """Make sure that the expansion of the attention mask used in the decoder has the output for 1D
    and 2D attention masks."""
    if one_d_attention_mask.dtype != dtype:
        one_d_attention_mask = one_d_attention_mask.to(dtype)
    if two_d_attention_mask.dtype != dtype:
        two_d_attention_mask = two_d_attention_mask.to(dtype)
    one_d_expanded = expand_mask_and_transform_values(
        attention_mask=one_d_attention_mask,
        dtype=one_d_attention_mask.dtype,
        tgt_len=tgt_len,
        expand=False,
    )
    two_d_expanded = expand_mask_and_transform_values(
        attention_mask=two_d_attention_mask,
        dtype=two_d_attention_mask.dtype,
        tgt_len=tgt_len,
        expand=False,
    )
    assert torch.equal(one_d_expanded, torch.diag(two_d_expanded.squeeze(0)).unsqueeze(0))


@parametrize(
    "two_d_attention_mask,target_indices",
    [
        (
            torch.tensor(
                [
                    # Tokens: History, Future, Future, Global
                    [
                        [0, -float("inf"), -float("inf"), float("inf")],
                        [0, 0, 0, float("inf")],
                        [0, 0, 0, float("inf")],
                        [float("inf"), -float("inf"), -float("inf"), float("inf")],
                    ],
                    # Tokens: History, Future, Global, Global
                    [
                        [0, -float("inf"), float("inf"), float("inf")],
                        [0, 0, float("inf"), float("inf")],
                        [float("inf"), -float("inf"), float("inf"), float("inf")],
                        [float("inf"), -float("inf"), float("inf"), float("inf")],
                    ],
                    # Tokens: Future, Future, Future, Global
                    [
                        [0, 0, 0, float("inf")],
                        [0, 0, 0, float("inf")],
                        [0, 0, 0, float("inf")],
                        [-float("inf"), -float("inf"), -float("inf"), float("inf")],
                    ],
                ],
            ),
            # Indices of future tokens
            (torch.tensor([0, 0, 1, 2, 2, 2]), torch.tensor([1, 2, 1, 0, 1, 2])),  # noqa: WPS221
        )
    ],
)
def test_encoder_global_future_attn_indices(
    two_d_attention_mask: torch.Tensor,
    target_indices: tuple[torch.Tensor, torch.Tensor],
    window_overlap: int = 2,
) -> None:
    config = EmmaConfig()
    config.attention_window = [window_overlap]
    self_attention = EmmaEncoderSelfAttention(config=config, layer_id=0)
    is_local_index_global_attn_future = self_attention._get_global_future_attn_indices(
        two_d_attention_mask
    )
    assert torch.equal(is_local_index_global_attn_future[0], target_indices[0])
    assert torch.equal(is_local_index_global_attn_future[1], target_indices[1])
