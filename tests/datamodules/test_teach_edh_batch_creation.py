import torch
from pytest_cases import parametrize

from emma_policy.datamodules.collate import (
    make_encoder_causal_mask_batch,
    make_text_history_global_pattern,
)
from emma_policy.datamodules.teach_edh_datamodule import TeachEdhDataModule


@parametrize(
    "total_seq_len,text_temporal_ids,target_mask",
    [
        (
            4,
            torch.tensor([[-1, -1, -1, 0], [-1, 0, 0, 0]]),
            torch.tensor([[1, 1, 1, 0], [1, 0, 0, 0]]),
        ),
        (
            5,
            torch.tensor([[-1, -1, -1], [0, 0, 0], [-1, -1, 0]]),  # noqa: WPS221
            torch.tensor([[0, 0, 1, 1, 1], [0, 0, 0, 0, 0], [0, 0, 1, 1, 0]]),  # noqa: WPS221
        ),
    ],
)
def test_text_history_global_attention(
    total_seq_len: int, text_temporal_ids: torch.Tensor, target_mask: torch.Tensor
) -> None:
    """Check global attention output for dummy inputs."""
    output = make_text_history_global_pattern(
        total_seq_len=total_seq_len,
        text_temporal_ids=text_temporal_ids,
        dtype=text_temporal_ids.dtype,
    )
    assert torch.equal(output, target_mask)


def test_text_history_global_attention_counts(
    teach_edh_datamodule: TeachEdhDataModule,
) -> None:
    """Ensure that the global attention mask has as many 1s as text tokens."""
    for batch in teach_edh_datamodule.train_dataloader():
        assert batch.global_attention_mask.sum() == batch.text_attention_mask.sum()


@parametrize(
    "scene_temporal_ids,object_temporal_ids,text_temporal_ids, target_mask",
    [
        (
            torch.tensor([[-1, 1, 0]]),
            torch.tensor([[-1, -1, 1, 2]]),
            torch.tensor([[-1, -1]]),
            torch.tensor(
                [
                    [1, 0, 0, 1, 1, 0, 0, 1, 1],
                    [1, 1, 0, 1, 1, 1, 0, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 1, 1, 0, 0, 1, 1],
                    [1, 0, 0, 1, 1, 0, 0, 1, 1],
                    [1, 1, 0, 1, 1, 1, 0, 1, 1],
                    [1, 1, 0, 1, 1, 1, 1, 1, 1],
                    [1, 0, 0, 1, 1, 0, 0, 1, 1],
                    [1, 0, 0, 1, 1, 0, 0, 1, 1],
                ]
            ).unsqueeze(0),
        ),
        (
            torch.tensor([[1, 2], [-1, 1]]),
            torch.tensor([[1, 1, 2], [-1, 1, 0]]),
            torch.tensor([[0], [-1]]),
            torch.tensor(
                [
                    [
                        [1, 0, 1, 1, 0, 0],
                        [1, 1, 1, 1, 1, 0],
                        [1, 0, 1, 1, 0, 0],
                        [1, 0, 1, 1, 0, 0],
                        [1, 1, 1, 1, 1, 0],
                        [0, 0, 0, 0, 0, 0],
                    ],
                    [
                        [1, 0, 1, 0, 0, 1],
                        [1, 1, 1, 1, 0, 1],
                        [1, 0, 1, 0, 0, 1],
                        [1, 1, 1, 1, 0, 1],
                        [0, 0, 0, 0, 0, 0],
                        [1, 0, 1, 0, 0, 1],
                    ],
                ]
            ),
        ),
    ],
)
def test_encoder_full_attention_mask(
    scene_temporal_ids: torch.Tensor,
    object_temporal_ids: torch.Tensor,
    text_temporal_ids: torch.Tensor,
    target_mask: torch.Tensor,
) -> None:
    """Check 2D attention output for dummy inputs."""
    output = make_encoder_causal_mask_batch(
        scene_temporal_ids=scene_temporal_ids,
        object_temporal_ids=object_temporal_ids,
        text_temporal_ids=text_temporal_ids,
        dtype=text_temporal_ids.dtype,
    )
    assert torch.equal(output, target_mask)
