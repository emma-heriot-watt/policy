import torch
from torch.nn import CrossEntropyLoss

from emma_policy.datamodules.emma_dataclasses import EmmaDatasetPadding


def tiny_value_of_dtype(dtype: torch.dtype) -> float:
    """Returns a moderately tiny value for a given PyTorch data type.

    This is used to avoid numerical issues such as division by zero. This is different from
    `info_value_of_dtype(dtype).tiny` because it causes some NaN bugs. Only supports floating point
    dtypes. Implementation from AllenNLP: https://github.com/allenai/allennlp/blob/39c40fe38cd2fd36b3465b0b3c031f54ec824160/allennlp/nn/util.py#L2010-L2024
    """
    if not dtype.is_floating_point:
        raise TypeError("Only supports floating point dtypes.")
    if dtype in {torch.float, torch.double}:
        return 1e-13  # noqa: WPS432
    elif dtype == torch.half:
        return 1e-4  # noqa: WPS432
    raise TypeError(f"Does not support dtype {str(dtype)}")


def masked_mean(
    vector: torch.Tensor, mask: torch.Tensor, dim: int, keepdim: bool = False
) -> torch.Tensor:
    """To calculate mean along certain dimensions on masked values.

    Implementation from AllenNLP: https://github.com/allenai/allennlp/blob/39c40fe38cd2fd36b3465b0b3c031f54ec824160/allennlp/nn/util.py#L351-L377
    Args:
        vector (torch.Tensor): The vector to calculate mean.
        mask (torch.Tensor): The mask of the vector. It must be broadcastable with vector.
        dim (int): The dimension to calculate mean
        keepdim (bool): Whether to keep dimension
    Returns:
        (torch.Tensor): Masked mean tensor
    """
    replaced_vector = vector.masked_fill(~mask, 0.0)  # noqa: WPS358

    value_sum = torch.sum(replaced_vector, dim=dim, keepdim=keepdim)
    value_count = torch.sum(mask, dim=dim, keepdim=keepdim)
    return value_sum / value_count.float().clamp(min=tiny_value_of_dtype(torch.float))


def average_task_loss(
    labels: torch.Tensor, lm_logits: torch.Tensor, vocab_size: int
) -> torch.Tensor:
    """Compute cross-entropy averaged by sequence length and batch size."""
    loss_fct = CrossEntropyLoss(reduction="none", ignore_index=EmmaDatasetPadding.target_token_ids)
    batch_size, seq_len = labels.shape
    labels_mask = labels != EmmaDatasetPadding.target_token_ids
    # flat_labels shape (batch_size, seq_len) -> (batch_size * seq_len)
    flat_labels = labels.view(-1)
    # flat_logits shape (batch_size, seq_len, vocab_size) -> (batch_size * seq_len, vocab_size)
    flat_logits = lm_logits.view(-1, vocab_size)
    # lm_loss shape (batch_size, seq_len)
    lm_loss = loss_fct(flat_logits, flat_labels).view(batch_size, seq_len)
    # averages over the sequence length dimension first and then over the batch dimension
    masked_lm_loss = masked_mean(lm_loss, labels_mask, dim=-1).mean()
    return masked_lm_loss
