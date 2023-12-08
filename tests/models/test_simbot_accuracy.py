import torch

from emma_policy.utils.simbot_action_metrics import SimbotActionExactMatch


def test_simbot_action_accuracy_with_no_mask() -> None:
    predicted = torch.tensor(
        [[2, 3, 4, 5], [1, 2, 3, 4], [5, 6, 7, 8], [5, 6, 7, 8]]  # noqa: WPS221
    )

    targets = torch.tensor(
        [[2, 4, 5, 3], [1, 3, 2, 4], [5, 6, 7, 8], [5, 6, 7, 8]]  # noqa: WPS221
    )

    mask = torch.ones_like(targets)

    accuracy = SimbotActionExactMatch()

    accuracy(predicted, targets, mask)

    assert accuracy.compute().item() == float(0.5)


def test_simbot_action_accuracy_with_mask() -> None:
    predicted = torch.tensor(
        [[2, 3, 4, 5, 6], [1, 2, 3, 4, 0], [5, 6, 7, 8, 0], [5, 6, 7, 8, 0]]  # noqa: WPS221
    )

    targets = torch.tensor(
        [[2, 4, 5, 3, 6], [1, 3, 2, 4, 0], [5, 6, 7, 8, 0], [5, 6, 7, 8, 0]]  # noqa: WPS221
    )

    mask = torch.tensor(
        [[1, 1, 1, 1, 1], [1, 1, 1, 1, 0], [1, 1, 1, 1, 0], [1, 1, 1, 1, 0]]  # noqa: WPS221
    )
    accuracy = SimbotActionExactMatch()

    accuracy(predicted, targets, mask)

    assert accuracy.compute().item() == float(0.5)
