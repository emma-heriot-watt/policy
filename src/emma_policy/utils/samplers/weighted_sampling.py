from collections import Counter
from typing import Any, Optional

import numpy as np


def compute_weights(
    data_list: list[Any], temperature: float = 1.0, subsampled_weight: Optional[list[int]] = None
) -> list[float]:
    """Proportional temperature scaling to mitigate action type imbalance.

    data_list: A list of the type based on which we are grouping the data
    temperature: Temperature of weight scaling
    subsampled_weight: List of 0s and 1s, where 0s indicate skipping a data point
    """
    if subsampled_weight is None:
        subsampled_weight = [1 for _ in data_list]
    data_groups = [group for weight, group in zip(subsampled_weight, data_list) if weight == 1]
    counts = Counter(data_groups)
    group_names = list(counts.keys())
    probas = 1 / np.array([counts[action] for action in group_names])

    # Update the sampling probabilities through temperature scaling
    scaled_probas = probas ** (1 / temperature)
    scaled_probas = scaled_probas / scaled_probas.sum()
    group_weights = dict(zip(group_names, scaled_probas))

    # Second pass to get the final weight of each sample
    data_weights = []
    for weight, group in zip(subsampled_weight, data_list):
        data_weights.append(float(weight) * group_weights[group])
    return data_weights
