import math
from collections.abc import Iterator, Sequence
from typing import Optional

import torch
from torch.utils.data import Sampler


class DistributedWeightedSampler(Sampler[int]):
    """Distributed Weighted Sampler."""

    def __init__(
        self,
        weights: Sequence[float],
        total_size: int,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        replacement: bool = True,
    ) -> None:
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        self.num_replicas = num_replicas
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.rank = rank
        if not isinstance(replacement, bool):
            raise ValueError(
                f"replacement should be a boolean value, but got replacement={replacement}"
            )
        self.epoch = 0
        self._weights = torch.as_tensor(weights, dtype=torch.double)
        self._num_samples = int(math.ceil(total_size / self.num_replicas))
        self._total_size = self._num_samples * self.num_replicas
        self._replacement = replacement

    def __iter__(self) -> Iterator[int]:
        """Sample dataset indices."""
        # deterministically sample based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        weights = self._weights.clone()
        # subsample based in rank and set all other samples' weight to 0
        rank_weights = weights[self.rank : self._total_size : self.num_replicas]
        sampler_weights = torch.zeros_like(weights)
        sampler_weights[  # noqa: WPS362
            self.rank : self._total_size : self.num_replicas
        ] = rank_weights
        # sample the data
        sampled_indices = torch.multinomial(
            sampler_weights, self._num_samples, self._replacement, generator=g
        ).tolist()
        return iter(sampled_indices)

    def __len__(self) -> int:
        """Get the number of samples."""
        return self._num_samples

    def set_epoch(self, epoch: int) -> None:  # noqa: WPS615
        """Update the epoch."""
        self.epoch = epoch
