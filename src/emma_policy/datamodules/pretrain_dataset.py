from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from emma_datasets.db import DatasetDb
from torch.utils.data import Dataset

from emma_policy.datamodules.pretrain_instances import PretrainInstance, Task


@dataclass
class EmmaDatasetOutput:
    """Output for the dataset reader."""

    input_token_ids: torch.Tensor
    target_token_ids: torch.Tensor
    attention_mask: torch.Tensor
    global_attention_mask: torch.Tensor
    visual_features: torch.Tensor


class EmmaPretrainDataset(Dataset[EmmaDatasetOutput]):
    """Pretrain dataset reader for the EMMA model.

    Each task in the `self.task_process_map` corresponds to a method which will take the instance
    and return an instance of the `EmmaDatasetOutput`.
    """

    def __init__(self, dataset_db_path: Path) -> None:
        self.db = DatasetDb(dataset_db_path)

        self.task_process_map: dict[Task, Callable[[PretrainInstance], EmmaDatasetOutput]] = {
            Task.mlm: self.mlm,
            Task.itm: self.itm,
            Task.visual_grounding: self.visual_grounding,
            Task.dense_captioning: self.dense_captioning,
            Task.captioning: self.captioning,
            Task.vqa: self.vqa,
            Task.instruction_prediction: self.instruction_prediction,
            Task.action_execution: self.action_execution,
            Task.vtm: self.vtm,
        }

    def __len__(self) -> int:
        """Return the total number of instances within the database."""
        return len(self.db)

    def __getitem__(self, index: int) -> EmmaDatasetOutput:
        """Get a single instance from the dataset."""
        with self.db:
            _, _, instance_str = self.db[index]
            instance = PretrainInstance.parse_raw(instance_str)

        return self.task_process_map[instance.task](instance)

    def mlm(self, instance: PretrainInstance) -> EmmaDatasetOutput:
        """Process the instance for the MLM task."""
        raise NotImplementedError

    def itm(self, instance: PretrainInstance) -> EmmaDatasetOutput:
        """Process the instance for the ITM task."""
        raise NotImplementedError

    def visual_grounding(self, instance: PretrainInstance) -> EmmaDatasetOutput:
        """Process the instance for the Visual Grounding task."""
        raise NotImplementedError

    def dense_captioning(self, instance: PretrainInstance) -> EmmaDatasetOutput:
        """Process the instance for the dense captioning task."""
        raise NotImplementedError

    def captioning(self, instance: PretrainInstance) -> EmmaDatasetOutput:
        """Process the instance for the captioning task."""
        raise NotImplementedError

    def vqa(self, instance: PretrainInstance) -> EmmaDatasetOutput:
        """Process the instance for the VQA task."""
        raise NotImplementedError

    def instruction_prediction(self, instance: PretrainInstance) -> EmmaDatasetOutput:
        """Process the instance for the instruction prediction task."""
        raise NotImplementedError

    def action_execution(self, instance: PretrainInstance) -> EmmaDatasetOutput:
        """Process the instance for the action execution task."""
        raise NotImplementedError

    def vtm(self, instance: PretrainInstance) -> EmmaDatasetOutput:
        """Process the instance for the VTM task."""
        raise NotImplementedError
