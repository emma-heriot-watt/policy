import json
from pathlib import Path
from typing import Optional, Union

from emma_datasets.common import Settings
from emma_datasets.datamodels import Instance
from emma_datasets.datamodels.constants import DatasetName
from emma_datasets.db import DatasetDb
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from emma_policy.common import get_progress
from emma_policy.datamodules.collate import collate_fn
from emma_policy.datamodules.emma_dataclasses import EmmaDatasetBatch
from emma_policy.datamodules.pretrain_dataset import EmmaPretrainDataset
from emma_policy.datamodules.pretrain_instances import convert_instance_to_pretrain_instances


DEFAULT_DATASET_DB_PATH = Settings().paths.databases.joinpath("instances.db")
DEFAULT_COCO_SPLITS_PATH = Settings().paths.storage.joinpath("data", "vl-t5-splits")


def load_ref_coco_images(coco_splits_path: Path = DEFAULT_COCO_SPLITS_PATH) -> dict[str, set[str]]:
    """Loads the pretraining splits used by the VL-T5 paper (https://arxiv.org/abs/2102.02779)."""
    valid_image_ids = set()
    train_image_ids = set()

    for split_file in coco_splits_path.glob("*.json"):
        with open(split_file) as in_file:
            data_list = json.load(in_file)

            # from this dataset list we extract only the image ids
            for data in data_list:
                img_id_str = data["img_id"]
                # COCO ids are in the form: COCO_val2014_000000238836
                coco_id = str(int(img_id_str.split("_")[-1]))

                if "train" in split_file.name:
                    # this is a training file
                    train_image_ids.add(coco_id)
                else:
                    valid_image_ids.add(coco_id)

    return {"train": train_image_ids, "valid": valid_image_ids}


def is_train_instance(coco_ref_images: dict[str, set[str]], instance: Instance) -> bool:
    """Checks whether a given pretraining instance belongs to the *train* split.

    COCO is the most problematic dataset because many others are based on it. Therefore, to avoid
    downstream task contamination, we make sure to put in our validation set all those instances
    that are present in the test set of the downstream tasks. We follow the same procedure used by
    UNITER/VL-T5.
    """
    is_train = True
    coco_metadata = instance.dataset.get(DatasetName.coco, None)
    # only instances associated with COCO images are problematic
    if coco_metadata is not None and coco_metadata.id in coco_ref_images["valid"]:
        is_train = False

    # in any other case the instance can be part of the training
    return is_train


class EmmaPretrainDataModule(LightningDataModule):
    """DataModule to load data for the EMMA Pretraining Model."""

    def __init__(
        self,
        pretrain_train_db_file: Union[str, Path],
        pretrain_valid_db_file: Union[str, Path],
        instances_db_file: Union[str, Path] = DEFAULT_DATASET_DB_PATH,
        force_prepare_data: bool = False,
        load_valid_data: bool = False,
        num_workers: int = 0,
        batch_size: int = 8,
        coco_split_path: Union[str, Path] = DEFAULT_COCO_SPLITS_PATH,
        model_name: str = "heriot-watt/emma-base",
        mlm_probability: float = 0.3,
    ) -> None:
        self.model_name = model_name
        self.mlm_probability = mlm_probability

        if isinstance(instances_db_file, str):
            instances_db_file = Path(instances_db_file)

        if isinstance(pretrain_train_db_file, str):
            pretrain_train_db_file = Path(pretrain_train_db_file)

        if isinstance(pretrain_valid_db_file, str):
            pretrain_valid_db_file = Path(pretrain_valid_db_file)

        if isinstance(coco_split_path, str):
            coco_split_path = Path(coco_split_path)

        self._instances_db_file = instances_db_file
        self._pretrain_train_db_file = pretrain_train_db_file
        self._pretrain_valid_db_file = pretrain_valid_db_file

        self._instances_db_file_exists = (
            self._instances_db_file is not None and self._instances_db_file.exists()
        )

        no_pretrain_db_files = (
            not self._pretrain_train_db_file.exists() or not self._pretrain_valid_db_file.exists()
        )

        if no_pretrain_db_files and not self._instances_db_file_exists:
            raise ValueError(
                "Both `instances_db_file` and `pretrain_*_db_file` cannot be None. At least one MUST be provided."
            )

        self._force_prepare_data = force_prepare_data

        self._batch_size = batch_size
        self._num_workers = num_workers
        self._coco_ref_images = load_ref_coco_images(coco_split_path)
        self.load_valid_data = load_valid_data

    def prepare_data(self) -> None:
        """Prepare the DatasetDb for the pretraining.

        This will only create the pretraining instances db file if it does not already exist.
        """
        super().prepare_data()

        if not self._pretrain_train_db_file.exists() or self._force_prepare_data:
            self._prepare_pretrain_instances_db()

        # make sure to trigger the tokenizer download on the main process
        AutoTokenizer.from_pretrained(self.model_name)

    def setup(self, stage: Optional[str] = None) -> None:
        """Setup datasets for the dataloaders."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._train_dataset = EmmaPretrainDataset(
            dataset_db_path=self._pretrain_train_db_file,
            tokenizer=self.tokenizer,
            mlm_probability=self.mlm_probability,
        )

        if self.load_valid_data:
            self._val_dataset = EmmaPretrainDataset(
                dataset_db_path=self._pretrain_valid_db_file,
                tokenizer=self.tokenizer,
                mlm_probability=self.mlm_probability,
            )

    def train_dataloader(self) -> DataLoader[EmmaDatasetBatch]:
        """Generate train dataloader."""
        return DataLoader(
            self._train_dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            collate_fn=collate_fn,
        )

    def val_dataloader(self) -> DataLoader[EmmaDatasetBatch]:
        """Generate val dataloader."""
        return DataLoader(
            self._val_dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
        )

    def _prepare_pretrain_instances_db(self) -> None:
        if self.load_valid_data:
            self._prepare_train_valid_instances()
        else:
            self._prepare_train_instances()

    def _prepare_train_valid_instances(self) -> None:

        instances_db = DatasetDb(self._instances_db_file)
        train_instances_db = DatasetDb(self._pretrain_train_db_file, readonly=False)

        valid_instances_db = DatasetDb(self._pretrain_valid_db_file, readonly=False)
        progress = get_progress()

        get_instance_task_id = progress.add_task("Getting instances", total=len(instances_db))
        create_train_task_id = progress.add_task("Creating train instances", total=float("inf"))
        create_valid_task_id = progress.add_task("Creating valid instances", total=float("inf"))

        with instances_db, train_instances_db, valid_instances_db, progress:  # noqa: WPS316
            data_idx = 0

            for _, _, instance_str in instances_db:
                instance = Instance.parse_raw(instance_str)
                is_train = is_train_instance(self._coco_ref_images, instance)

                progress.advance(get_instance_task_id)

                current_instances = convert_instance_to_pretrain_instances(instance)

                for pretrain_instance in current_instances:
                    if is_train:
                        train_instances_db[  # noqa: WPS220
                            (data_idx, f"pretrain_train_{data_idx}")
                        ] = pretrain_instance

                        progress.advance(create_train_task_id)  # noqa: WPS220
                    else:
                        valid_instances_db[  # noqa: WPS220
                            (data_idx, f"pretrain_valid_{data_idx}")
                        ] = pretrain_instance

                        progress.advance(create_valid_task_id)  # noqa: WPS220
                    data_idx += 1

    def _prepare_train_instances(self) -> None:

        instances_db = DatasetDb(self._instances_db_file)
        train_instances_db = DatasetDb(self._pretrain_train_db_file, readonly=False)

        progress = get_progress()

        get_instance_task_id = progress.add_task("Getting instances", total=len(instances_db))
        create_train_task_id = progress.add_task("Creating train instances", total=float("inf"))

        with instances_db, train_instances_db, progress:  # noqa: WPS316
            data_idx = 0

            for _, _, instance_str in instances_db:
                instance = Instance.parse_raw(instance_str)
                is_train = is_train_instance(self._coco_ref_images, instance)

                progress.advance(get_instance_task_id)

                current_instances = convert_instance_to_pretrain_instances(instance)

                for pretrain_instance in current_instances:
                    if is_train:
                        train_instances_db[  # noqa: WPS220
                            (data_idx, f"pretrain_train_{data_idx}")
                        ] = pretrain_instance

                        progress.advance(create_train_task_id)  # noqa: WPS220
                    data_idx += 1
