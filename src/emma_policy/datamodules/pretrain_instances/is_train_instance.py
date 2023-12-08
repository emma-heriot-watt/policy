import json
from functools import lru_cache
from pathlib import Path

from emma_datasets.common import Settings
from emma_datasets.datamodels import DatasetName, DatasetSplit, Instance


settings = Settings()


def load_coco_ids(coco_splits_path: Path) -> set[str]:
    """Load coco ids.

    We only extract the image ID's, which are in the form `COCO_val2014_000000238836`.
    """
    with open(coco_splits_path) as in_file:
        data_list = json.load(in_file)

    image_ids: set[str] = set()

    # from this dataset list we extract only the image ids
    for data in data_list:
        img_id_str = data["img_id"]
        # COCO ids are in the form: COCO_val2014_000000238836
        coco_id = str(int(img_id_str.split("_")[-1]))

        image_ids.add(coco_id)
    return image_ids


@lru_cache(maxsize=1)
def get_validation_coco_ids() -> set[str]:
    """Loads the validation split used by the VL-T5 paper (https://arxiv.org/abs/2102.02779)."""
    coco_splits_path = settings.paths.storage.joinpath("constants", "mscoco_resplit_val.json")
    return load_coco_ids(coco_splits_path)


@lru_cache(maxsize=1)
def get_train_coco_ids() -> set[str]:
    """Loads the train split used by the VL-T5 paper (https://arxiv.org/abs/2102.02779)."""
    coco_splits_path = settings.paths.storage.joinpath("constants", "mscoco_resplit_train.json")
    return load_coco_ids(coco_splits_path)


def is_train_instance(instance: Instance) -> bool:
    """Checks whether a given pretraining instance belongs to the *train* split.

    COCO is the most problematic dataset because many others are based on it. Therefore, to avoid
    downstream task contamination, we make sure to put in our validation set all those instances
    that are present in the test set of the downstream tasks. We follow the same procedure used by
    UNITER/VL-T5.
    """
    is_train = True

    validation_coco_ids = get_validation_coco_ids()

    coco_metadata = instance.dataset.get(DatasetName.coco, None)

    if coco_metadata is None:
        is_train = all(
            [dm.split == DatasetSplit.train for dm in instance.dataset.values() if dm.split]
        )
    elif coco_metadata.id in validation_coco_ids:
        is_train = False

    return is_train


def should_keep_instance(instance: Instance) -> bool:
    """Check whether an instance should be included in the pretraining dataset.

    All instances outside of COCO instances are considered valid by default. A COCO instance is
    included in the pretraining if the image id belongs either in the train or validation ids.
    """
    validation_coco_ids = get_validation_coco_ids()
    train_coco_ids = get_train_coco_ids()

    keep_instance = True
    coco_metadata = instance.dataset.get(DatasetName.coco, None)
    if coco_metadata is not None:
        keep_instance = (
            coco_metadata.id in validation_coco_ids or coco_metadata.id in train_coco_ids
        )
    return keep_instance
