import json
from pathlib import Path
from typing import NamedTuple

from emma_datasets.common import Settings
from emma_datasets.datamodels import DatasetName, Instance


DEFAULT_COCO_SPLITS_PATH = Settings().paths.storage.joinpath("data", "vl-t5-splits")


class CocoRefImages(NamedTuple):
    """Tuple of COCO Ref images."""

    train: set[str]
    valid: set[str]


def load_ref_coco_images(coco_splits_path: Path = DEFAULT_COCO_SPLITS_PATH) -> CocoRefImages:
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

    return CocoRefImages(train=train_image_ids, valid=valid_image_ids)


def is_train_instance(coco_ref_images: CocoRefImages, instance: Instance) -> bool:
    """Checks whether a given pretraining instance belongs to the *train* split.

    COCO is the most problematic dataset because many others are based on it. Therefore, to avoid
    downstream task contamination, we make sure to put in our validation set all those instances
    that are present in the test set of the downstream tasks. We follow the same procedure used by
    UNITER/VL-T5.
    """
    # TODO(amit): What about non-coco-related instances? Aren't then also part of the valid set?

    is_train = True
    coco_metadata = instance.dataset.get(DatasetName.coco, None)
    # only instances associated with COCO images are problematic
    if coco_metadata is not None and coco_metadata.id in coco_ref_images.valid:
        is_train = False

    # in any other case the instance can be part of the training
    return is_train
