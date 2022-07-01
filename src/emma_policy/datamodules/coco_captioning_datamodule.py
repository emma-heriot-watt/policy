from pathlib import Path
from typing import Literal, Optional, Union

from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader
from transformers import AutoTokenizer

from emma_policy.datamodules.coco_captioning_dataset import COCOCaptioningDataset
from emma_policy.datamodules.collate import collate_fn
from emma_policy.datamodules.emma_dataclasses import EmmaDatasetBatch


class COCOCaptioningDataModule(LightningDataModule):
    """Data module to load COCO captions for the EMMA Policy model."""

    def __init__(
        self,
        coco_cap_train_db_file: Union[str, Path],
        coco_cap_restvalid_db_file: Union[str, Path],
        coco_cap_valid_db_file: Union[str, Path],
        coco_cap_test_db_file: Union[str, Path],
        use_restval: bool = True,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        num_workers: int = 0,
        model_name: str = "heriot-watt/emma-base",
        max_lang_tokens: Optional[int] = None,
        tokenizer_truncation_side: Literal["left", "right"] = "right",
    ) -> None:
        super().__init__()
        if isinstance(coco_cap_train_db_file, str):
            coco_cap_train_db_file = Path(coco_cap_train_db_file)
        if isinstance(coco_cap_restvalid_db_file, str):
            coco_cap_restvalid_db_file = Path(coco_cap_restvalid_db_file)
        if isinstance(coco_cap_valid_db_file, str):
            coco_cap_valid_db_file = Path(coco_cap_valid_db_file)
        if isinstance(coco_cap_test_db_file, str):
            coco_cap_test_db_file = Path(coco_cap_test_db_file)

        self._coco_cap_train_db_file = coco_cap_train_db_file
        self._coco_cap_restvalid_db_file = coco_cap_restvalid_db_file
        self._coco_cap_valid_db_file = coco_cap_valid_db_file
        self._coco_cap_test_db_file = coco_cap_test_db_file

        # Dataloader constraints
        self._max_lang_tokens = max_lang_tokens
        self._tokenizer_truncation_side = tokenizer_truncation_side
        self._num_workers = num_workers
        self._train_batch_size = train_batch_size
        self._val_batch_size = val_batch_size
        self._use_restval = use_restval

        # Model
        self._model_name = model_name

    def prepare_data(self) -> None:
        """Perform any preparation steps necessary before loading the data to the model."""
        super().prepare_data()

        AutoTokenizer.from_pretrained(self._model_name)

    def setup(self, stage: Optional[str] = None) -> None:
        """Setup datasets for the dataloaders."""
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._tokenizer.truncation_side = self._tokenizer_truncation_side

        if self._max_lang_tokens:
            self._tokenizer.model_max_length = self._max_lang_tokens

        self._train_dataset = COCOCaptioningDataset(
            dataset_db_path=self._coco_cap_train_db_file,
            tokenizer=self._tokenizer,
        )

        if self._use_restval:
            self._restval_dataset = COCOCaptioningDataset(
                dataset_db_path=self._coco_cap_restvalid_db_file,
                tokenizer=self._tokenizer,
            )

        self._valid_dataset = COCOCaptioningDataset(
            dataset_db_path=self._coco_cap_valid_db_file,
            tokenizer=self._tokenizer,
        )

        self._test_dataset = COCOCaptioningDataset(
            dataset_db_path=self._coco_cap_test_db_file,
            tokenizer=self._tokenizer,
        )

    def train_dataloader(self) -> DataLoader[EmmaDatasetBatch]:
        """Generate train dataloader for COCO captioning instances."""
        dataset = (
            ConcatDataset([self._train_dataset, self._restval_dataset])
            if self._use_restval
            else self._train_dataset
        )
        return DataLoader(
            dataset,  # type: ignore[arg-type]
            batch_size=self._train_batch_size,
            num_workers=self._num_workers,
            collate_fn=collate_fn,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader[EmmaDatasetBatch]:
        """Generate valid dataloader for COCO captioning instances."""
        return DataLoader(
            self._valid_dataset,  # type: ignore[arg-type]
            batch_size=self._val_batch_size,
            num_workers=self._num_workers,
            collate_fn=collate_fn,
            shuffle=False,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader[EmmaDatasetBatch]:
        """Generate test dataloader for COCO captioning instances."""
        return DataLoader(
            self._test_dataset,  # type: ignore[arg-type]
            batch_size=self._val_batch_size,
            num_workers=self._num_workers,
            collate_fn=collate_fn,
            shuffle=False,
            pin_memory=True,
        )
