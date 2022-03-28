import pytest
from pytest_cases import parametrize_with_cases
from pytorch_lightning import LightningDataModule, LightningModule, Trainer

from tests.fixtures.datamodules import DataModuleCases


@parametrize_with_cases("datamodule", cases=DataModuleCases)
def test_trainer_fast_dev_run_does_not_error(
    emma_policy_model: LightningModule, datamodule: LightningDataModule
) -> None:
    """Ensure the model can go through a quick training and validation batch.

    For more as to how and why, see:
    https://pytorch-lightning.readthedocs.io/en/stable/common/debugging.html#fast-dev-run
    """
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(model=emma_policy_model, datamodule=datamodule)


@pytest.mark.skip(reason="CI runner is currently not powerful enough to run this test")
@parametrize_with_cases("datamodule", cases=DataModuleCases)
def test_trainer_with_ddp_on_cpu_does_not_error(
    emma_policy_model: LightningModule, datamodule: LightningDataModule
) -> None:
    """Test the trainer can fit the model using DDP on a CPU.

    For more as to how and why, see:
    https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#num-processes
    """
    trainer = Trainer(accelerator="cpu", strategy="ddp", num_processes=2, fast_dev_run=True)
    trainer.fit(model=emma_policy_model, datamodule=datamodule)
