from pathlib import Path

from omegaconf import OmegaConf
from pytest_cases import parametrize
from transformers import AutoConfig, AutoModelForCausalLM

from emma_policy.datamodules.pretrain_datamodule import EmmaPretrainDataModule
from emma_policy.models.model_output_emma import EmmaSeq2SeqLMOutput


@parametrize("enabled_tasks_path", [Path("storage/fixtures/enabled_tasks.yaml")])
def test_model_forward(tmp_path: Path, enabled_tasks_path: Path) -> None:
    """Make sure model forward pass works."""
    config = AutoConfig.from_pretrained("heriot-watt/emma-small")
    model = AutoModelForCausalLM.from_config(config)
    instances_db_path = "storage/fixtures/instances_tiny_batch.db"
    enabled_tasks_dict = OmegaConf.load(enabled_tasks_path)
    # TODO: Remove enabled_tasks arguement once all pre-trained tasks are implemented
    dm = EmmaPretrainDataModule(
        tmp_path.joinpath("pretrain_train.db"),
        tmp_path.joinpath("pretrain_valid.db"),
        instances_db_path,
        model_name="heriot-watt/emma-small",
        force_prepare_data=True,
        batch_size=2,
        load_valid_data=False,
        enabled_tasks=enabled_tasks_dict["enabled_tasks"],  # type: ignore[index]
    )
    dm.prepare_data()
    dm.setup()

    train_loader = dm.train_dataloader()
    # once the data are ready, we inspect the train dataloader

    batch = next(iter(train_loader))
    output = model(
        scene_features=batch.scene_features,
        scene_coordinates=batch.scene_coordinates,
        scene_frame_ids=batch.scene_frame_ids,
        object_features=batch.object_features,
        object_coordinates=batch.object_coordinates,
        object_frame_ids=batch.object_frame_ids,
        visual_token_ids=batch.visual_token_ids,
        language_token_ids=batch.input_token_ids,
        attention_mask=batch.attention_mask,
        global_attention_mask=batch.global_attention_mask,
        # =batch.text_attention_mask,
        labels=batch.target_token_ids,
        decoder_attention_mask=batch.decoder_attention_mask,
        # =batch.object_attention_mask,
        # =batch.scene_attention_mask,
    )

    assert isinstance(output, EmmaSeq2SeqLMOutput)
