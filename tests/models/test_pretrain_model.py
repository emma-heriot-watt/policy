from typing import Union

import torch
from pytest_cases import parametrize_with_cases
from transformers import PreTrainedModel

from emma_policy.datamodules.pretrain_datamodule import EmmaPretrainDataModule
from emma_policy.models.model_output_emma import EmmaSeq2SeqLMOutput


# ---------------------------- Cases for the tests --------------------------- #
def case_pretrain_datamodule(
    emma_pretrain_datamodule: EmmaPretrainDataModule,
) -> EmmaPretrainDataModule:
    return emma_pretrain_datamodule


# ----------------------------------- Tests ---------------------------------- #
@parametrize_with_cases("datamodule", cases=".", glob="*_datamodule")
def test_pretrain_model_forward_works_on_train_data(
    emma_model_for_causal_lm: PreTrainedModel,
    datamodule: EmmaPretrainDataModule,
) -> None:
    train_loader = datamodule.train_dataloader()
    batch = next(iter(train_loader))

    output = emma_model_for_causal_lm(
        scene_features=batch.scene_features,
        scene_coordinates=batch.scene_coordinates,
        scene_frame_tokens=batch.scene_frame_tokens,
        object_features=batch.object_features,
        object_coordinates=batch.object_coordinates,
        object_frame_tokens=batch.object_frame_tokens,
        visual_token_ids=batch.visual_token_ids,
        language_token_ids=batch.input_token_ids,
        attention_mask=batch.attention_mask,
        global_attention_mask=batch.global_attention_mask,
        # =batch.text_attention_mask
        labels=batch.target_token_ids,
        decoder_attention_mask=batch.decoder_attention_mask,
        decoder_encoder_attention_mask=batch.decoder_encoder_attention_mask,
        # =batch.object_attention_mask,
        # =batch.scene_attention_mask,
    )

    assert isinstance(output, EmmaSeq2SeqLMOutput)

    # Verify the loss exists and is not nan
    assert output.loss is not None
    assert not torch.isnan(output.loss)
