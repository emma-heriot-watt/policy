from typing import Union

import torch
from torch.nn import Embedding, Linear
from transformers import PreTrainedModel

from emma_policy.models.configuration_emma import EmmaConfig
from emma_policy.models.encoder_decoder_emma import EmmaDecoder, EmmaEncoder


class EmmaPreTrainedModel(PreTrainedModel):
    """Stores the configuration of the model and handles methods common to all models."""

    config_class = EmmaConfig  # type: ignore[assignment]
    base_model_prefix = "emma"
    supports_gradient_checkpointing = True

    @property
    def dummy_inputs(self) -> dict[str, torch.Tensor]:
        """Dummy inputs for Emma."""
        batch_size = 4
        num_frames = 5
        num_objects = 36
        num_lang_tokens = 10
        num_total_tokens = num_frames * (num_objects + 1) + num_lang_tokens

        scene_features = torch.randn(batch_size, num_frames, self.config.scene_features_dim, 2, 2)
        scene_coordinates = torch.tile(torch.tensor([0, 0, 1.0, 1.0]), (batch_size, num_frames, 1))
        object_features = torch.randn(
            batch_size, num_frames, num_objects, self.config.object_features_dim
        )
        image_coordinates = torch.randn(
            batch_size, num_frames, num_objects, self.config.image_coordinates_dim
        )
        visual_token_ids = torch.randint(
            low=0, high=num_objects, size=(batch_size, num_frames, num_objects)
        )
        language_token_ids = torch.randint(low=0, high=100, size=(batch_size, num_lang_tokens))
        attention_mask = torch.ones(batch_size, num_total_tokens)
        dummy_inputs = {
            "scene_features": scene_features,
            "scene_coordinates": scene_coordinates,
            "object_features": object_features,
            "image_coordinates": image_coordinates,
            "visual_token_ids": visual_token_ids,
            "language_token_ids": language_token_ids,
            "attention_mask": attention_mask,
        }
        return dummy_inputs

    def _init_weights(self, module: Union[Linear, Embedding]) -> None:
        std = self.config.init_std
        if isinstance(module, Linear):
            module.weight.data.normal_(mean=0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, Embedding):
            module.weight.data.normal_(mean=0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _set_gradient_checkpointing(
        self, module: Union[EmmaEncoder, EmmaDecoder], value: bool = False  # noqa: WPS110
    ) -> None:
        if isinstance(module, (EmmaDecoder, EmmaEncoder)):
            module.gradient_checkpointing = value
