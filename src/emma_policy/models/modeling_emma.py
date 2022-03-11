from typing import Any, Optional

import torch
from overrides import overrides
from torch.nn import Embedding

from emma_policy.models.configuration_emma import EmmaConfig
from emma_policy.models.embeddings_emma import (
    EmmaImagePositionEmbeddings,
    EmmaObjectEmbeddings,
    EmmaSceneEmbeddings,
)
from emma_policy.models.encoder_decoder_emma import (
    EmmaDecoder,
    EmmaEncoder,
    EmmaEncoderBaseModelOutput,
)
from emma_policy.models.model_output_emma import EmmaSeq2SeqModelOutput
from emma_policy.models.pretrained_emma import EmmaPreTrainedModel


def shift_tokens_right(
    input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int
) -> torch.Tensor:
    """Shift input ids one token to the right."""
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise AssertionError("`config.pad_token_id` has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


class EmmaModel(EmmaPreTrainedModel):
    """[`EmmaModel`] is a multimodal extension of [`LEDModel`]."""

    def __init__(self, config: EmmaConfig) -> None:
        super().__init__(config=config)
        word_embeddings = Embedding(
            config.vocab_size, config.d_model, padding_idx=config.pad_token_id
        )
        self.image_position_embeddings = EmmaImagePositionEmbeddings(config=config)
        self.scene_embeddings = EmmaSceneEmbeddings(
            config=config,
            image_position_embeddings=self.image_position_embeddings,
        )
        self.object_embeddings = EmmaObjectEmbeddings(
            config=config,
            word_embeddings=word_embeddings,
            image_position_embeddings=self.image_position_embeddings,
        )

        self.encoder = EmmaEncoder(config=config, embed_tokens=word_embeddings)
        self.decoder = EmmaDecoder(config=config, embed_tokens=word_embeddings)

    def get_encoder(self) -> EmmaEncoder:  # noqa: WPS615
        """Return the encoder (required for generation)."""
        return self.encoder

    def get_decoder(self) -> EmmaDecoder:  # noqa: WPS615
        """Return the decoder."""
        return self.decoder

    def get_input_embeddings(self) -> Embedding:  # noqa: WPS615
        """Get word embeddings."""
        return self.encoder.embed_tokens

    @overrides(check_signature=False)
    def set_input_embeddings(self, value: Embedding) -> None:  # noqa: WPS110, WPS615
        """Set word embeddings."""
        self.encoder.embed_tokens = value
        self.decoder.embed_tokens = value

    def embed_inputs(
        self,
        scene_features: torch.Tensor,
        scene_coordinates: torch.Tensor,
        scene_frame_ids: torch.Tensor,
        object_features: torch.Tensor,
        object_coordinates: torch.Tensor,
        object_frame_ids: torch.Tensor,
        visual_token_ids: torch.Tensor,
        language_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Embed all inputs."""
        scene_embeddings = self.scene_embeddings(
            cnn_features=scene_features,
            image_coordinates=scene_coordinates,
            frame_ids=scene_frame_ids,
        )

        object_embeddings = self.object_embeddings(
            object_features=object_features,
            image_coordinates=object_coordinates,
            visual_token_ids=visual_token_ids,
            frame_ids=object_frame_ids,
        )

        language_embeddings = self.encoder.text_embeddings(language_token_ids)
        inputs_embeds = torch.cat(
            [scene_embeddings, object_embeddings, language_embeddings], dim=1
        )
        return inputs_embeds

    def forward(  # noqa: WPS231
        self,
        scene_features: torch.Tensor,
        scene_coordinates: torch.Tensor,
        scene_frame_ids: torch.Tensor,
        object_features: torch.Tensor,
        object_coordinates: torch.Tensor,
        object_frame_ids: torch.Tensor,
        visual_token_ids: torch.Tensor,
        language_token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        global_attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Any] = None,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Any:
        """Forward pass."""
        if encoder_outputs is None:
            inputs_embeds = self.embed_inputs(
                scene_features=scene_features,
                scene_coordinates=scene_coordinates,
                scene_frame_ids=scene_frame_ids,
                object_features=object_features,
                object_coordinates=object_coordinates,
                object_frame_ids=object_frame_ids,
                visual_token_ids=visual_token_ids,
                language_token_ids=language_token_ids,
            )

        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Using this like Bart, as LED is derived from it. So far
        # No checkpoint on the hub exists that uses that in practice.
        # https://github.com/huggingface/transformers/blob/ac3cb660cad283163f7c73cad511124e845ca388/src/transformers/models/bart/modeling_bart.py#L1153
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = shift_tokens_right(
                language_token_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=None,
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a EmmaEncoderBaseModelOutput when return_dict=False
        elif return_dict and not isinstance(encoder_outputs, EmmaEncoderBaseModelOutput):
            encoder_outputs = EmmaEncoderBaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
                global_attentions=encoder_outputs[3] if len(encoder_outputs) > 3 else None,
            )

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return EmmaSeq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            encoder_global_attentions=encoder_outputs.global_attentions,
        )
