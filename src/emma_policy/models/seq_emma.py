from typing import Any, Optional, Union

import torch
from overrides import overrides
from torch.nn import CrossEntropyLoss, Embedding, Linear

from emma_policy.models.configuration_emma import EmmaConfig
from emma_policy.models.encoder_decoder_emma import EmmaDecoder, EmmaEncoder
from emma_policy.models.model_output_emma import EmmaSeq2SeqLMOutput
from emma_policy.models.modeling_emma import EmmaModel, shift_tokens_right
from emma_policy.models.pretrained_emma import EmmaPreTrainedModel


class EmmaForConditionalGeneration(EmmaPreTrainedModel):
    """EmmaModel with LM head."""

    base_model_prefix = "emma"
    _keys_to_ignore_on_load_missing = [  # type: ignore[assignment]
        "final_logits_bias",
        r"encoder\.version",
        r"decoder\.version",
        r"lm_head\.weight",
    ]

    def __init__(self, config: EmmaConfig, **kwargs: dict[str, Any]) -> None:
        super().__init__(config=config)

        self.emma = EmmaModel(config)
        num_embeddings = self.emma.encoder.embed_tokens.num_embeddings
        self.register_buffer("final_logits_bias", torch.zeros((1, num_embeddings)))
        self.lm_head = Linear(config.d_model, num_embeddings, bias=False)

        self.final_logits_bias: "torch.Tensor"
        self.main_input_name = "input_embeds"
        # Initialize weights and apply final processing
        self.post_init()

    @overrides(check_signature=False)
    def resize_token_embeddings(self, new_num_tokens: int) -> Embedding:
        """Resize the embedding layer."""
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def get_output_embeddings(self) -> Linear:  # noqa: WPS615
        """Get the linear layer."""
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: Linear) -> None:  # noqa: WPS615
        """Get the new embeddings as the linear layer."""
        self.lm_head = new_embeddings

    def get_encoder(self) -> EmmaEncoder:
        """Return the encoder."""
        return self.emma.get_encoder()

    def get_decoder(self) -> EmmaDecoder:
        """Return the decoder."""
        return self.emma.get_decoder()

    def forward(
        self,
        scene_features: torch.Tensor,
        scene_coordinates: torch.Tensor,
        scene_frame_tokens: torch.Tensor,
        object_features: torch.Tensor,
        object_coordinates: torch.Tensor,
        object_frame_tokens: torch.Tensor,
        visual_token_ids: torch.Tensor,
        language_token_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
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
        """Forward pass and compute loss.

        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*): Labels
        for computing the masked language modeling loss. Indices should either be in `[0, ...,
        config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100`
        are ignored     (masked), the loss is only computed for the tokens with labels in `[0, ...,
        config.vocab_size]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = self.prepare_decoder_input_ids_from_labels(labels)

        outputs = self.emma(
            scene_features=scene_features,
            scene_coordinates=scene_coordinates,
            scene_frame_tokens=scene_frame_tokens,
            object_features=object_features,
            object_coordinates=object_coordinates,
            object_frame_tokens=object_frame_tokens,
            visual_token_ids=visual_token_ids,
            language_token_ids=language_token_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            global_attention_mask=global_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return EmmaSeq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
            encoder_global_attentions=outputs.encoder_global_attentions,
        )

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """Get decoder input from labels."""
        return shift_tokens_right(
            labels, self.config.pad_token_id, self.config.decoder_start_token_id
        )

    @overrides(check_signature=False)
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids: torch.Tensor,
        past: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> dict[str, Union[None, torch.Tensor]]:
        """Prepare for generation."""
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "scene_features": None,  # encoder_outputs is defined.
            "scene_coordinates": None,
            "scene_frame_tokens": None,
            "object_features": None,
            "object_coordinates": None,
            "object_frame_tokens": None,
            "visual_token_ids": None,
            "language_token_ids": None,
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    @staticmethod
    def _reorder_cache(past: Any, beam_idx: int) -> tuple[Any, ...]:  # noqa: WPS602
        reordered_past = ()
        for layer_past in past:  # noqa: WPS519
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (  # type: ignore[assignment]
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2])
                + layer_past[2:],
            )
        return reordered_past

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            num_zeros = new_num_tokens - old_num_tokens
            extra_bias = torch.zeros((1, num_zeros), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)
