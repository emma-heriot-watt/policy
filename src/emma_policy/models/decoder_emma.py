import random
from typing import Any, Callable, Optional, Union

import torch
from torch.nn import Embedding, functional
from transformers.models.led.modeling_led import (  # noqa: WPS450
    BaseModelOutputWithPastAndCrossAttentions,
    LEDDecoder,
    LEDDecoderLayer,
    _expand_mask,
    _make_causal_mask,
)

from emma_policy.models.configuration_emma import EmmaConfig
from emma_policy.models.expand_mask import expand_mask_and_transform_values
from emma_policy.utils import get_logger


logger = get_logger(__name__)


class EmmaDecoder(LEDDecoder):
    """[`EmmaDecoder`] is identical to [`LEDDecoder`]."""

    def __init__(self, config: EmmaConfig, embed_tokens: Optional[Embedding] = None) -> None:
        super().__init__(config=config, embed_tokens=embed_tokens)  # type: ignore[arg-type]

    def check_head_mask_shapes(
        self, attention_masks: list[Optional[torch.Tensor]], mask_names: list[str]
    ) -> None:
        """Check that the head mask and cross attention head masks have the correct shape."""
        for attention_mask, mask_name in zip(attention_masks, mask_names):
            if attention_mask is not None and attention_mask.size()[0] != len(self.layers):
                raise AssertionError(
                    f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for {attention_mask.size()[0]}."
                )

    def decoder_layer_outputs(
        self,
        decoder_layer: LEDDecoderLayer,
        hidden_states: torch.Tensor,
        combined_attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[tuple[torch.Tensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
    ) -> tuple[torch.FloatTensor, ...]:
        """Get output from a single decoder layer."""
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

            def create_custom_forward(  # noqa: WPS430
                module: LEDDecoderLayer,
            ) -> Callable[..., tuple[torch.Tensor]]:
                """Custom forward."""

                def custom_forward(*inputs: Any) -> tuple[torch.Tensor]:  # noqa: WPS430
                    """Helper fn."""
                    return module(*inputs, output_attentions, use_cache)

                return custom_forward

            layer_outputs = torch.utils.checkpoint.checkpoint(
                create_custom_forward(decoder_layer),
                hidden_states,
                combined_attention_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                head_mask,
                cross_attn_head_mask,
                None,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=combined_attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                layer_head_mask=head_mask,
                cross_attn_layer_head_mask=cross_attn_head_mask,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
        return layer_outputs

    def prepare_decoder_input(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Size]:
        """Prepare the decoder input from the input_ids and input_embeds."""
        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            input_embeddings = self.embed_tokens(input_ids)
            return (input_embeddings, input_shape)
        elif inputs_embeds is not None:
            input_embeddings = inputs_embeds
            input_shape = inputs_embeds.size()[:-1]
            return (input_embeddings, input_shape)
        raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

    def forward(  # noqa: WPS231
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        global_attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[tuple[tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple[Any, ...], BaseModelOutputWithPastAndCrossAttentions]:
        """Decoder forward pass."""
        output_attentions, output_hidden_states, return_dict = self._set_return_forward_args(
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # retrieve input_ids and inputs_embeds
        (inputs_embeds, input_shape) = self.prepare_decoder_input(input_ids, inputs_embeds)

        # past_key_values_length
        past_key_values_length = (
            past_key_values[0][0].shape[2] if past_key_values is not None else 0
        )

        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask: Optional[torch.Tensor] = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, inputs_embeds.dtype, past_key_values_length=past_key_values_length
            ).to(self.device)

        if attention_mask is not None and combined_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            combined_attention_mask = combined_attention_mask + _expand_mask(
                attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
            )

        # expand encoder attention mask and change values from 0,1 to -inf, 0
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            encoder_attention_mask = expand_mask_and_transform_values(
                attention_mask=encoder_attention_mask,
                dtype=inputs_embeds.dtype,
                tgt_len=input_shape[-1],
                expand=True,
            )
        # embed positions
        hidden_states = inputs_embeds + self.embed_positions(input_shape, past_key_values_length)
        hidden_states = self.layernorm_embedding(hidden_states)

        hidden_states = functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # decoder layers
        all_hidden_states: Optional[tuple[torch.FloatTensor, ...]] = (
            () if output_hidden_states else None
        )
        all_self_attns: Optional[tuple[torch.FloatTensor, ...]] = () if output_attentions else None
        all_cross_attentions: Optional[tuple[torch.FloatTensor, ...]] = (
            () if output_attentions else None
        )
        next_decoder_cache: Optional[tuple[tuple[torch.FloatTensor], ...]] = (
            () if use_cache else None
        )

        # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
        self.check_head_mask_shapes(
            attention_masks=[head_mask, cross_attn_head_mask],
            mask_names=["head_mask", "cross_attn_head_mask"],
        )

        for layer_idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if all_hidden_states is not None:
                all_hidden_states += (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            layer_outputs = self.decoder_layer_outputs(
                decoder_layer=decoder_layer,
                hidden_states=hidden_states,
                combined_attention_mask=combined_attention_mask,
                encoder_attention_mask=encoder_attention_mask,
                head_mask=head_mask[layer_idx] if head_mask is not None else None,
                cross_attn_head_mask=cross_attn_head_mask[layer_idx]
                if cross_attn_head_mask is not None
                else None,
                encoder_hidden_states=encoder_hidden_states,
                past_key_value=past_key_values[layer_idx] if past_key_values is not None else None,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs[0]

            if next_decoder_cache is not None:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)  # type: ignore[operator]

            if all_self_attns is not None:
                all_self_attns += (layer_outputs[1],)
            if all_cross_attentions is not None:
                all_cross_attentions += (layer_outputs[2],)

        # add hidden states from the last decoder layer
        if all_hidden_states is not None:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            out_list = [
                hidden_states,
                next_cache,
                all_hidden_states,
                all_self_attns,
                all_cross_attentions,
            ]
            return tuple(v for v in out_list if v is not None)
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,  # type: ignore[arg-type]
            hidden_states=all_hidden_states,  # type: ignore[arg-type]
            attentions=all_self_attns,  # type: ignore[arg-type]
            cross_attentions=all_cross_attentions,  # type: ignore[arg-type]
        )

    def _set_return_forward_args(
        self,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> tuple[bool, bool, bool]:
        """Sets the optional boolean values obtained during forward to True or False."""
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        return output_attentions, output_hidden_states, return_dict
