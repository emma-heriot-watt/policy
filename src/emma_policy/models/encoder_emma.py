import random
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

import torch
from overrides import overrides
from torch.nn import Embedding, LayerNorm, ModuleList, functional
from transformers.file_utils import ModelOutput
from transformers.models.led.modeling_led import LEDEncoder, LEDLearnedPositionalEmbedding

from emma_policy.models.configuration_emma import EmmaConfig
from emma_policy.models.encoder_layers import EmmaEncoderLayer
from emma_policy.models.expand_mask import expand_mask_and_transform_values


@dataclass
class EmmaEncoderBaseModelOutput(ModelOutput):
    """Base class for outputs, with potential hidden states, local and global attentions.

    Copied from transformers.models.longformer.modeling_longformer.LongformerBaseModelOutput
    with Longformer->LEDEncoder

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x +
            attention_window + 1)`, where `x` is the number of tokens with global attention mask.
            Local attentions weights after the attention softmax, used to compute the weighted average in the
            self-attention heads. Those are the attention weights from every token in the sequence to every token with
            global attention (first `x` values) and to every token in the attention window (remaining `attention_window
            + 1` values). Note that the first `x` values refer to tokens with fixed positions in the text, but the
            remaining `attention_window + 1` values refer to tokens with relative positions: the attention weight of a
            token to itself is located at index `x + attention_window / 2` and the `attention_window / 2` preceding
            (succeeding) values are the attention weights to the `attention_window / 2` preceding (succeeding) tokens.
            If the attention window contains a token with global attention, the attention weight at the corresponding
            index is set to 0; the value should be accessed from the first `x` attention weights. If a token has global
            attention, the attention weights to all other tokens in `attentions` is set to 0, the values should be
            accessed from `global_attentions`.
        global_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x)`,
            where `x` is the number of tokens with global attention mask.
            Global attentions weights after the attention softmax, used to compute the weighted average in the
            self-attention heads. Those are the attention weights from every token with global attention to every token
            in the sequence.
    """

    last_hidden_state: torch.FloatTensor
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None
    global_attentions: Optional[tuple[torch.FloatTensor, ...]] = None


class EmmaEncoder(LEDEncoder):  # noqa: WPS230
    """[`EmmaEncoder`] is identical to [`LEDEncoder`]."""

    def __init__(
        self,
        config: EmmaConfig,
        embed_tokens: Optional[Embedding] = None,
    ) -> None:
        super().__init__(config=config, embed_tokens=embed_tokens)  # type: ignore[arg-type]

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id

        if isinstance(config.attention_window, int):
            if config.attention_window <= 0:
                raise AssertionError("`config.attention_window` has to be positive")
            # one value per layer
            config.attention_window = [
                config.attention_window for _ in range(config.encoder_layers)
            ]
        elif len(config.attention_window) != config.encoder_layers:
            raise AssertionError(
                "`len(config.attention_window)` should equal `config.encoder_layers`. "
                + f"Expected {config.encoder_layers}, given {len(config.attention_window)}"
            )

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = Embedding(config.vocab_size, embed_dim, self.padding_idx)

        self.embed_positions = LEDLearnedPositionalEmbedding(
            config.max_encoder_position_embeddings,
            embed_dim,
        )
        self.embed_text_positions = LEDLearnedPositionalEmbedding(
            config.max_encoder_position_embeddings,
            embed_dim,
        )
        self.layers = ModuleList(
            [EmmaEncoderLayer(config, idx) for idx in range(config.encoder_layers)]
        )
        self.layernorm_embedding = LayerNorm(embed_dim)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def text_embeddings(
        self, token_ids: torch.Tensor, past_key_values_length: int = 0
    ) -> torch.Tensor:
        """Returns textual embeddings for the given token ids."""
        embed_pos = self.embed_text_positions(token_ids.size())

        inputs_embeds = self.embed_tokens(token_ids)
        hidden_states = inputs_embeds + embed_pos
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = functional.dropout(hidden_states, p=self.dropout, training=self.training)

        return hidden_states

    def encoder_layer_outputs(
        self,
        encoder_layer: EmmaEncoderLayer,
        hidden_states: torch.Tensor,
        is_global_attn: bool,
        attention_mask: torch.Tensor,
        is_index_global_attn: torch.Tensor,
        is_index_masked: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
    ) -> tuple[torch.FloatTensor, ...]:
        """Get output from a single encoder layer."""
        if self.gradient_checkpointing and self.training:

            def create_custom_forward(  # noqa: WPS430
                module: EmmaEncoderLayer,
            ) -> Callable[..., tuple[torch.Tensor]]:
                """Custom forward."""

                def custom_forward(*inputs: Any) -> tuple[torch.Tensor]:  # noqa: WPS430
                    """Helper fn."""
                    return module(*inputs, is_global_attn, output_attentions)

                return custom_forward

            layer_outputs = torch.utils.checkpoint.checkpoint(
                create_custom_forward(encoder_layer),
                hidden_states,
                attention_mask,
                head_mask,
                is_index_masked,
                is_index_global_attn,
            )
        else:
            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                layer_head_mask=head_mask,
                is_index_masked=is_index_masked,
                is_index_global_attn=is_index_global_attn,
                is_global_attn=is_global_attn,
                output_attentions=output_attentions,
            )
        return layer_outputs

    def prepare_enc_inputs(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        global_attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[Optional[torch.Tensor], ...]:
        """Prepare the inputs for the encoder forward pass."""
        # check input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # create default attention_mask
        if attention_mask is None:
            attention_mask = torch.ones(
                inputs_embeds.size()[:-1], device=inputs_embeds.device, dtype=torch.long
            )

        # merge `global_attention_mask` and `attention_mask`
        if global_attention_mask is not None:
            attention_mask = self._merge_to_attention_mask(attention_mask, global_attention_mask)

        return input_ids, inputs_embeds, attention_mask, global_attention_mask

    def forward(  # noqa: WPS231
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        global_attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple[Any, ...], EmmaEncoderBaseModelOutput]:
        """Encoder forward pass."""
        output_attentions, output_hidden_states, return_dict = self._set_return_forward_args(
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        input_ids, inputs_embeds, attention_mask, global_attention_mask = self.prepare_enc_inputs(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
        )

        # pad input if necessary
        (  # noqa: WPS236
            padding_len,
            attention_mask,
            input_ids,
            inputs_embeds,
            global_attention_mask,
        ) = self._pad_to_window_size(
            attention_mask=attention_mask,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            pad_token_id=self.config.pad_token_id,
            global_attention_mask=global_attention_mask,
        )

        # expand attention mask and change values from 0,1 to -inf, 0.
        attention_mask = expand_mask_and_transform_values(
            attention_mask=attention_mask,  # type: ignore[arg-type]
            dtype=inputs_embeds.dtype,  # type: ignore[union-attr]
            expand=False,
        )
        is_index_masked: Optional[torch.Tensor] = None
        is_index_masked = self._get_is_index_masked(
            attention_mask=attention_mask, global_attention_mask=global_attention_mask
        )

        is_index_global_attn: torch.Tensor = global_attention_mask > 0  # type: ignore[assignment, operator]
        is_global_attn: bool = is_index_global_attn.flatten().any().item()  # type: ignore[assignment]

        hidden_states = inputs_embeds + self.embed_positions(inputs_embeds.size()[:-1])  # type: ignore[union-attr]
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = functional.dropout(hidden_states, p=self.dropout, training=self.training)

        encoder_states: Optional[tuple[torch.FloatTensor, ...]] = (
            () if output_hidden_states else None
        )
        all_attentions: Optional[tuple[torch.FloatTensor, ...]] = () if output_attentions else None
        all_global_attentions: Optional[tuple[torch.FloatTensor, ...]] = (
            () if (output_attentions and is_global_attn) else None
        )
        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None and head_mask.size()[0] != len(self.layers):
            raise AssertionError(
                f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
            )
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states and encoder_states is not None:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)

            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                continue
            else:
                layer_outputs = self.encoder_layer_outputs(
                    encoder_layer=encoder_layer,
                    hidden_states=hidden_states,
                    is_global_attn=is_global_attn,
                    attention_mask=attention_mask,
                    is_index_global_attn=is_index_global_attn,
                    is_index_masked=is_index_masked,
                    head_mask=head_mask[idx] if head_mask is not None else None,
                    output_attentions=output_attentions,
                )
                hidden_states = layer_outputs[0]

            if all_attentions is not None:
                # bzs x seq_len x num_attn_heads x (num_global_attn + attention_window_len + 1) => bzs x num_attn_heads x seq_len x (num_global_attn + attention_window_len + 1)
                all_attentions = all_attentions + (layer_outputs[1].transpose(1, 2),)  # type: ignore[operator]

                if all_global_attentions is not None:
                    # bzs x num_attn_heads x num_global_attn x seq_len => bzs x num_attn_heads x seq_len x num_global_attn
                    all_global_attentions = all_global_attentions + (  # type: ignore[operator]
                        layer_outputs[2].transpose(2, 3),
                    )

        if encoder_states is not None:
            encoder_states = encoder_states + (hidden_states,)

        # undo padding
        if padding_len > 0:
            # unpad `hidden_states` because the calling function is expecting a length == input_ids.size(1)
            hidden_states = hidden_states[:, :-padding_len]
            if output_hidden_states and encoder_states is not None:
                encoder_states = tuple(state[:, :-padding_len] for state in encoder_states)  # type: ignore[misc]

            if output_attentions and all_attentions is not None:
                all_attentions = tuple(state[:, :, :-padding_len, :] for state in all_attentions)  # type: ignore[misc]

        if not return_dict:
            out_list = [hidden_states, encoder_states, all_attentions, all_global_attentions]
            return tuple(v for v in out_list if v is not None)
        return EmmaEncoderBaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
            attentions=all_attentions,
            global_attentions=all_global_attentions,
        )

    def _get_is_index_masked(
        self, attention_mask: torch.Tensor, global_attention_mask: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """Get is_index_masked."""
        is_index_masked = None
        if len(attention_mask.shape) == 2:
            is_index_masked = attention_mask < 0
        elif global_attention_mask is not None and len(attention_mask.shape) == 3:
            diag = torch.arange(0, attention_mask.shape[-1])
            is_index_masked = torch.logical_and(
                attention_mask[:, diag, diag] < 0, global_attention_mask == 1
            )
        return is_index_masked

    def _merge_to_attention_mask(
        self, attention_mask: torch.Tensor, global_attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Merges the attention and the globla attention mask."""
        # longformer self attention expects attention mask to have 0 (no attn), 1 (local attn), 2 (global attn)
        # (global_attention_mask + 1) => 1 for local attention, 2 for global attention
        # => final attention_mask => 0 for no attention, 1 for local attention 2 for global attention
        if len(attention_mask.shape) == 3:
            cols = global_attention_mask[:, None, :].expand(
                attention_mask.shape[0], attention_mask.shape[1], attention_mask.shape[2]
            )
            rows = global_attention_mask[:, :, None].expand(
                attention_mask.shape[0], attention_mask.shape[1], attention_mask.shape[2]
            )
            global_attention_mask_expanded = (rows + cols) * attention_mask
            attention_mask[global_attention_mask_expanded > 0] = 2
        else:
            attention_mask = attention_mask * (global_attention_mask + 1)
        return attention_mask

    @overrides(check_signature=False)
    def _pad_to_window_size(  # noqa: WPS320
        self,
        attention_mask: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        pad_token_id: int = 0,
        global_attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[
        int, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]
    ]:
        """Pad tokens and mask to work with implementation of Longformer self-attention."""
        # padding
        attention_window = (
            self.config.attention_window
            if isinstance(self.config.attention_window, int)
            else max(self.config.attention_window)
        )

        if attention_window % 2 != 0:
            raise AssertionError(
                f"`attention_window` should be an even value. Given {attention_window}"
            )
        input_shape = (
            input_ids.shape if input_ids is not None else inputs_embeds.shape  # type: ignore[union-attr]
        )
        batch_size, seq_len = input_shape[:2]

        padding_len: int = (attention_window - seq_len % attention_window) % attention_window
        if padding_len > 0:
            (input_ids, inputs_embeds) = self._pad_input_ids_and_inputs_embeds(
                batch_size=batch_size,
                pad_token_id=pad_token_id,
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                padding_len=padding_len,
            )
            if len(attention_mask.shape) == 2:
                padding_tuple = (0, padding_len)
            else:
                padding_tuple = (0, padding_len, 0, padding_len)  # type: ignore[assignment]
            attention_mask = functional.pad(
                attention_mask, padding_tuple, value=False
            )  # no attention on the padding tokens
            if global_attention_mask is not None:
                global_attention_mask = functional.pad(
                    global_attention_mask, (0, padding_len), value=False
                )  # no attention on the padding tokens

        return padding_len, attention_mask, input_ids, inputs_embeds, global_attention_mask

    def _pad_input_ids_and_inputs_embeds(
        self,
        batch_size: int,
        pad_token_id: int,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        padding_len: int = 0,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Pads input ids and embeddings to attention window size."""
        if input_ids is not None:
            input_ids = functional.pad(input_ids, (0, padding_len), value=pad_token_id)
        if inputs_embeds is not None:
            input_ids_padding = inputs_embeds.new_full(
                size=(batch_size, padding_len),
                fill_value=self.config.pad_token_id,
                dtype=torch.long,
            )
            inputs_embeds_padding = self.embed_tokens(input_ids_padding)
            inputs_embeds = torch.cat([inputs_embeds, inputs_embeds_padding], dim=-2)
        return input_ids, inputs_embeds

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
