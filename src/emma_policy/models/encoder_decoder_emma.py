from dataclasses import dataclass
from typing import Optional

import torch
from torch.nn import Embedding, LayerNorm, ModuleList
from transformers.file_utils import ModelOutput
from transformers.models.led.modeling_led import (
    LEDDecoder,
    LEDEncoder,
    LEDEncoderLayer,
    LEDLearnedPositionalEmbedding,
)

from emma_policy.models.configuration_emma import EmmaConfig


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

    def __init__(  # noqa: WPS231
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
            if config.attention_window % 2 != 0:
                raise AssertionError("`config.attention_window` must be even.")
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
        self.layers = ModuleList(
            [
                LEDEncoderLayer(config, idx)  # type: ignore[arg-type]
                for idx in range(config.encoder_layers)
            ]
        )
        self.layernorm_embedding = LayerNorm(embed_dim)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def text_embeddings(
        self, token_ids: torch.Tensor, past_key_values_length: int = 0
    ) -> torch.Tensor:
        """Returns textual embeddings for the given token ids."""
        _, seq_len = token_ids.shape[:2]
        positions = torch.arange(
            past_key_values_length,
            past_key_values_length + seq_len,
            dtype=torch.long,
            device=self.embed_positions.weight.device,
        )
        embed_pos = self.embed_positions(positions)

        inputs_embeds = self.embed_tokens(token_ids)
        hidden_states = inputs_embeds + embed_pos
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = torch.nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )

        return hidden_states


class EmmaDecoder(LEDDecoder):
    """[`EmmaDecoder`] is identical to [`LEDDecoder`]."""

    def __init__(self, config: EmmaConfig, embed_tokens: Optional[Embedding] = None) -> None:
        super().__init__(config=config, embed_tokens=embed_tokens)  # type: ignore[arg-type]
