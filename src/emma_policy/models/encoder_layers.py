import math
from typing import Optional

import torch
from torch.nn import functional
from transformers.models.led.modeling_led import (
    LEDEncoderAttention,
    LEDEncoderLayer,
    LEDEncoderSelfAttention,
)

from emma_policy.models.configuration_emma import EmmaConfig


class EmmaEncoderSelfAttention(LEDEncoderSelfAttention):
    """[`EmmaEncoderSelfAttention`] module."""

    def __init__(self, config: EmmaConfig, layer_id: int) -> None:
        super().__init__(config=config, layer_id=layer_id)
        self.mask_value = -10000.0

    def get_diagonal_mask(
        self, attention_mask: torch.Tensor, query_vectors: torch.Tensor
    ) -> torch.Tensor:
        """Get the diagonal mask from the attention mask."""
        # values to pad for attention probs
        if len(attention_mask.shape) == 2:
            remove_from_windowed_attention_mask = (attention_mask != 0)[:, :, None, None]
            # cast to fp32/fp16 then replace 1's with -inf
            float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
                remove_from_windowed_attention_mask, self.mask_value
            )
            # diagonal mask with zeros everywhere and -inf inplace of padding
            diagonal_mask = self._sliding_chunks_query_key_matmul(
                float_mask.new_ones(size=float_mask.size()),
                float_mask,
                self.one_sided_attn_window_size,
            )
        else:
            remove_from_windowed_attention_mask = attention_mask != 0
            # cast to fp32/fp16 then replace 1's with -inf
            float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
                remove_from_windowed_attention_mask, self.mask_value
            )
            diagonal_mask = self._diagonal_from_full_mask(
                float_mask, self.one_sided_attn_window_size
            )

        return diagonal_mask

    def forward(  # noqa: WPS231
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        is_index_masked: Optional[torch.Tensor] = None,
        is_index_global_attn: Optional[torch.Tensor] = None,
        is_global_attn: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        """Expects *len(hidden_states)* to be multiple of *attention_window*.

        Padding to *attention_window* happens in [`LEDEncoderModel.forward`] to avoid redoing the padding on each layer.
        The *attention_mask* is changed in [`LEDEncoderModel.forward`] from 0, 1, 2 to:
            - -10000: no attention
            - 0: local attention
            - +10000: global attention
        """
        hidden_states = hidden_states.transpose(0, 1)

        # project hidden states
        query_vectors = self.query(hidden_states)
        key_vectors = self.key(hidden_states)
        value_vectors = self.value(hidden_states)

        seq_len, batch_size, embed_dim = hidden_states.size()
        if embed_dim != self.embed_dim:
            raise AssertionError(
                f"hidden_states should have embed_dim = {self.embed_dim}, but has {embed_dim}"
            )

        # normalize query
        query_vectors /= math.sqrt(self.head_dim)

        query_vectors = query_vectors.view(
            seq_len, batch_size, self.num_heads, self.head_dim
        ).transpose(0, 1)
        key_vectors = key_vectors.view(
            seq_len, batch_size, self.num_heads, self.head_dim
        ).transpose(0, 1)

        # [bsz, 2 * self.one_sided_attn_window_size, num_heads, 2 * self.one_sided_attn_window_size + 1]
        attn_scores = self._sliding_chunks_query_key_matmul(
            query_vectors, key_vectors, self.one_sided_attn_window_size
        )

        if attention_mask is not None:
            # pad local attention probs
            attn_scores += self.get_diagonal_mask(
                attention_mask=attention_mask, query_vectors=query_vectors
            )

        if list(attn_scores.size()) != [  # noqa: WPS337
            batch_size,
            seq_len,
            self.num_heads,
            self.one_sided_attn_window_size * 2 + 1,
        ]:
            raise AssertionError(
                f"local_attn_probs should be of size ({batch_size}, {seq_len}, {self.num_heads}, "
                + f"{self.one_sided_attn_window_size * 2 + 1}), but is of size {attn_scores.size()}"
            )

        # compute local attention probs from global attention keys and contact over window dim
        if is_global_attn:
            # compute global attn indices required through out forward fn
            (
                max_num_global_attn_indices,
                is_index_global_attn_nonzero,
                is_local_index_global_attn_nonzero,
                is_local_index_no_global_attn_nonzero,
            ) = self._get_global_attn_indices(is_index_global_attn)

            # calculate global attn probs from global key
            global_key_attn_scores = self._concat_with_global_key_attn_probs(
                query_vectors=query_vectors,
                key_vectors=key_vectors,
                max_num_global_attn_indices=max_num_global_attn_indices,
                is_index_global_attn_nonzero=is_index_global_attn_nonzero,
                is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
                is_local_index_no_global_attn_nonzero=is_local_index_no_global_attn_nonzero,
            )

            # concat to local_attn_probs
            # (batch_size, seq_len, num_heads, extra attention count + 2*window+1)
            attn_scores = torch.cat((global_key_attn_scores, attn_scores), dim=-1)
            # free memory
            del global_key_attn_scores  # noqa: WPS420

        attn_probs = functional.softmax(
            attn_scores, dim=-1, dtype=torch.float32
        )  # use fp32 for numerical stability

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise AssertionError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, "
                    + f"but is {layer_head_mask.size()}"
                )
            attn_probs = layer_head_mask.view(1, 1, -1, 1) * attn_probs

        # softmax sometimes inserts NaN if all positions are masked, replace them with 0
        if is_index_masked is not None:
            attn_probs = torch.masked_fill(
                attn_probs, is_index_masked[:, :, None, None], 0.0  # noqa: WPS358
            )
        attn_probs = attn_probs.type_as(attn_scores)

        # free memory
        del attn_scores  # noqa: WPS420

        # apply dropout
        attn_probs = functional.dropout(attn_probs, p=self.dropout, training=self.training)

        value_vectors = value_vectors.view(
            seq_len, batch_size, self.num_heads, self.head_dim
        ).transpose(0, 1)

        # compute local attention output with global attention value and add
        # expected shape: (batch_size, seq_len, self.num_heads, self.head_dim)
        if is_global_attn:
            # compute sum of global and local attn
            attn_output = self._compute_attn_output_with_global_indices(
                value_vectors=value_vectors,
                attn_probs=attn_probs,
                max_num_global_attn_indices=max_num_global_attn_indices,
                is_index_global_attn_nonzero=is_index_global_attn_nonzero,
                is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
            )
        else:
            # compute local attn only
            attn_output = self._sliding_chunks_matmul_attn_probs_value(
                attn_probs, value_vectors, self.one_sided_attn_window_size
            )

        attn_output = (
            attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
        )

        # compute value for global attention and overwrite to attention output
        if is_global_attn:
            is_local_index_global_attn_future: Optional[tuple[torch.Tensor, ...]] = None
            if attention_mask is not None and len(attention_mask.shape) == 3:
                is_local_index_global_attn_future = self._get_global_future_attn_indices(
                    attention_mask
                )

            global_attn_output, global_attn_probs = self._compute_global_attn_output_from_hidden(
                hidden_states=hidden_states,
                max_num_global_attn_indices=max_num_global_attn_indices,
                layer_head_mask=layer_head_mask,
                is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
                is_index_global_attn_nonzero=is_index_global_attn_nonzero,
                is_local_index_no_global_attn_nonzero=is_local_index_no_global_attn_nonzero,
                is_index_masked=is_index_masked,
                is_local_index_global_attn_future=is_local_index_global_attn_future,
            )
            # get only non zero global attn output
            nonzero_global_attn_output = global_attn_output[
                is_local_index_global_attn_nonzero[0], :, is_local_index_global_attn_nonzero[1]
            ]

            # overwrite values with global attention
            attn_output[is_index_global_attn_nonzero[::-1]] = nonzero_global_attn_output.view(
                len(is_local_index_global_attn_nonzero[0]), -1
            )
            # The attention weights for tokens with global attention are
            # just filler values, they were never used to compute the output.
            # Fill with 0 now, the correct values are in 'global_attn_probs'.
            attn_probs[is_index_global_attn_nonzero] = 0

        outputs: tuple[torch.Tensor, ...] = (attn_output.transpose(0, 1),)

        if output_attentions:
            outputs += (attn_probs,)

        return (
            outputs + (global_attn_probs,) if (is_global_attn and output_attentions) else outputs
        )

    def _get_global_future_attn_indices(
        self, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        """Get the indices for all future-global attention indices."""
        # (batch_size, seq_len, seq_len)
        attention_mask_sum = attention_mask.new_zeros(attention_mask.size())
        # Set allowed local and global attention elements to 1
        attention_mask_sum[attention_mask >= 0] = 1

        # (batch_size, seq_len, seq_len) -> (batch_size, seq_len)
        # Compute how many tokens attend to each token
        attention_mask_sum = attention_mask_sum.sum(dim=1)
        # History tokens are attended by the maximum number of tokens
        future_mask = attention_mask_sum < attention_mask_sum.max(dim=-1).values[:, None]
        is_local_index_global_attn_future = future_mask.nonzero(as_tuple=True)
        return is_local_index_global_attn_future

    def _diagonal_from_full_mask(self, mask: torch.Tensor, window_overlap: int) -> torch.Tensor:
        """Get diagonal from full 2D attention mask."""
        batch_size = mask.size(0)
        seq_len = mask.size(1)
        if seq_len % (window_overlap * 2) != 0:
            raise AssertionError(
                f"Sequence length should be multiple of {window_overlap * 2}. Given {seq_len}"
            )
        # convert diagonals into columns
        mask = self._pad_and_transpose_last_two_dims(mask, padding=(0, 0, 0, 1))
        attention_scores = mask.new_empty(
            batch_size,
            seq_len,
            2 * window_overlap + 1,
        )
        attention_scores[:, :, window_overlap:] = mask[:, :, : window_overlap + 1]
        attention_scores[:, 1:, :window_overlap] = mask[:, :-1, -window_overlap:]  # noqa: WPS221

        attention_scores = attention_scores[:, :, : 2 * window_overlap + 1]
        attention_scores = (
            attention_scores.view(
                batch_size,
                1,
                seq_len,
                2 * window_overlap + 1,
            ).transpose(2, 1)
            * 1.0  # noqa: WPS345
        )
        self._mask_invalid_locations(attention_scores, window_overlap)
        return attention_scores

    def _compute_global_attn_output_from_hidden(
        self,
        hidden_states: torch.Tensor,
        max_num_global_attn_indices: int,
        layer_head_mask: Optional[torch.Tensor],
        is_local_index_global_attn_nonzero: tuple[torch.Tensor, torch.Tensor],
        is_index_global_attn_nonzero: tuple[torch.Tensor, torch.Tensor],
        is_local_index_no_global_attn_nonzero: tuple[torch.Tensor, torch.Tensor],
        is_index_masked: Optional[torch.Tensor] = None,
        is_local_index_global_attn_future: Optional[tuple[torch.Tensor, ...]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute attention from global to all tokens."""
        seq_len, batch_size = hidden_states.shape[:2]

        # prepare global hidden states
        global_attn_hidden_states = hidden_states.new_zeros(
            max_num_global_attn_indices, batch_size, self.embed_dim
        )
        global_attn_hidden_states[is_local_index_global_attn_nonzero[::-1]] = hidden_states[
            is_index_global_attn_nonzero[::-1]
        ]

        # global key, query, value
        global_query_vectors_only_global = self.query_global(global_attn_hidden_states)
        global_key_vectors = self.key_global(hidden_states)
        global_value_vectors = self.value_global(hidden_states)

        # normalize
        global_query_vectors_only_global /= math.sqrt(self.head_dim)

        # reshape

        # (batch_size * self.num_heads, max_num_global_attn_indices, head_dim)
        global_query_vectors_only_global = (
            global_query_vectors_only_global.contiguous()
            .view(max_num_global_attn_indices, batch_size * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )

        # batch_size * self.num_heads, seq_len, head_dim)
        global_key_vectors = (
            global_key_vectors.contiguous()
            .view(-1, batch_size * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )

        # batch_size * self.num_heads, seq_len, head_dim)
        global_value_vectors = (
            global_value_vectors.contiguous()
            .view(-1, batch_size * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )

        # compute attn scores
        global_attn_scores = torch.bmm(
            global_query_vectors_only_global, global_key_vectors.transpose(1, 2)
        )

        if list(global_attn_scores.size()) != [  # noqa: WPS337
            batch_size * self.num_heads,
            max_num_global_attn_indices,
            seq_len,
        ]:
            raise AssertionError(
                f"global_attn_scores have the wrong size. Size should be {(batch_size * self.num_heads, max_num_global_attn_indices, seq_len)}, but is {global_attn_scores.size()}."
            )

        global_attn_scores = global_attn_scores.view(
            batch_size, self.num_heads, max_num_global_attn_indices, seq_len
        )

        global_attn_scores[
            is_local_index_no_global_attn_nonzero[0],
            :,
            is_local_index_no_global_attn_nonzero[1],
            :,
        ] = self.mask_value

        # Pad for future tokens: Modification of LED code.
        if is_local_index_global_attn_future is not None:
            global_attn_scores[
                is_local_index_global_attn_future[0], :, :, is_local_index_global_attn_future[1]
            ] = self.mask_value

        global_attn_scores = global_attn_scores.masked_fill(
            is_index_masked[:, None, None, :],  # type: ignore[index]
            self.mask_value,
        )

        global_attn_scores = global_attn_scores.view(
            batch_size * self.num_heads, max_num_global_attn_indices, seq_len
        )

        # compute global attn probs
        global_attn_probs_float = functional.softmax(
            global_attn_scores, dim=-1, dtype=torch.float32
        )  # use fp32 for numerical stability

        # apply layer head masking
        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise AssertionError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is {layer_head_mask.size()}"
                )
            global_attn_probs_float = layer_head_mask.view(
                1, -1, 1, 1
            ) * global_attn_probs_float.view(
                batch_size, self.num_heads, max_num_global_attn_indices, seq_len
            )
            global_attn_probs_float = global_attn_probs_float.view(
                batch_size * self.num_heads, max_num_global_attn_indices, seq_len
            )

        global_attn_probs = functional.dropout(
            global_attn_probs_float.type_as(global_attn_scores),
            p=self.dropout,
            training=self.training,
        )

        # global attn output
        global_attn_output = torch.bmm(global_attn_probs, global_value_vectors)

        if list(global_attn_output.size()) != [  # noqa: WPS337
            batch_size * self.num_heads,
            max_num_global_attn_indices,
            self.head_dim,
        ]:
            raise AssertionError(
                f"global_attn_output tensor has the wrong size. Size should be {(batch_size * self.num_heads, max_num_global_attn_indices, self.head_dim)}, but is {global_attn_output.size()}."
            )

        global_attn_probs = global_attn_probs.view(
            batch_size, self.num_heads, max_num_global_attn_indices, seq_len
        )
        global_attn_output = global_attn_output.view(
            batch_size, self.num_heads, max_num_global_attn_indices, self.head_dim
        )
        return global_attn_output, global_attn_probs


class EmmaEncoderAttention(LEDEncoderAttention):
    """[`EmmaEncoderAttention`] is identical to [`LEDEncoderAttention`]."""

    def __init__(self, config: EmmaConfig, layer_id: int) -> None:
        super().__init__(config=config, layer_id=layer_id)
        self.longformer_self_attn = EmmaEncoderSelfAttention(config, layer_id=layer_id)


class EmmaEncoderLayer(LEDEncoderLayer):
    """[`EmmaEncoderLayer`] is identical to [`LEDEncoderLayer`]."""

    def __init__(self, config: EmmaConfig, layer_id: int) -> None:
        super().__init__(config=config, layer_id=layer_id)  # type: ignore[arg-type]
        self.self_attn = EmmaEncoderAttention(config, layer_id)
