import logging
from typing import Any, Union

from transformers import PretrainedConfig


logger = logging.getLogger(__name__)


EMMA_PRETRAINED_CONFIG_ARCHIVE_MAP = {  # noqa: WPS407
    "heriot-watt/emma-small": "heriot-watt/emma-small/config.json",
    "heriot-watt/emma-base": "heriot-watt/emma-base/config.json",
}


class EmmaConfig(PretrainedConfig):  # noqa: WPS230
    """Emma model configuration, modified from LED."""

    model_type = "emma"
    attribute_map = {
        "num_attention_heads": "encoder_attention_heads",
        "hidden_size": "d_model",
        "attention_probs_dropout_prob": "attention_dropout",
        "initializer_range": "init_std",
        "hidden_dropout_prob": "dropout",
    }

    def __init__(
        self,
        vocab_size: int = 50265,
        max_encoder_position_embeddings: int = 16384,
        max_decoder_position_embeddings: int = 1024,
        max_frame_embeddings: int = 512,
        encoder_layers: int = 12,
        encoder_ffn_dim: int = 4096,
        encoder_attention_heads: int = 16,
        decoder_layers: int = 12,
        decoder_ffn_dim: int = 4096,
        decoder_attention_heads: int = 16,
        encoder_layerdrop: float = 0,
        decoder_layerdrop: float = 0,
        use_cache: bool = True,
        is_encoder_decoder: bool = True,
        activation_function: str = "gelu",
        d_model: int = 1024,
        dropout: float = 0.1,
        attention_dropout: float = 0,
        activation_dropout: float = 0,
        init_std: float = 0.02,
        decoder_start_token_id: int = 2,
        classifier_dropout: float = 0,
        pad_token_id: int = 1,
        bos_token_id: int = 0,
        eos_token_id: int = 2,
        attention_window: Union[list[int], int] = 512,
        scene_features_dim: int = 1024,
        object_features_dim: int = 2048,
        image_coordinates_dim: int = 4,
        **kwargs: Any,
    ) -> None:
        self.vocab_size = vocab_size
        self.max_encoder_position_embeddings = max_encoder_position_embeddings
        self.max_decoder_position_embeddings = max_decoder_position_embeddings
        self.max_frame_embeddings = max_frame_embeddings
        self.d_model = d_model
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_ffn_dim = decoder_ffn_dim
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.activation_function = activation_function
        self.init_std = init_std
        self.encoder_layerdrop = encoder_layerdrop
        self.decoder_layerdrop = decoder_layerdrop
        self.classifier_dropout = classifier_dropout
        self.use_cache = use_cache
        self.num_hidden_layers = encoder_layers
        self.attention_window = attention_window
        self.scene_features_dim = scene_features_dim
        self.object_features_dim = object_features_dim
        self.image_coordinates_dim = image_coordinates_dim

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            decoder_start_token_id=decoder_start_token_id,
            **kwargs,
        )
