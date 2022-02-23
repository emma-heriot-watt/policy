import torch
from torch.nn import Dropout, Embedding, LayerNorm, Linear, Module

from emma_policy.models.configuration_emma import EmmaConfig


class EmmaImagePositionEmbeddings(Module):
    """Learned positional embeddings for image frames represented as a set of regions."""

    def __init__(self, config: EmmaConfig) -> None:
        super().__init__()

        self.position_embeddings = Embedding(config.max_frame_embeddings, config.d_model)
        self.register_buffer(
            "position_ids", torch.arange(config.max_frame_embeddings).expand((1, -1))
        )

        self.projection = Linear(config.image_coordinates_dim, config.d_model)

    def forward(self, frame_idx: torch.Tensor, image_coordinates: torch.Tensor) -> torch.Tensor:
        """Embed the frame index and the region coordinates.

        Args:
            frame_idx (torch.Tensor): Frame indices.
            image_coordinates (torch.Tensor): Region coordinates of shape `(*, image_coordinates_dim)`.

        Returns:
            torch.Tensor: Position emebedding of shape `(*, d_model)`.
        """
        projected_coordinates = self.projection(image_coordinates)
        return self.position_embeddings(frame_idx) + projected_coordinates


class EmmaSceneEmbeddings(Module):
    """Embed a sequence of frames represented with grid features."""

    def __init__(
        self,
        image_position_embeddings: EmmaImagePositionEmbeddings,
        config: EmmaConfig,
    ) -> None:
        super().__init__()

        self.obj_embedding = Linear(config.scene_features_dim, config.d_model)
        self.layer_norm = LayerNorm(config.d_model)
        self.dropout = Dropout(config.hidden_dropout_prob)
        self.image_position_embeddings = image_position_embeddings

    def forward(
        self,
        cnn_features: torch.Tensor,
        image_coordinates: torch.Tensor,
        frame_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Embed a sequence of scenes.

        Args:
            cnn_features (torch.Tensor): Scene grid features of shape `(batch_size, num_frames, scene_features_dim)`.
            image_coordinates (torch.Tensor): Coordinates for the entire scene of shape `(batch_size, num_frames, image_coordinates_dim)`.
            frame_ids (torch.Tensor): Frame ids.

        Returns:
            torch.Tensor: Output embedding of shape `(batch_size, num_frames, d_model)`.
        """
        cnn_embeddings = self.obj_embedding(cnn_features)

        position_embeddings = self.image_position_embeddings(
            frame_idx=frame_ids, image_coordinates=image_coordinates
        )
        embeddings = cnn_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class EmmaObjectEmbeddings(Module):
    """Embed a sequence of frames represented as a set of regions."""

    def __init__(
        self,
        word_embeddings: Embedding,
        image_position_embeddings: EmmaImagePositionEmbeddings,
        config: EmmaConfig,
    ) -> None:
        super().__init__()

        self.obj_embedding = Linear(config.object_features_dim, config.d_model)
        self.layer_norm = LayerNorm(config.d_model)
        self.dropout = Dropout(config.hidden_dropout_prob)
        self.word_embeddings = word_embeddings
        self.image_position_embeddings = image_position_embeddings

    def forward(
        self,
        object_features: torch.Tensor,
        image_coordinates: torch.Tensor,
        visual_token_ids: torch.Tensor,
        frame_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Embed a sequence of object regions.

        Args:
            object_features (torch.Tensor): Region features of shape `(batch_size, num_frames * num_objects, object_features_dim)`.
            image_coordinates (torch.Tensor): Coordinates for the regions of shape `(batch_size, num_frames * num_objects, image_coordinates_dim)`.
            visual_token_ids (torch.Tensor): Visual sentinel tokens of shape `(batch_size, num_frames, num_objects)`.
            frame_ids (torch.Tensor): Frame id for each token.


        Returns:
            torch.Tensor: Output embedding of shape `(batch_size, num_frames, num_objects, d_model)`.
        """
        visual_sentinel = self.word_embeddings(visual_token_ids)

        obj_embedding = self.obj_embedding(object_features)

        position_embeddings = self.image_position_embeddings(
            frame_idx=frame_ids, image_coordinates=image_coordinates
        )
        embeddings = obj_embedding + visual_sentinel + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
