import dataclasses
import logging
from argparse import ArgumentParser
from pathlib import Path
from random import randint
from typing import Optional

import numpy as np
import torch
from emma_datasets.datamodels import TeachEdhInstance
from PIL import Image
from pytorch_lightning import LightningModule
from transformers.generation_stopping_criteria import StoppingCriteriaList

from emma_policy.datamodules.batch_attention_masks import make_batch_attention_masks
from emma_policy.datamodules.emma_dataclasses import (
    EmmaDatasetBatch,
    EmmaDatasetItem,
    EmmaDatasetPadding,
)
from emma_policy.inference.actions import AgentAction, teach_action_types
from emma_policy.inference.api.settings import ApiSettings
from emma_policy.inference.decoded_trajectory_parser import DecodedTrajectoryParser
from emma_policy.inference.model_wrapper.base import BaseModelWrapper, SimulatorAction
from emma_policy.inference.model_wrapper.stopping_criteria import ActionStopCriteria
from emma_policy.inference.model_wrapper.teach_edh_inference_dataset import (
    TeachEdhInferenceDataset,
)
from emma_policy.inference.model_wrapper.teach_edh_inference_state import EdhInstanceInferenceState
from emma_policy.models.emma_policy import EmmaPolicy


logger = logging.getLogger(__name__)

IMAGE_SIMILARITY_ABSOLUTE_THRESHOLD = 1e-5


class PolicyModelWrapper(BaseModelWrapper):
    """Wrapper around the EMMA Policy model for performing inference."""

    def __init__(
        self,
        process_index: int,
        num_processes: int,
        model_checkpoint_path: Path,
        model_name: str = "heriot-watt/emma-base",
        max_frames: int = 100,
        max_target_len: int = 10,
        max_lang_tokens: Optional[int] = None,
        device_id: int = -1,
        generation_num_beams: int = 1,
    ) -> None:

        self._device = self._get_device(process_index, device_id)
        self._model_name = model_name
        self._model = self._setup_model(model_checkpoint_path)

        feature_extractor_settings = ApiSettings().feature_extractor_api
        logger.info(f"Using feature extractor API at `{feature_extractor_settings.endpoint}`")

        self._teach_edh_inference_dataset = TeachEdhInferenceDataset.from_model_name(
            model_name=model_name,
            max_frames=max_frames,
            max_lang_tokens=max_lang_tokens,
            feature_extractor_settings=feature_extractor_settings,
        )

        self._tokenizer = self._teach_edh_inference_dataset.tokenizer
        self._parse_decoded_trajectory = DecodedTrajectoryParser(
            execution_domain="TEACh", action_delimiter=".", eos_token=self._tokenizer.eos_token
        )

        self._edh_instance_state = EdhInstanceInferenceState(
            max_target_len,
            max_target_len,
            max_past_decoding_steps=max_frames - 1,
            eos_token_id=self._tokenizer.eos_token_id,
        )

        self.action_stop = StoppingCriteriaList(
            [
                ActionStopCriteria(
                    action_sep_token_id=self._tokenizer.sep_token_id,  # type: ignore[arg-type]
                    eos_token_id=self._tokenizer.eos_token_id,  # type: ignore[arg-type]
                )
            ]
        )
        self._generation_num_beams = generation_num_beams

        # Update the torch device used by the Perception API to ensure they're the same
        self._teach_edh_inference_dataset.client.update_device(self._device)
        self._action_types = teach_action_types()

    @classmethod
    def from_argparse(
        cls, process_index: int, num_processes: int, model_args: list[str]
    ) -> "PolicyModelWrapper":
        """Create the policy model from argparse."""
        arg_parser = ArgumentParser("EMMA Policy Model Wrapper")

        arg_parser.add_argument(
            "--model_checkpoint_path",
            type=Path,
            required=True,
            help="Path to the model checkpoint file.",
        )
        arg_parser.add_argument(
            "--model_name",
            type=str,
            default="heriot-watt/emma-base",
            help="Name of the pretrained model to setup the correct checkpoint",
        )
        arg_parser.add_argument(
            "--device_id", type=int, default=-1, help="CPU/GPU device id. Use -1 for CPU"
        )
        arg_parser.add_argument(
            "--generation_num_beams",
            type=int,
            default=1,
            help="Number of beams for beam search. 1 means no beam search.",
        )
        arg_parser.add_argument(
            "--max_frames",
            type=int,
            default=32,  # noqa: WPS432
            help="Set max number of frames for the model to decode for.",
        )
        parsed_model_args = arg_parser.parse_args(model_args)

        logger.debug(parsed_model_args)

        return cls(
            process_index=process_index,
            num_processes=num_processes,
            model_checkpoint_path=parsed_model_args.model_checkpoint_path,
            model_name=parsed_model_args.model_name,
            device_id=parsed_model_args.device_id,
            generation_num_beams=parsed_model_args.generation_num_beams,
            max_frames=parsed_model_args.max_frames,
        )

    def start_new_edh_instance(
        self,
        edh_instance: TeachEdhInstance,
        edh_history_images: list[Image.Image],
        edh_name: Optional[str] = None,
    ) -> bool:
        """Reset the model ready for a new EDH instance."""
        self._teach_edh_inference_dataset.start_new_edh_instance(edh_instance, edh_history_images)

        self._edh_instance_state.reset_state()

        return True

    def get_next_action(
        self,
        img: Image,
        edh_instance: TeachEdhInstance,
        prev_action: Optional[SimulatorAction],
        img_name: Optional[str] = None,
        edh_name: Optional[str] = None,
    ) -> tuple[str, Optional[tuple[float, float]]]:
        """Get the next predicted action from the model.

        Called at each timestep.

        Args:
            img: Agent's egocentric image.
            edh_instance: EDH Instance from the file.
            prev_action: Previous action taken by the agent, if any
            img_name: File name of the image
            edh_name: File name for the EDH instance

        Returns:
            - action name for the TEACh API
            - obj_relative_coord: A relative (x, y) coordinate (values between 0 and 1)
                indicating an object in the image if available
        """
        logger.info(f"Getting next action for EDH instance `{edh_instance.instance_id}`")

        dataset_instance = self._convert_edh_to_dataset_instance(current_frame=img)

        model_input_batch = self._create_model_input_from_dataset_instance(dataset_instance)

        output_token_ids = self._predict_action_trajectory(model_input_batch)

        next_action = self._parse_predicted_action(output_token_ids)

        is_agent_stuck = self._is_agent_stuck(
            previous_action=prev_action, next_action=next_action, current_frame=img
        )

        if is_agent_stuck:
            next_action, output_token_ids = self.handle_repeatedly_walking_into_obstruction(
                model_input_batch
            )

        # TODO: Any other cases that need handling?

        self._edh_instance_state.update_state(
            instance=dataset_instance, output_token_ids=output_token_ids
        )

        # Update the previous frame
        self._teach_edh_inference_dataset.previous_frame = img

        return next_action.action, self._get_object_relative_coordinates_from_action(
            next_action, dataset_instance
        )

    def handle_repeatedly_walking_into_obstruction(
        self, model_input_batch: EmmaDatasetBatch
    ) -> tuple[AgentAction, torch.Tensor]:
        """Get the agent to turn if it keeps walking into an obstacle."""
        turn_token = self._tokenizer.encode(
            " turn", add_special_tokens=False, return_tensors="pt"
        )[0]

        extended_decoding_input_ids = self._edh_instance_state.decoding_input_ids.copy()
        extended_decoding_input_ids.append(turn_token)
        decoding_input_ids = torch.cat(extended_decoding_input_ids)

        output_token_ids = self._predict_action_trajectory(
            model_input_batch, decoding_input_ids=decoding_input_ids
        )
        new_action = self._parse_predicted_action(output_token_ids)
        return new_action, output_token_ids

    def _is_agent_stuck(
        self,
        previous_action: Optional[SimulatorAction],
        next_action: AgentAction,
        current_frame: Image,
    ) -> bool:
        """Determine whether or not the agent is stuck.

        Perform two main checks:
            - Does the previous frame match the current frame?
            - Did the agent predict `Forward` again?
        """
        is_agent_predicting_the_same_action = (
            previous_action is not None and previous_action.action == next_action.action
        )

        is_images_identical = np.allclose(
            np.asarray(self._teach_edh_inference_dataset.previous_frame),
            np.asarray(current_frame),
            atol=IMAGE_SIMILARITY_ABSOLUTE_THRESHOLD,
        )

        return is_agent_predicting_the_same_action and bool(is_images_identical)

    def _compute_center_from_bbox(self, bbox_coordinates: torch.Tensor) -> tuple[float, float]:
        """Compute the centroid of a given bounding box.

        Args:
            bbox_coordinates (torch.Tensor): Coordinates as XYXY (x1, y1, x2, y2)

        Returns:
            Relative (x, y) coordinates of the center of the bounding box.
        """
        x_center = ((bbox_coordinates[0] + bbox_coordinates[2]) / 2).item()
        y_center = ((bbox_coordinates[1] + bbox_coordinates[3]) / 2).item()
        return (y_center, x_center)

    def _setup_model(self, model_checkpoint_path: Path) -> LightningModule:
        """Setup the model from the checkpoint."""
        model = EmmaPolicy(model_name=self._model_name)
        model = model.load_from_checkpoint(
            model_checkpoint_path.as_posix(), map_location=self._device
        )
        model.eval()

        return model

    def _get_device(self, process_index: int, device_id: int = -1) -> torch.device:
        """Get the device for the model.

        This does it the exact same way they did it for ET, provided the device_id is not greater
        than -1.
        """
        if not torch.cuda.is_available():
            return torch.device("cpu")

        gpu_count = torch.cuda.device_count()
        logger.info(f"{gpu_count} GPUs detected")

        model_device_id = device_id if device_id > -1 else process_index % gpu_count

        device = torch.device(f"cuda:{model_device_id}")
        logger.info(f"Device used: {device}")

        return device

    def _convert_edh_to_dataset_instance(self, current_frame: Image.Image) -> EmmaDatasetItem:
        """Convert the TEACh EDH instance to the EmmaDatasetItem for the model."""
        dataset_instance = self._teach_edh_inference_dataset.get_next_dataset_instance(
            current_frame=current_frame
        )
        # Add some dummy token ids for predicting the next action
        extended_decoding_input_ids = self._edh_instance_state.decoding_input_ids.copy()
        extended_decoding_input_ids.append(
            torch.zeros(self._edh_instance_state.step_max_target_length, dtype=torch.int64)
        )
        dataset_instance.target_token_ids = torch.cat(extended_decoding_input_ids)
        # Add some dummy target temporal ids for next prediction
        extended_target_temporal_ids = self._edh_instance_state.target_temporal_ids.copy()
        extended_target_temporal_ids.append(
            torch.full(
                size=(self._edh_instance_state.step_max_target_length,),
                fill_value=self._edh_instance_state.decoding_step,
                dtype=torch.int64,
            )
        )
        dataset_instance.target_temporal_ids = torch.cat(extended_target_temporal_ids)
        dataset_instance.decoder_attention_mask = torch.ones_like(
            dataset_instance.target_temporal_ids, dtype=torch.int64
        )
        return dataset_instance

    def _create_model_input_from_dataset_instance(
        self, dataset_instance: EmmaDatasetItem
    ) -> EmmaDatasetBatch:
        """Create the batched input for the model from the dataset instance.

        Collate lists of samples into batches after padding.
        """
        fields = dataclasses.fields(EmmaDatasetItem)
        padding = EmmaDatasetPadding()

        raw_batch = {
            field.name: getattr(dataset_instance, field.name).unsqueeze(0)
            for field in fields
            if getattr(dataset_instance, field.name) is not None
        }
        make_batch_attention_masks(raw_batch, padding_value=padding.attention_mask)
        return EmmaDatasetBatch(**raw_batch)

    def _predict_action_trajectory(
        self, model_input: EmmaDatasetBatch, decoding_input_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Get the model to predict the action trajectory."""
        if decoding_input_ids is None:
            decoding_input_ids = torch.cat(self._edh_instance_state.decoding_input_ids)
        with torch.no_grad():
            output_token_ids = self._model.inference_step(  # type: ignore[operator]
                model_input,
                decoder_input_ids=decoding_input_ids.unsqueeze(0),
                max_length_per_action_sequence=self._edh_instance_state.total_max_target_length,
                action_stop=self.action_stop,
                num_beams=self._generation_num_beams,
            )[0]

        return output_token_ids

    def _parse_predicted_action(self, model_output_tokens: torch.Tensor) -> AgentAction:
        """Convert the predicted action from the model into the actual action.

        If it's the first decoding step, ignore the initial special tokens (e.g. <s></s>).
        """
        next_action_token_ids = model_output_tokens[
            self._edh_instance_state.previous_decoded_token_length :
        ]

        if self._edh_instance_state.is_first_decoding_step:
            next_action_token_ids = next_action_token_ids[1:]

        next_action_raw_string = self._tokenizer.decode(
            next_action_token_ids, skip_special_tokens=False
        )

        next_action = self._parse_decoded_trajectory(next_action_raw_string)

        return next_action

    def _get_object_relative_coordinates_from_action(
        self, action: AgentAction, teach_item: EmmaDatasetItem
    ) -> Optional[tuple[float, float]]:
        """Return relative (x, y) coordinates indicating the position of the object in the image.

        Note:   The TEACh wrapper on AI2-THOR examines the ground truth segmentation mask of the
                agent's egocentric image, selects an object in a 10x10 pixel patch around the pixel
                indicated by the coordinate if the desired action can be performed on it, and
                executes the action in AI2-THOR.
        """
        action_type = self._action_types.get(action.action)
        if action_type is not None and action_type != "ObjectInteraction":
            return None

        # Attempt to index the object label
        object_index = action.get_object_index_from_label(
            bbox_probas=self._teach_edh_inference_dataset.get_current_object_probas()
        )
        if object_index is not None:
            return self._compute_center_from_bbox(
                bbox_coordinates=self._teach_edh_inference_dataset.get_current_coordinates()[
                    object_index
                ]
            )

        # Attempt to index the visual token
        object_index = action.get_object_index_from_visual_token()

        if object_index is not None:
            return self._compute_center_from_bbox(
                bbox_coordinates=self._teach_edh_inference_dataset.get_current_coordinates()[
                    object_index
                ]
            )

        # Attempt to get an object with the most similar label
        object_index = action.get_similarity_based_object_index(
            bbox_probas=self._teach_edh_inference_dataset.get_current_object_probas()
        )

        if object_index is not None:
            return self._compute_center_from_bbox(
                bbox_coordinates=teach_item.object_coordinates[object_index]
            )

        # Attempt to get an object with the most similar name that was not an AI2THOR label
        object_index = action.get_similarity_based_raw_object_index(
            bbox_probas=self._teach_edh_inference_dataset.get_current_object_probas()
        )

        # Pick a random object
        if object_index is None:
            object_index = randint(0, len(teach_item.object_coordinates) - 1)

        return self._compute_center_from_bbox(
            bbox_coordinates=self._teach_edh_inference_dataset.get_current_coordinates()[
                object_index
            ],
        )
