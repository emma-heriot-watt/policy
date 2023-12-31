import logging
from typing import Any, Optional

import torch
from emma_common.datamodels import EmmaPolicyRequest
from pytorch_lightning.utilities import move_data_to_device
from transformers import BatchEncoding, PreTrainedTokenizer

from emma_policy.datamodules.base_dataset import prepare_emma_visual_features
from emma_policy.datamodules.collate import collate_fn
from emma_policy.datamodules.emma_dataclasses import (
    EmmaDatasetBatch,
    EmmaDatasetItem,
    EmmaVisualFeatures,
)
from emma_policy.datamodules.pretrain_instances import TASK_TEMPLATES_MAP, Task
from emma_policy.inference.api.simbot_state import SPEAKER_TOKEN_MAP
from emma_policy.utils.datamodels.simbot import EMPTY_INVENTORY


logger = logging.getLogger(__name__)

ActionBuilderOutput = tuple[
    Optional[str], Optional[EmmaDatasetBatch], Optional[torch.Tensor], Optional[list[int]]
]


class SimBotActionInputBuilder:
    """Build the input for the SimBotAction model."""

    def __init__(self, tokenizer: PreTrainedTokenizer, device: str = "cpu") -> None:
        self._tokenizer = tokenizer
        self._device = device
        self._input_prompt = {
            Task.action_execution: TASK_TEMPLATES_MAP[Task.action_execution][0],
            Task.visual_grounding: TASK_TEMPLATES_MAP[Task.visual_grounding][0],
        }
        # these are generally instructions that mix the policy tasks and confuse the model
        # e.g. "locate and pick up the bowl" triggers the search with <act><search> but then this utterance does not go to the utterance queue
        # this is problematic as we do half the instruction, we just find the bowl but then we dont pick it up
        self._skip_instruction_phrase_list = [
            "locate and",
            "find and",
            "search and",
            "look and",
            "trace and",
            "seek and",
        ]

    def __call__(self, request: EmmaPolicyRequest, task: Task) -> ActionBuilderOutput:
        """Process the environment output into a batch for the model.

        The sample batch provides the set of previous observations and previous actions taken by
        the agent in the environment.
        """
        inventory = EMPTY_INVENTORY if request.inventory is None else request.inventory
        instruction = self._parse_dialogue_from_request(
            request, task=task
        )  # @TODO check whether gfh should change instructions
        (
            feature_dicts,
            previous_actions,
            step_index,
        ) = self._parse_environment_history_from_request(request)
        # assert len(feature_dicts) == len(step_index)
        batch: Optional[EmmaDatasetBatch] = None
        decoder_input_ids: Optional[torch.Tensor] = None

        if instruction is not None and instruction:
            instruction = f"Inventory: {inventory}. {instruction}"
            logger.debug(f"Predicting action for instruction: {instruction}")

            encoded_inputs = self._prepare_input_text(instruction=instruction, task=task)
            if task == Task.action_execution:
                visual_features = prepare_emma_visual_features(
                    feature_dicts=feature_dicts,
                    tokenizer=self._tokenizer,
                )

                dataset_item = self._create_emma_dataset_item(
                    visual_features=visual_features,
                    encoded_inputs=encoded_inputs,
                    minimum_frame_index=self._get_minimum_predicted_frame_index(
                        feature_dicts=feature_dicts, request=request
                    ),
                )
                decoder_input_ids = self._prepare_decoder_input_ids(
                    previous_actions=previous_actions
                )

                batch = self._create_emma_dataset_batch([dataset_item])
            elif task == Task.visual_grounding:
                dataset_items = []
                for feature_dict in feature_dicts:
                    visual_features = prepare_emma_visual_features(
                        feature_dicts=[feature_dict],
                        tokenizer=self._tokenizer,
                    )

                    dataset_item = self._create_emma_dataset_item(
                        visual_features=visual_features,
                        encoded_inputs=encoded_inputs,
                        minimum_frame_index=1,
                    )
                    dataset_items.append(dataset_item)

                batch = self._create_emma_dataset_batch(dataset_items)  # type: ignore[arg-type]

                decoder_input_ids = self._prepare_decoder_input_ids(
                    previous_actions=previous_actions
                )
            else:
                logger.error(f"Found unsupported task: {task}")
        return (instruction, batch, decoder_input_ids, step_index)

    def _prepare_decoder_input_ids(
        self, previous_actions: Optional[str] = None
    ) -> Optional[torch.Tensor]:
        """Prepare decoder input ids."""
        if previous_actions:
            decoder_encoding = self._tokenizer.encode_plus(
                previous_actions, return_tensors="pt", truncation=True
            )
            decoder_input_ids = decoder_encoding.input_ids
            shifted_decoder_input_ids = decoder_input_ids.new_full(
                decoder_input_ids.shape,
                fill_value=self._tokenizer.eos_token_id,
            )
            shifted_decoder_input_ids[:, 1:] = decoder_input_ids[:, :-1].clone()

            decoder_input_ids = move_data_to_device(shifted_decoder_input_ids, self._device)
            return decoder_input_ids
        return None

    def _parse_dialogue_from_request(
        self, request: EmmaPolicyRequest, task: Task
    ) -> Optional[str]:
        """Parse the dialogue for the current request."""
        instruction = None
        dialogue = []
        for utterance in request.dialogue_history:
            if not utterance.utterance:
                continue
            instruction_utterance = utterance.utterance
            for phrase in self._skip_instruction_phrase_list:
                instruction_utterance = instruction_utterance.replace(phrase, "").strip()

            utterance_text = f"{SPEAKER_TOKEN_MAP[utterance.role]} {instruction_utterance}"
            if not utterance_text.endswith(("?", ".")):
                utterance_text = f"{utterance_text}."
            dialogue.append(utterance_text)

        instruction = self._get_instruction_from_dialogue(dialogue, task)

        if dialogue:
            logger.debug(f"Found instruction: {instruction}")
        else:
            logger.debug(f"No instruction for request: {request}")
        return instruction

    def _get_instruction_from_dialogue(self, dialogue: list[str], task: Task) -> Optional[str]:
        if not dialogue:
            return None
        if task == Task.action_execution:
            instruction = " ".join(dialogue).lower()
        else:
            if len(dialogue) > 1:
                raise ValueError("Found more than one utterance in the dialogue history for VG.")
            instruction = dialogue[-1].lower()
            instruction = instruction.split("<<driver>>")[0].strip()
        return instruction

    def _parse_environment_history_from_request(
        self, request: EmmaPolicyRequest
    ) -> tuple[list[dict[str, torch.Tensor]], Optional[str], Optional[list[int]]]:  # noqa: WPS221
        """Parse the feature dicts and actions from the current request."""
        feature_dicts: list[dict[str, torch.Tensor]] = []
        step_index: list[int] = []
        previous_actions: list[str] = []
        total_steps = len(request.environment_history)
        for idx, step in enumerate(request.environment_history, 1):
            if step.output is None and idx < total_steps:
                msg = "Found unexpected 'None' as a previous action. Verify that the received request contains string values for previous actions."
                logger.debug(msg)

            feature_dicts.extend(feature.dict() for feature in step.features)
            step_index.extend(idx - 1 for _ in step.features)

            if idx < total_steps:
                output = step.output.replace("</s>", "")
                previous_actions.append(output)

        # concatenate all the previous actions to a string
        # ignore the last element as it is None (the current action has not been predicted yet)
        previous_actions_str = None
        if previous_actions:
            # Currently the implementation allows None previous actios
            # but in practice this should never happen.
            previous_actions_str = " ".join(previous_actions)
        return (feature_dicts, previous_actions_str, step_index)

    def _prepare_input_text(self, instruction: str, task: Task) -> BatchEncoding:
        """Prepare the input text for the SimBotAction model.

        The input text follows the same template as with the action execution with the addition of
        the clarification question and answer (if provided).
        """
        if task == Task.action_execution:
            source_text = self._input_prompt[task].format(instruction=instruction)
        elif task == Task.visual_grounding:
            source_text = self._input_prompt[task].format(caption=instruction)
        else:
            logger.error(
                f"Found unsupported task: {task}. Using empty string as input to the model"
            )
            source_text = ""

        return self._tokenizer.encode_plus(
            source_text.lower(), return_tensors="pt", truncation=True
        )

    def _create_emma_dataset_item(
        self,
        visual_features: EmmaVisualFeatures,
        encoded_inputs: BatchEncoding,
        minimum_frame_index: int = 0,
    ) -> EmmaDatasetItem:
        """Create the `EmmaDatasetItem` for a given set of observations and actions."""
        return EmmaDatasetItem(
            # Text
            input_token_ids=encoded_inputs.input_ids.squeeze(0),
            text_attention_mask=encoded_inputs.attention_mask.squeeze(0),
            # Visual features
            object_attention_mask=visual_features.object_attention_mask,
            object_coordinates=visual_features.object_coordinates,
            object_features=visual_features.object_features,
            object_frame_tokens=visual_features.object_frame_tokens,
            scene_attention_mask=visual_features.scene_attention_mask,
            scene_coordinates=visual_features.scene_coordinates,
            scene_features=visual_features.scene_features,
            scene_frame_tokens=visual_features.scene_frame_tokens,
            visual_token_ids=visual_features.visual_token_ids,
            raw_target={"minimum_frame_index": minimum_frame_index},
        )

    def _create_emma_dataset_batch(
        self,
        dataset_items: list[Optional[EmmaDatasetItem]],
    ) -> EmmaDatasetBatch:
        """Create the `EmmaDatasetBatch` for a given set of observations and actions."""
        batch = collate_fn(dataset_items)
        prev_device = batch.input_token_ids.device
        batch = move_data_to_device(batch, self._device)
        logger.debug(f"Moved batch from {prev_device} to {batch.input_token_ids.device}")
        return batch

    def _get_minimum_predicted_frame_index(
        self, feature_dicts: list[dict[str, Any]], request: EmmaPolicyRequest
    ) -> int:
        """Force the predicted frame indices to be larger than past frames."""
        total_frames = len(feature_dicts)
        num_frames_in_current_turn = len(request.environment_history[-1].features)
        minimum_frame_index = total_frames - num_frames_in_current_turn + 1
        return minimum_frame_index
