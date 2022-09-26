import logging
from typing import Any, Optional, Union

import torch
from pytorch_lightning.utilities import move_data_to_device
from transformers import BatchEncoding, PreTrainedTokenizer

from emma_policy.datamodules.base_dataset import prepare_emma_visual_features
from emma_policy.datamodules.collate import collate_fn
from emma_policy.datamodules.emma_dataclasses import (
    EmmaDatasetBatch,
    EmmaDatasetItem,
    EmmaVisualFeatures,
)
from emma_policy.inference.api.simbot_state import GenerateRequest


logger = logging.getLogger(__name__)


class SimBotActionInputBuilder:
    """Build the input for the SimBotAction model."""

    def __init__(self, tokenizer: PreTrainedTokenizer, device: str = "cpu") -> None:
        self._tokenizer = tokenizer
        self._device = device
        self._action_execution_prompt = "Act according to the instruction: {instruction}"
        self._question_answer_prompt = "With question: {question} and answer: {answer}"

    def __call__(
        self, request: GenerateRequest
    ) -> tuple[Optional[EmmaDatasetBatch], Optional[torch.Tensor]]:
        """Process the environment output into a batch for the model.

        The sample batch provides the set of previous observations and previous actions taken by
        the agent in the environment.
        """
        (instruction, question, answer) = self._parse_dialogue_from_request(request)
        (feature_dicts, previous_actions) = self._parse_environment_history_from_request(request)

        batch: Optional[EmmaDatasetBatch] = None
        decoder_input_ids: Optional[torch.Tensor] = None
        if instruction is not None:
            logger.debug(f"Predicting action for instruction: {instruction}")

            encoded_inputs = self._prepare_input_text(
                instruction=instruction, question=question, answer=answer
            )
            visual_features = prepare_emma_visual_features(
                feature_dicts=feature_dicts,
                tokenizer=self._tokenizer,
            )
            dataset_item = self._create_emma_dataset_item(
                visual_features=visual_features, encoded_inputs=encoded_inputs
            )
            decoder_input_ids = self._prepare_decoder_input_ids(previous_actions=previous_actions)

            batch = self._create_emma_dataset_batch(dataset_item)
        return (batch, decoder_input_ids)

    def _prepare_decoder_input_ids(
        self, previous_actions: Optional[str] = None
    ) -> Optional[torch.Tensor]:
        """Prepare decoder input ids."""
        if previous_actions is not None:
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

    def _parse_dialogue_from_request(  # noqa: WPS231
        self,
        request: GenerateRequest,
    ) -> tuple[Optional[str], Optional[str], Optional[str]]:
        """Parse the dialogue for the current request."""
        instruction = None
        question = None
        answer = None
        instruction_pos = None
        dialogue_history = request.dialogue_history
        for idx, utterance in enumerate(dialogue_history[::-1], 1):
            if utterance.role == "user" and utterance.intent == "instruction":
                instruction = utterance.utterance
                instruction_pos = idx
                break

        if instruction_pos is None:
            logger.debug(f"No instruction for request: {request}")

        else:
            for utterance in dialogue_history[-instruction_pos:]:  # noqa: WPS440
                if utterance.role == "agent" and utterance.intent == "clarify_question":
                    question = utterance.utterance

                if utterance.role == "user" and utterance.intent == "clarify_answer":
                    answer = utterance.utterance

            logger.debug(
                f"Found instruction: {instruction}, question: {question} answer: {answer}"
            )
        return (instruction, question, answer)

    def _parse_environment_history_from_request(
        self, request: GenerateRequest
    ) -> tuple[list[dict[str, Any]], Optional[str]]:
        """Parse the feature dicts and actions from the current request."""
        feature_dicts = []
        previous_actions = []
        for step in request.environment_history:
            if step.output is None:
                msg = "Found unexpected 'None' as a previous action. Verify that the received request contains string values for previous actions."
                logger.debug(msg)

            # TODO: fix this when porting the causal mask
            for features in step.features:
                feature_dict: dict[str, Union[torch.Tensor, int]] = {
                    "bbox_features": torch.tensor(features["bbox_features"]),
                    "bbox_probas": torch.tensor(features["bbox_probas"]),
                    "bbox_coords": torch.tensor(features["bbox_coords"]),
                    "cnn_features": torch.tensor(features["cnn_features"]),
                    "width": features["width"],
                    "height": features["height"],
                }
                feature_dicts.append(feature_dict)
            previous_actions.append(step.output)

        # concatenate all the previous actions to a string
        # ignore the last element as it is None (the current action has not been predicted yet)
        previous_actions_str = None
        if previous_actions:
            # Currently the implementation allows None previous actios
            # but in practice this should never happen.
            previous_actions_str = " ".join(previous_actions[:-1])  # type: ignore[arg-type]
        return (feature_dicts, previous_actions_str)

    def _prepare_input_text(
        self,
        instruction: str,
        question: Optional[str] = None,
        answer: Optional[str] = None,
    ) -> BatchEncoding:
        """Prepare the input text for the SimBotAction model.

        The input text follows the same template as with the action execution with the addition of
        the clarification question and answer (if provided).
        """
        if not instruction.endswith("."):
            instruction = f"{instruction}."
        source_text = self._action_execution_prompt.format(instruction=instruction)
        if question is not None and answer is not None:
            source_question_answer = self._question_answer_prompt.format(
                question=question, answer=answer
            )
            if not source_question_answer.endswith("."):
                source_question_answer = f"{source_question_answer}."
            source_text = f"{source_text} {source_question_answer}"

        return self._tokenizer.encode_plus(source_text, return_tensors="pt", truncation=True)

    def _create_emma_dataset_item(
        self,
        visual_features: EmmaVisualFeatures,
        encoded_inputs: BatchEncoding,
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
        )

    def _create_emma_dataset_batch(
        self,
        dataset_item: EmmaDatasetItem,
    ) -> EmmaDatasetBatch:
        """Create the `EmmaDatasetBatch` for a given set of observations and actions."""
        batch = collate_fn([dataset_item])

        return move_data_to_device(batch, self._device)
