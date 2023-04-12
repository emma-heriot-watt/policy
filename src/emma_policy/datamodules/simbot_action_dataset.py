import random
from pathlib import Path
from typing import Optional

import torch
from emma_datasets.constants.simbot.simbot import get_arena_definitions
from emma_datasets.datamodels.datasets.utils.simbot_utils.instruction_processing import (
    get_object_label_from_object_id,
    get_object_readable_name_from_object_id,
)
from emma_datasets.datamodels.datasets.utils.simbot_utils.paraphrasers import (
    InstructionParaphraser,
)
from emma_datasets.datamodels.datasets.utils.simbot_utils.simbot_datamodels import (
    SimBotAction,
    SimBotInstructionInstance,
    SimBotObjectAttributes,
)
from overrides import overrides
from torchvision.ops import masks_to_boxes
from transformers import PreTrainedTokenizer

from emma_policy.datamodules.base_dataset import EmmaBaseDataset
from emma_policy.datamodules.emma_dataclasses import (
    EmmaDatasetItem,
    EmmaDatasetPadding,
    EmmaVisualFeatures,
)
from emma_policy.datamodules.pretrain_instances import Task
from emma_policy.utils import decompress_simbot_mask, get_logger
from emma_policy.utils.datamodels.simbot import (
    SearchNegativeSampler,
    compressed_mask_is_bbox,
    format_instruction,
    get_object_for_search,
    get_simbot_instruction_paraphrase,
    mask_past_target_actions,
)


logger = get_logger(__name__)


class SimBotActionDataset(EmmaBaseDataset[EmmaDatasetItem]):
    """Dataset for SimBotAction.

    Each instance is loaded from the DatasetDb file and converted to an instance of
    `EmmaDatasetItem` before being returned.
    """

    def __init__(
        self,
        dataset_db_path: Path,
        tokenizer: PreTrainedTokenizer,
        iou_threshold: float = 0.5,
        search_negative_proba: float = 0.5,
        max_frames: int = 15,
        use_only_necessary_questions: bool = True,
        allow_paraphrasing: bool = False,
        shuffle_objects: bool = False,
    ) -> None:

        super().__init__(
            dataset_db_path=dataset_db_path,
            tokenizer=tokenizer,
            max_frames=max_frames,
            shuffle_objects=shuffle_objects,
        )

        self._iou_threshold = iou_threshold
        self._goto_proba = 0
        self._search_negative_proba = search_negative_proba
        self._use_only_necessary_questions = use_only_necessary_questions
        self.question_answer_prompt = "<<driver>> {question} <<commander>> {answer}"

        arena_definitions = get_arena_definitions()
        self._object_assets_to_names = arena_definitions["asset_to_label"]
        self._image_width = arena_definitions["image_width"]
        self._image_height = arena_definitions["image_height"]
        self._allow_paraphrasing = allow_paraphrasing
        self._search_negative_sampler = SearchNegativeSampler(self.db)
        self._special_name_cases = arena_definitions["special_asset_to_readable_name"]
        self.paraphraser = InstructionParaphraser()
        self._search_positive_type = "search_positive"
        self._search_negative_type = "search_negative"

    @overrides(check_signature=False)
    def __getitem__(self, index: int) -> EmmaDatasetItem:
        """Get the `EmmaDatasetItem` for a SimBotInstructionInstance."""
        instance_str: str
        with self.db:
            instance_str = self.db[index]
        instance = SimBotInstructionInstance.parse_raw(instance_str)
        if instance.vision_augmentation:
            return self.simbot_vision_augmentation(instance)
        return self.simbot_action_execution(instance)

    def simbot_vision_augmentation(  # noqa: WPS210, WPS231
        self, instance: SimBotInstructionInstance
    ) -> EmmaDatasetItem:
        """Process a visual augmentation instance for the SimBot action task."""
        if instance.actions[-1].type == "Search":
            action_object_metadata = instance.actions[-1].get_action_data["object"]
            object_id, object_index, attributes = get_object_for_search(
                instance.actions[-1].search, action_object_metadata, instance.paraphrasable
            )

            # All the instances comming from augmentations are paraphrasable
            # The ones comming from annotations are not.
            if self._allow_paraphrasing and instance.paraphrasable:
                instruction = self.paraphraser(
                    action_type=instance.actions[-1].type.lower(),
                    object_id=object_id,
                    object_attributes=SimBotObjectAttributes(**attributes),
                )
            else:
                instruction = instance.instruction.instruction

            source_text = f"<<commander>> {instruction}"
            source_text = self._get_random_template_for_task(Task.visual_grounding).format(
                caption=source_text
            )
            source_text = format_instruction(source_text)

            object_name = get_object_readable_name_from_object_id(
                object_id=object_id,
                object_assets_to_names=self._object_assets_to_names,
                special_name_cases=self._special_name_cases,
            )

            object_token = None

            # We need to skip the instances that are from annotations aka paraphrasable
            # TODO: we need to make this easier
            select_negative = (
                random.random() >= self._search_negative_proba and instance.paraphrasable
            )
            if select_negative:
                negative_idx = self._search_negative_sampler(object_name)

                instance_str = self.db[negative_idx]
                negative_instance = SimBotInstructionInstance.parse_raw(instance_str)
                visual_features, _, _ = self._load_visual_features(
                    features_path=negative_instance.features_path,
                    target_frames=[0 for _ in negative_instance.actions],
                )
            else:
                visual_features, _, _ = self._load_visual_features(
                    features_path=instance.features_path,
                    target_frames=[0 for _ in instance.actions],
                )

            ground_truth_bboxes = action_object_metadata["mask"]
            if ground_truth_bboxes is None or select_negative:
                target_text = f"no {object_name} <stop>."
                action_type = self._search_negative_type
            else:
                ground_truth_bbox = ground_truth_bboxes[object_index]
                ground_truth_bbox = torch.tensor(
                    ground_truth_bbox,
                    dtype=torch.float32,
                ).unsqueeze(0)

                ground_truth_bbox[:, (0, 2)] /= self._image_width
                ground_truth_bbox[:, (1, 3)] /= self._image_height

                matched_indices, ground_truth_flags = self._best_match_features(
                    ground_truth_bbox=ground_truth_bbox,
                    object_coordinates_bbox=visual_features.object_coordinates,
                    threshold=self._iou_threshold,
                )
                # If there is a matching bounding box, append its visual token to the target text
                if ground_truth_flags[0]:
                    object_token = self.tokenizer.decode(
                        visual_features.visual_token_ids[matched_indices[0]]
                    )
                    scene_frame_token = self.tokenizer.decode(
                        visual_features.scene_frame_tokens[0]
                    )
                    target_text = f"{scene_frame_token} {object_token} <stop>."
                    action_type = self._search_positive_type
                else:
                    target_text = f"no {object_name} <stop>."
                    action_type = self._search_negative_type

            target_text = target_text.lower()

            input_encoding = self.tokenizer.encode_plus(
                source_text, return_tensors=self._return_tensor_type, truncation=True
            )

            target_encoding = self.tokenizer.encode_plus(
                target_text, return_tensors=self._return_tensor_type, truncation=True
            )

            full_target_token_ids = target_encoding.input_ids.squeeze(0)

            target_token_ids = mask_past_target_actions(
                full_target_token_ids,
                sep_token_id=self.tokenizer.sep_token_id,  # type: ignore[arg-type]
                masking_value=EmmaDatasetPadding.target_token_ids,
            )
            decoder_input_ids = torch.full_like(
                full_target_token_ids,
                fill_value=self.tokenizer.eos_token_id,  # type: ignore[arg-type]
            )
            # Now shift them to the right
            decoder_input_ids[1:] = full_target_token_ids[:-1].clone()  # noqa: WPS362
            decoder_attention_mask = torch.ones_like(decoder_input_ids)
            raw_target = {
                "instance_id": self._get_instance_id(instance),
                "instruction": source_text,
                "target": target_text,
                "action_type": action_type,
                "object_type": object_name,
            }

            return EmmaDatasetItem(
                input_token_ids=input_encoding.input_ids.squeeze(0),
                text_attention_mask=input_encoding.attention_mask.squeeze(0),
                target_token_ids=target_token_ids,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                object_attention_mask=visual_features.object_attention_mask,
                object_coordinates=visual_features.object_coordinates,
                object_features=visual_features.object_features,
                object_frame_tokens=visual_features.object_frame_tokens,
                scene_attention_mask=visual_features.scene_attention_mask,
                scene_coordinates=visual_features.scene_coordinates,
                scene_features=visual_features.scene_features,
                scene_frame_tokens=visual_features.scene_frame_tokens,
                visual_token_ids=visual_features.visual_token_ids,
                task=self._get_task_as_tensor(Task.visual_grounding),
                raw_target=raw_target,
            )
        # All other augmentations are covered by the action execution
        return self.simbot_action_execution(instance)

    def simbot_action_execution(self, instance: SimBotInstructionInstance) -> EmmaDatasetItem:
        """Process the instance for the SimBot action task."""
        if instance.instruction is None:
            raise AssertionError("Instructions for this instance must exist.")
        actions_ids_in_instruction = instance.instruction.actions
        action_ids_in_instance = [action.id for action in instance.actions]
        if action_ids_in_instance != actions_ids_in_instruction:
            raise AssertionError(
                f"Instructions have {instance.instruction.actions} but found {action_ids_in_instance}."
            )

        target_frames = self.get_target_frames(instance)

        visual_features, frames, objects_per_frame = self._load_visual_features(
            features_path=instance.features_path,
            target_frames=target_frames,
        )

        source_text = self._prepare_source_text(instance=instance)

        target_text = self._prepare_target_text(
            instance=instance,
            visual_features=visual_features,
            frames=frames,
            objects_per_frame=objects_per_frame,
        )

        input_encoding = self.tokenizer.encode_plus(
            source_text, return_tensors=self._return_tensor_type, truncation=True
        )

        target_encoding = self.tokenizer.encode_plus(
            target_text, return_tensors=self._return_tensor_type, truncation=True
        )

        full_target_token_ids = target_encoding.input_ids.squeeze(0)

        target_token_ids = mask_past_target_actions(
            full_target_token_ids,
            sep_token_id=self.tokenizer.sep_token_id,  # type: ignore[arg-type]
            masking_value=EmmaDatasetPadding.target_token_ids,
        )
        decoder_input_ids = torch.full_like(
            full_target_token_ids,
            fill_value=self.tokenizer.eos_token_id,  # type: ignore[arg-type]
        )
        # Now shift them to the right
        decoder_input_ids[1:] = full_target_token_ids[:-1].clone()  # noqa: WPS362
        decoder_attention_mask = torch.ones_like(decoder_input_ids)
        raw_target = {
            "instance_id": self._get_instance_id(instance),
            "instruction": source_text,
            "target": target_text,
            "action_type": instance.actions[-1].type,
            "object_type": self._get_target_object(instance.actions[-1]),
        }

        return EmmaDatasetItem(
            input_token_ids=input_encoding.input_ids.squeeze(0),
            text_attention_mask=input_encoding.attention_mask.squeeze(0),
            target_token_ids=target_token_ids,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            object_attention_mask=visual_features.object_attention_mask,
            object_coordinates=visual_features.object_coordinates,
            object_features=visual_features.object_features,
            object_frame_tokens=visual_features.object_frame_tokens,
            scene_attention_mask=visual_features.scene_attention_mask,
            scene_coordinates=visual_features.scene_coordinates,
            scene_features=visual_features.scene_features,
            scene_frame_tokens=visual_features.scene_frame_tokens,
            visual_token_ids=visual_features.visual_token_ids,
            task=self._get_task_as_tensor(Task.action_execution),
            raw_target=raw_target,
        )

    def get_target_frames(self, instance: SimBotInstructionInstance) -> list[int]:
        """Get the frame indices of the target frames for the actions."""
        target_frames = []
        for action in instance.actions:
            action_object_metadata = action.get_action_data.get("object", None)
            if action_object_metadata is not None:
                target_frames.append(action_object_metadata.get("colorImageIndex", 0))
            else:
                target_frames.append(0)
        return target_frames

    def map_object_to_visual_token(
        self,
        object_name: str,
        action: SimBotAction,
        image_index: int,
        visual_features: EmmaVisualFeatures,
        frames: list[str],
        objects_per_frame: list[int],
    ) -> str:
        """Map the name of an object with its frame and visual token.

        If the object cannot be mapped return the object name alone.
        """
        # Get the index of the color image
        color_image = action.color_images[image_index]
        frame_index = frames.index(color_image)
        # Find the offset position of the frame objects within the visual features
        offset_idx = sum(objects_per_frame[:frame_index])
        object_coordinates_bbox = visual_features.object_coordinates[
            offset_idx : offset_idx + objects_per_frame[frame_index], :
        ]

        gt_object_dict = action.get_action_data
        # If the groundtruth object is a sticky note, the groundtruth bbox
        # coordinates are currently provided directly in the mask
        # TODO: this could potentially be improved if we have the segmentation masks for the sticky notes as well instead of the bounding boxes
        object_mask = gt_object_dict["object"]["mask"]
        if object_name == "Sticky Note":
            ground_truth_bbox = torch.tensor(object_mask[0]).float()

        else:
            if compressed_mask_is_bbox(object_mask):
                ground_truth_bbox = torch.tensor(object_mask, dtype=torch.float32).unsqueeze(0)
            else:
                gt_binary_mask = decompress_simbot_mask(object_mask)
                ground_truth_bbox = masks_to_boxes(torch.tensor(gt_binary_mask).unsqueeze(0))

        ground_truth_bbox[:, (0, 2)] /= self._image_width
        ground_truth_bbox[:, (1, 3)] /= self._image_height

        matched_indices, ground_truth_flags = self._best_match_features(
            ground_truth_bbox=ground_truth_bbox,
            object_coordinates_bbox=object_coordinates_bbox,
            threshold=self._iou_threshold,
        )
        scene_frame_token = self.tokenizer.decode(visual_features.scene_frame_tokens[frame_index])

        # If there is a matching bounding box, append its visual token to the target text
        if ground_truth_flags[0]:
            # Get the visual token ids for the corresponding frame
            vis_tokens = visual_features.visual_token_ids[
                offset_idx : offset_idx + objects_per_frame[frame_index]
            ]
            object_token = self.tokenizer.decode(vis_tokens[matched_indices[0]])
            return f"{object_name} {scene_frame_token} {object_token}"
        return f"{object_name} {scene_frame_token}"

    def _prepare_source_text(self, instance: SimBotInstructionInstance) -> str:
        """Prepare the source text.

        The source text allows the same template as the action execution task
        with the addition of the question and answer pair.

        Example:
            source_text = Execute the instruction: go to the desk with a hammer on it. With question
            and answer: where is the hammer? the hammer is on the table in the robotics lab.
        """
        if self._allow_paraphrasing and instance.paraphrasable:
            action_object_metadata = instance.actions[-1].get_action_data["object"]
            object_name = get_object_label_from_object_id(
                object_id=action_object_metadata["id"],
                object_assets_to_names=self._object_assets_to_names,
            )

            instruction = get_simbot_instruction_paraphrase(
                self.paraphraser, instance, object_name
            )
        else:
            instruction = instance.instruction.instruction

        source_text = f"<<commander>> {instruction}"
        source_text = self._get_random_template_for_task(Task.action_execution).format(
            instruction=source_text
        )
        if instance.instruction.question_answers is not None:
            question_answer_candidates = instance.instruction.question_answers
            if self._use_only_necessary_questions:
                question_answer_candidates = [
                    qa for qa in instance.instruction.question_answers if qa.question_necessary
                ]
            if question_answer_candidates:
                question_answer = random.choice(question_answer_candidates)
                question_answer_text = self.question_answer_prompt.format(
                    question=question_answer.question.lower(), answer=question_answer.answer
                )
                source_text = f"{source_text}. {question_answer_text}"

        source_text = format_instruction(source_text)

        return source_text

    def _prepare_target_text(  # noqa: WPS231
        self,
        instance: SimBotInstructionInstance,
        visual_features: EmmaVisualFeatures,
        frames: list[str],
        objects_per_frame: list[int],
    ) -> str:
        """Prepare the target text.

        The target text if the sequence of actions for the instruction instance separated with '.'.

        Example:
            target_text = Look Around. Goto Desk.
        """
        target_text = []
        for action in instance.actions:
            action_type = action.type
            action_metadata = action.get_action_data
            action_object_metadata = action_metadata.get("object", None)
            # case 1: navigation actions except GoTo
            if action_type in {"Look", "Move", "Rotate", "Turn"}:
                target_text.append(f"{action_type} {action_metadata['direction']}.")
            # case 2: room/object navigation or interaction action
            elif action_object_metadata is not None:
                object_id = action_object_metadata.get("id", None)
                # action with a specific object
                if object_id is not None:
                    object_name = get_object_label_from_object_id(
                        object_id=action_object_metadata["id"],
                        object_assets_to_names=self._object_assets_to_names,
                    )
                    image_index = action_object_metadata["colorImageIndex"]
                    object_name_with_tokens = self.map_object_to_visual_token(
                        object_name=object_name,
                        action=action,
                        image_index=image_index,
                        visual_features=visual_features,
                        frames=frames,
                        objects_per_frame=objects_per_frame,
                    )

                    target_text.append(f"{action_type} {object_name_with_tokens}.")
                # action without an object (e.g, Goto Office)
                else:
                    # {'object': {'officeRoom': 'Lab1'}}
                    object_name = list(action_object_metadata.values())[0]
                    target_text.append(f"{action_type} {object_name}.")
            # no other action types, sanity check.
            else:
                raise AssertionError(f"Unsupported action {action}.")

        if instance.actions[-1].final:
            target_text[-1] = f"{target_text[-1][:-1]} <stop>."
        return " ".join(target_text).lower()

    def _ignore_goto_redundant_frames(
        self,
        action: SimBotAction,
        visual_features: EmmaVisualFeatures,
        frames: list[str],
        objects_per_frame: list[int],
    ) -> tuple[EmmaVisualFeatures, list[str], list[int]]:
        """Remove the additional input frames from the goto synthetic instructions."""
        # remove the irrelevant frames from the visual features
        # find the number of objects for the golden frame and keep only
        # the relevant object features, classes, coordinates, visual tokens, and masks
        # We only need the scene features, coordinates for a single scene.

        frame_index = action.goto["object"]["colorImageIndex"]
        offset_idx = sum(objects_per_frame[:frame_index])

        frame_token = self.tokenizer.convert_tokens_to_ids("<frame_token_1>")
        object_frame_tokens = (
            torch.ones((objects_per_frame[frame_index]), dtype=torch.long) * frame_token
        )

        truncated_visual_features = EmmaVisualFeatures(
            scene_features=visual_features.scene_features[frame_index, :].unsqueeze(0),
            scene_coordinates=torch.tensor(
                [[0, 0, 1, 1]], dtype=visual_features.scene_coordinates.dtype
            ),
            object_classes=visual_features.object_classes[
                offset_idx : offset_idx + objects_per_frame[frame_index]
            ],
            object_features=visual_features.object_features[
                offset_idx : offset_idx + objects_per_frame[frame_index], :
            ],
            object_coordinates=visual_features.object_coordinates[
                offset_idx : offset_idx + objects_per_frame[frame_index], :
            ],
            visual_token_ids=visual_features.visual_token_ids[
                offset_idx : offset_idx + objects_per_frame[frame_index]
            ],
            scene_frame_tokens=torch.tensor(
                [frame_token], dtype=visual_features.scene_frame_tokens.dtype
            ),
            object_frame_tokens=object_frame_tokens,
            scene_attention_mask=torch.tensor([True]),
            object_attention_mask=visual_features.object_attention_mask[
                offset_idx : offset_idx + objects_per_frame[frame_index]
            ],
            original_frame_order=torch.tensor([0]),
        )

        return truncated_visual_features, [frames[frame_index]], [objects_per_frame[frame_index]]

    @overrides(check_signature=False)
    def _load_visual_features(
        self,
        features_path: list[Path],
        target_frames: list[int],
    ) -> tuple[EmmaVisualFeatures, list[str], list[int]]:
        """Get all the visual features from the given instance."""
        feature_dicts = []
        objects_per_frame = []
        frames = []
        for fpath, target_frame in zip(features_path, target_frames):
            frame_dict = torch.load(fpath)["frames"][target_frame]
            feature_dicts.append(frame_dict["features"])
            objects_per_frame.append(frame_dict["features"]["bbox_coords"].shape[0])
            frames.append(frame_dict["image"])

        visual_features = self._prepare_emma_visual_features(feature_dicts=feature_dicts)

        return visual_features, frames, objects_per_frame

    def _get_instance_id(self, instance: SimBotInstructionInstance) -> str:
        """Construct the instance id."""
        instruction_id = f"mission{instance.mission_id}_instr{instance.instruction_id}"
        return f"{instruction_id}_ann{instance.annotation_id}_action{instance.actions[-1].type}"

    def _get_target_object(self, action: SimBotAction) -> Optional[str]:
        """Prepare the object name."""
        action_type = action.type
        # case 1: navigation actions except GoTo
        if action_type in {"Look", "Move", "Rotate", "Turn"}:
            return None

        action_object_metadata = action.get_action_data["object"]
        # case 2: room/object navigation or interaction action
        object_id = action_object_metadata.get("id", None)
        # action with a specific object
        if object_id is not None:
            object_name = get_object_readable_name_from_object_id(
                object_id=action_object_metadata["id"],
                object_assets_to_names=self._object_assets_to_names,
                special_name_cases=self._special_name_cases,
            )
        # action without an object (e.g, Goto Office)
        else:
            # {'object': {'officeRoom': 'Lab1'}}
            object_name = list(action_object_metadata.values())[0]

        return object_name
