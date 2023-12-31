import random
from enum import Enum
from pathlib import Path
from typing import Literal, Optional

import torch
from emma_datasets.constants.simbot.simbot import get_arena_definitions
from emma_datasets.datamodels.datasets.utils.simbot_utils.high_level_key_processor import (
    HighLevelKeyProcessor,
)
from emma_datasets.datamodels.datasets.utils.simbot_utils.instruction_processing import (
    get_object_label_from_object_id,
    get_object_readable_name_from_object_id,
)
from emma_datasets.datamodels.datasets.utils.simbot_utils.paraphrasers import (
    InstructionParaphraser,
)
from emma_datasets.datamodels.datasets.utils.simbot_utils.simbot_datamodels import (
    SimBotAction,
    SimBotClarificationTypes,
    SimBotInstructionInstance,
    SimBotObjectAttributes,
)
from overrides import overrides
from transformers import PreTrainedTokenizer

from emma_policy.datamodules.base_dataset import EmmaBaseDataset
from emma_policy.datamodules.emma_dataclasses import EmmaDatasetItem, EmmaVisualFeatures
from emma_policy.datamodules.pretrain_instances import Task
from emma_policy.utils import get_logger
from emma_policy.utils.datamodels.simbot import (
    SearchNegativeSampler,
    add_inventory_to_instruction,
    format_instruction,
    get_object_for_search,
    get_simbot_instruction_paraphrase,
)


logger = get_logger(__name__)


class SimBotCRIntents(Enum):
    """SimBot CR intent types."""

    act = "<act>"
    search = "<search>"
    match = "<one_match>"
    no_match = "<no_match>"
    too_many_matches = "<too_many_matches>"
    missing_inventory = "<missing_inventory>"

    act_one_match = "<act><one_match>"
    act_no_match = "<act><no_match>"
    act_too_many_matches = "<act><too_many_matches>"
    act_missing_inventory = "<act><missing_inventory>"

    search_one_match = "<search><one_match>"
    search_no_match = "<search><no_match>"
    search_too_many_matches = "<search><too_many_matches>"

    @property
    def is_special_token(self) -> bool:
        """The name of the intent corresponds to a special token."""
        return self in {
            self.act,
            self.search,
            self.match,
            self.no_match,
            self.too_many_matches,
            self.missing_inventory,
        }

    @property
    def is_cr_output(self) -> bool:
        """Wether an intent is a valid output."""
        return self in {
            self.act_one_match,
            self.act_no_match,
            self.act_too_many_matches,
            self.act_missing_inventory,
            self.search_one_match,
            self.search_no_match,
        }


def action_is_object_interaction(action: SimBotAction) -> bool:
    """Check if an instruction is an object interaction.

    1. Not a Search or Examine action
    2. Has object metadata - but not room
    """
    if action.type in {"Search", "Examine", "Look"}:
        return False
    object_metadata = action.get_action_data.get("object", None)
    if object_metadata is None:
        return False
    return "officeRoom" not in object_metadata


class SimBotCRDataset(EmmaBaseDataset[EmmaDatasetItem]):
    """Dataset for AreanCR.

    Each instance is loaded from the DatasetDb file and converted to an instance of
    `EmmaDatasetItem` before being returned.
    """

    def __init__(
        self,
        dataset_db_path: Path,
        tokenizer: PreTrainedTokenizer,
        max_frames: int = 4,
        is_train: bool = True,
        iou_threshold: float = 0.5,
        search_negative_proba: float = 0.5,
        _act_one_match_proba: float = 0.6,
        _one_match_to_missining_inventory_proba: float = 0.35,
        _no_match_to_missining_inventory_proba: float = 0.2,
        shuffle_objects: bool = False,
    ) -> None:
        super().__init__(
            dataset_db_path=dataset_db_path,
            tokenizer=tokenizer,
            max_frames=max_frames,
            shuffle_objects=shuffle_objects,
        )

        self.is_train = is_train
        self.data_intents: list[SimBotCRIntents] = []
        self._synthetic_negative_candidates: list[int] = []
        self._question_type_intent_map = {
            SimBotClarificationTypes.location: SimBotCRIntents.act_no_match,
            SimBotClarificationTypes.disambiguation: SimBotCRIntents.act_too_many_matches,
        }
        self._prepare_data()
        arena_definitions = get_arena_definitions()
        self._object_assets_to_names = arena_definitions["asset_to_label"]
        self._label_to_idx: dict[str, int] = arena_definitions["label_to_idx"]
        self._special_name_cases = arena_definitions["special_asset_to_readable_name"]
        self._image_width = arena_definitions["image_width"]
        self._image_height = arena_definitions["image_height"]
        self._iou_threshold = iou_threshold
        self._search_negative_proba = search_negative_proba
        self._act_one_match_proba = _act_one_match_proba
        self._one_match_to_missining_inventory_proba = _one_match_to_missining_inventory_proba
        self._no_match_to_missining_inventory_proba = _no_match_to_missining_inventory_proba
        self._search_negative_sampler = SearchNegativeSampler(self.db)
        self.low_level_paraphraser = InstructionParaphraser()
        self.high_level_paraphraser = HighLevelKeyProcessor()

    @overrides(check_signature=False)
    def __len__(self) -> int:
        """Return the total number of instances within the database."""
        return len(self.db)

    @overrides(check_signature=False)
    def __getitem__(self, index: int) -> EmmaDatasetItem:
        """Get the SimBot CR instance at the given index as an instance of `EmmaDatasetItem`."""
        with self.db:
            instance_str = self.db[index]
        instance = SimBotInstructionInstance.parse_raw(instance_str)
        frame_idx = 0
        if instance.actions[0].type == "Search":  # noqa: WPS204
            instruction, visual_features, target_text = self.prepare_search_instance(instance)
        else:
            instruction, visual_features, target_text = self.prepare_action_instance(instance)

        source_text = f"Predict the system act: {format_instruction(instruction)}"
        if "inventory:" not in source_text:
            raise AssertionError(f"{source_text}")
        target_text = target_text.lower()
        input_encoding = self.tokenizer.encode_plus(
            source_text, return_tensors=self._return_tensor_type, truncation=True
        )
        target_encoding = self.tokenizer.encode_plus(
            target_text, return_tensors=self._return_tensor_type, truncation=True
        )

        target_token_ids = target_encoding.input_ids.squeeze(0)
        # Prepare the decoder input ids
        # We have to do this for multitask training
        decoder_input_ids = torch.full_like(
            target_token_ids,
            fill_value=self.tokenizer.eos_token_id,  # type: ignore[arg-type]
        )
        # Now shift them to the right
        decoder_input_ids[1:] = target_token_ids[:-1].clone()  # noqa: WPS362

        raw_target = {
            "instance_id": self._get_instance_id(instance),
            "instruction": source_text,
            "target": target_text,
            "action_type": instance.actions[0].type,
            "object_type": " ".join(target_text.split()[1:]),
            "frame_idx": frame_idx,
            "features_path": instance.features_path[0],
            "color_images": instance.actions[0].color_images[frame_idx],
            "task": "vad",
        }

        return EmmaDatasetItem(
            input_token_ids=input_encoding.input_ids.squeeze(0),
            text_attention_mask=input_encoding.attention_mask.squeeze(0),
            target_token_ids=target_token_ids,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=target_encoding.attention_mask.squeeze(0),
            object_attention_mask=visual_features.object_attention_mask,
            object_coordinates=visual_features.object_coordinates,
            object_features=visual_features.object_features,
            object_frame_tokens=visual_features.object_frame_tokens,
            scene_attention_mask=visual_features.scene_attention_mask,
            scene_coordinates=visual_features.scene_coordinates,
            scene_features=visual_features.scene_features,
            scene_frame_tokens=visual_features.scene_frame_tokens,
            visual_token_ids=visual_features.visual_token_ids,
            task=self._get_task_as_tensor(Task.vad),
            raw_target=raw_target,
        )

    def prepare_action_instance(
        self, instance: SimBotInstructionInstance
    ) -> tuple[str, EmmaVisualFeatures, str]:
        """Prepare the source and target text for an action instance."""
        if instance.cdf_augmentation:
            instruction, visual_features, target_text = self.prepare_cdf_action_instance(instance)
        else:
            if instance.synthetic:
                instruction, target_text = self.prepare_synthetic_action_instance(instance)
            else:
                instruction, target_text = self.prepare_human_action_instance(instance)
            frame_idx = self._get_instance_frame(instance, target_text.lower())
            visual_features = self._load_visual_features(
                features_path=instance.features_path[0], frame_idx=frame_idx
            )

        return instruction, visual_features, target_text

    def prepare_cdf_action_instance(
        self, instance: SimBotInstructionInstance
    ) -> tuple[str, EmmaVisualFeatures, str]:
        """Prepare the instruction and target text for a human instruction."""
        if random.random() < self._act_one_match_proba:
            high_level_key = self.high_level_paraphraser(instance.cdf_highlevel_key)
            instruction = random.choice(high_level_key.paraphrases)

            holding_object = instance.actions[0].inventory_object_id
            missing_holding_object = (
                holding_object is not None
                and random.random() < self._one_match_to_missining_inventory_proba
            )
            if missing_holding_object:
                instance.actions[0].inventory_object_id = None
                target_text = SimBotCRIntents.act_missing_inventory.value

                object_readable_name = get_object_readable_name_from_object_id(
                    object_id=holding_object,
                    object_assets_to_names=self._object_assets_to_names,
                    special_name_cases=self._special_name_cases,
                )

            else:
                target_text = SimBotCRIntents.act_one_match.value
                object_readable_name = self._get_target_object_name(
                    action=instance.actions[0],
                    name_type="readable",
                )

            frame_idx = self._get_instance_frame(instance, target_text.lower())
            visual_features = self._load_visual_features(
                features_path=instance.features_path[0], frame_idx=frame_idx
            )
            instruction = add_inventory_to_instruction(
                object_assets_to_names=self._object_assets_to_names,
                special_name_cases=self._special_name_cases,
                instruction=instruction,
                inventory_object_id=instance.actions[0].inventory_object_id,
            )

            if object_readable_name:
                target_text = f"{target_text} {object_readable_name}"

        else:
            instruction, visual_features, target_text = self._sample_cdf_no_match(instance)

        return instruction, visual_features, target_text

    def prepare_human_action_instance(
        self, instance: SimBotInstructionInstance
    ) -> tuple[str, str]:
        """Prepare the instruction and target text for a human instruction."""
        instruction = instance.instruction.instruction
        instruction = add_inventory_to_instruction(
            object_assets_to_names=self._object_assets_to_names,
            special_name_cases=self._special_name_cases,
            instruction=instruction,
            inventory_object_id=instance.actions[0].inventory_object_id,
        )
        # First try to get the target for a clarification
        target_text = self._get_cr_human_question(instance)
        # If target_text is an empty list, we have an action
        if not target_text:
            target_text = SimBotCRIntents.act_one_match.value
            object_readable_name = self._get_target_object_name(
                action=instance.actions[0],
                name_type="readable",
            )
            if object_readable_name:
                target_text = f"{target_text} {object_readable_name}"

        return instruction, target_text

    def prepare_synthetic_action_instance(
        self, instance: SimBotInstructionInstance
    ) -> tuple[str, str]:
        """Prepare the instruction and target text for a synthetic instruction."""
        if instance.ambiguous:
            instance.actions[
                0
            ].inventory_object_id = self.low_level_paraphraser.sample_inventory_object(
                instance.actions[0].type.lower()
            )
            instruction = self._get_synthectic_action_instruction(
                instance, allow_paraphrasing=False
            )
            instruction = add_inventory_to_instruction(
                object_assets_to_names=self._object_assets_to_names,
                special_name_cases=self._special_name_cases,
                instruction=instruction,
                inventory_object_id=instance.actions[0].inventory_object_id,
            )
            target_text = self._get_cr_synthetic_too_many_matches(instance)
        else:
            instruction, target_text = self._augment_synthetic_action(instance)
        return instruction, target_text

    def prepare_search_instance(
        self, instance: SimBotInstructionInstance
    ) -> tuple[str, EmmaVisualFeatures, str]:
        """Get source and target text for Search instructions."""
        # Select the object
        action = instance.actions[0]
        action_object_metadata = action.get_action_data["object"]
        object_id, object_index, attributes = get_object_for_search(
            action.search, action_object_metadata, instance.paraphrasable
        )

        # Prepare the visual_features and target
        object_readable_name = get_object_readable_name_from_object_id(
            object_id=object_id,
            object_assets_to_names=self._object_assets_to_names,
            special_name_cases=self._special_name_cases,
        )

        negative_proba = random.random() >= self._search_negative_proba
        # We need to skip the instances that are from annotations aka paraphrasable
        select_negative = negative_proba and instance.paraphrasable
        is_negative = False
        if select_negative:
            negative_idx = self._search_negative_sampler(object_readable_name)
            negative_instance = SimBotInstructionInstance.parse_raw(self.db[negative_idx])
            visual_features = self._load_visual_features(
                features_path=negative_instance.features_path[0]
            )
            target_text = f"{SimBotCRIntents.search_no_match.value} {object_readable_name}"
            is_negative = True
        elif action_object_metadata["mask"] is None:
            # A negative search sample
            visual_features = self._load_visual_features(features_path=instance.features_path[0])
            target_text = f"{SimBotCRIntents.search_no_match.value} {object_readable_name}"
            is_negative = True

        else:
            # A positive search sample
            ground_truth_bbox = torch.tensor(
                action_object_metadata["mask"][object_index],
                dtype=torch.float32,
            ).unsqueeze(0)

            ground_truth_bbox[:, (0, 2)] /= self._image_width
            ground_truth_bbox[:, (1, 3)] /= self._image_height
            visual_features = self._load_visual_features(features_path=instance.features_path[0])

            _, ground_truth_flags = self._best_match_features(
                ground_truth_bbox=ground_truth_bbox,
                object_coordinates_bbox=visual_features.object_coordinates,
                threshold=self._iou_threshold,
            )
            # If there is a matching bounding box, append its visual token to the target text
            if ground_truth_flags.shape[0] == 0:
                target_text = f"{SimBotCRIntents.search_no_match.value} {object_readable_name}"
                is_negative = True
            elif ground_truth_flags.shape[0] == 1:
                target_text = f"{SimBotCRIntents.search_one_match.value} {object_readable_name}"
            else:
                target_text = (
                    f"{SimBotCRIntents.search_too_many_matches.value} {object_readable_name}"
                )

        instruction = self._prepare_search_instruction(
            instance,
            object_id=object_id,
            is_negative=is_negative,
            attributes=attributes,
            inventory_object_id=action.inventory_object_id,
        )

        return instruction, visual_features, target_text

    def _get_instance_id(self, instance: SimBotInstructionInstance) -> str:
        """Construct the instance id."""
        instruction_id = f"mission{instance.mission_id}_instr{instance.instruction_id}"
        return f"{instruction_id}_ann{instance.annotation_id}_action{instance.actions[-1].type}"

    def _prepare_search_instruction(
        self,
        instance: SimBotInstructionInstance,
        object_id: str,
        is_negative: bool,
        attributes: dict[str, str],
        inventory_object_id: Optional[str],
    ) -> str:
        """Prepare the instruction."""
        if instance.paraphrasable:
            if is_negative and "location" in attributes:
                attributes.pop("location")
            inventory_object_id = self.low_level_paraphraser.sample_inventory_object("search")
            instruction = self.low_level_paraphraser(
                action_type="search",
                object_id=object_id,
                object_attributes=SimBotObjectAttributes(**attributes),
                inventory_object_id=inventory_object_id,
            )
        else:
            instruction = instance.instruction.instruction

        instruction = add_inventory_to_instruction(
            object_assets_to_names=self._object_assets_to_names,
            special_name_cases=self._special_name_cases,
            instruction=instruction,
            inventory_object_id=inventory_object_id,
        )
        return instruction

    def _augment_synthetic_action(self, instance: SimBotInstructionInstance) -> tuple[str, str]:
        """Prepare the instruction and target text for a synthetic unambiguous instruction.

        With some probability sample an instruction that doesn't match the detected objects.
        """
        if random.random() < self._act_one_match_proba:
            instruction, target_text = self._augment_synthetic_inventory(
                instance=instance,
                missing_inventory_proba=self._one_match_to_missining_inventory_proba,
                target_text=SimBotCRIntents.act_one_match.value,
            )
        else:
            visual_features = self._load_visual_features(features_path=instance.features_path[0])
            object_labels = visual_features.object_classes
            # Sample a negative candidate
            rand_idx = int(len(self._synthetic_negative_candidates) * random.random())
            new_instruction, new_action = self._synthetic_act_no_match(rand_idx, object_labels)
            while new_instruction is None:
                rand_idx = int(len(self._synthetic_negative_candidates) * random.random())
                new_instruction, new_action = self._synthetic_act_no_match(rand_idx, object_labels)
            instance.actions = [self._create_act_no_match_action(instance.actions[0], new_action)]
            instance.instruction.instruction = new_instruction
            instruction, target_text = self._augment_synthetic_inventory(
                instance=instance,
                missing_inventory_proba=self._no_match_to_missining_inventory_proba,
                target_text=SimBotCRIntents.act_no_match.value,
                include_location_in_attributes=False,
            )

        return instruction, target_text

    def _get_synthectic_action_instruction(
        self,
        instance: SimBotInstructionInstance,
        allow_paraphrasing: bool = False,
        include_location_in_attributes: bool = True,
    ) -> str:
        """Get source text for all actions except Search."""
        if allow_paraphrasing and instance.paraphrasable:
            object_name = self._get_target_object_name(instance.actions[0], name_type="class")
            if object_name:
                instruction = get_simbot_instruction_paraphrase(
                    self.low_level_paraphraser,
                    instance,
                    object_name,
                    include_location=include_location_in_attributes,
                )
            else:
                instruction = instance.instruction.instruction
        else:
            instruction = instance.instruction.instruction

        return instruction

    def _get_cr_human_question(self, instance: SimBotInstructionInstance) -> Optional[str]:
        """Get the target text and question type vector from a human question.

        Examples to avoid:
        1. water: Fill mug with water --> Where is the water?
        2. door: Close the fridge door/ --> Where is the door?
        3. Where is the office / room?
        """
        if len(instance.instruction.necessary_question_answers) > 1:
            raise AssertionError("Expected one question per instance.")
        if not instance.instruction.necessary_question_answers:
            return None
        question_as_target: Optional[str] = None
        qa_pair = instance.instruction.necessary_question_answers[0]
        if qa_pair.question_type not in self._question_type_intent_map:
            return question_as_target
        if qa_pair.question_target in {"water", "door", "office", "room"}:
            return question_as_target
        question_as_target = self._question_type_intent_map[qa_pair.question_type].value
        if qa_pair.question_target:
            question_as_target = f"{question_as_target} {qa_pair.question_target}"

        return question_as_target

    def _get_cr_synthetic_too_many_matches(self, instance: SimBotInstructionInstance) -> str:
        """Get the target text and question type vector from a synthetic question."""
        question_as_target = SimBotCRIntents.act_too_many_matches.value
        object_name = self._get_target_object_name(instance.actions[0], name_type="class")
        if object_name:
            question_as_target = f"{question_as_target} {object_name}"
        return question_as_target

    def _synthetic_act_no_match(
        self, rand_index: int, existing_objects: list[str]
    ) -> tuple[Optional[str], SimBotAction]:
        """Prepare a synthetic <act><no_match> instance."""
        db_index = self._synthetic_negative_candidates[rand_index]
        with self.db:
            new_instance = SimBotInstructionInstance.parse_raw(self.db[db_index])

        # Get the corresponding bounding box label
        label = self._get_target_object_name(new_instance.actions[0], name_type="class")
        # If there are matching bounding boxes in the current image, discard this candidate
        if label is not None and self._label_to_idx[label] in existing_objects:
            return None, new_instance.actions[0]
        return new_instance.instruction.instruction, new_instance.actions[0]

    def _augment_synthetic_inventory(
        self,
        instance: SimBotInstructionInstance,
        missing_inventory_proba: float,
        target_text: str,
        include_location_in_attributes: bool = True,
    ) -> tuple[str, str]:
        """Add an inventory object to the synthetic low-level instance."""
        is_inventory_required = self.low_level_paraphraser.is_inventory_required(
            instance.actions[0].type.lower()
        )
        if instance.paraphrasable:
            instance.actions[
                0
            ].inventory_object_id = self.low_level_paraphraser.sample_inventory_object(
                instance.actions[0].type.lower()
            )
        instruction = self._get_synthectic_action_instruction(
            instance,
            allow_paraphrasing=True,
            include_location_in_attributes=include_location_in_attributes,
        )
        # If the inventory is not required for the action, we should NOT have an act_missing_inventory target
        cond1 = random.random() < missing_inventory_proba
        cond2 = is_inventory_required and instance.paraphrasable
        if cond1 and cond2:
            # Get the object_readable_name from the inventory readable name
            instruction = add_inventory_to_instruction(
                object_assets_to_names=self._object_assets_to_names,
                special_name_cases=self._special_name_cases,
                inventory_object_id=None,
                instruction=instruction,
            )
            target_text = SimBotCRIntents.act_missing_inventory.value
            object_readable_name = get_object_readable_name_from_object_id(
                object_id=instance.actions[0].inventory_object_id,
                object_assets_to_names=self._object_assets_to_names,
                special_name_cases=self._special_name_cases,
            )
        else:
            instruction = add_inventory_to_instruction(
                object_assets_to_names=self._object_assets_to_names,
                special_name_cases=self._special_name_cases,
                inventory_object_id=instance.actions[0].inventory_object_id,
                instruction=instruction,
            )
            # Get the object_readable_name from the action target readable name
            object_readable_name = self._get_target_object_name(
                action=instance.actions[0], name_type="readable"
            )

        if object_readable_name:
            target_text = f"{target_text} {object_readable_name.lower()}"
        return instruction, target_text

    def _prepare_data(self) -> None:
        """Prepare the data intents and the negative candidates.

        Precompile a list of possible negative candidates for synthetic instructions.
        An instance can be a negative candidate if the following conditions are true:
        1. It is synthetic
        2. It is not a Search or Examine instruction
        3. It is an object interaction
        """
        db_size = len(self.db)
        with self.db:
            for index in range(db_size):
                instance_str: str = self.db[index]
                instance = SimBotInstructionInstance.parse_raw(instance_str)

                self.data_intents.append(self._get_data_intent(instance))

                if not instance.synthetic or not action_is_object_interaction(instance.actions[0]):
                    continue
                self._synthetic_negative_candidates.append(index)

    def _get_data_intent(self, instance: SimBotInstructionInstance) -> SimBotCRIntents:
        if instance.actions[0].type == "Search":
            action_object_metadata = instance.actions[0].get_action_data["object"]
            if action_object_metadata["mask"] is None:
                return SimBotCRIntents.search_no_match
            return SimBotCRIntents.search_one_match
        if instance.instruction.necessary_question_answers:
            qa_pair = instance.instruction.necessary_question_answers[0]
            return self._question_type_intent_map.get(
                qa_pair.question_type, SimBotCRIntents.act_one_match
            )

        elif instance.ambiguous:
            return SimBotCRIntents.act_too_many_matches
        return SimBotCRIntents.act_one_match

    def _get_instance_frame(self, instance: SimBotInstructionInstance, target_text: str) -> int:
        """Get either the image infront of you or the image with the target object."""
        frame_idx = 0
        # Always return the first frame if the intent is no_match
        if self._is_no_match(target_text):
            return frame_idx
        action_object_metadata = instance.actions[0].get_action_data.get("object", None)
        if action_object_metadata is not None:
            frame_idx = action_object_metadata.get("colorImageIndex", 0)
        return frame_idx

    @overrides(check_signature=False)
    def _load_visual_features(self, features_path: Path, frame_idx: int = 0) -> EmmaVisualFeatures:
        """Get the visual features just for the first frame."""
        frame_dict = torch.load(features_path)["frames"][frame_idx]
        feature_dicts = [frame_dict["features"]]

        visual_features = self._prepare_emma_visual_features(feature_dicts=feature_dicts)

        return visual_features

    def _is_no_match(self, target_text: str) -> bool:
        """Check if the instance CR label is no_match."""
        return SimBotCRIntents.no_match.value in target_text

    def _get_target_object_name(
        self, action: SimBotAction, name_type: Literal["class", "readable"] = "readable"
    ) -> Optional[str]:
        """Get the object name for a given SimBot action.

        The name can be either the class name which matches the bounding box classes, or the
        readable name which includes special names such as 'laser monitor'.
        """
        target_object_name = None
        object_metadata = action.get_action_data.get("object", None)
        if object_metadata is not None and "id" in object_metadata:
            if name_type == "readable":
                target_object_name = get_object_readable_name_from_object_id(
                    object_id=object_metadata["id"],
                    object_assets_to_names=self._object_assets_to_names,
                    special_name_cases=self._special_name_cases,
                )
            else:
                target_object_name = get_object_label_from_object_id(
                    object_metadata["id"], self._object_assets_to_names
                )

        return target_object_name

    def _create_act_no_match_action(
        self, old_action: SimBotAction, negative_example_action: SimBotAction
    ) -> SimBotAction:
        """Modify the action selectively based on the sampled negative example."""
        # Keep the action id and color images of the old action to load the correct features
        # Update all other fields to get the correct target
        modified_action = old_action.copy(
            update={
                "type": negative_example_action.type,
                "final": negative_example_action.final,
                "inventory_object_id": negative_example_action.inventory_object_id,
                negative_example_action.type.lower(): negative_example_action.get_action_data,
            }
        )
        return modified_action

    def _sample_cdf_no_match(
        self, instance: SimBotInstructionInstance
    ) -> tuple[str, EmmaVisualFeatures, str]:
        """Sample a negative image and a CDF paraphrase for the no_match intent."""
        high_level_key = self.high_level_paraphraser(instance.cdf_highlevel_key)
        instruction = random.choice(high_level_key.paraphrases)

        # Get the target object name from the CDF high level key
        # Prioritize anything but the target object
        candidates = [
            high_level_key.decoded_key.from_container,
            high_level_key.decoded_key.from_receptacle,
            high_level_key.decoded_key.to_container,
            high_level_key.decoded_key.to_receptacle,
            high_level_key.decoded_key.interaction_object,
            high_level_key.decoded_key.stacked_object,
        ]
        target_asset_ids = [c for c in candidates if c is not None]
        if target_asset_ids:
            target_asset_id = target_asset_ids[0]
        else:
            target_asset_id = high_level_key.decoded_key.target_object

        if target_asset_id is None:
            raise AssertionError(f"No target from CDF high level key {instance.cdf_highlevel_key}")
        target_object_name = get_object_label_from_object_id(
            target_asset_id, self._object_assets_to_names
        )

        object_readable_name = get_object_readable_name_from_object_id(
            object_id=target_object_name,
            object_assets_to_names=self._object_assets_to_names,
            special_name_cases=self._special_name_cases,
        )
        # Sample a negative image
        visual_features = None
        while visual_features is None:
            rand_index = int(len(self._synthetic_negative_candidates) * random.random())
            db_index = self._synthetic_negative_candidates[rand_index]
            with self.db:
                new_instance = SimBotInstructionInstance.parse_raw(self.db[db_index])
            temp_visual_features = self._load_visual_features(
                features_path=new_instance.features_path[0], frame_idx=0
            )
            tmp_object_classes = temp_visual_features.object_classes.tolist()
            if self._label_to_idx[target_object_name] not in tmp_object_classes:
                visual_features = temp_visual_features

        instruction = add_inventory_to_instruction(  # type: ignore[unreachable]
            object_assets_to_names=self._object_assets_to_names,
            special_name_cases=self._special_name_cases,
            instruction=instruction,
            inventory_object_id=instance.actions[0].inventory_object_id,
        )

        target_text = SimBotCRIntents.act_no_match.value
        if object_readable_name:
            target_text = f"{target_text} {object_readable_name}"

        return instruction, visual_features, target_text
