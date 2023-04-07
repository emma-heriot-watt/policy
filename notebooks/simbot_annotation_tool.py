import argparse
import glob
import json
import logging
import os
import subprocess  # noqa: S404
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional

import boto3
import cv2
import gradio as gr
import numpy as np
import pandas as pd
import torch
from boto3.dynamodb.conditions import Key
from botocore.exceptions import ClientError
from emma_datasets.constants.simbot.simbot import get_arena_definitions

from emma_policy.commands.decode_images import decode_images_for_file
from emma_policy.commands.plot_bb import PlotBoundingBoxes


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


TurnOut = tuple[str, dict[str, Any], list[str], int]
SessionOut = tuple[int, str, str, list[Any], str, dict[str, Any], list[str], int]  # noqa: WPS221


class SessionClient:
    """A simple client for retrieving sessions from the s3 bucket and dynamo db."""

    def __init__(
        self,
        primary_key: str = "session_id",
        resource_region: str = "us-east-1",
        table_name: str = "SIMBOT_MEMORY_TABLE",
        s3_sessions_bucket_url: str = "s3://emma-simbot-live-challenge",
        sessions_file: str = "./notebooks/sessions.txt",
    ) -> None:

        self._primary_key = primary_key
        self._resource_region = resource_region
        self._table_name = table_name

        self._db = boto3.resource("dynamodb", self._resource_region)
        self._table = self._db.Table(self._table_name)

        self._s3_sessions_bucket_url = s3_sessions_bucket_url
        self._sessions_file = sessions_file

    def get_all_session_turns_for_session(self, session_id: str) -> list[Any]:
        """Get all the turns for a given session."""
        try:
            response = self._table.query(
                KeyConditionExpression=Key(self._primary_key).eq(session_id)
            )
        except ClientError as err:
            error_code = err.response["Error"]["Code"]

            if error_code != "ConditionalCheckFailedException":
                logger.exception("Could not add turn to table.", exc_info=err)
                raise err
            return []

        parsed_responses = response["Items"]
        logger.debug(f"Successfully got previous {len(parsed_responses)} turns")
        return parsed_responses

    def get_all_session_ids_from_bucket(self) -> dict[str, str]:
        """Get all the session ids from the s3 bucket."""
        command = f"aws s3 ls --recursive {self._s3_sessions_bucket_url}"
        with open(self._sessions_file, "w") as fpw:
            subprocess.call(command.split(), stdout=fpw)  # noqa: S603

        df_csv = pd.read_csv(self._sessions_file, sep=r"\s+")

        session_days = df_csv.iloc[:, 0].tolist()
        session_times = df_csv.iloc[:, 1].tolist()
        session_files = df_csv.iloc[:, 3].tolist()

        sessions: dict[str, str] = {}

        session_metadata = zip(session_days, session_times, session_files)
        for session_day, session_time, session_file in session_metadata:
            if not session_file.startswith("amzn1"):
                continue
            session_name = os.path.dirname(session_file)

            timestamp = sessions.get(session_name, None)
            if timestamp is not None:
                t1 = datetime.strptime(timestamp, "%Y-%m-%d_%H:%M:%S")
                t2 = datetime.strptime(timestamp, "%Y-%m-%d_%H:%M:%S")

                earliest_datetime = min((t1, t2))
                sessions[session_name] = datetime.strftime(earliest_datetime, "%Y-%m-%d_%H:%M:%S")
            else:
                sessions[session_name] = f"{session_day}_{session_time}"

        return sessions

    def download_from_s3(
        self, local_cache_path: str, s3_object_url: str, is_folder: bool = False
    ) -> None:
        """Download a file or folder from the s3 bucket."""
        local_path = os.path.join(local_cache_path, s3_object_url)
        if os.path.exists(local_path):
            logger.debug(f"{s3_object_url} has been download in {local_path}")
            return
        s3_url = os.path.join(self._s3_sessions_bucket_url, s3_object_url)
        command = f"aws s3 cp {s3_url} {local_path}"
        if is_folder:
            command = f"{command} --recursive"
        logger.debug(f"Downloading {s3_url} into {local_path}")
        subprocess.call(  # noqa: S603
            command.split(), stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL
        )


class ArenaSessionAnnotation:
    """Class for visualising and annotating turns from arena sessions."""

    def __init__(
        self,
        output_annotation_json: str,
        output_features_directory: str,
        s3_sessions_bucket_url: str = "s3://emma-simbot-live-challenge",
        cache_dir: str = "sessions",
        max_bboxes: int = 36,
    ) -> None:
        self.output_annotation_json = output_annotation_json

        os.makedirs(output_features_directory, exist_ok=True)
        self.output_features_directory = output_features_directory

        os.makedirs(cache_dir, exist_ok=True)
        self.cache_dir = cache_dir

        arena_definitions = get_arena_definitions()
        self.actions = sorted(arena_definitions["action_list"] + ["Search"])
        self.assets = list(arena_definitions["asset_to_label"].keys())
        self.max_bboxes = max_bboxes

        self._session_client = SessionClient(s3_sessions_bucket_url=s3_sessions_bucket_url)
        sessions_dict = self._session_client.get_all_session_ids_from_bucket()
        self._session_ids = list(sessions_dict.keys())
        self._session_timestamps = list(sessions_dict.values())
        self._bbox_plot = PlotBoundingBoxes()

    def __len__(self) -> int:
        """Return the number of sessions."""
        return len(self._session_ids)

    def sort_sessions(self, session_index: int, key: Literal["alphabetical", "timestamp"]) -> int:
        """Sort the sessions depending on the key and update the current session index."""
        if key == "alphabetical":
            order = np.argsort(self._session_ids)
        elif key == "timestamp":
            predicate = [
                datetime.strptime(timestamp, "%Y-%m-%d_%H:%M:%S")
                for timestamp in self._session_timestamps
            ]
            order = np.argsort(predicate)  # type: ignore[arg-type]

        self._session_ids = np.array(self._session_ids)[order].tolist()
        self._session_timestamps = np.array(self._session_timestamps)[order].tolist()
        return order.tolist().index(session_index)

    def get_user_utterance_for_turn(self, current_session_turn: dict[str, Any]) -> str:
        """Get the user utterance for the current turn."""
        metadata_current_turn = json.loads(current_session_turn["turn"])
        if metadata_current_turn["speech"] is None:
            return ""

        utterance_metadata = metadata_current_turn["speech"].get("utterance", None)
        if utterance_metadata is not None:
            return metadata_current_turn["speech"]["utterance"]
        return " ".join([token["value"] for token in metadata_current_turn["speech"]["tokens"]])

    def get_agent_metadata_for_turn(self, current_session_turn: dict[str, Any]) -> dict[str, Any]:
        """Get the metadata dict for the current turn."""
        metadata_current_turn = json.loads(current_session_turn["turn"])

        agent_turn_metadata = deepcopy(metadata_current_turn)
        agent_turn_metadata.pop("timestamp", None)
        agent_turn_metadata.pop("environment", None)
        agent_turn_metadata.pop("auxiliary_metadata_uri", None)
        agent_turn_metadata.pop("viewpoints", None)
        agent_turn_metadata.pop("unique_room_names", None)
        agent_turn_metadata["actions"].pop("dialog", None)
        agent_turn_metadata.pop("state", None)
        return agent_turn_metadata

    def get_images_for_turn(self, current_session_turn: dict[str, Any]) -> list[str]:
        """Get the images for the current turn."""
        turn_metadata = json.loads(current_session_turn["turn"])
        session_id = turn_metadata["session_id"]
        prediction_id = turn_metadata["prediction_request_id"]

        local_image_path = Path(os.path.join(self.cache_dir, session_id, "images"))
        os.makedirs(local_image_path, exist_ok=True)

        local_json_image_path = Path(
            os.path.join(self.cache_dir, session_id, f"{prediction_id}.json")
        )
        if not local_json_image_path.exists():
            logger.debug(f"{local_json_image_path} does not exist")
            return []

        decode_images_for_file(local_json_image_path, local_image_path)
        images_pattern = f"{prediction_id}*.png"
        images = glob.glob(f"{os.path.join(self.cache_dir, session_id, 'images', images_pattern)}")
        return sorted(images)

    def prepare_output_for_turn(
        self, current_session_turn: dict[str, Any]
    ) -> tuple[str, dict[str, Any], list[str]]:
        """Prepare the output for the current turn."""
        user_utterance = self.get_user_utterance_for_turn(current_session_turn)
        agent_turn_metadata = self.get_agent_metadata_for_turn(current_session_turn)
        images = self.get_images_for_turn(current_session_turn)
        return (user_utterance, agent_turn_metadata, images)

    def on_previous_turn(self, session_turns: list[Any], turn_index: int) -> Optional[TurnOut]:
        """Get the previous turn."""
        # The next turn is either the turn with a -1 index or the first element in the list.
        new_turn_index = max(0, turn_index - 1)
        if session_turns:
            new_session_turn = session_turns[new_turn_index]
            (user_utterance, agent_turn_metadata, images) = self.prepare_output_for_turn(
                new_session_turn
            )
            return (user_utterance, agent_turn_metadata, images, new_turn_index)
        return None

    def on_next_turn(self, session_turns: list[Any], turn_index: int) -> Optional[TurnOut]:
        """Get the next turn."""
        # The next turn is either the turn with a +1 index or the last element in the list.
        new_turn_index = min(len(session_turns) - 1, turn_index + 1)
        if session_turns:
            new_session_turn = session_turns[new_turn_index]
            (user_utterance, agent_turn_metadata, images) = self.prepare_output_for_turn(
                new_session_turn
            )
            return (user_utterance, agent_turn_metadata, images, new_turn_index)
        return None

    def on_previous_session_id(self, session_index: int) -> Optional[SessionOut]:
        """Get the previous session id."""
        new_session_index = max(0, session_index - 1)

        session_id = self._session_ids[new_session_index]
        session_timestamp = self._session_timestamps[new_session_index]
        self._session_client.download_from_s3(
            local_cache_path=self.cache_dir, s3_object_url=session_id, is_folder=True
        )

        session_turns = self._session_client.get_all_session_turns_for_session(session_id)
        if session_turns:
            new_session_turn = session_turns[0]
            (user_utterance, agent_turn_metadata, images) = self.prepare_output_for_turn(
                new_session_turn
            )

            return (  # noqa: WPS227
                new_session_index,
                session_id,
                session_timestamp,
                session_turns,
                user_utterance,
                agent_turn_metadata,
                images,
                0,
            )
        return None

    def on_next_session_id(self, session_index: int) -> Optional[SessionOut]:
        """Get the next session id."""
        new_session_index = min(len(self._session_ids) - 1, session_index + 1)

        session_id = self._session_ids[new_session_index]
        session_timestamp = self._session_timestamps[new_session_index]
        self._session_client.download_from_s3(
            local_cache_path=self.cache_dir, s3_object_url=session_id, is_folder=True
        )

        session_turns = self._session_client.get_all_session_turns_for_session(session_id)
        if session_turns:
            new_session_turn = session_turns[0]
            (user_utterance, agent_turn_metadata, images) = self.prepare_output_for_turn(
                new_session_turn
            )

            return (  # noqa: WPS227
                new_session_index,
                session_id,
                session_timestamp,
                session_turns,
                user_utterance,
                agent_turn_metadata,
                images,
                0,
            )
        return None

    def on_jump_session_id_slider(self, session_index: int) -> Optional[SessionOut]:
        """Go to a session provided by its index."""
        session_id = self._session_ids[session_index]
        session_timestamp = self._session_timestamps[session_index]
        self._session_client.download_from_s3(
            local_cache_path=self.cache_dir, s3_object_url=session_id, is_folder=True
        )
        session_turns = self._session_client.get_all_session_turns_for_session(session_id)
        if session_turns:
            new_session_turn = session_turns[0]
            (user_utterance, agent_turn_metadata, images) = self.prepare_output_for_turn(
                new_session_turn
            )

            return (  # noqa: WPS227
                session_index,
                session_id,
                session_timestamp,
                session_turns,
                user_utterance,
                agent_turn_metadata,
                images,
                0,
            )
        return None

    def on_jump_session_id_textbox(self, session_id: str) -> Optional[SessionOut]:
        """Go to a session provided by its id."""
        self._session_client.download_from_s3(
            local_cache_path=self.cache_dir, s3_object_url=session_id, is_folder=True
        )
        session_index = self._session_ids.index(session_id)
        session_timestamp = self._session_timestamps[session_index]
        session_turns = self._session_client.get_all_session_turns_for_session(session_id)
        if session_turns:
            new_session_turn = session_turns[0]
            (user_utterance, agent_turn_metadata, images) = self.prepare_output_for_turn(
                new_session_turn
            )

            return (  # noqa: WPS227
                session_index,
                session_id,
                session_timestamp,
                session_turns,
                user_utterance,
                agent_turn_metadata,
                images,
                0,
            )
        return None

    def on_hide_all_boxes(
        self, session_id: str, session_turns: list[Any], session_turn_index: int
    ) -> tuple[list[int], list[str]]:
        """Disable all the bounding boxes."""
        indices: list[int] = []
        return indices, self.on_show_specific_boxes(
            session_id, session_turns, session_turn_index, indices
        )

    def on_show_all_boxes(
        self, session_id: str, session_turns: list[Any], session_turn_index: int
    ) -> tuple[list[int], list[str]]:
        """Show all the bounding boxes."""
        indices = [idx + 1 for idx in range(self.max_bboxes)]
        return indices, self.on_show_specific_boxes(
            session_id, session_turns, session_turn_index, list(range(self.max_bboxes))
        )

    def on_show_specific_boxes(
        self,
        session_id: str,
        session_turns: list[Any],
        session_turn_index: int,
        indices: list[int],
    ) -> list[str]:
        """Show only a subset of the available bounding boxes."""
        images = self.get_images_for_turn(session_turns[session_turn_index])

        local_image_bbox_path = Path(os.path.join(self.cache_dir, "images_bboxes"))
        os.makedirs(local_image_bbox_path, exist_ok=True)
        images_bboxes = []
        for idx, image in enumerate(images):
            image_bname = os.path.splitext(os.path.basename(image))[0]
            (feature_basename, image_index) = image_bname.split("_")
            feature_path = os.path.join(self.cache_dir, session_id, f"{feature_basename}.pt")

            if os.path.exists(feature_path):
                image_features = torch.load(feature_path)[int(image_index)]

                boxes_coords = image_features["bbox_coords"].cpu().numpy()

                num_boxes = boxes_coords.shape[0]
                boxes_indices = [idx for idx in indices if 0 <= idx < num_boxes]

                image_cv = cv2.imread(image)
                self._bbox_plot.draw_bb(
                    image=image_cv,
                    boxes_coords=boxes_coords[boxes_indices],
                    boxes_labels=[f"{idx + 1}" for idx in boxes_indices],
                    draw_label=True,
                )
                image_bbox_path = os.path.join(local_image_bbox_path, f"{session_id}_{idx}.png")
                cv2.imwrite(image_bbox_path, image_cv)
                images_bboxes.append(image_bbox_path)
            else:
                logger.debug(f"Feature path {feature_path} does not exist")
        return images_bboxes

    def on_update_annotation(
        self,
        session_id: str,
        session_turns: list[Any],
        session_turn_index: int,
        user_utterance: str,
        action_type: Optional[str] = None,
        object_id: Optional[str] = None,
        visual_token: Optional[int] = None,
    ) -> dict[str, Any]:
        """Update the annotation for a turn."""
        instruction_metadata: dict[str, Any] = {
            "instruction": {
                "instruction": user_utterance,
                "actions": [0],
            }
        }
        if action_type:
            images = self.get_images_for_turn(session_turns[session_turn_index])
            if images:
                image = images[0]
                image_bname = os.path.splitext(os.path.basename(image))[0]
                (feature_basename, image_index) = image_bname.split("_")
                feature_path = os.path.join(self.cache_dir, session_id, f"{feature_basename}.pt")

                mask = None
                if os.path.exists(feature_path) and visual_token:
                    image_features = torch.load(feature_path)[int(image_index)]
                    boxes_coords = image_features["bbox_coords"].cpu().numpy()
                    mask = boxes_coords[int(visual_token - 1)].astype(int).tolist()

                # The search metadata are slightly different from the other actions.
                # The object dictionary has multiple object ids and object masks.
                if action_type == "Search":
                    instruction_metadata["actions"] = [
                        {
                            "id": 0,
                            "type": action_type,
                            action_type.lower(): {
                                "object": {"id": [object_id], "mask": [mask], "colorImageIndex": 0}
                            },
                            "colorImages": [os.path.basename(image)],
                            "final": True,
                            "positive": True,
                        }
                    ]
                else:
                    instruction_metadata["actions"] = [
                        {
                            "id": 0,
                            "type": action_type,
                            action_type.lower(): {
                                "object": {"id": object_id, "mask": mask, "colorImageIndex": 0}
                            },
                            "colorImages": [os.path.basename(image)],
                            "final": True,
                        }
                    ]

        # Fill in required metadata so that the dictionary can be parsed by the SimBotInstructionInstance
        instruction_metadata["annotation_id"] = 0
        instruction_metadata["instruction_id"] = 0
        instruction_metadata["synthetic"] = False
        instruction_metadata["mission_id"] = session_id
        # This needs to be set to true so that we can get the correct features path from
        # https://github.com/emma-simbot/datasets/blob/19db6ef9244e2e78acf2cb36a1c2f1bd6be799cd/src/emma_datasets/datamodels/datasets/utils/simbot_utils/simbot_datamodels.py#L149-L162
        instruction_metadata["vision_augmentation"] = True
        return instruction_metadata

    def on_save_annotation_for_turn(
        self,
        session_id: str,
        session_turns: list[Any],
        session_turn_index: int,
        instruction_metadata: dict[str, Any],
    ) -> None:
        """Save the annotations for a turn."""
        images = self.get_images_for_turn(session_turns[session_turn_index])

        prediction_request_id = json.loads(session_turns[session_turn_index]["turn"])[
            "prediction_request_id"
        ]
        data = {}
        # Allow for multiple annotations per prediction id
        date_time = datetime.now().strftime("%m/%d/%Y-%H:%M:%S")
        data_key = f"session_id_{session_id}_prediction_id_{prediction_request_id}_{date_time}"
        if os.path.exists(self.output_annotation_json):
            with open(self.output_annotation_json) as fpr:
                data = json.load(fpr)

        data[data_key] = instruction_metadata
        with open(self.output_annotation_json, "w") as fpw:
            json.dump(data, fpw, indent=4)

        features = torch.load(
            os.path.join(self.cache_dir, session_id, f"{prediction_request_id}.pt")
        )
        features_formatted: dict[str, Any] = {"frames": []}
        for feature_idx, feature_dict in features.items():
            feature_dict_formatted = {
                "image": os.path.basename(images[feature_idx]),
                "features": {
                    "bbox_features": feature_dict["bbox_features"].cpu(),
                    "bbox_coords": feature_dict["bbox_coords"].cpu(),
                    "bbox_probas": feature_dict["bbox_probas"].cpu(),
                    "cnn_features": feature_dict["cnn_features"].cpu(),
                    "width": 300,
                    "height": 300,
                },
            }
            features_formatted["frames"].append(feature_dict_formatted)

            torch.save(
                features_formatted,
                os.path.join(
                    self.output_features_directory, f"{prediction_request_id}_{feature_idx}.pt"
                ),
            )


def main(args: argparse.Namespace) -> None:  # noqa: WPS210
    """Main."""
    session_visualizer = ArenaSessionAnnotation(
        output_annotation_json=args.output_annotation_json,
        output_features_directory=args.output_features_directory,
        cache_dir=args.cache_dir,
    )

    with gr.Blocks() as block:
        session_id_turns = gr.State([])
        session_turn_index = gr.State(0)
        with gr.Row():
            session_id_textbox = gr.Textbox(label="Session ID \U0000270D", interactive=True)

            session_timestamp_textbox = gr.Textbox(label="Session Timestamp", interactive=False)

            sort_sessions_dropdown = gr.Radio(
                label="Sort Sessions", choices=["alphabetical", "timestamp"], value="alphabetical"
            )
        with gr.Row():
            previous_session_id_button = gr.Button(
                "Previous Session ID", label="Previous Session ID"
            )
            next_session_id_button = gr.Button(
                "Next Session ID",
                label="Next Session ID",
            )
            jump_session_id_button = gr.Button(
                "Go To Session ID",
                label="Go To Session ID",
            )
        with gr.Row():
            jump_session_id_slider = gr.Slider(
                minimum=0,
                maximum=len(session_visualizer) - 1,
                label="Jump To Session",
                value=0,
                step=1,
            )

        with gr.Row():
            previous_turn_button = gr.Button("Previous Turn", label="Previous Turn")
            next_turn_button = gr.Button("Next Turn", label="Next Turn")

        with gr.Row():
            agent_turn_session_id_textbox = gr.JSON(label="Agent Turn Metadata")

            with gr.Column():
                output_image_gallery = gr.Gallery(label="Images For Current Turn")

        with gr.Row():
            checkboxgroup_bboxes = gr.CheckboxGroup(
                choices=[f"{idx + 1}" for idx in range(session_visualizer.max_bboxes)],
                type="index",
                label="Show specific bounding boxes",
                interactive=True,
            )

        with gr.Row():
            disable_all_boxes_button = gr.Button("Hide all bounding boxes")
            show_all_boxes_button = gr.Button("Show all bounding boxes")

        with gr.Row():
            with gr.Column():
                user_turn_session_id_textbox = gr.Textbox(
                    label="User Turn Utterance \U0000270D", interactive=True
                )

                action_type_dropdown = gr.Dropdown(
                    label="Action Type", choices=session_visualizer.actions
                )

                object_id_dropdown = gr.Dropdown(
                    label="Object ID", choices=session_visualizer.assets
                )

                visual_token_dropdown = gr.Dropdown(
                    label="Visual Token",
                    choices=list(range(1, session_visualizer.max_bboxes + 1)),
                )

            with gr.Column():
                instruction_annotation_json = gr.JSON(label="Instruction metadata", value={})

        with gr.Row():
            save_turn_button = gr.Button(
                "Save Annotation For Turn",
                label="Save Annotation For Turn",
                variant="primary",
            )

        previous_session_id_button.click(
            fn=session_visualizer.on_previous_session_id,
            inputs=[jump_session_id_slider],
            outputs=[
                jump_session_id_slider,
                session_id_textbox,
                session_timestamp_textbox,
                session_id_turns,
                user_turn_session_id_textbox,
                agent_turn_session_id_textbox,
                output_image_gallery,
                session_turn_index,
            ],
        )

        next_session_id_button.click(
            fn=session_visualizer.on_next_session_id,
            inputs=[jump_session_id_slider],
            outputs=[
                jump_session_id_slider,
                session_id_textbox,
                session_timestamp_textbox,
                session_id_turns,
                user_turn_session_id_textbox,
                agent_turn_session_id_textbox,
                output_image_gallery,
                session_turn_index,
            ],
        )

        jump_session_id_button.click(
            fn=session_visualizer.on_jump_session_id_textbox,
            inputs=[session_id_textbox],
            outputs=[
                jump_session_id_slider,
                session_id_textbox,
                session_timestamp_textbox,
                session_id_turns,
                user_turn_session_id_textbox,
                agent_turn_session_id_textbox,
                output_image_gallery,
                session_turn_index,
            ],
        )

        jump_session_id_slider.change(
            fn=session_visualizer.on_jump_session_id_slider,
            inputs=[jump_session_id_slider],
            outputs=[
                jump_session_id_slider,
                session_id_textbox,
                session_timestamp_textbox,
                session_id_turns,
                user_turn_session_id_textbox,
                agent_turn_session_id_textbox,
                output_image_gallery,
                session_turn_index,
            ],
        )

        sort_sessions_dropdown.change(
            fn=session_visualizer.sort_sessions,
            inputs=[jump_session_id_slider, sort_sessions_dropdown],
            outputs=[jump_session_id_slider],
        )

        previous_turn_button.click(
            fn=session_visualizer.on_previous_turn,
            inputs=[session_id_turns, session_turn_index],
            outputs=[
                user_turn_session_id_textbox,
                agent_turn_session_id_textbox,
                output_image_gallery,
                session_turn_index,
            ],
        )

        next_turn_button.click(
            fn=session_visualizer.on_next_turn,
            inputs=[session_id_turns, session_turn_index],
            outputs=[
                user_turn_session_id_textbox,
                agent_turn_session_id_textbox,
                output_image_gallery,
                session_turn_index,
            ],
        )

        disable_all_boxes_button.click(
            fn=session_visualizer.on_hide_all_boxes,
            inputs=[
                session_id_textbox,
                session_id_turns,
                session_turn_index,
            ],
            outputs=[checkboxgroup_bboxes, output_image_gallery],
        )

        show_all_boxes_button.click(
            fn=session_visualizer.on_show_all_boxes,
            inputs=[
                session_id_textbox,
                session_id_turns,
                session_turn_index,
            ],
            outputs=[checkboxgroup_bboxes, output_image_gallery],
        )

        checkboxgroup_bboxes.change(
            fn=session_visualizer.on_show_specific_boxes,
            inputs=[
                session_id_textbox,
                session_id_turns,
                session_turn_index,
                checkboxgroup_bboxes,
            ],
            outputs=[output_image_gallery],
        )

        user_turn_session_id_textbox.change(
            fn=session_visualizer.on_update_annotation,
            inputs=[
                session_id_textbox,
                session_id_turns,
                session_turn_index,
                user_turn_session_id_textbox,
                action_type_dropdown,
                object_id_dropdown,
                visual_token_dropdown,
            ],
            outputs=[instruction_annotation_json],
        )

        action_type_dropdown.change(
            fn=session_visualizer.on_update_annotation,
            inputs=[
                session_id_textbox,
                session_id_turns,
                session_turn_index,
                user_turn_session_id_textbox,
                action_type_dropdown,
                object_id_dropdown,
                visual_token_dropdown,
            ],
            outputs=[instruction_annotation_json],
        )

        object_id_dropdown.change(
            fn=session_visualizer.on_update_annotation,
            inputs=[
                session_id_textbox,
                session_id_turns,
                session_turn_index,
                user_turn_session_id_textbox,
                action_type_dropdown,
                object_id_dropdown,
                visual_token_dropdown,
            ],
            outputs=[instruction_annotation_json],
        )
        visual_token_dropdown.change(
            fn=session_visualizer.on_update_annotation,
            inputs=[
                session_id_textbox,
                session_id_turns,
                session_turn_index,
                user_turn_session_id_textbox,
                action_type_dropdown,
                object_id_dropdown,
                visual_token_dropdown,
            ],
            outputs=[instruction_annotation_json],
        )

        save_turn_button.click(
            fn=session_visualizer.on_save_annotation_for_turn,
            inputs=[
                session_id_textbox,
                session_id_turns,
                session_turn_index,
                instruction_annotation_json,
            ],
        )

        block.launch(share=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_annotation_json",
        default="session_annotations/session_annotations.json",
        help="Path to output annotation json file.",
    )

    parser.add_argument(
        "--output_features_directory",
        default="session_annotations/features/",
        help="Path to output annotation feature directory.",
    )

    parser.add_argument(
        "--cache_dir",
        default="sessions",
        help="Path to cache directory storing raw session metadata while annotating.",
    )

    args = parser.parse_args()
    main(args)
