import argparse
import json
import logging
from pathlib import Path
from typing import Any, Callable, Literal, Optional

import gradio as gr


logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

VisualizerOutput = tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    list[str],
    list[str],
]


class ArenaVisualizer:
    """Class to visualize groundtruth arena instructions."""

    def __init__(self, input_images: Optional[Path] = None):
        self._mission_name = ""
        self._mission_len = -1
        self._mission_metadata: dict[str, Any] = {}
        self._current_pos = 0
        self._num_images = 1
        self._input_images = input_images

    def get_paths_for_images(
        self, instructions_dict: Optional[dict[str, Any]] = None
    ) -> list[str]:
        """Get the paths for images associated with actions for an instruction."""
        shown_path_images = []
        if instructions_dict is not None and self._input_images is not None:
            mission_actions = self._mission_metadata["actions"]
            actions_ids_in_instruction = instructions_dict["actions"]
            for mission_action in mission_actions:
                mission_action_id = mission_action["id"]
                if mission_action_id in actions_ids_in_instruction:
                    log.info(
                        f"Mission id: {mission_action_id} is within instruction: {instructions_dict}"
                    )
                    images_for_action = [
                        str(Path(self._input_images, self._mission_name, image))
                        for image in mission_action["colorImages"]
                    ]
                    shown_path_images.extend(images_for_action)

        return shown_path_images

    def get_instruction(
        self, dataset: dict[str, Any], step: Literal["submit", "previous", "next"] = "next"
    ) -> Callable[[str], VisualizerOutput]:
        """Prepare the output for an instruction."""

        def generate_response(  # noqa: WPS430
            mission_name: str,
        ) -> VisualizerOutput:
            log.info(f"mission name: {mission_name}")
            if self._mission_name == mission_name:
                if step == "next":
                    self._current_pos = min(self._mission_len - 1, self._current_pos + 1)
                else:
                    self._current_pos = max(0, self._current_pos - 1)
            else:
                self._mission_name = mission_name
                if mission_name not in dataset.keys():
                    log.info(f"Mission: {mission_name} is not found in dataset metadata")
                self._mission_metadata = dataset[mission_name]
                self._current_pos = 0
                self._mission_len = len(
                    self._mission_metadata["human_annotations"][0]["instructions"]
                )

            human_annotations = self._mission_metadata["human_annotations"]
            action_start = self._mission_metadata["human_annotations"][0][  # noqa: WPS219
                "instructions"
            ][self._current_pos]["actions"][0]
            action_end = self._mission_metadata["human_annotations"][0][  # noqa: WPS219
                "instructions"
            ][self._current_pos]["actions"][-1]
            action_path_images = self.get_paths_for_images(
                human_annotations[0]["instructions"][self._current_pos]
            )
            return (  # noqa: WPS227
                self._action2dict(
                    self._mission_metadata["actions"][action_start : action_end + 1]
                ),
                human_annotations[0]["instructions"][self._current_pos],
                human_annotations[1]["instructions"][self._current_pos],
                human_annotations[2]["instructions"][self._current_pos],
                action_path_images,
                action_path_images,
            )

        return generate_response

    def _action2dict(self, actions_in_instruction: list[dict[str, Any]]) -> dict[str, Any]:
        """Get actions for instruction.

        Remove unnecessary fields like mask to avoid huge dictionaries at the interface.
        """
        action_dict = {}
        for action in actions_in_instruction:
            action_dict[str(action["id"])] = {"type": action["type"]}
            type_in_dict = action["type"].lower()
            action_dict[str(action["id"])][type_in_dict] = action[type_in_dict]
            if "object" in action_dict[str(action["id"])][type_in_dict]:
                action_dict[str(action["id"])][type_in_dict]["object"].pop(  # noqa: WPS529
                    "mask", None
                )
        return action_dict


def main(args: argparse.Namespace) -> None:
    """Main."""
    with open(args.input_json) as fp:
        dataset = json.load(fp)

    visualizer = ArenaVisualizer(input_images=args.input_images)
    with gr.Blocks() as block:
        with gr.Row():
            input_text = gr.Textbox(label="Input mission id \U0000270D", interactive=True)
        with gr.Row():
            output_actions = gr.JSON(label="Actions in instruction", interactive=False)
            output_instructions1 = gr.JSON(label="Human instructions1", interactive=False)
            output_instructions2 = gr.JSON(label="Human instructions2", interactive=False)
            output_instructions3 = gr.JSON(label="Human instructions3", interactive=False)
        with gr.Row():
            output_image_gallery = gr.Gallery(
                label="Image for action",
                value=visualizer.get_paths_for_images,
            )
            output_input_images = gr.JSON(label="Image paths", interactive=False)
        with gr.Row():
            with gr.Column():
                submit_button = gr.Button(
                    "Submit mission",
                    variant="primary",
                )
            with gr.Column():
                previous_button = gr.Button("Previous instruction")
            with gr.Column():
                next_button = gr.Button("Next instruction")

        example_names = list(dataset.keys())
        gr.Examples(
            label="Mission ids",
            examples=example_names,
            inputs=[input_text],
        )

        submit_button.click(
            fn=visualizer.get_instruction(dataset, step="submit"),
            inputs=input_text,
            outputs=[
                output_actions,
                output_instructions1,
                output_instructions2,
                output_instructions3,
                output_image_gallery,
                output_input_images,
            ],
        )

        previous_button.click(
            fn=visualizer.get_instruction(dataset, step="previous"),
            inputs=input_text,
            outputs=[
                output_actions,
                output_instructions1,
                output_instructions2,
                output_instructions3,
                output_image_gallery,
                output_input_images,
            ],
        )

        next_button.click(
            fn=visualizer.get_instruction(dataset, step="next"),
            inputs=input_text,
            outputs=[
                output_actions,
                output_instructions1,
                output_instructions2,
                output_instructions3,
                output_image_gallery,
                output_input_images,
            ],
        )

        block.launch(share=args.share)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_json",
        required=True,
        help="Path input simbot json file",
    )
    parser.add_argument(
        "--input_images",
        help="Path input images. If specified the app shows the images for each action",
    )
    parser.add_argument(
        "--share",
        help="Create a publicly shareable link from your computer for the interface",
        action="store_true",
    )
    args = parser.parse_args()
    main(args)
