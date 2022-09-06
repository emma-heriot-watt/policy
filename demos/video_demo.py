import argparse
import glob
import json
import logging
from pathlib import Path
from typing import Any, Callable

import gradio as gr
import torch
from transformers import AutoTokenizer, PreTrainedTokenizer

from emma_policy.datamodules.batch_attention_masks import make_text_history_global_pattern
from emma_policy.datamodules.emma_dataclasses import EmmaDatasetBatch
from emma_policy.models.emma_policy import EmmaPolicy


logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


input_prompts = [
    "Provide an instruction",
    "Act according to the instruction:",
    "Execute the instruction",
]

css = """
    .gradio-container {
        background: rgb(47, 79, 79);
    }
    #button_style {
        background: rgba(255, 217, 102, 1.00);
        border-color: rgba(255, 204, 51 1.00);
        border-width: thin;
    }
"""


def change_textbox(choice: str) -> dict[str, Any]:
    """Update a textbox with a given choice."""
    return gr.Textbox.update(value=choice, visible=True)


def run_model(
    root_examples_path: str,
    video_id_map: dict[str, str],
    model: EmmaPolicy,
    tokenizer: PreTrainedTokenizer,
) -> Callable[[str, str], str]:
    """Prepare the response generation."""

    def generate_response(input_text: str, input_video_path: str) -> str:  # noqa: WPS430
        """Generate a response to the text input."""
        video_id = f"{Path(input_video_path).name[:3]}"

        video_features_path = Path(
            root_examples_path, "frame_features", video_id_map[video_id]
        ).with_suffix(".pt")

        log.info(
            f"Input path {input_video_path} Video id: {video_id} Video features path: {video_features_path}"
        )

        feature_dicts = [
            feature_dict["features"] for feature_dict in torch.load(video_features_path)["frames"]
        ]

        visual_input: dict[str, list[torch.Tensor]] = {
            "object_features": [],
            "object_classes": [],
            "object_coordinates": [],
            "vis_tokens": [],
            "object_frame_tokens": [],
            "object_attention_mask": [],
            "scene_features": [],
            "scene_frame_tokens": [],
        }

        for frame_idx, feature_dict in enumerate(feature_dicts):
            visual_input["object_features"].append(feature_dict["bbox_features"])
            visual_input["object_classes"].append(
                torch.tensor([torch.argmax(proba, -1) for proba in feature_dict["bbox_probas"]])
            )
            image_coords = feature_dict["bbox_coords"]

            # normalized coordinates
            image_coords[:, (0, 2)] /= feature_dict["width"]
            image_coords[:, (1, 3)] /= feature_dict["height"]
            visual_input["object_coordinates"].append(image_coords)

            visual_input["scene_features"].append(feature_dict["cnn_features"].unsqueeze(0))

            feature_count = visual_input["object_features"][-1].shape[0]

            curr_vis_tokens = torch.tensor(
                tokenizer.convert_tokens_to_ids(
                    [f"<vis_token_{idx+1}>" for idx in range(feature_count)]
                ),
                dtype=torch.long,
            )
            visual_input["vis_tokens"].append(curr_vis_tokens)

            frame_token = tokenizer.convert_tokens_to_ids(f"<frame_token_{frame_idx+1}>")
            visual_input["object_frame_tokens"].append(
                curr_vis_tokens.new_full(
                    curr_vis_tokens.shape,
                    fill_value=frame_token,  # type: ignore[arg-type]
                )
            )
            visual_input["scene_frame_tokens"].append(frame_token)  # type: ignore[arg-type]
            visual_input["object_attention_mask"].append(
                torch.ones_like(curr_vis_tokens, dtype=torch.bool)
            )

        num_frames = len(visual_input["scene_features"])
        scene_attention_mask = torch.ones(num_frames, dtype=torch.bool)
        scene_coordinates = torch.tensor([0, 0, 1.0, 1.0]).repeat(num_frames, 1)

        inputs = tokenizer.encode_plus(input_text, return_tensors="pt")
        log.info(f"Input text: {input_text} Input token ids: {inputs.input_ids}")
        attention_mask = torch.cat(
            [
                scene_attention_mask,
                torch.cat(visual_input["object_attention_mask"]),
                inputs.attention_mask.squeeze(0),
            ],
            dim=-1,
        ).unsqueeze(0)

        global_attention_mask = make_text_history_global_pattern(
            total_seq_len=attention_mask.shape[-1],
            text_attention_mask=inputs.attention_mask,
            dtype=attention_mask.dtype,
        )
        sample = EmmaDatasetBatch(
            input_token_ids=inputs.input_ids,
            text_attention_mask=inputs.attention_mask,
            target_token_ids=torch.empty_like(inputs.input_ids),
            decoder_attention_mask=torch.empty_like(inputs.attention_mask),
            task=torch.empty(1),
            object_attention_mask=torch.cat(visual_input["object_attention_mask"]).unsqueeze(0),
            object_coordinates=torch.cat(visual_input["object_coordinates"]).unsqueeze(0),
            object_frame_tokens=torch.cat(visual_input["object_frame_tokens"]).unsqueeze(0),
            object_features=torch.cat(visual_input["object_features"]).unsqueeze(0),
            scene_attention_mask=scene_attention_mask.unsqueeze(0),
            scene_coordinates=scene_coordinates.unsqueeze(0),
            scene_features=torch.cat(visual_input["scene_features"]).unsqueeze(0),
            scene_frame_tokens=torch.tensor(visual_input["scene_frame_tokens"]).unsqueeze(0),
            visual_token_ids=torch.cat(visual_input["vis_tokens"]).unsqueeze(0),
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
        )
        outputs = model.predict_step(sample, 0)
        preds = tokenizer.batch_decode(outputs, skip_special_tokens=False)
        response = str(preds)
        response = response.replace("'", "")
        response = response.replace("<pad>", "")
        response = response.replace("<unk>", "")
        response = response.replace("</s>", "")
        response = response.replace("<s>", "")
        response = response.replace('"', "")
        response = response.replace("[", "")
        response = response.replace("]", "")
        response = response.replace(".", "")
        # this is a bug in the metadata of epic kitchens where consecutive narrations
        # with the same action (e.g, washing hands). The second narration may have
        # still washing hands as groundturth label.
        response = response.replace("still", "")
        log.info(f"Response: {response}")
        return response

    return generate_response


def main(args: argparse.Namespace) -> None:
    """Main."""
    model = EmmaPolicy(args.model_name).load_from_checkpoint(args.ckpt_path)
    if args.max_length is not None:
        model.emma.config.max_length = args.max_length
        log.info(f"Max number of generated tokens set to: {model.emma.config.max_length}")
    if args.min_length is not None:
        model.emma.config.max_length = args.max_length
        log.info(f"Min number of generated tokens set to: {model.emma.config.min_length}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model.eval()

    with gr.Blocks(css=css) as block:
        with gr.Row():
            input_video = gr.inputs.Video(label="Input Video \U0001F4F7")
            with gr.Column():
                dropdown_prompts = gr.Dropdown(
                    label="Example Input Prompts \U0001F4E3", choices=input_prompts
                )
                with gr.Row():
                    input_text = gr.Textbox(label="Input Text \U0000270D", interactive=True)
        with gr.Row():
            model_button = gr.Button("Run model", elem_id="button_style")
        with gr.Row():
            out_text = gr.Label(label="Output Text", interactive=False)

        example_template_path = str(Path(args.root_examples_path, "videos", "*.mp4"))
        example_names = glob.glob(example_template_path)

        gr.Examples(
            label="Video examples",
            examples=example_names,
            inputs=[input_video],
        )

        with open(Path(args.root_examples_path, "dict_map.json")) as fp:
            video_id_map = json.load(fp)

        dropdown_prompts.change(fn=change_textbox, inputs=dropdown_prompts, outputs=input_text)
        model_button.click(
            run_model(
                root_examples_path=args.root_examples_path,
                video_id_map=video_id_map,
                model=model,
                tokenizer=tokenizer,
            ),
            [input_text, input_video],
            out_text,
        )
        block.launch(share=args.share)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        help="Model name",
        default="heriot-watt/emma-small",
    )
    parser.add_argument(
        "--ckpt_path",
        help="Path to model_ckpt",
    )
    parser.add_argument(
        "--root_examples_path",
        required=True,
        help="Path to examples folder containing images",
    )
    parser.add_argument(
        "--max_length",
        help="Optionally override the max length of the generated sequence of the model config",
    )
    parser.add_argument(
        "--min_length",
        help="Optionally override the min length of the generated sequence of the model config",
    )
    parser.add_argument(
        "--share",
        help="Create a publicly shareable link from your computer for the interface",
        action="store_true",
    )
    args = parser.parse_args()
    main(args)
