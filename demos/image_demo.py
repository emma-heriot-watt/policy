import argparse
import glob
import logging
import os
from io import BytesIO
from pathlib import Path
from typing import Any, Callable

import gradio as gr
import numpy as np
import requests
import torch
from PIL import Image
from transformers import AutoTokenizer, PreTrainedTokenizer

from emma_policy.commands.plot_bb import PlotBoundingBoxes
from emma_policy.datamodules.batch_attention_masks import make_text_history_global_pattern
from emma_policy.datamodules.emma_dataclasses import EmmaDatasetBatch
from emma_policy.models.emma_policy import EmmaPolicy


logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


input_prompts = [
    "Assess the statement: A cat standing on a sofa",
    "Evaluate the description: A dog catching a frisbee",
    "Find the white spoon",
    "Locate the object brown dog",
    "Caption <vis_token_3>",
    "Describe object <vis_token_4>",
    "Caption this image",
    "Describe this image",
    "Answer the question: What color is the ball?",
    "What is the answer to the question: What material is the table made of?",
    "Describe the relationship between <vis_token_1> and <vis_token_2>",
    "Describe how <vis_token_2> relates to <vis_token_3>",
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
    model: EmmaPolicy,
    tokenizer: PreTrainedTokenizer,
    endpoint: str,
) -> Callable[[str, str], str]:
    """Prepare the response generation."""

    def generate_response(input_text: str, input_image_path: str) -> str:  # noqa: WPS430
        """Generate a response to the text input."""
        input_image = Image.open(input_image_path)
        feature_dict = extract_single_image(input_image, endpoint=endpoint)

        vis_tokens = torch.tensor(
            tokenizer.convert_tokens_to_ids(
                [f"<vis_token_{idx+1}>" for idx in range(feature_dict["bbox_features"].shape[0])]
            ),
            dtype=torch.long,
        )
        width, height = input_image.size
        object_coordinates = feature_dict["bbox_coords"]

        object_coordinates[:, (0, 2)] /= width
        object_coordinates[:, (1, 3)] /= height

        feature_dict["visual_tokens"] = vis_tokens
        frame_token = tokenizer.convert_tokens_to_ids(f"<frame_token_{1}>")
        feature_dict["obj_frame_tokens"] = vis_tokens.new_full(
            vis_tokens.shape, fill_value=frame_token  # type: ignore[arg-type]
        )
        feature_dict["scene_attention_mask"] = torch.ones(1, dtype=torch.bool)
        feature_dict["scene_coordinates"] = torch.tensor([0, 0, 1.0, 1.0]).repeat(1, 1)
        feature_dict["scene_frame_tokens"] = torch.tensor(frame_token)
        feature_dict["object_attention_mask"] = torch.ones_like(vis_tokens, dtype=torch.bool)

        inputs = tokenizer.encode_plus(input_text, return_tensors="pt")
        log.info(f"Input text: {input_text} Input token ids: {inputs.input_ids}")
        attention_mask = torch.cat(
            [
                feature_dict["scene_attention_mask"],
                feature_dict["object_attention_mask"],
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
            object_attention_mask=feature_dict["object_attention_mask"].unsqueeze(0),
            object_coordinates=feature_dict["bbox_coords"].unsqueeze(0),
            object_frame_tokens=feature_dict["obj_frame_tokens"].unsqueeze(0),
            object_features=feature_dict["bbox_features"].unsqueeze(0),
            scene_attention_mask=feature_dict["scene_attention_mask"].unsqueeze(0),
            scene_coordinates=feature_dict["scene_coordinates"].unsqueeze(0),
            scene_features=feature_dict["cnn_features"].unsqueeze(0),
            scene_frame_tokens=feature_dict["scene_frame_tokens"].unsqueeze(0),
            visual_token_ids=feature_dict["visual_tokens"].unsqueeze(0),
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
        log.info(f"Response: {response}")
        return response

    return generate_response


def run_bboxes(endpoint: str) -> Callable[[str], Image.Image]:
    """Prepare bboxes in output plot."""

    def _plot_bboxes(input_image_path: str) -> Image:  # noqa: WPS430
        """Plot the bounding boxes on image."""
        input_image = Image.open(input_image_path)
        feature_dict = extract_single_image(input_image, endpoint=endpoint)

        cv_image = np.array(input_image)
        num_bboxes = feature_dict["bbox_features"].shape[0]
        vis_tokens = [f"<vis_token_{idx+1}>" for idx in range(num_bboxes)]
        PlotBoundingBoxes().draw_bb(
            image=cv_image,
            boxes_coords=feature_dict["bbox_coords"].numpy(),
            boxes_labels=vis_tokens,
            draw_label=True,
        )

        return Image.fromarray(cv_image)

    return _plot_bboxes


def extract_single_image(
    image: Image, endpoint: str = "http://0.0.0.0:5500/features"
) -> dict[str, torch.Tensor]:
    """Submit a request to the feature extraction server for a single image."""
    image_bytes = convert_single_image_to_bytes(image)
    request_files = {"input_file": image_bytes}
    response = requests.post(endpoint, files=request_files, timeout=5)
    log.info(f"Response: {response}")

    data = response.json()
    feature_response = {
        "bbox_features": torch.tensor(data["bbox_features"]),
        "bbox_coords": torch.tensor(data["bbox_coords"]),
        "bbox_probas": torch.tensor(data["bbox_probas"]),
        "cnn_features": torch.tensor(data["cnn_features"]),
    }

    return feature_response


def convert_single_image_to_bytes(image: Image) -> bytes:
    """Converts a single image to bytes."""
    image_bytes = BytesIO()

    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.array(image))

    image.save(image_bytes, format=image.format)
    return image_bytes.getvalue()


def main(args: argparse.Namespace) -> None:
    """Main."""
    model = EmmaPolicy(args.model_name).load_from_checkpoint(args.ckpt_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model.eval()

    with gr.Blocks(css=css) as block:
        with gr.Row():
            input_image = gr.inputs.Image(label="Input Image \U0001F4F7", type="filepath")
            with gr.Column():
                dropdown_prompts = gr.Dropdown(
                    label="Example Input Prompts \U0001F4E3", choices=input_prompts
                )
                with gr.Row():
                    input_text = gr.Textbox(label="Input Text \U0000270D", interactive=True)
        with gr.Row():
            with gr.Column():
                bboxes_button = gr.Button("Show Bounding Boxes", elem_id="button_style")
            with gr.Column():
                model_button = gr.Button("Run model", elem_id="button_style")
        with gr.Row():
            out_image = gr.inputs.Image(label="Output Image", type="filepath")
            out_text = gr.Label(label="Output Text", interactive=False)

        if args.examples_path is not None and os.path.exists(args.examples_path):
            example_names = (
                glob.glob(str(Path(args.examples_path, "*.jpg")))
                + glob.glob(str(Path(args.examples_path, "*.png")))
                + glob.glob(str(Path(args.examples_path, "*.jpeg")))
            )
            gr.Examples(
                label="Image examples",
                examples=example_names,
                inputs=[input_image],
            )
        dropdown_prompts.change(fn=change_textbox, inputs=dropdown_prompts, outputs=input_text)
        model_button.click(
            fn=run_model(
                model=model,
                tokenizer=tokenizer,
                endpoint=args.endpoint,
            ),
            inputs=[input_text, input_image],
            outputs=out_text,
        )
        bboxes_button.click(fn=run_bboxes(args.endpoint), inputs=input_image, outputs=out_image)
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
        "--examples_path",
        help="Path to examples folder containing images",
    )
    parser.add_argument(
        "--endpoint", help="Feature extraction endpoint", default="http://0.0.0.0:5500/features"
    )
    parser.add_argument(
        "--share",
        help="Create a publicly shareable link from your computer for the interface",
        action="store_true",
    )
    args = parser.parse_args()
    main(args)
