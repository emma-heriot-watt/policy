import argparse
import json
import logging
import os
import shutil
from typing import Any, Callable, Union

import gradio as gr
import numpy as np
import requests
import torch
from emma_datasets.common import get_progress
from emma_datasets.datamodels import Instance
from emma_datasets.db import DatasetDb
from pydantic import HttpUrl
from transformers import AutoTokenizer, PreTrainedTokenizer

from emma_policy.datamodules.emma_dataclasses import EmmaDatasetBatch
from emma_policy.datamodules.pretrain_instances import is_train_instance, load_ref_coco_images
from emma_policy.models.emma_policy import EmmaPolicy
from emma_policy.utils import get_logger


log = get_logger(__name__)

logging.getLogger("boto3").setLevel(logging.CRITICAL)
logging.getLogger("botocore").setLevel(logging.CRITICAL)
logging.getLogger("nose").setLevel(logging.CRITICAL)
logging.getLogger("s3transfer").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)


def load_image_features(
    features_path: str, tokenizer: PreTrainedTokenizer
) -> dict[str, torch.Tensor]:
    """Loads and prepares the image features."""
    feature_dict = torch.load(features_path)
    object_coordinates = feature_dict["bbox_coords"]
    # normalized coordinates
    object_coordinates[:, (0, 2)] /= feature_dict["width"]
    object_coordinates[:, (1, 3)] /= feature_dict["height"]
    feature_dict["bbox_coords"] = object_coordinates
    feature_dict["cnn_features"] = feature_dict["cnn_features"].unsqueeze(0)
    vis_tokens = torch.tensor(
        tokenizer.convert_tokens_to_ids(
            [f"<vis_token_{idx+1}>" for idx in range(feature_dict["bbox_features"].shape[0])]
        ),
        dtype=torch.long,
    )
    feature_dict["viz_tokens"] = vis_tokens
    frame_token = tokenizer.convert_tokens_to_ids(f"<frame_token_{1}>")
    feature_dict["obj_frame_tokens"] = vis_tokens.new_full(
        vis_tokens.shape, fill_value=frame_token  # type: ignore[arg-type]
    )
    num_frames = 1
    feature_dict["scene_attention_mask"] = torch.ones(num_frames, dtype=torch.bool)
    feature_dict["scene_coordinates"] = torch.tensor([0, 0, 1.0, 1.0]).repeat(num_frames, 1)
    feature_dict["scene_frame_tokens"] = torch.tensor(frame_token)
    feature_dict["object_attention_mask"] = torch.ones_like(vis_tokens, dtype=torch.bool)

    return feature_dict


def get_generate_response(
    model: EmmaPolicy,
    tokenizer: PreTrainedTokenizer,
    model_config: dict[str, Any],
) -> Callable[[Any, str, str], list[str]]:
    """Prepare the response generation."""

    def generate_response(  # noqa: WPS430
        image: Any, text_input: str, features_path: str
    ) -> list[str]:
        """Generate a response to the text input."""
        feature_dict = load_image_features(features_path, tokenizer)
        inputs = tokenizer.encode_plus(text_input, return_tensors="pt")
        attention_mask = torch.cat(
            [
                feature_dict["scene_attention_mask"],
                feature_dict["object_attention_mask"],
                inputs.attention_mask.squeeze(0),
            ],
            dim=-1,
        ).unsqueeze(0)

        global_attention_mask = torch.zeros_like(attention_mask)
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
            visual_token_ids=feature_dict["viz_tokens"].unsqueeze(0),
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
        )
        outputs = model.predict_step(sample, 0)

        preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        response = str(preds)
        response = response.replace("'", "")
        response = response.replace("<pad>", "")
        response = response.replace("<unk>", "")
        response = response.replace("</s>", "")
        response = response.replace("<s>", "")
        response = response.replace('"', "")
        response = response.replace("[", "")
        response = response.replace("]", "")
        return [response, json.dumps(model_config, indent=2)]

    return generate_response


def download_image(http_url: HttpUrl) -> str:
    """Download an image from url."""
    img_name = os.path.basename(http_url)
    image_path = os.path.join(args.local_path, img_name)
    # This is a hacky way to avoid blocking requests
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36"
    }

    r = requests.get(http_url, stream=True, headers=headers)
    if r.status_code != requests.codes.ok:
        return ""

    r.raw.decode_content = True

    # Open a local file with wb ( write binary ) permission.
    with open(image_path, "wb") as f:
        shutil.copyfileobj(r.raw, f)
    return image_path


def get_url_from_instance(instance: Instance) -> Union[None, HttpUrl]:
    """Get url from instance metadata."""
    http_url = None
    for dataset in instance.dataset.values():
        if dataset.media.url:
            http_url = dataset.media.url
            break
    return http_url


def get_dowloaded_image_path(instance: Instance, is_train: bool) -> Union[None, str]:
    """Download a validation image."""
    if is_train:
        return None
    http_url = get_url_from_instance(instance)
    if not http_url:
        return None
    image_path = download_image(http_url)
    return image_path


def prepare_examples(args: argparse.Namespace) -> list[list[str]]:
    """Select validation samples and download the corresponding images."""
    os.makedirs(args.local_path, exist_ok=True)
    ref_coco_images = load_ref_coco_images()
    examples = []
    progress = get_progress()
    with progress:
        with DatasetDb(args.input_db) as in_db:
            task_id = progress.add_task(
                f"Downloading images for {args.input_db}", total=len(in_db)
            )
            indices = np.arange(1, len(in_db))
            np.random.shuffle(indices)
            for idx in indices:
                data = in_db[int(idx)]
                instance = Instance.parse_raw(data)
                image_path = get_dowloaded_image_path(
                    instance, is_train_instance(ref_coco_images, instance)
                )

                if not image_path:
                    continue

                examples.append(
                    [
                        image_path,
                        "Example question: Is the photo black and white?",
                        str(instance.features_path),
                    ]
                )
                progress.advance(task_id)

                if len(examples) == args.total:
                    break
    return examples


def main(args: argparse.Namespace) -> None:
    """Launch a gradio application to inspect model outputs."""
    examples = prepare_examples(args)
    model = EmmaPolicy(args.model_name).load_from_checkpoint(args.ckpt_path)
    # model = pl.emma
    model_config = model.emma.config.to_dict()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model.eval()
    iface = gr.Interface(
        fn=get_generate_response(model=model, tokenizer=tokenizer, model_config=model_config),
        inputs=[gr.inputs.Image(type="pil", shape=(124, 124)), "text", "text"],
        outputs=["text", "text"],
        allow_screenshot=False,
        allow_flagging="never",
        examples=examples,
    )
    iface.launch(
        share=args.share,
    )


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
        "--input_db",
        help="Path to the input database",
        default="storage/fixtures/instances.db",
    )
    parser.add_argument(
        "--local_path",
        help="Path to local feature path",
        default="storage/tmp_images/",
    )
    parser.add_argument(
        "--share",
        help="Create a publicly shareable link from your computer for the interface",
        action="store_true",
    )
    parser.add_argument("--total", type=int, default=4, help="Total number of examples")
    args = parser.parse_args()
    main(args)
