import argparse
import json
from base64 import b64decode
from io import BytesIO
from pathlib import Path

from PIL import Image


def decode_images(encoded_images: dict[str, str]) -> dict[str, Image.Image]:
    """Decode the images from metadata dictionary."""
    ordered_encoded_images = sorted(encoded_images.items())
    decoded_images = {
        image_key: Image.open(BytesIO(b64decode(image_str)))
        for image_key, image_str in ordered_encoded_images
    }
    return decoded_images


def decode_images_for_file(input_json_path: Path, output_image_directory: Path) -> None:
    """Decode the images for a single json file."""
    with open(input_json_path) as fp:
        data = json.load(fp)

    decoded_images = decode_images(data["encoded_images"])
    for image_key, image in decoded_images.items():
        image_path = output_image_directory.joinpath(f"{input_json_path.stem}_{image_key}.png")
        image.save(image_path)


def main(input_json_directory: Path, output_image_directory: Path) -> None:
    """Iterates over all json files and decodes the images."""
    output_image_directory.mkdir(parents=True, exist_ok=True)

    for input_json_path in input_json_directory.iterdir():
        if not input_json_path.name.endswith(".json"):
            continue

        decode_images_for_file(input_json_path, output_image_directory)


def parse_api_args() -> argparse.Namespace:
    """Parse any arguments."""
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--input_json_directory",
        type=Path,
        required=True,
        help="Path to the directory containing json files for a session",
    )
    arg_parser.add_argument(
        "--output_image_directory",
        type=Path,
        required=True,
        help="Path to output image directory",
    )
    return arg_parser.parse_args()


if __name__ == "__main__":
    args = parse_api_args()
    main(args.input_json_directory, args.output_image_directory)
