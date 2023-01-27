import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from emma_datasets.db import DatasetDb
from matplotlib import pyplot as plt
from tqdm import tqdm


temperatures = [1, 1.3, 1.5, 1.7, 2]


def get_action_type(action: dict[str, Any], separate_gotos: bool = False) -> str:
    """Get the action type from an action dict."""
    if action["type"] != "Goto":
        return action["type"]
    if "officeRoom" in action["goto"]["object"]:
        if separate_gotos:
            return "Goto-Room"
        return "Goto"
    if separate_gotos:
        return "Goto-Object"
    return "Goto"


def main(  # noqa: WPS213
    input_db: Path,
    output_image_directory: Path,
    show_plots: bool = False,
    separate_gotos: bool = False,
    xtick_label_rotation: int = 45,
    bar_width: float = 0.8,
) -> None:
    """Main."""
    output_image_directory.mkdir(parents=True, exist_ok=True)
    db = DatasetDb(input_db)
    actions = []
    action_to_indices = defaultdict(list)
    for data_index in tqdm(range(len(db))):
        data = json.loads(db[data_index])
        action_type = get_action_type(data["actions"][-1], separate_gotos=separate_gotos)
        actions.append(action_type)
        action_to_indices[action_type].append(data_index)

    action_counts = Counter(actions).most_common()

    plt.bar([cc[0] for cc in action_counts], [cc[1] for cc in action_counts], bar_width)
    plt.xlabel("Actions")
    plt.ylabel("Counts")
    plt.title("Action counts in db")
    plt.grid(True)
    plt.xticks(rotation=xtick_label_rotation)
    plt.tight_layout()
    plt.savefig(output_image_directory.joinpath("action_dist.png"))
    if show_plots:
        plt.show()

    cc = np.array([cc[1] for cc in action_counts])
    probas = 1 / np.array(cc)
    action_types = [cc[0] for cc in action_counts]
    for temperature in temperatures:
        temperature_weights = probas ** (1 / temperature)
        probas_weighted_by_temperature = temperature_weights / temperature_weights.sum()
        expected_samples = cc * probas_weighted_by_temperature

        plt.figure()
        plt.bar(action_types, expected_samples, bar_width)
        plt.xlabel("Actions")
        plt.ylabel("Counts")
        plt.title(f"Action probas for T={temperature} in training data")
        plt.grid(True)
        plt.xticks(rotation=xtick_label_rotation)
        plt.tight_layout()
        plt.savefig(
            output_image_directory.joinpath(f"action_probas_temperature_{temperature}.png")
        )
        if show_plots:
            plt.show()


def parse_args() -> argparse.Namespace:
    """Parse any arguments."""
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--input_db",
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
    arg_parser.add_argument(
        "--show_plots",
        action="store_true",
        help="Path to output image directory",
    )
    arg_parser.add_argument(
        "--separate_gotos",
        action="store_true",
        help="Path to output image directory",
    )
    return arg_parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        input_db=args.input_db,
        output_image_directory=args.output_image_directory,
        show_plots=args.show_plots,
        separate_gotos=args.separate_gotos,
    )
