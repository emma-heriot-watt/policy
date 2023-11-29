import argparse
import json
import logging
from collections import defaultdict
from collections.abc import Collection
from pathlib import Path
from typing import Any

import gradio as gr
import plotly
from plotly.subplots import make_subplots
from tqdm import tqdm
from transformers import AutoTokenizer

from emma_policy.datamodules.simbot_action_dataset import SimBotActionDataset
from emma_policy.datamodules.simbot_cr_dataset import SimBotCRDataset


logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

data_type = dict[str, Collection[Any]]


class DatasetVisualizer:
    """Visualize the SimBot dataset distribution."""

    def __init__(self, data: data_type, total_examples: int) -> None:
        self.data = data
        self.total_examples = total_examples
        self._accent_color = "#ffb400"  # purple
        self._base_color = "#9080ff"  # mustard

    def get_data_visualization(self, subset_name: str = "") -> plotly.graph_objs.Figure:
        """Prepare the output for a subset of the data."""
        data_subset = self.data.get(subset_name, None)
        fig = make_subplots(rows=2, cols=1, specs=[[{"type": "histogram"}], [{"type": "pie"}]])
        if not data_subset:
            return fig
        fig.append_trace(
            plotly.graph_objects.Histogram(
                x=data_subset, showlegend=False, marker={"color": self._accent_color}
            ),
            row=1,
            col=1,
        )
        fig.append_trace(
            plotly.graph_objects.Pie(
                values=[len(data_subset), self.total_examples - len(data_subset)],
                labels=[subset_name, "other"],
                marker={"colors": [self._accent_color, self._base_color]},
            ),
            row=2,
            col=1,
        )
        return fig


def get_data_from_action_dataset(args: argparse.Namespace) -> dict[str, Any]:
    """Get the visualization data from the action dataset."""
    train_dataset = SimBotActionDataset(
        dataset_db_path=args.dataset_db,
        tokenizer=AutoTokenizer.from_pretrained("heriot-watt/emma-base"),
    )
    data = []
    data_per_object = defaultdict(list)
    data_per_action = defaultdict(list)

    for index, instance in tqdm(enumerate(train_dataset)):  # type: ignore[arg-type]
        data.append(instance.raw_target["action_type"])
        if instance.raw_target["object_type"] is None:
            continue
        data_per_object[instance.raw_target["object_type"]].append(
            instance.raw_target["action_type"]
        )
        data_per_action[instance.raw_target["action_type"]].append(
            instance.raw_target["object_type"]
        )

        if index > len(train_dataset) - 1:
            break
    data_dict = {"overall": data, "per_object": data_per_object, "per_action": data_per_action}
    with open(args.cache_dir, "w") as file_out:
        json.dump(data_dict, file_out)
    return data_dict


def get_data_from_cr_dataset(args: argparse.Namespace) -> dict[str, Any]:
    """Get the visualization data from the CR dataset."""
    train_dataset = SimBotCRDataset(
        dataset_db_path=args.dataset_db,
        tokenizer=AutoTokenizer.from_pretrained("heriot-watt/emma-base"),
        is_train=True,
    )
    data = []
    data_per_object = defaultdict(list)
    data_per_action = defaultdict(list)

    for index, instance in tqdm(enumerate(train_dataset)):  # type: ignore[arg-type]
        data.append(instance.raw_target["cr_class"])
        data_per_object[instance.raw_target["object_type"]].append(instance.raw_target["cr_class"])
        data_per_action[instance.raw_target["action_type"]].append(instance.raw_target["cr_class"])

        if index == len(train_dataset) - 1:
            break
    data_dict = {"overall": data, "per_object": data_per_object, "per_action": data_per_action}
    with open(args.cache_dir, "w") as file_out:
        json.dump(data_dict, file_out)
    return data_dict


def get_data_for_visualization(args: argparse.Namespace) -> dict[str, Any]:
    """Get the data for the visualization."""
    if args.cache_dir.exists():
        with open(args.cache_dir) as file_in:
            return json.load(file_in)
    elif args.dataset_type == "cr":
        return get_data_from_cr_dataset(args)
    return get_data_from_action_dataset(args)


def main(args: argparse.Namespace) -> None:
    """Main."""
    data = get_data_for_visualization(args)
    total_examples = len(data["overall"])
    object_visualizer = DatasetVisualizer(data["per_object"], total_examples=total_examples)
    action_visualizer = DatasetVisualizer(data["per_action"], total_examples=total_examples)
    with gr.Blocks() as block:
        with gr.Row():
            gr.Plot(
                plotly.graph_objects.Figure(
                    data=[
                        plotly.graph_objects.Histogram(
                            x=data["overall"], marker={"color": "#9080ff"}
                        )
                    ]
                ),
                label="Overall Label Distribution",
            )
        with gr.Row():
            object_types = sorted(set(data["per_object"].keys()))
            input_object = gr.Dropdown(object_types, label="Object")
            object_plot = gr.Plot(label="Distribution per Object")
        with gr.Row():
            action_types = sorted(set(data["per_action"].keys()))
            input_action = gr.Dropdown(action_types, label="Action")
            action_plot = gr.Plot(label="Distribution per Action")

        input_object.change(
            fn=object_visualizer.get_data_visualization,
            inputs=[input_object],
            outputs=[object_plot],
        )
        input_action.change(
            fn=action_visualizer.get_data_visualization,
            inputs=[input_action],
            outputs=[action_plot],
        )
        block.launch(share=args.share)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_db",
        default=Path("storage/db/simbot_actions_train.db"),
        type=Path,
        help="Path the simbot dataset db.",
    )
    parser.add_argument(
        "--cache_dir",
        default=Path("storage/db/action_app_cache1.json"),
        type=Path,
        help="Path the simbot dataset cache.",
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        choices=["cr", "action"],
        help="Type of the dataset",
    )
    parser.add_argument(
        "--share",
        help="Create a publicly shareable link from your computer for the interface",
        action="store_true",
    )
    args = parser.parse_args()
    main(args)
