import argparse
import json
from pathlib import Path
from re import sub

import spacy
import torch
from scipy.spatial.distance import cdist

from emma_policy.common.settings import Settings


settings = Settings()
AI2THOR_CLASS_DICT_FILE = settings.paths.constants.joinpath("ai2thor_labels.json")


def splitted_object_name(identifier: str) -> str:
    """Split a action to lower case words."""
    # Split camel case
    matches = sub(
        "([A-Z][a-z]+)",
        r" \1",
        sub("([A-Z]+)", r" \1", identifier),
    )
    return " ".join([match.lower() for match in matches.split()])


def cache_ai2thor_object_vectors_and_similarities(
    output_path: Path, vector_dim: int = 300
) -> None:
    """Get the word2vec vectors for AI2THOR object names."""
    with open(AI2THOR_CLASS_DICT_FILE) as in_file:
        indices_labels_map = json.load(in_file)["idx_to_label"]
    nlp = spacy.load("en_core_web_lg")
    num_objects = len(indices_labels_map)
    # Sort labels according to their index
    sorted_labels = [indices_labels_map[str(other_index)] for other_index in range(num_objects)]
    # Get word2vec vectors for labels
    vectors = torch.zeros((num_objects, vector_dim))
    for index, label in enumerate(sorted_labels):
        object_name = nlp(splitted_object_name(label))
        vectors[index] = torch.tensor(object_name.vector)

    # Get the cosine similarities between labels
    similarities = torch.tensor(1 - cdist(vectors, vectors, "cosine"))
    torch.save({"vectors": vectors, "similarities": similarities}, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cache the word embeddings for ai2thor objects and their similariies."
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        help="Path where the outputs will be cached",
    )
    parser.add_argument(
        "--vector_dim",
        type=int,
        default=300,  # noqa: WPS432
        help="Dimensionality of word vectors",
    )
    args = parser.parse_args()
    cache_ai2thor_object_vectors_and_similarities(
        args.output_file,
        args.vector_dim,
    )
