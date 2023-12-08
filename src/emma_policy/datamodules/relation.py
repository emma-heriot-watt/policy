from dataclasses import dataclass

from emma_datasets.datamodels import Region


@dataclass
class Relation:
    """Relation from scene_graph."""

    subject_attr: list[str]
    subject: Region
    predicate: str
    object: Region
    object_attr: list[str]
