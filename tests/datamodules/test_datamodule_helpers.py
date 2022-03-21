from pytest_cases import parametrize

from emma_policy.datamodules.pretrain_dataset import split_action_name
from emma_policy.datamodules.pretrain_instances import get_validation_coco_ids


def test_valid_ids_from_coco_load() -> None:
    valid_ids = get_validation_coco_ids()

    assert valid_ids


@parametrize(
    "action_name, action_text",
    [
        ("Pickup", "pick up"),
        ("PickupObject", "pick up"),
        ("Stop", "stop"),
        ("Move to", "move to"),
        ("Forward", "forward"),
        ("Turn Left", "turn left"),
        ("Look Up", "look up"),
        ("ToggleOn", "toggle on"),
        ("ToggleOff", "toggle off"),
        ("BehindAboveOn", "behind above on"),
        ("OpenProgressCheck", "open progress check"),
    ],
)
def test_convert_action_name_to_consistent_form(action_name: str, action_text: str) -> None:
    assert split_action_name(action_name) == action_text.split(" ")
