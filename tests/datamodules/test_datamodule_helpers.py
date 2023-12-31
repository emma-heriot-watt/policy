import torch
from pytest_cases import parametrize

from emma_policy.datamodules.pretrain_dataset import split_action_name
from emma_policy.datamodules.pretrain_instances import get_validation_coco_ids
from emma_policy.datamodules.simbot_action_dataset import mask_past_target_actions
from emma_policy.models.tokenizer_emma import EmmaTokenizer
from emma_policy.utils import DistributedWeightedSampler


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


@parametrize(
    "target_text, action_text",
    [
        ("pick up bowl.", "pick up bowl."),
        ("look around.", "look around."),
        ("pick up bowl. look around.", " look around."),
        ("pick up bowl. look around. move forward.", " move forward."),
    ],
)
def test_simbot_target_tokens(
    target_text: str,
    action_text: str,
    emma_tokenizer: EmmaTokenizer,
) -> None:
    target_encoding = emma_tokenizer.encode_plus(target_text, return_tensors="pt", truncation=True)
    full_target_token_ids = target_encoding.input_ids.squeeze(0)
    target_token_ids = mask_past_target_actions(
        full_target_token_ids,
        emma_tokenizer.sep_token_id,  # type: ignore[arg-type]
    )
    start_index = 1
    if not torch.equal(full_target_token_ids, target_token_ids):
        start_index += torch.where(target_token_ids < 0)[0][-1]  # type: ignore[assignment]
    check_tokens = target_token_ids[start_index:-1]
    assert emma_tokenizer.decode(check_tokens) == action_text


@parametrize("len_weights", [2, 3, 4, 5])
def test_distributed_weighted_sampler(len_weights: int) -> None:
    """Test that the distributed weighted sampler will only sample from positive weights."""
    # Get a random binary vector of size len_weights
    weights = torch.bernoulli(torch.randn(len_weights).uniform_(0, 1)).tolist()
    while sum(weights) == 0:
        weights = torch.bernoulli(torch.randn(len_weights).uniform_(0, 1)).tolist()
    sampler = DistributedWeightedSampler(weights, len(weights), num_replicas=1, rank=0)
    total_iterations = len_weights * 10
    # All samples should have weights > 0
    for idx in sampler:
        assert weights[idx]
        total_iterations -= 1
        if not total_iterations:
            break
