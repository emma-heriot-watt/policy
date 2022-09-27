from pytest_cases import parametrize

from emma_policy.utils.masks import compress_simbot_mask, decompress_simbot_mask


@parametrize(
    "segmentation_mask",
    [
        [
            [49437, 1],
            [49741, 5],
            [50037, 10],
            [50063, 4],
            [50337, 18],
            [50360, 10],
            [50374, 8],
            [50638, 32],
            [50675, 22],
            [50938, 1],
            [50953, 47],
            [51238, 1],
            [51267, 33],
            [51538, 1],
            [51579, 21],
            [51838, 1],
            [51889, 11],
            [52138, 1],
            [52198, 2],
            [52438, 1],
            [52738, 1],
            [53338, 1],
            [53638, 1],
            [53938, 1],
        ],
    ],
)
def test_segmentation_simbot_mask(
    segmentation_mask: list[list[int]],
) -> None:
    """Test that decompressing and compressing a mask results in identical output."""
    decompressed_mask = decompress_simbot_mask(segmentation_mask, return_tensor=False)
    compressed_mask = compress_simbot_mask(decompressed_mask)  # type: ignore[arg-type]
    assert compressed_mask == segmentation_mask
