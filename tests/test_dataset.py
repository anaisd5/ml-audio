import numpy as np

from ml_audio.dataset import (
    PAD_VALUE,
    SEGMENT_WIDTH,
    GTZANDataset,
    load_split,
    spec_augment,
    split_into_segments,
)


def test_split_into_segments_shape():
    """
    Check that a scalogram of a whole track (30 s) is cut into
    10 segments of SEGMENT_WIDTH frames.
    """
    scalogram = np.random.randn(84, 1293)

    segments = split_into_segments(scalogram)

    assert segments.shape == (10, 84, SEGMENT_WIDTH)
    # The first segment is the beginning of the scalogram
    assert np.array_equal(segments[0], scalogram[:, :SEGMENT_WIDTH])
    # The second segment follows the first one
    assert np.array_equal(
        segments[1], scalogram[:, SEGMENT_WIDTH : 2 * SEGMENT_WIDTH]
    )


def test_split_into_segments_padding():
    """
    Check that a scalogram shorter than one segment is padded
    with silence (PAD_VALUE) and not with 0 dB (loudest value).
    """
    scalogram = np.zeros((84, 50))

    segments = split_into_segments(scalogram)

    assert segments.shape == (1, 84, SEGMENT_WIDTH)
    assert (segments[0, :, 50:] == PAD_VALUE).all()


def test_splits_are_disjoint():
    """
    Check that the fault-filtered splits are loaded and that a track
    is never in two splits.
    """
    train = load_split("train")
    valid = load_split("valid")
    test = load_split("test")

    assert len(train) == 443
    assert len(valid) == 197
    assert len(test) == 290
    assert not train & valid
    assert not train & test
    assert not valid & test


def test_spec_augment():
    """
    Check that SpecAugment keeps the shape, only hides values (set to 0)
    and does not modify the original segment.
    """
    segment = np.random.randn(84, SEGMENT_WIDTH) + 10  # no value is 0

    augmented = spec_augment(segment, num_masks=1, time_mask_width=20)

    assert augmented.shape == segment.shape
    # The original segment is not modified
    assert (segment != 0).all()
    # The values which are not hidden are unchanged
    visible = augmented != 0
    assert np.array_equal(augmented[visible], segment[visible])


def test_dataset_augment(tmp_path):
    """
    Check that the dataset gives segments of the right shape with and
    without augmentation, and that the segments without augmentation
    are always the same.
    """
    (tmp_path / "blues").mkdir()
    np.save(tmp_path / "blues" / "blues.00000.npy", np.random.randn(84, 1293))

    dataset = GTZANDataset(tmp_path)
    augmented_dataset = GTZANDataset(tmp_path, augment=True)

    assert len(dataset) == 10
    data, label = dataset[3]
    augmented_data, _ = augmented_dataset[3]
    assert data.shape == (1, 84, SEGMENT_WIDTH)
    assert augmented_data.shape == (1, 84, SEGMENT_WIDTH)
    assert label == 0
    # Without augmentation, the result is always the same
    assert (dataset[3][0] == data).all()
