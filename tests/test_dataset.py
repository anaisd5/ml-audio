import numpy as np

from ml_audio.dataset import (
    PAD_VALUE,
    SEGMENT_WIDTH,
    load_split,
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
