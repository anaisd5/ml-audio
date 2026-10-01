import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# Logging
logger = logging.getLogger(__name__)

# --- Parameters ---

# Width of a segment (in frames): with sr=22050 and hop_length=512,
# 129 frames correspond to about 3 seconds of audio
SEGMENT_WIDTH = 129
# Value used for padding: -80 dB is the silence (0 dB is the loudest value)
PAD_VALUE = -80.0
# Folder containing the lists of the fault-filtered split
SPLITS_DIR = Path(__file__).parent / "splits"

# --- Data augmentation parameters (SpecAugment) ---
NUM_MASKS = 2  # Number of masks of each type (frequency and time)
FREQ_MASK_WIDTH = 8  # Maximal number of frequency bins hidden by a mask
TIME_MASK_WIDTH = 20  # Maximal number of frames hidden by a mask


def load_split(split_name):
    """
    Load the names of the tracks of a split of the fault-filtered
    GTZAN partition (Sturm, 2013; Kereliuk et al., 2015). This partition
    removes duplicates and prevents an artist from being in two
    different splits.

    :param split_name: name of the split ('train', 'valid' or 'test')
    :type split_name: str
    :raises FileNotFoundError: if the split file does not exist
    :returns: the names of the tracks (e.g. 'blues.00029')
    :rtype: set[str]
    """

    split_path = SPLITS_DIR / f"{split_name}_filtered.txt"
    with open(split_path, "r") as f:
        # Each line looks like 'blues/blues.00029.wav'
        return {Path(line.strip()).stem for line in f if line.strip()}


def split_into_segments(scalogram, segment_width=SEGMENT_WIDTH):
    """
    Cut a scalogram into consecutive segments of the same width.
    The end of the scalogram that does not fill a whole segment is
    dropped. A scalogram shorter than one segment is padded with
    PAD_VALUE.

    :param scalogram: the scalogram (Height, Width)
    :type scalogram: numpy.ndarray
    :param segment_width: the width of a segment (in frames)
    :type segment_width: int
    :returns: the segments (Number of segments, Height, segment_width)
    :rtype: numpy.ndarray
    """

    current_width = scalogram.shape[1]

    if current_width < segment_width:
        # Too short: add silence at the end (padding)
        pad_width = segment_width - current_width
        # padding((top, bottom), (left, right))
        scalogram = np.pad(
            scalogram,
            ((0, 0), (0, pad_width)),
            mode="constant",
            constant_values=PAD_VALUE,
        )

    num_segments = scalogram.shape[1] // segment_width
    # Only keep the frames filling whole segments
    scalogram = scalogram[:, : num_segments * segment_width]

    # (H, N * W) -> (H, N, W) -> (N, H, W)
    height = scalogram.shape[0]
    segments = scalogram.reshape(height, num_segments, segment_width)
    return segments.transpose(1, 0, 2)


def spec_augment(
    segment,
    num_masks=NUM_MASKS,
    freq_mask_width=FREQ_MASK_WIDTH,
    time_mask_width=TIME_MASK_WIDTH,
):
    """
    Data augmentation (SpecAugment, Park et al., 2019): hide random bands
    of frequencies and random intervals of time. The model learns not to
    rely on a single detail of the scalogram. The hidden values are
    replaced by 0, the mean of a standardised scalogram.

    :param segment: the standardised segment (Height, Width)
    :type segment: numpy.ndarray
    :param num_masks: number of masks of each type (frequency and time)
    :type num_masks: int
    :param freq_mask_width: maximal number of frequency bins of a mask
    :type freq_mask_width: int
    :param time_mask_width: maximal number of frames of a mask
    :type time_mask_width: int
    :returns: a copy of the segment with the masks
    :rtype: numpy.ndarray
    """

    # torch.randint is used (and not numpy) since PyTorch gives a
    # different seed to each process loading the data
    segment = segment.copy()
    height, width = segment.shape

    for _ in range(num_masks):
        # Frequency mask: rows f0 to f0 + f are hidden
        f = torch.randint(0, freq_mask_width + 1, (1,)).item()
        f0 = torch.randint(0, height - f + 1, (1,)).item()
        segment[f0 : f0 + f, :] = 0.0

        # Time mask: columns t0 to t0 + t are hidden
        t = torch.randint(0, time_mask_width + 1, (1,)).item()
        t0 = torch.randint(0, width - t + 1, (1,)).item()
        segment[:, t0 : t0 + t] = 0.0

    return segment


class GTZANDataset(Dataset):
    def __init__(
        self,
        data_dir,
        track_names=None,
        segment_width=SEGMENT_WIDTH,
        mean=0.0,
        std=1.0,
        augment=False,
    ):
        """
        Function called at initialisation. Each scalogram is cut into
        segments of segment_width frames: an item of the dataset is a
        segment, not a whole track.

        :param data_dir: path to the folder containing the
                         .npy files (ex: data/processed/scalograms)
        :type data_dir: str | Path
        :param track_names: names of the tracks to keep (e.g. the result
                            of load_split). If None, all tracks are kept.
        :type track_names: set[str] | None
        :param segment_width: the width of a segment (in frames)
        :type segment_width: int
        :param mean: mean used to standardise the scalograms
        :type mean: float
        :param std: standard deviation used to standardise the scalograms
        :type std: float
        :param augment: if True, data augmentation is applied (random
                        crop in time and SpecAugment). It should only be
                        used for the training set.
        :type augment: bool
        :raises RuntimeError: if no .npy file is found in data_dir
        :returns: None
        """

        # Store the parameters
        self.data_dir = Path(data_dir)
        self.segment_width = segment_width
        self.mean = mean
        self.std = std
        self.augment = augment

        # Find all .npy files

        # .rglob("*.npy") searches recursively in all subfolders
        # sorted ensures the order is always the same
        all_files = sorted(self.data_dir.rglob("*.npy"))

        if not all_files:
            raise RuntimeError(f"No .npy file found in {data_dir}")

        # Create labels from names of parent files (blues, rock...)

        # The classes are computed on all files, so that the mapping
        # is the same for every split (blues=0, classical=1...)
        self.classes = sorted(list(set(f.parent.name for f in all_files)))
        # create the 'translation' dictionary
        self.class_to_idx = {
            cls_name: i for i, cls_name in enumerate(self.classes)
        }

        # Only keep the tracks of the split
        if track_names is None:
            self.files = all_files
        else:
            self.files = [f for f in all_files if f.stem in track_names]
            missing = set(track_names) - {f.stem for f in self.files}
            for name in sorted(missing):
                logger.warning(f"Track {name} not found in {data_dir}")

        # List all segments as (index of the file, index of the segment)
        self.segments = []
        for file_idx, file_path in enumerate(self.files):
            # mmap_mode only reads the header of the file (fast)
            width = np.load(file_path, mmap_mode="r").shape[1]
            num_segments = max(1, width // self.segment_width)
            for segment_idx in range(num_segments):
                self.segments.append((file_idx, segment_idx))

        logger.info(
            f"Dataset loaded : {len(self.files)} files, "
            f"{len(self.segments)} segments."
        )
        logger.info(f"Classes found : {self.classes}")

    def __len__(self):
        """
        Give the number of samples (segments of scalograms).

        :returns: number of samples
        :rtype: int
        """
        return len(self.segments)

    def get_label(self, file_idx):
        """
        Get the label of a file based on its index.

        :param file_idx: index of the file in self.files
        :type file_idx: int
        :returns: the label as an integer
        :rtype: int
        """
        label_name = self.files[file_idx].parent.name
        return self.class_to_idx[label_name]  # conversion into integer

    def __getitem__(self, idx):
        """
        Get an item (segment of scalogram) based on its index.

        :param idx: index of the item to retrieve
        :type idx: int
        :returns: a tuple (data_tensor, label)
                  - data_tensor: the segment as a PyTorch tensor
                  - label: the corresponding label as an integer
        :rtype: tuple[torch.Tensor, int]
        """

        # Get the file and the position of the segment
        file_idx, segment_idx = self.segments[idx]

        # Load data (mmap_mode only reads the segment from the disk)
        scalogram = np.load(self.files[file_idx], mmap_mode="r")

        if self.augment:
            # Random crop: the segment can start anywhere in the track
            max_start = max(0, scalogram.shape[1] - self.segment_width)
            start = torch.randint(0, max_start + 1, (1,)).item()
        else:
            start = segment_idx * self.segment_width
        segment = scalogram[:, start : start + self.segment_width]

        # Padding for size standardisation (if the track is too short)
        segment = split_into_segments(np.array(segment), self.segment_width)[0]

        # Standardisation (mean 0 and standard deviation 1)
        segment = (segment - self.mean) / self.std

        # Hide random frequencies and times
        if self.augment:
            segment = spec_augment(segment)

        # Convert into a PyTorch tensor

        # unsqueeze add a dimension (of 1) at the beginning
        # since the model expects (Channel, Height, Width)
        # and the tensors are (Height, Width)
        data_tensor = torch.from_numpy(segment).float().unsqueeze(0)

        return data_tensor, self.get_label(file_idx)


def compute_mean_std(dataset):
    """
    Compute the mean and standard deviation of all the values of the
    scalograms of a dataset. It should only be called on the training
    set (to avoid information leaks from the validation and test sets).

    :param dataset: the dataset
    :type dataset: GTZANDataset
    :returns: a tuple (mean, std)
    :rtype: tuple[float, float]
    """

    total = 0.0
    total_squared = 0.0
    count = 0

    for file_path in dataset.files:
        scalogram = np.load(file_path).astype(np.float64)
        total += scalogram.sum()
        total_squared += (scalogram**2).sum()
        count += scalogram.size

    mean = total / count
    std = np.sqrt(total_squared / count - mean**2)
    return float(mean), float(std)
