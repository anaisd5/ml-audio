import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# Logging
logger = logging.getLogger(__name__)


class GTZANDataset(Dataset):
    def __init__(self, data_dir, fixed_width=1280):
        """
        Function called at initialisation.

        :param data_dir: path to the folder containing the
                         .npy files (ex: data/processed/scalograms)
        :type data_dir: str | Path
        :param fixed_width: the fixed temporal width for all
                            scalograms
        :type fixed_width: int
        :raises RuntimeError: if no .npy file is found in data_dir
        :returns: None
        """

        # Store the parameters
        self.data_dir = Path(data_dir)
        self.fixed_width = fixed_width

        # Find all .npy files

        # .rglob("*.npy") searches recursively in all subfolders
        self.files = list(self.data_dir.rglob("*.npy"))

        if not self.files:
            raise RuntimeError(f"No .npy file found in {data_dir}")

        # Create labels from names of parent files (blues, rock...)

        # sorted ensures the order is always the same (blues=0, classical=1...)
        self.classes = sorted(list(set(f.parent.name for f in self.files)))
        # create the 'translation' dictionary
        self.class_to_idx = {
            cls_name: i for i, cls_name in enumerate(self.classes)
        }

        logger.info(f"Dataset loaded : {len(self.files)} files.")
        logger.info(f"Classes found : {self.classes}")

    def __len__(self):
        """
        Give the number of samples (scalograms).

        :returns: number of samples
        :rtype: int
        """
        return len(self.files)

    def __getitem__(self, idx):
        """
        Get an item (scalogram) based on its index.

        :param idx: index of the item to retrieve
        :type idx: int
        :returns: a tuple (data_tensor, label)
                  - data_tensor: the scalogram as a PyTorch tensor
                  - label: the corresponding label as an integer
        :rtype: tuple[torch.Tensor, int]
        """

        # Get the path of the file (the scalogram)
        file_path = self.files[idx]

        # Load data
        scalogram = np.load(file_path)

        # Get the corresponding label
        label_name = file_path.parent.name
        label = self.class_to_idx[label_name]  # conversion into integer

        # Padding of truncating for size standardisation
        # (width expected: self.fixed_width)
        current_width = scalogram.shape[1]

        if current_width > self.fixed_width:
            # Too long: only keep the beginning
            scalogram = scalogram[:, : self.fixed_width]
        else:
            # Too long: add zeros at the end (padding)
            pad_width = self.fixed_width - current_width
            # padding((top, bottom), (left, right))
            scalogram = np.pad(
                scalogram, ((0, 0), (0, pad_width)), mode="constant"
            )

        # Convert into a PyTorch tensor

        # unsqueeze add a dimension (of 1) at the beginning
        # since the model expects (Channel, Height, Width)
        # and the tensors are (Height, Width)
        data_tensor = torch.from_numpy(scalogram).float().unsqueeze(0)

        return data_tensor, label
