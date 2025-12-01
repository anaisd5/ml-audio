from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class GTZANDataset(Dataset):
    def __init__(self, data_dir, fixed_width=1280):
        """
        Function called at initialisation.

        Args:
            data_dir (string): Path to the folder containing the .npy files
            (ex: data/processed/scalograms)
            fixed_width (int): The fixed temporal width for all scalograms.
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

        print(f"Dataset loaded : {len(self.files)} files.")
        print(f"Classes found : {self.classes}")

    def __len__(self):
        """
        Give the number of samples (scalograms).
        """
        return len(self.files)

    def __getitem__(self, idx):
        """
        Get an item (scalogram) based on its index

        Args:
            data_dir (int): The index of the item to retrieve
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
