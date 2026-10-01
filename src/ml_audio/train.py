import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import personalised modules
from .dataset import GTZANDataset, compute_mean_std, load_split
from .evaluate import evaluate, plot_history
from .features import DEFAULT_FEATURES, FEATURES_DIRS
from .model import get_audio_resnet

# Declare the logger at module level
logger = logging.getLogger(__name__)

# --- Hyperparameters ---

# This values can be modified
NUM_CLASSES = 10  # 10 genres
BATCH_SIZE = 16  # Size of batches (default, see --batch-size)
NUM_WORKERS = 2  # Processes loading the data (default, see --num-workers)
NUM_THREADS = 4  # CPU cores used by PyTorch (default, see --num-threads)
NUM_EPOCHS = 50  # Maximal number of epochs
LEARNING_RATE = 0.0001  # Learning rate for the AdamW optimiser
WEIGHT_DECAY = 0.0001  # Weight decay (regularisation) for AdamW
PATIENCE = 5  # Stop if the validation loss does not improve for 5 epochs
SEED = 42  # For reproductible results
MIXUP_ALPHA = 0.4  # Parameter of the Beta distribution for mixup
FROZEN_LAYERS = 0  # Groups of layers frozen (default, see --frozen-layers)
MODEL_SAVE_PATH = "model_trained.pth"
MAP_SAVE_PATH = "class_map.json"
RESULTS_DIR = "results"  # Folder for the history, reports and figures


def mixup(inputs, labels, alpha=MIXUP_ALPHA):
    """
    Data augmentation (mixup, Zhang et al., 2018): each example of the
    batch is mixed with another random example of the batch. The loss
    is then the same mix of the losses of the two labels.

    :param inputs: the batch of segments
    :type inputs: torch.Tensor
    :param labels: the labels of the batch
    :type labels: torch.Tensor
    :param alpha: parameter of the Beta distribution giving the
                  proportion of the mix
    :type alpha: float
    :returns: a tuple (mixed_inputs, labels_a, labels_b, lam)
              - mixed_inputs: lam * inputs + (1 - lam) * other inputs
              - labels_a: the labels of the inputs
              - labels_b: the labels of the other inputs
              - lam: the proportion of the mix
    :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]
    """

    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    # Random order of the batch: example i is mixed with example perm[i]
    perm = torch.randperm(inputs.size(0), device=inputs.device)
    mixed_inputs = lam * inputs + (1 - lam) * inputs[perm]
    return mixed_inputs, labels, labels[perm], lam


def train(
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    augment=True,
    num_threads=NUM_THREADS,
    frozen_layers=FROZEN_LAYERS,
    features=DEFAULT_FEATURES,
):
    """
    Function for training the model. The best model (lowest validation
    loss) is saved, then evaluated on the test set.

    :param batch_size: size of batches
    :type batch_size: int
    :param num_workers: number of processes loading the data
                        (0 to use less memory)
    :type num_workers: int
    :param augment: if True, data augmentation is used on the training
                    set (random crop, SpecAugment and mixup)
    :type augment: bool
    :param num_threads: number of CPU cores used by PyTorch (a low value
                        keeps the computer usable during the training)
    :type num_threads: int
    :param frozen_layers: number of groups of layers of the ResNet (from
                          0 to 4) that keep their pretrained weights
    :type frozen_layers: int
    :param features: the type of features ('cqt' or 'mel'), the files
                     should first be created by preprocess.py
    :type features: str
    :raises RuntimeError: if dataset cannot be loaded
    :returns: None
    """

    logger.info("Beginning training")

    # Folder of .npy files
    DATA_DIR = FEATURES_DIRS[features]
    logger.info(f"Features: {features} ({DATA_DIR})")

    # Fix the seed for reproductible results
    torch.manual_seed(SEED)

    # Limit the CPU cores used (by default PyTorch uses all of them,
    # which can freeze the computer)
    torch.set_num_threads(num_threads)
    logger.info(f"CPU cores used by PyTorch: {num_threads}")

    # Hardware configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using: {device}")

    # --- Preparing data ---

    # Load the datasets with the fault-filtered split
    # (the tracks of an artist are all in the same split)
    try:
        train_dataset = GTZANDataset(
            DATA_DIR, track_names=load_split("train"), augment=augment
        )
    except RuntimeError as e:
        logger.critical(f"Error loading the dataset: {e}")
        logger.critical(
            f"Check if the folder '{DATA_DIR}' contains the files .npy"
        )
        sys.exit(1)

    # Standardisation values, computed on the training set only
    mean, std = compute_mean_std(train_dataset)
    train_dataset.mean, train_dataset.std = mean, std
    logger.info(f"Standardisation: mean={mean:.2f}, std={std:.2f}")
    logger.info(f"Data augmentation: {augment}")

    val_dataset = GTZANDataset(
        DATA_DIR, track_names=load_split("valid"), mean=mean, std=std
    )
    test_dataset = GTZANDataset(
        DATA_DIR, track_names=load_split("test"), mean=mean, std=std
    )

    # Save the class mapping, the standardisation values and the type
    # of features (for predict.py)
    class_map = {
        "classes": train_dataset.classes,
        "mean": mean,
        "std": std,
        "features": features,
    }
    with open(MAP_SAVE_PATH, "w") as f:
        json.dump(class_map, f)
    logger.info(f"Mapping classes saved in: {MAP_SAVE_PATH}")

    logger.info(
        f"Training set size: {len(train_dataset.files)} tracks, "
        f"{len(train_dataset)} segments"
    )
    logger.info(
        f"Validation set size: {len(val_dataset.files)} tracks, "
        f"{len(val_dataset)} segments"
    )
    logger.info(
        f"Test set size: {len(test_dataset.files)} tracks, "
        f"{len(test_dataset)} segments"
    )

    # Create Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    # --- Initialise model, loss and optimiser ---

    model = get_audio_resnet(
        num_classes=NUM_CLASSES, frozen_layers=frozen_layers
    )
    logger.info(f"Groups of layers frozen: {frozen_layers}")
    model = model.to(device)  # Send the model on GPU

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        # only the parameters that are not frozen
        [param for param in model.parameters() if param.requires_grad],
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    # Divide the learning rate by 2 if the validation loss does not
    # improve for 2 epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    # --- Traning and validation loop ---

    history = []  # one line per epoch
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(NUM_EPOCHS):
        logger.info(f"--- Starting Epoch {epoch + 1}/{NUM_EPOCHS} ---")

        # Training phase
        model.train()  # Put the model on training mode
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Mix the examples of the batch (without augmentation,
            # lam = 1 so nothing changes)
            if augment:
                inputs, labels_a, labels_b, lam = mixup(inputs, labels)
            else:
                labels_a, labels_b, lam = labels, labels, 1.0

            # Forward pass, backward pass, and optimisation
            outputs = model(inputs)
            loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(
                outputs, labels_b
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            # With mixup, a prediction is right in proportion of its label
            correct_train += (
                lam * (predicted == labels_a).sum().item()
                + (1 - lam) * (predicted == labels_b).sum().item()
            )

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = 100 * correct_train / total_train
        logger.info(
            f"Epoch {epoch+1} - Train Loss: {epoch_loss:.4f}, \
                Acc: {epoch_acc:.2f}%"
        )

        # --- Validation phase ---
        model.eval()  # Put the model on validation phase
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        epoch_val_loss = val_loss / len(val_dataset)
        epoch_val_acc = 100 * correct_val / total_val
        logger.info(
            f"Epoch {epoch+1} - Val Loss: {epoch_val_loss:.4f}, \
                Acc: {epoch_val_acc:.2f}%"
        )

        # Store the statistics of the epoch
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": epoch_loss,
                "train_acc": epoch_acc,
                "val_loss": epoch_val_loss,
                "val_acc": epoch_val_acc,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )

        scheduler.step(epoch_val_loss)

        # --- Save the best model and early stopping ---
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            logger.info(f"Best model so far saved in: {MODEL_SAVE_PATH}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                logger.info(
                    f"No improvement for {PATIENCE} epochs: early stopping."
                )
                break

    logger.info("Training done.")

    # --- Save the history and the curves ---
    results_dir = Path(RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)
    history = pd.DataFrame(history)
    history.to_csv(results_dir / "training_history.csv", index=False)
    plot_history(history, results_dir / "training_curves.png")
    logger.info(f"Training history and curves saved in: {results_dir}")

    # --- Evaluate the best model on the test set ---
    logger.info("Evaluating the best model on the test set")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    evaluate(model, test_dataset, device, batch_size, num_workers, results_dir)


if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="Train the model.")
    parser.add_argument(
        "--log",
        default="INFO",
        help="Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Size of batches (default: {BATCH_SIZE})",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=NUM_WORKERS,
        help=f"Number of processes loading the data "
        f"(default: {NUM_WORKERS}, 0 to use less memory)",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=NUM_THREADS,
        help=f"Number of CPU cores used by PyTorch (default: {NUM_THREADS}, "
        f"reduce it to keep the computer usable)",
    )
    parser.add_argument(
        "--frozen-layers",
        type=int,
        default=FROZEN_LAYERS,
        choices=range(5),
        help=f"Number of groups of layers of the ResNet that keep their "
        f"pretrained weights (default: {FROZEN_LAYERS})",
    )
    parser.add_argument(
        "--features",
        default=DEFAULT_FEATURES,
        choices=FEATURES_DIRS.keys(),
        help=f"Type of features: 'cqt' (scalogram) or 'mel' (mel "
        f"spectrogram) (default: {DEFAULT_FEATURES})",
    )
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Train without data augmentation (random crop, SpecAugment "
        "and mixup)",
    )

    args = parser.parse_args()

    loglevel = args.log
    numeric_level = getattr(logging, loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError("Invalid log level: %s" % loglevel)

    logging.getLogger().setLevel(numeric_level)

    # Configuration of the logging
    logging.basicConfig(
        level=numeric_level,
        format="[%(levelname)s]\t%(asctime)s\t%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,  # delete any previous configuration
    )
    logger = logging.getLogger(__name__)

    # Call the training function
    train(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=not args.no_augment,
        num_threads=args.num_threads,
        frozen_layers=args.frozen_layers,
        features=args.features,
    )
