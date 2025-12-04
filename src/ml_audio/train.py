import argparse
import json
import logging
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

# Import personalised modules
from .dataset import GTZANDataset
from .model import get_audio_resnet

# Declare the logger at module level
logger = logging.getLogger(__name__)

# --- Hyperparameters ---

# This values can be modified
NUM_CLASSES = 10  # 10 genres
BATCH_SIZE = 16  # Size of batches
NUM_EPOCHS = 15  # 15 epochs
LEARNING_RATE = 0.001  # Learning rate for the Adam optimiser
DATA_DIR = "data/processed/scalograms"  # Folder of .npy files
MODEL_SAVE_PATH = "model_trained.pth"
MAP_SAVE_PATH = "class_map.json"


def train():
    """
    Function for training the model.

    :raises RuntimeError: if dataset cannot be loaded
    :returns: None
    """

    logger.info("Beginning training")

    # Hardware configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using: {device}")

    # --- Preparing data ---

    # Load the dataset
    try:
        full_dataset = GTZANDataset(data_dir=DATA_DIR)
    except RuntimeError as e:
        logger.critical(f"Error loading the dataset: {e}")
        logger.critical(
            f"Check if the folder '{DATA_DIR}' contains the files .npy"
        )
        sys.exit(1)

    # Save the class mapping (for predict.py)
    class_map = {"classes": full_dataset.classes}
    with open(MAP_SAVE_PATH, "w") as f:
        json.dump(class_map, f)
    logger.info(f"Mapping classes saved in: {MAP_SAVE_PATH}")

    # Separate training (80%) and validation (20%)
    labels = [f.parent.name for f in full_dataset.files]
    train_idx, val_idx = train_test_split(
        list(range(len(full_dataset))),
        test_size=0.2,  # 20% for validation
        stratify=labels,  # ensure a fair repartition of genres
        random_state=42,  # for reproductible results
    )

    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)

    logger.info(f"Training set size: {len(train_dataset)}")
    logger.info(f"Validation set size: {len(val_dataset)}")

    # Create Dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )

    # --- Initialise model, loss and optimiser ---

    model = get_audio_resnet(num_classes=NUM_CLASSES)
    model = model.to(device)  # Send the model on GPU

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- Traning and validation loop ---

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

            # Forward pass, backward pass, and optimisation
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

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

    logger.info("Training done.")

    # --- Save the model ---
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    logger.info(f"Model saved in: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="Preprocess audio files.")
    parser.add_argument(
        "--log",
        default="INFO",
        help="Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
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
    train()
