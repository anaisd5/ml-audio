import argparse
import json
import logging
import sys

import librosa
import numpy as np
import torch

from .model import get_audio_resnet

# Declare the logger at module level
logger = logging.getLogger(__name__)

# --- Parameters ---
MODEL_PATH = "model_trained.pth"
CLASS_MAP_PATH = "class_map.json"
FIXED_WIDTH = 1280  # should be the same during training


def preprocess_single_file(file_path):
    """
    Function preprocessing one file.

    :param file_path: the path to the audio file to process
    :type file_path: str | Path
    :raises Exception: if the audio file cannot be loaded or processed
    :returns: a tensor ready for model input
    :rtype: torch.Tensor
    """

    logger.info(f"Loading and processing the file {file_path}")
    try:
        y, sr = librosa.load(file_path, sr=None)

        # If the file is very short, warn the user
        if librosa.get_duration(y=y, sr=sr) < 1.0:
            logger.warning(
                f"File {file_path} is very short (< 1s). \
                           Quality might be poor."
            )

        # Calculate the scalogram
        C = librosa.cqt(y, sr=sr, fmin=librosa.note_to_hz("C1"))
        C_db = librosa.amplitude_to_db(np.abs(C), ref=np.max)

        # Manage padding / truncating (as in the dataset)
        current_width = C_db.shape[1]
        if current_width > FIXED_WIDTH:
            C_db = C_db[:, :FIXED_WIDTH]
        else:
            pad_width = FIXED_WIDTH - current_width
            C_db = np.pad(C_db, ((0, 0), (0, pad_width)), mode="constant")

        # Format for PyTorch (Batch=1, Channel=1, H, W)
        tensor_input = torch.from_numpy(C_db).float().unsqueeze(0).unsqueeze(0)

        return tensor_input

    except Exception as e:
        logger.error(f"Error processing the file: {e}")
        return None


def predict(file_to_predict):
    """
    Principal prediction function.

    :param file_to_predict: the path to the audio file to
                            make prediction on
    :type file_to_predict: str | Path
    :raises FileNotFoundError: if model files are not found
    :returns: None
    """

    # Loading classes list (from JSON)
    logger.info(f"Loading classes list from {CLASS_MAP_PATH}")
    try:
        with open(CLASS_MAP_PATH, "r") as f:
            classes = json.load(f)["classes"]
        num_classes = len(classes)
    except FileNotFoundError:
        logger.critical(f"Error : File {CLASS_MAP_PATH} not found.")
        logger.critical("Please launch train.py first to generate the model.")
        sys.exit(1)

    # Load model architecture
    logger.info("Loading the model architecture")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_audio_resnet(num_classes=num_classes).to(device)

    # Load the trained weights
    logger.info(f"Loading weights from {MODEL_PATH}")
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except FileNotFoundError:
        logger.critical(f"Error: File {MODEL_PATH} not found.")
        logger.critical("Please launch train.py first to generate the model.")
        sys.exit(1)

    model.eval()  # put the model in validation mode

    # Preprocess the audio file
    input_tensor = preprocess_single_file(file_to_predict)
    if input_tensor is None:
        return

    # Make prediction
    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        output_logits = model(input_tensor)
        probabilities = torch.softmax(output_logits, dim=1)
        predicted_index = probabilities.argmax(dim=1).item()

        predicted_class = classes[predicted_index]
        confidence = probabilities[0][predicted_index].item()

    print("\n--- Prediction results ---")
    print(f"File: {file_to_predict}")
    print(f"Prediction: {predicted_class.upper()}")
    print(f"Confidence: {confidence * 100:.2f}%")


if __name__ == "__main__":

    # Argument parser
    parser = argparse.ArgumentParser(description="Preprocess audio files.")

    # Obligatory argument: file path
    parser.add_argument(
        "file_path", type=str, help="Path to the audio file to predict"
    )

    # Optional argument: logging level
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

    predict(args.file_path)
