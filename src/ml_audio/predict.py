import argparse
import json
import logging
import sys

import librosa
import torch

from .dataset import split_into_segments
from .features import (
    BOTH_FEATURES,
    DEFAULT_FEATURES,
    FEATURES_DIRS,
    compute_features,
    get_output_paths,
)
from .model import get_audio_resnet

# Declare the logger at module level
logger = logging.getLogger(__name__)


def preprocess_single_file(file_path, mean, std, features=DEFAULT_FEATURES):
    """
    Function preprocessing one file. The scalogram is cut into segments
    of about 3 seconds (as during training).

    :param file_path: the path to the audio file to process
    :type file_path: str | Path
    :param mean: mean used to standardise the scalogram
                 (computed during training)
    :type mean: float
    :param std: standard deviation used to standardise the scalogram
                (computed during training)
    :type std: float
    :param features: the type of features ('cqt' or 'mel'), it should be
                     the same as during training
    :type features: str
    :raises Exception: if the audio file cannot be loaded or processed
    :returns: a tensor ready for model input (one item per segment)
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

        # Calculate the scalogram (or the mel spectrogram)
        features_db = compute_features(y, sr, features)

        # Cut into segments (as in the dataset)
        segments = split_into_segments(features_db)

        # Standardisation (as in the dataset)
        segments = (segments - mean) / std

        # Format for PyTorch (Batch=Number of segments, Channel=1, H, W)
        tensor_input = torch.from_numpy(segments).float().unsqueeze(1)

        return tensor_input

    except Exception as e:
        logger.error(f"Error processing the file: {e}")
        return None


def predict(file_to_predict, features=DEFAULT_FEATURES):
    """
    Principal prediction function.

    :param file_to_predict: the path to the audio file to
                            make prediction on
    :type file_to_predict: str | Path
    :param features: the model to use: the one trained on 'cqt' or on
                     'mel' features, or 'both' for the mean of the
                     probabilities of the two models
    :type features: str
    :raises FileNotFoundError: if model files are not found
    :returns: None
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Types of features of the models to use
    if features == BOTH_FEATURES:
        features_list = list(FEATURES_DIRS.keys())
    else:
        features_list = [features]

    # Probabilities given by each model
    all_probabilities = []

    for model_features in features_list:
        model_path, class_map_path, _ = get_output_paths(model_features)

        # Loading classes list and standardisation values (from JSON)
        logger.info(f"Loading classes list from {class_map_path}")
        try:
            with open(class_map_path, "r") as f:
                class_map = json.load(f)
            classes = class_map["classes"]
            num_classes = len(classes)
        except FileNotFoundError:
            logger.critical(f"Error : File {class_map_path} not found.")
            logger.critical(
                f"Please launch train.py with --features={model_features} "
                f"first to generate the model."
            )
            sys.exit(1)

        # Load model architecture
        logger.info("Loading the model architecture")
        model = get_audio_resnet(num_classes=num_classes).to(device)

        # Load the trained weights
        logger.info(f"Loading weights from {model_path}")
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
        except FileNotFoundError:
            logger.critical(f"Error: File {model_path} not found.")
            logger.critical(
                f"Please launch train.py with --features={model_features} "
                f"first to generate the model."
            )
            sys.exit(1)

        model.eval()  # put the model in validation mode

        # Preprocess the audio file
        # (with the type of features used during training)
        input_tensor = preprocess_single_file(
            file_to_predict,
            class_map["mean"],
            class_map["std"],
            model_features,
        )
        if input_tensor is None:
            return

        # Make prediction
        with torch.no_grad():
            input_tensor = input_tensor.to(device)
            output_logits = model(input_tensor)
            # Mean of the probabilities of all segments
            all_probabilities.append(
                torch.softmax(output_logits, dim=1).mean(dim=0)
            )

    # Mean of the probabilities of the models
    probabilities = torch.stack(all_probabilities).mean(dim=0)
    predicted_index = probabilities.argmax().item()

    predicted_class = classes[predicted_index]
    confidence = probabilities[predicted_index].item()

    print("\n--- Prediction results ---")
    print(f"File: {file_to_predict}")
    print(f"Features: {' + '.join(features_list)}")
    print(f"Number of segments (3 s): {input_tensor.shape[0]}")
    print(f"Prediction: {predicted_class.upper()}")
    print(f"Confidence: {confidence * 100:.2f}%")


if __name__ == "__main__":

    # Argument parser
    parser = argparse.ArgumentParser(
        description="Predict the genre of an audio file."
    )

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

    # Optional argument: model to use
    parser.add_argument(
        "--features",
        default=DEFAULT_FEATURES,
        choices=[*FEATURES_DIRS.keys(), BOTH_FEATURES],
        help=f"Model to use: the one trained on 'cqt' or on 'mel' "
        f"features, or '{BOTH_FEATURES}' for the mean of the probabilities "
        f"of the two models (default: {DEFAULT_FEATURES})",
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

    predict(args.file_path, args.features)
