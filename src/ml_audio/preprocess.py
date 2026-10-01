import argparse
import logging
import sys
from pathlib import Path

import librosa
import numpy as np
from tqdm import tqdm

from .features import DEFAULT_FEATURES, FEATURES_DIRS, compute_features

# Declare the logger at module level
logger = logging.getLogger(__name__)

# Files definition
SOURCE_DIR = Path("data/gtzan/audio/")


def process_file(file_path, target_path, features=DEFAULT_FEATURES):
    """
    Load an audio file, transform it into a CQT scalogram
    (or a mel spectrogram) and save it in a Numpy array.

    :param file_path: the path to the audio file to process
    :type file_path: str | Path
    :param target_path: the path to save the output file
    :type target_path: str | Path
    :param features: the type of features ('cqt' or 'mel')
    :type features: str
    :raises Exception: if the audio file cannot be loaded or processed
    :returns: None
    """

    try:
        # Load the audio file
        logger.debug(f"Processing file: {file_path}")

        # y is the audio signal, sr is the sample rate
        y, sr = librosa.load(file_path, sr=None)

        # If the file is very short, warn the user
        if librosa.get_duration(y=y, sr=sr) < 1.0:
            logger.warning(
                f"File {file_path} is very short (< 1s). \
                           Quality might be poor."
            )

        # Do the Constant Q Transform (or the mel spectrogram), in dB
        features_db = compute_features(y, sr, features)

        # Save raw data (Numpy array) at target_path
        np.save(target_path, features_db)

    except Exception as e:
        logger.error(f"Failed to process {file_path}: {e}")


# Main code
if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="Preprocess audio files.")
    parser.add_argument(
        "--log",
        default="INFO",
        help="Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    parser.add_argument(
        "--features",
        default=DEFAULT_FEATURES,
        choices=FEATURES_DIRS.keys(),
        help=f"Type of features: 'cqt' (scalogram) or 'mel' (mel "
        f"spectrogram) (default: {DEFAULT_FEATURES})",
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

    # Check if the output directory exist and if not create it with parents
    TARGET_DIR = Path(FEATURES_DIRS[args.features])
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    # Start preprocessing
    logger.info("Starting preprocessing")
    logger.info(f"Features: {args.features}")
    logger.info(f"Source: {SOURCE_DIR}")
    logger.info(f"Destination: {TARGET_DIR}")

    # Find all audio files

    # .rglob("*.wav") searches recursively in all subfolders
    audio_files = list(SOURCE_DIR.rglob("*.wav"))

    if not audio_files:
        logger.critical(f"Error : No .wav file found in {SOURCE_DIR}")
        sys.exit(1)

    logger.info(f"{len(audio_files)} audio files found.")

    # Main loop for preprocessing

    # tqdm adds a progress bar
    for file_path in tqdm(audio_files, desc="Files preprocessing"):

        # Create the output path keeping the folder structure
        # e.g. data/gtzan/audio/blues/file.wav
        # -> data/processed/scalograms/blues/file.npy

        # Calculate the relative path (e.g. 'blues/file.wav')
        relative_path = file_path.relative_to(SOURCE_DIR)

        # Create the destination path
        # (e.g. 'data/processed/scalograms/blues/file.npy')
        target_path = TARGET_DIR / relative_path.with_suffix(".npy")

        # Create the parent folder if necessary
        # (e.g. 'data/processed/scalograms/blues/')
        target_path.parent.mkdir(parents=True, exist_ok=True)

        # Call the function to process the file
        process_file(file_path, target_path, args.features)

    logger.info("Preprocessing done.")
