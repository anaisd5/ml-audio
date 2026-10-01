import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from matplotlib.figure import Figure
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

from .dataset import GTZANDataset, load_split
from .features import (
    BOTH_FEATURES,
    DEFAULT_FEATURES,
    FEATURES_DIRS,
    get_output_paths,
)
from .model import get_audio_resnet

# Declare the logger at module level
logger = logging.getLogger(__name__)

# --- Parameters ---
BATCH_SIZE = 16
NUM_WORKERS = 2
NUM_THREADS = 4  # CPU cores used by PyTorch (default, see --num-threads)


def predict_dataset(model, dataset, device, batch_size, num_workers):
    """
    Compute the probabilities of each class for every segment of a
    dataset, then for every track (mean of the probabilities of its
    segments).

    :param model: the trained model
    :type model: torch.nn.Module
    :param dataset: the dataset to predict on
    :type dataset: GTZANDataset
    :param device: the device used for the computation
    :type device: torch.device
    :param batch_size: size of batches
    :type batch_size: int
    :param num_workers: number of processes loading the data
    :type num_workers: int
    :returns: a tuple (segment_probs, segment_labels,
              track_probs, track_labels)
    :rtype: tuple[numpy.ndarray, numpy.ndarray,
            numpy.ndarray, numpy.ndarray]
    """

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    model.eval()  # put the model in validation mode
    segment_probs = []
    segment_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs.to(device))
            segment_probs.append(torch.softmax(outputs, dim=1).cpu().numpy())
            segment_labels.append(labels.numpy())

    segment_probs = np.concatenate(segment_probs)
    segment_labels = np.concatenate(segment_labels)

    # Aggregate the segments of each track (mean of the probabilities)
    file_indices = np.array([file_idx for file_idx, _ in dataset.segments])
    track_probs = np.stack(
        [
            segment_probs[file_indices == i].mean(axis=0)
            for i in range(len(dataset.files))
        ]
    )
    track_labels = np.array(
        [dataset.get_label(i) for i in range(len(dataset.files))]
    )

    return segment_probs, segment_labels, track_probs, track_labels


def plot_history(history, save_path):
    """
    Plot the training and validation curves (loss and accuracy).

    :param history: the history of training, with the columns 'epoch',
                    'train_loss', 'val_loss', 'train_acc' and 'val_acc'
    :type history: pandas.DataFrame
    :param save_path: the path of the image to save
    :type save_path: str | Path
    :returns: None
    """

    fig = Figure(figsize=(11, 4))
    ax_loss, ax_acc = fig.subplots(1, 2)

    ax_loss.plot(history["epoch"], history["train_loss"], label="Training")
    ax_loss.plot(history["epoch"], history["val_loss"], label="Validation")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Loss")
    ax_loss.legend()

    ax_acc.plot(history["epoch"], history["train_acc"], label="Training")
    ax_acc.plot(history["epoch"], history["val_acc"], label="Validation")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.set_title("Accuracy")
    ax_acc.legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=120)


def plot_confusion_matrix(matrix, classes, save_path):
    """
    Plot a confusion matrix (rows: true labels, columns: predictions).

    :param matrix: the confusion matrix
    :type matrix: numpy.ndarray
    :param classes: the names of the classes
    :type classes: list[str]
    :param save_path: the path of the image to save
    :type save_path: str | Path
    :returns: None
    """

    fig = Figure(figsize=(7, 6))
    ax = fig.subplots()

    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax)

    ax.set_xticks(range(len(classes)), labels=classes, rotation=45)
    ax.set_yticks(range(len(classes)), labels=classes)
    ax.set_xlabel("Predicted genre")
    ax.set_ylabel("True genre")
    ax.set_title("Confusion matrix (test set, tracks)")

    # Write the number of tracks in each cell
    for i in range(len(classes)):
        for j in range(len(classes)):
            color = "white" if matrix[i, j] > matrix.max() / 2 else "black"
            ax.text(j, i, matrix[i, j], ha="center", va="center", color=color)

    fig.tight_layout()
    fig.savefig(save_path, dpi=120)


def plot_roc_curves(labels, probs, classes, save_path):
    """
    Plot the ROC curve of each class (one class versus the others).

    :param labels: the true labels
    :type labels: numpy.ndarray
    :param probs: the predicted probabilities (Samples, Classes)
    :type probs: numpy.ndarray
    :param classes: the names of the classes
    :type classes: list[str]
    :param save_path: the path of the image to save
    :type save_path: str | Path
    :returns: None
    """

    fig = Figure(figsize=(7, 6))
    ax = fig.subplots()

    for i, cls_name in enumerate(classes):
        fpr, tpr, _ = roc_curve(labels == i, probs[:, i])
        auc = roc_auc_score(labels == i, probs[:, i])
        ax.plot(fpr, tpr, label=f"{cls_name} (AUC = {auc:.3f})")

    # Diagonal: random classifier
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey")
    ax.set_xlabel("False positive rate (1 - specificity)")
    ax.set_ylabel("True positive rate (sensitivity)")
    ax.set_title("ROC curves (test set, tracks)")
    ax.legend(loc="lower right", fontsize="small")

    fig.tight_layout()
    fig.savefig(save_path, dpi=120)


def evaluate(model, dataset, device, batch_size, num_workers, results_dir):
    """
    Evaluate a model on a dataset (usually the test set). It saves a
    text report (accuracy, AUC, sensitivity and specificity of each
    class...), the confusion matrix and the ROC curves in results_dir.

    :param model: the trained model
    :type model: torch.nn.Module
    :param dataset: the dataset to evaluate on
    :type dataset: GTZANDataset
    :param device: the device used for the computation
    :type device: torch.device
    :param batch_size: size of batches
    :type batch_size: int
    :param num_workers: number of processes loading the data
    :type num_workers: int
    :param results_dir: folder where the results are saved
    :type results_dir: str | Path
    :returns: the accuracy on the tracks
    :rtype: float
    """

    segment_probs, segment_labels, track_probs, track_labels = predict_dataset(
        model, dataset, device, batch_size, num_workers
    )

    return evaluate_predictions(
        segment_probs,
        segment_labels,
        track_probs,
        track_labels,
        dataset.classes,
        results_dir,
    )


def evaluate_predictions(
    segment_probs,
    segment_labels,
    track_probs,
    track_labels,
    classes,
    results_dir,
):
    """
    Evaluate predictions (given by predict_dataset). The probabilities
    can come from one model, or be the mean of the probabilities of
    several models. It saves a text report (accuracy, AUC, sensitivity
    and specificity of each class...), the confusion matrix and the ROC
    curves in results_dir.

    :param segment_probs: the probabilities of the segments
                          (Segments, Classes)
    :type segment_probs: numpy.ndarray
    :param segment_labels: the true labels of the segments
    :type segment_labels: numpy.ndarray
    :param track_probs: the probabilities of the tracks (Tracks, Classes)
    :type track_probs: numpy.ndarray
    :param track_labels: the true labels of the tracks
    :type track_labels: numpy.ndarray
    :param classes: the names of the classes
    :type classes: list[str]
    :param results_dir: folder where the results are saved
    :type results_dir: str | Path
    :returns: the accuracy on the tracks
    :rtype: float
    """

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    track_preds = track_probs.argmax(axis=1)

    # Accuracy on the segments (3 s) and on the tracks (30 s)
    segment_acc = (segment_probs.argmax(axis=1) == segment_labels).mean()
    track_acc = (track_preds == track_labels).mean()
    auc = roc_auc_score(track_labels, track_probs, multi_class="ovr")

    # Sensitivity and specificity of each class
    matrix = confusion_matrix(
        track_labels, track_preds, labels=range(len(classes))
    )
    true_positives = np.diag(matrix)
    false_negatives = matrix.sum(axis=1) - true_positives
    false_positives = matrix.sum(axis=0) - true_positives
    true_negatives = matrix.sum() - (
        true_positives + false_negatives + false_positives
    )
    sensitivity = true_positives / (true_positives + false_negatives)
    specificity = true_negatives / (true_negatives + false_positives)

    # Write the report
    lines = [
        f"Number of tracks: {len(track_labels)}",
        f"Number of segments: {len(segment_labels)}",
        f"Accuracy (segments): {segment_acc * 100:.2f}%",
        f"Accuracy (tracks): {track_acc * 100:.2f}%",
        f"Macro ROC AUC (tracks, one vs rest): {auc:.3f}",
        "",
        classification_report(
            track_labels,
            track_preds,
            labels=range(len(classes)),
            target_names=classes,
            digits=3,
            zero_division=0,
        ),
        f"{'':<12}{'sensitivity':>12}{'specificity':>12}{'AUC':>8}",
    ]
    for i, cls_name in enumerate(classes):
        class_auc = roc_auc_score(track_labels == i, track_probs[:, i])
        lines.append(
            f"{cls_name:<12}{sensitivity[i]:>12.3f}"
            f"{specificity[i]:>12.3f}{class_auc:>8.3f}"
        )
    report = "\n".join(lines)

    report_path = results_dir / "test_report.txt"
    with open(report_path, "w") as f:
        f.write(report + "\n")
    logger.info(f"Test report saved in: {report_path}")
    print(report)

    # Save the figures
    plot_confusion_matrix(
        matrix, classes, results_dir / "confusion_matrix.png"
    )
    plot_roc_curves(
        track_labels, track_probs, classes, results_dir / "roc_curves.png"
    )
    logger.info(f"Confusion matrix and ROC curves saved in: {results_dir}")

    return track_acc


def load_model_and_dataset(features, device, split="test"):
    """
    Load the model trained on a type of features, and the dataset of a
    split with the same type of features.

    :param features: the type of features ('cqt' or 'mel')
    :type features: str
    :param device: the device used for the computation
    :type device: torch.device
    :param split: name of the split ('train', 'valid' or 'test')
    :type split: str
    :raises FileNotFoundError: if model files are not found
    :returns: a tuple (model, dataset)
    :rtype: tuple[torch.nn.Module, GTZANDataset]
    """

    model_path, class_map_path, _ = get_output_paths(features)

    # Loading classes list and standardisation values (from JSON)
    with open(class_map_path, "r") as f:
        class_map = json.load(f)

    dataset = GTZANDataset(
        FEATURES_DIRS[features],
        track_names=load_split(split),
        mean=class_map["mean"],
        std=class_map["std"],
    )

    model = get_audio_resnet(num_classes=len(class_map["classes"]))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)

    return model, dataset


if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(
        description="Evaluate the trained model on the test set."
    )
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
        "--features",
        default=DEFAULT_FEATURES,
        choices=[*FEATURES_DIRS.keys(), BOTH_FEATURES],
        help=f"Model to evaluate: the one trained on 'cqt' or on 'mel' "
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

    # Limit the CPU cores used (by default PyTorch uses all of them,
    # which can freeze the computer)
    torch.set_num_threads(args.num_threads)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Types of features of the models to evaluate
    if args.features == BOTH_FEATURES:
        features_list = list(FEATURES_DIRS.keys())
    else:
        features_list = [args.features]

    # Compute the probabilities given by each model
    predictions = []
    for features in features_list:
        logger.info(f"Loading the model trained on '{features}' features")
        try:
            model, test_dataset = load_model_and_dataset(features, device)
        except FileNotFoundError as e:
            logger.critical(f"Error : File {e.filename} not found.")
            logger.critical(
                f"Please launch train.py with --features={features} first "
                f"to generate the model."
            )
            sys.exit(1)

        predictions.append(
            predict_dataset(
                model, test_dataset, device, args.batch_size, args.num_workers
            )
        )

    # The segments should be the same for all models
    segment_probs, segment_labels, track_probs, track_labels = predictions[0]
    for other in predictions[1:]:
        if not np.array_equal(other[1], segment_labels):
            logger.critical(
                "Error : the models were not evaluated on the same segments."
            )
            logger.critical(
                "Please launch preprocess.py again for each type of features."
            )
            sys.exit(1)

    # Mean of the probabilities of the models
    segment_probs = np.mean([p[0] for p in predictions], axis=0)
    track_probs = np.mean([p[2] for p in predictions], axis=0)

    _, _, results_dir = get_output_paths(args.features)
    evaluate_predictions(
        segment_probs,
        segment_labels,
        track_probs,
        track_labels,
        test_dataset.classes,
        results_dir,
    )
