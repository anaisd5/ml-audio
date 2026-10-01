import librosa
import numpy as np

# --- Parameters ---

# Folder of the .npy files for each type of features
FEATURES_DIRS = {
    "cqt": "data/processed/scalograms",
    "mel": "data/processed/melspectrograms",
}
DEFAULT_FEATURES = "cqt"  # Type of features used by default
# Value of --features to combine the models of all types of features
# (mean of their probabilities)
BOTH_FEATURES = "both"
N_MELS = 128  # Number of frequency bands of the mel spectrogram


def get_output_paths(features=DEFAULT_FEATURES):
    """
    Give the paths of the files created for a type of features, so that
    the models of the different types can be kept at the same time.
    The names of the default type have no suffix (model_trained.pth),
    the others end with the type (model_trained_mel.pth).

    :param features: the type of features ('cqt' or 'mel'), or 'both'
                     for the folder of the results of the combined models
    :type features: str
    :returns: a tuple (model_path, class_map_path, results_dir)
    :rtype: tuple[str, str, str]
    """

    suffix = "" if features == DEFAULT_FEATURES else f"_{features}"
    return (
        f"model_trained{suffix}.pth",
        f"class_map{suffix}.json",
        f"results{suffix}",
    )


def compute_features(y, sr, features=DEFAULT_FEATURES):
    """
    Transform an audio signal into an image (frequencies x time) in dB.
    Two types of features are available:

    - 'cqt': scalogram from the Constant Q Transform (84 frequency bins,
      one per semitone, so it follows the musical notes);
    - 'mel': mel spectrogram (128 frequency bands, spaced as the human
      ear perceives the pitches).

    Both have the same number of frames per second (hop_length=512), and
    their values are between -80 dB (silence) and 0 dB (loudest value).

    :param y: the audio signal
    :type y: numpy.ndarray
    :param sr: the sample rate of the audio signal
    :type sr: int
    :param features: the type of features ('cqt' or 'mel')
    :type features: str
    :raises ValueError: if the type of features is unknown
    :returns: the features in dB (Height, Width)
    :rtype: numpy.ndarray
    """

    if features == "cqt":
        # Do the Constant Q Transform

        # fmin is a filter: only consider notes higher than C1
        # C is a 2D Numpy array with complex numbers
        C = librosa.cqt(y, sr=sr, fmin=librosa.note_to_hz("C1"))
        # only consider the amplitude of the signal (not the phase)
        # convert into dB (negative since the reference is
        # max amplitude of the signal)
        return librosa.amplitude_to_db(np.abs(C), ref=np.max)

    if features == "mel":
        # Do the mel spectrogram

        # S is a 2D Numpy array with the power of each frequency band
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
        # convert into dB (negative since the reference is
        # max power of the signal)
        return librosa.power_to_db(S, ref=np.max)

    raise ValueError(f"Unknown type of features: {features}")
