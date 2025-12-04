# Audio Classification Project (Machine Learning)

![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Poetry](https://img.shields.io/badge/packaging-poetry-cyan)

This project is an implementation of an audio classification system
using `librosa` for scalogram creation and `PyTorch` for the
machine learning model (based on `ResNet-18`). It is managed with
[Poetry](https://python-poetry.org/) for dependency and environment
management. The `.wav` files from the [GTZAN database](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification?resource=download-directory)
were used as an input.

**Goal:** Accurately classify audio tracks into 10 musical genres (blues, classical,
rock, etc.) using Deep Learning on the GTZAN dataset.

**Target audience:** Students and developers interested in audio processing pipelines or
`PyTorch` implementation of CNNs for spectrograms.

## Installation

This project uses Poetry. Dependencies are specified in the
`pyproject.toml` file and locked in `poetry.lock`.

### Clone the repository

```
git clone [GITHUB_LINK]
cd ml-audio
```

### Install dependencies

This command will create a virtual environment and install
all required libraries (PyTorch, Librosa, etc.).

```
poetry install
```

### Download `.wav` files

For this project, you should download the GTZAN dataset by
[follow this link](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification?resource=download-directory).

You should unzip the file and copy paste the folders from the `genres_original` folder
into the `data/gtzan/audio` folder. You should obtain this tree (from the root
of the project):

```
├───data
│   └───gtzan
│       └───audio
│           ├───blues
│           │   ├───blues.00000.wav
│           │   └───...
│           ├───classical
│           ├───country
│           ├───disco
│           ├───hiphop
│           ├───jazz
│           ├───metal
│           ├───pop
│           ├───reggae
│           └───rock
├───src
│   └───...
└───...
```

## Usage

Go to the root directory of the project.

Load and unzip the GTZAN dataset and put it on the right place (see previous
section).

For a full pipeline, execute the following commands in order (more
descriptions in next sections):

```
# Run as modules to handle relative imports correctly
poetry run python -m ml_audio.preprocess
poetry run python -m ml_audio.train
poetry run python -m ml_audio.predict <path_to_audio_file>
```

For those three files, you can use the optional `--log` argument to indicate
the minimal level of logs. By default, it is set to `INFO`. You can choose
`DEBUG`, `INFO`, `WARNING`, `ERROR` or `CRITICAL`.

For example, to set it to `WARNING`, type:

```
poetry run python -m ml_audio.preprocess --log=WARNING
```

### Preprocessing

For preprocessing all audio files, you can run the `preprocess` module:

```
poetry run python -m ml_audio.preprocess
```

When you run this command, some files will fail. It is a known behaviour.
At the end though, the file preprocessing should be complete (but some files
may be missing).

```
poetry run python -m ml_audio.preprocess

[INFO]  2025-12-03 23:37:13     Starting preprocessing
[INFO]  2025-12-03 23:37:13     Source: data/gtzan/audio
[INFO]  2025-12-03 23:37:13     Destination: data/processed/scalograms
[INFO]  2025-12-03 23:37:13     1000 audio files found.
Files preprocessing:  55%|████████████████████████████████████████████▎                                   | 554/1000 [03:56<03:25,  2.17it/s]/mnt/c/Users/anais/Documents/Cours 3A/Majeure_info/Technological_Foundations_of_Software_Development/TODO4/ml-audio/src/ml_audio/preprocess.py:37: UserWarning: PySoundFile failed. Trying audioread instead.
  y, sr = librosa.load(file_path, sr=None)
/mnt/c/Users/anais/Documents/Cours 3A/Majeure_info/Technological_Foundations_of_Software_Development/TODO4/ml-audio/.venv/lib/python3.10/site-packages/librosa/core/audio.py:184: FutureWarning: librosa.core.audio.__audioread_load
        Deprecated as of librosa version 0.10.0.
        It will be removed in librosa version 1.0.
  y, sr_native = __audioread_load(path, offset, duration, dtype)
[ERROR] 2025-12-03 23:41:10     Failed to process data/gtzan/audio/jazz/jazz.00054.wav:
Files preprocessing: 100%|███████████████████████████████████████████████████████████████████████████████| 1000/1000 [06:06<00:00,  2.73it/s]
[INFO]  2025-12-03 23:43:20     Preprocessing done.
```

### Training the model

For training a model, you should run the following command:

```
poetry run python -m ml_audio.train
```

You can modify the parameters of the model from the `train.py` file:

```
# This values can be modified
NUM_CLASSES = 10      # 10 genres
BATCH_SIZE = 16       # Size of batches
NUM_EPOCHS = 15       # 15 epochs
LEARNING_RATE = 0.001 # Learning rate for the Adam optimiser
```

The file will create the files `model_trained.pth` (the trained model)
and `class_map.json` (the file listing labels in order).

### Prediction

For using the model, you should use this command:

```
poetry run python -m ml_audio.predict <path_to_audio_file>
```

Where `<path_to_audio_file>` is the path to you input `.wav` file.

This command will print the prediction results, including the
predicted label and the confidence.

**Example:**

Input:

```
poetry run python -m ml_audio.predict data/gtzan/audio/blues/blues.00010.wav
```

Output:

```
[INFO]  2025-12-04 00:24:54     Loading classes list from class_map.json
[INFO]  2025-12-04 00:24:54     Loading the model architecture
[INFO]  2025-12-04 00:24:55     Loading weights from model_trained.pth
[INFO]  2025-12-04 00:24:55     Loading and processing the file ./data/gtzan/audio/blues/blues.00010.wav

--- Prediction results ---
File: ./data/gtzan/audio/blues/blues.00010.wav
Prediction: BLUES
Confidence: 98.46%
```

### Other source files

**`dataset.py`**

This Python file contains the definition of the GTZANDataset class. It defines methods
`init`, `len` and `getitem` that the model will use.

**`model.py`**

This file loads the ReNet-18 model (transfer learning) and modifies it accordingly to
the needs of the project. It only defines a function and should not be called by a user
in command line (but it can be used in other scripts).

## Testing

Unit tests are implemented thanks to `pytest`. They are written in the `test` folder
and can be run thanks to this command:

```
poetry run pytest
```

## Packaging

To create a Python Wheel (.whl) of the project:

```
poetry build
```

It creates a `dist` directory with a `.whl` and a `.tar.gz` archive.

## Static Code Analysis & Automation

**Ruff** (linter and import sorter) and **Black** (code formatter) are used to ensure
code quality and consistency.

### Static analysis setup

* **Configuration:** The configuration is centralised in the `pyproject.toml` file.
* **Installation:** Dependencies are managed via Poetry, run `poetry install`.
* **Manual Execution:**
    * Run linter: `poetry run ruff check .` and if automatic correction is possible run
    `poetry run ruff check . --fix`.
    * Run formatter: `poetry run black --check .` for checking and `poetry run black .` for reformatting.

### Automation (pre-commit hooks)

The [`pre-commit` framework](https://github.com/pre-commit/pre-commit-hooks) is used to automatically run
checks and formatting before every commit.

* **Configuration:** See `.pre-commit-config.yaml` at the root.
* **Setup:** To activate the hooks locally, run `poetry run pre-commit install`.
* **Usage:** Once installed, the hooks will run automatically on `git commit`.
If formatting issues are found, the commit will be blocked, and the files will be automatically fixed.
You simply need to `git add` the fixed files and commit again.

## Contributing

Contributions are welcome! Please follow this rules for contributing:

1. Fork the repository.
2. Create a branch (`git checkout -b feature/amazing-feature`).
3. Make sure your code passes the **static analysis** and **tests** (see above).
4. Commit your changes (`git commit -m 'Adding some amazing feature'`).
5. Push to the branch (`git push origin feature/amazing-feature`).
6. Open a Pull Request.

## License

Distributed under the MIT License. See [`LICENSE`](https://github.com/anaisd5/ml-audio/blob/main/LICENSE)
for more information.

## Contact

Anaïs Dubois - anais.dubois5@outlook.fr

Project Link: [github.com/anaisd5/ml-audio](https://github.com/anaisd5/ml-audio)
