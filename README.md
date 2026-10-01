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

For those files, you can use the optional `--log` argument to indicate
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
Files preprocessing:  55%|████████████████████████████████████████████▎                                   | 554/1000 [03:56<03:25,  2.17it/s]/ml_audio/preprocess.py:37: UserWarning: PySoundFile failed. Trying audioread instead.
  y, sr = librosa.load(file_path, sr=None)
/ml-audio/.venv/lib/python3.10/site-packages/librosa/core/audio.py:184: FutureWarning: librosa.core.audio.__audioread_load
        Deprecated as of librosa version 0.10.0.
        It will be removed in librosa version 1.0.
  y, sr_native = __audioread_load(path, offset, duration, dtype)
[ERROR] 2025-12-03 23:41:10     Failed to process data/gtzan/audio/jazz/jazz.00054.wav:
Files preprocessing: 100%|███████████████████████████████████████████████████████████████████████████████| 1000/1000 [06:06<00:00,  2.73it/s]
[INFO]  2025-12-03 23:43:20     Preprocessing done.
```

By default, the audio files are transformed into CQT scalograms (84
frequency bins, one per semitone). With the `--features` argument, you can
create mel spectrograms instead (128 frequency bands), which are saved in
`data/processed/melspectrograms`:

```
poetry run python -m ml_audio.preprocess --features=mel
```

### Training the model

For training a model, you should run the following command:

```
poetry run python -m ml_audio.train
```

You can modify the parameters of the model from the `train.py` file:

```
# This values can be modified
NUM_CLASSES = 10        # 10 genres
BATCH_SIZE = 16         # Size of batches (default, see --batch-size)
NUM_WORKERS = 2         # Processes loading the data (default, see --num-workers)
NUM_THREADS = 4         # CPU cores used by PyTorch (default, see --num-threads)
NUM_EPOCHS = 50         # Maximal number of epochs
LEARNING_RATE = 0.0001  # Learning rate for the AdamW optimiser
WEIGHT_DECAY = 0.0001   # Weight decay (regularisation) for AdamW
PATIENCE = 5            # Stop if the validation loss does not improve for 5 epochs
SEED = 42               # For reproductible results
MIXUP_ALPHA = 0.4       # Parameter of the Beta distribution for mixup
FROZEN_LAYERS = 0       # Groups of layers frozen (default, see --frozen-layers)
```

The parameters of SpecAugment (number and size of the masks) can be modified
from the `dataset.py` file.

The size of batches, the number of processes loading the data and the
number of CPU cores used by PyTorch can also be changed in command line.
If your computer does not have a lot of memory (e.g. WSL with less than
8 GB) or freezes during the training, you can reduce them:

```
poetry run python -m ml_audio.train --batch-size=8 --num-workers=0 --num-threads=2
```

By default, only 4 CPU cores are used so that the computer stays usable
during the training. If you have more cores and do not need your computer
in the meantime, you can increase `--num-threads` to train faster.

**How the training works:**

* **Segments:** each track (30 s) is cut into 10 segments of about 3 s. The
model is trained on the segments, which gives 10 times more examples.
* **Split:** the tracks are split with the *fault-filtered* partition of
GTZAN ([Sturm, 2013](https://arxiv.org/abs/1306.1461);
[Kereliuk et al., 2015](https://github.com/coreyker/dnn-mgr)): 443 tracks
for training, 197 for validation and 290 for test. This partition removes
the duplicates and puts all the tracks of an artist in the same split, so
the results are not too optimistic. The lists of tracks are in the
`src/ml_audio/splits` folder. The segments of a track are always in the
same split.
* **Data augmentation:** for the training set only, three methods are
used to limit overfitting:
    * *random crop:* the segments start at a random position of the track;
    * *SpecAugment* ([Park et al., 2019](https://arxiv.org/abs/1904.08779)):
    random bands of frequencies and intervals of time are hidden;
    * *mixup* ([Zhang et al., 2018](https://arxiv.org/abs/1710.09412)):
    each segment of a batch is mixed with another one, and the loss is the
    same mix of the losses of the two labels.

  To train without data augmentation (e.g. for comparison), use the
  `--no-augment` argument:

  ```
  poetry run python -m ml_audio.train --no-augment
  ```
* **Frozen layers:** the `--frozen-layers` argument (from 0 to 4) freezes
the first groups of layers of the ResNet-18: they keep their pretrained
weights and only the last layers are trained. By default, nothing is
frozen.

  **Note:** freezing layers does not improve the performance. In our tests,
  freezing 2 or 3 groups gave a lower accuracy on the test set (61.0% and
  62.4% instead of 67.2%, one training each). The option is kept for
  experimentation.

  ```
  poetry run python -m ml_audio.train --frozen-layers=2
  ```
* **Features:** by default, the model is trained on the CQT scalograms. The
`--features=mel` argument trains it on the mel spectrograms (they should
first be created with `preprocess`, see above). The files of the mel model
have their own names (`model_trained_mel.pth`, `class_map_mel.json` and
`results_mel`), so the two models can be kept at the same time.

  ```
  poetry run python -m ml_audio.train --features=mel
  ```

  **Note:** alone, the mel spectrograms do not improve the performance. In
  our tests, the accuracy on the test set was 65.5% with mel and 67.2% with
  CQT (one training each), a difference too small to be significant. But
  the two models do not make the same errors (the mel spectrograms were
  better on pop and metal, the CQT scalograms on jazz and rock), so
  **combining them improves the performance** (see the `--features=both`
  argument in the Evaluation and Prediction sections).
* **Standardisation:** the scalograms are standardised (mean 0, standard
deviation 1) with values computed on the training set.
* **Optimisation:** the learning rate is divided by 2 when the validation
loss does not improve for 2 epochs. The training stops when the validation
loss does not improve for `PATIENCE` epochs (early stopping), and the
**best** model (lowest validation loss) is kept.

The file will create the files `model_trained.pth` (the best model),
`class_map.json` (the file listing labels in order and the standardisation
values) and a `results` folder containing:

* `training_history.csv`: loss, accuracy and learning rate of each epoch;
* `training_curves.png`: training and validation curves (loss and accuracy);
* the results of the evaluation on the test set (see next section).

### Evaluation

At the end of the training, the best model is evaluated on the test set.
You can also evaluate the saved model again with this command:

```
poetry run python -m ml_audio.evaluate
```

The `--batch-size`, `--num-workers` and `--num-threads` arguments are also
available.
The prediction of a track is the mean of the probabilities of its
segments. The command creates in the `results` folder:

* `test_report.txt`: accuracy (on segments and on tracks), ROC AUC,
precision, recall, F1-score, sensitivity and specificity of each genre;
* `confusion_matrix.png`: the confusion matrix;
* `roc_curves.png`: the ROC curve of each genre (one versus the others).

**Example (extract of `test_report.txt`):**

```
Number of tracks: 290
Number of segments: 2900
Accuracy (segments): 60.93%
Accuracy (tracks): 67.24%
Macro ROC AUC (tracks, one vs rest): 0.930
```

**Combining the CQT and mel models**

With the `--features` argument, you can choose the model to evaluate: `cqt`
(default), `mel` or `both`. With `both`, the probabilities given by the two
models are averaged. The two models should first be trained (`train` with
`--features=cqt` and `--features=mel`). The results are saved in the
`results_mel` and `results_both` folders.

```
poetry run python -m ml_audio.evaluate --features=both
```

Results on the test set (290 tracks, one training for each model):

| Model (`--features`) | Accuracy (tracks) | Macro ROC AUC |
|----------------------|-------------------|---------------|
| `cqt`                | 67.24%            | 0.930         |
| `mel`                | 65.52%            | 0.931         |
| `both`               | **70.34%**        | **0.946**     |

### Prediction

For using the model, you should use this command:

```
poetry run python -m ml_audio.predict <path_to_audio_file>
```

Where `<path_to_audio_file>` is the path to you input `.wav` file.

This command will print the prediction results, including the
predicted label and the confidence. The file is cut into segments of
about 3 s (as during training) and the probabilities of the segments
are averaged.

**Example** (`jazz.00073.wav` is in the test set, so the model has never
seen it during training):

Input:

```
poetry run python -m ml_audio.predict data/gtzan/audio/jazz/jazz.00073.wav
```

Output:

```
[INFO]  2026-10-01 09:08:38     Loading classes list from class_map.json
[INFO]  2026-10-01 09:08:38     Loading the model architecture
[INFO]  2026-10-01 09:08:38     Loading weights from model_trained.pth
[INFO]  2026-10-01 09:08:39     Loading and processing the file data/gtzan/audio/jazz/jazz.00073.wav

--- Prediction results ---
File: data/gtzan/audio/jazz/jazz.00073.wav
Features: cqt
Number of segments (3 s): 10
Prediction: JAZZ
Confidence: 84.64%
```

As for the evaluation, the `--features` argument chooses the model: `cqt`
(default), `mel` or `both` (mean of the probabilities of the two models,
which should both be trained first):

```
poetry run python -m ml_audio.predict data/gtzan/audio/jazz/jazz.00073.wav --features=both
```

### Other source files

**`features.py`**

This file contains the function transforming an audio signal into a CQT scalogram
or a mel spectrogram (`compute_features`). It is used by `preprocess.py` and
`predict.py`, so that the same transformation is applied to all files. It also gives
the names of the files created for each type of features (`get_output_paths`).

**`dataset.py`**

This Python file contains the definition of the GTZANDataset class. It defines methods
`init`, `len` and `getitem` that the model will use. An item of the dataset is a
segment (about 3 s) of a scalogram. The file also contains functions for loading the
fault-filtered split (`load_split`), cutting a scalogram into segments
(`split_into_segments`), hiding random frequencies and times (`spec_augment`) and
computing the standardisation values (`compute_mean_std`).

**`model.py`**

This file loads the ReNet-18 model (transfer learning) and modifies it accordingly to
the needs of the project. The first layer is adapted to 1 channel and keeps the
pretrained filters (summed over the 3 RGB channels). The first groups of layers
can be frozen (`frozen_layers`). It only defines a function and
should not be called by a user in command line (but it can be used in other scripts).

## Documentation

The project documentation is generated using **Sphinx**. It extracts docstrings from the code (reStructuredText format) and includes manually written tutorials.

### How to generate the documentation

1. Ensure development dependencies are installed: `poetry install`
2. Go to the `docs` directory: `cd docs`
3. Build the HTML files: `poetry run make html`
4. Open the generated file `docs/_build/html/index.html` in your web browser to
view the documentation.

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

Anaïs Dubois

Project Link: [github.com/anaisd5/ml-audio](https://github.com/anaisd5/ml-audio)
