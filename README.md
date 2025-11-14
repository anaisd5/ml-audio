# Audio Classification Project (Machine Learning)

This project is an implementation of an audio classification system
using `librosa` for scalogram creation and `PyTorch` for the
machine learning model.

It is managed with [Poetry](https://python-poetry.org/) for dependency
and environment management.

The `.wav` files from the [GTZAN database](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification?resource=download-directory) were used as an input.

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
poetry run python src/ml_audio/preprocess.py
poetry run python src/ml_audio/train.py
poetry run python src/ml_audio/predict.py <path_to_audio_file>
```

### Preprocessing

For preprocessing all audio files, you can run the `preprocess.py` script:

```
poetry run python src/ml_audio/preprocess.py
```

When you run this command, some files will fail. It is a known behaviour. 
At the end though, the file preprocessing should be complete (but some files
may be missing).

```
poetry run python src/ml_audio/preprocess.py 

Starting preprocessing...
Source: data/gtzan/audio
Destination: data/processed/scalograms
1000 audio files found.
Files preprocessing:  55%|██████████████████████████████████▋                            | 551/1000 [02:41<01:52,  3.99it/s]
ml-audio/src/ml_audio/preprocess.py:27: UserWarning: PySoundFile failed. Trying audioread instead.
  y, sr = librosa.load(file_path, sr=None) # y is the audio signal, sr is the sample rate
ml-audio/.venv/lib/python3.10/site-packages/librosa/core/audio.py:184: FutureWarning: librosa.core.audio.__audioread_load
        Deprecated as of librosa version 0.10.0.
        It will be removed in librosa version 1.0.
  y, sr_native = __audioread_load(path, offset, duration, dtype)
Error processing file data/gtzan/audio/jazz/jazz.00054.wav: 
Files preprocessing: 100%|██████████████████████████████████████████████████████████████| 1000/1000 [04:16<00:00,  3.90it/s]
Preprocessing done.
```

### Training the model

For training a model, you should run the following command:

```
poetry run python src/ml_audio/train.py
```

You can modify the parameters of the model from the `train.py` file:

```
# This values can be modified
NUM_CLASSES = 10      # 10 genres
BATCH_SIZE = 16       # Size of batches
NUM_EPOCHS = 20       # 20 epochs
LEARNING_RATE = 0.001 # Learning rate for the Adam optimiser
```

The file will create the files `model_trained.pth` (the trained model) 
and `class_map.json` (the file listing labels in order).

### Prediction

*TODO*

### Other files

**`dataset.py`**

This Python file contains the definition of the GTZANDataset class. It defines methods 
`init`, `len` and `getitem` that the model will use.

**`model.py`**

This file loads the ReNet-18 model (transfer learning) and modifies it accordingly to 
the needs of the project. It only defines a function and should not be called by a user 
in command line (but it can be used in other scripts).

## Packaging

To create a Python Wheel (.whl) of the project:

```
poetry build
```
