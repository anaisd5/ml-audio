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

*TODO*

### Preprocessing

For preprocessing all audio files, you can run the `preprocess.py` script:

```
poetry run python src/ml_audio/preprocess.py 
```

When you run this command, some files will fail. It is a known behaviour. 
At the end though, the file preprocessing should be complete.

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

## Packaging

To create a Python Wheel (.whl) of the project:

```
poetry build
```
