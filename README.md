# Audio Classification Project (Machine Learning)

This project is an implementation of an audio classification system
using `librosa` for scalogram creation and `PyTorch` for the
machine learning model.

It is managed with [Poetry](https://python-poetry.org/) for dependency
and environment management.

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

## Usage

*TODO*

## Packaging

To create a Python Wheel (.whl) of the project:

```
poetry build
```
