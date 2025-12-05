# Tutorial: Train your first Music Classifier

In this tutorial, you will learn how to use `ml-audio` to train a Deep Learning model capable of recognizing music genres.

We will go through the whole pipeline: from raw audio files to a trained neural network.

**Prerequisites:**
* Python 3.10+ installed
* Poetry installed
* The GTZAN dataset downloaded

## Step 1: Install dependencies

First, make sure your environment is ready. Run this command at the root of the project:

```
poetry install
```

## Step 2: Prepare the data

Download the GTZAN dataset. Extract the content of `genres_original` into the `data/gtzan/audio` folder of the project.

Your structure should look like this: `data/gtzan/audio/blues/blues.00000.wav`.

## Step 3: Turn audio into images

This model uses a Convolutional Neural Network (CNN), which is great at analyzing images. We need to convert the audio waves into "scalograms" (visual representations).

Run the preprocessor:

```
poetry run python -m ml_audio.preprocess
```

**Expected result:** You will see a progress bar. New `.npy` files will appear in `data/processed/scalograms`.

## Step 4: Train the model

Now comes the learning part. We will train a ResNet-18 model.

To make this tutorial quick, verify that `src/ml_audio/train.py` is configured with a small number of epochs (e.g., 5).

Run the training script:

```
poetry run python -m ml_audio.train
```

**Expected result:** The script will output the "Loss" and "Accuracy" for each epoch. After a few minutes, a `model_trained.pth` file is created.

## Step 5: Make a Prediction

Let's test your new model on a real file!

```
poetry run python -m ml_audio.predict data/gtzan/audio/jazz/jazz.00054.wav
```

You should see an output similar to:

```
--- Prediction results ---
File: ./data/gtzan/audio/jazz/jazz.00054.wav
Prediction: JAZZ
Confidence: 95.43%
```

Congratulations! You have successfully trained and tested an audio classification model.
