# Kaggle_Rainforest_Species_Audio
The code in this repository was used for the [Rainforest Connection Species Audio Kaggle Competition](https://www.kaggle.com/c/rfcx-species-audio-detection).

### Overview

Competitors were provided with audio samples of 24 different endangered species (birds and frogs). The challenge was to detect the presence of any of these species within a set of 60 second audio clips.

Simply put, the approach I took was to convert short clips of audio into images. When converted into images, each song becomes visually recognizable. I then used the same sort of technology used for image recognition to compare the songs to each other. 

More specifically...I converted the audio clips into Mel spectrograms. A [spectrogram](https://en.wikipedia.org/wiki/Spectrogram) is a visual representation of sound which shows the intensity of certain frequencies as they vary over time. A Mel spectrogram is a spectrogram that has been converted to the [Mel scale](https://en.wikipedia.org/wiki/Mel_scale). This scales the spectrogram so that the image provides more information about the range of frequencies that can be easily differentiated by the human ear, and doesn't waste much space on the frequencies outside that range. I then trained a [Convolutional Neural Net](https://en.wikipedia.org/wiki/Convolutional_neural_network) to classify the sounds.

### Code

The code provided contains several separate scripts which perform the following tasks:

##### config.yaml:
  Contains the settings for all the subsequent scripts. Edit this file to make any changes to how you want the scripts to run.
##### etl.py: 
  Pre-processes the audio data into Mel spectrograms and them to pickle files. 
  Optionally will create augmented data to use for training (adding noise, pitch shift, etc). 
  Splits the training set provided by Kaggle into a training and test set.
##### train.py:
  Trains an convolutional neural net and saves it to disk.
##### predict.py:
  Using the saved CNN model, predicts values for the held-out test set (as defined by the etl.py script), evaluates performance, and returns accuracy and loss metrics.
##### submission.py:
  Using the saved CNN model, predicts values for the final test set which will be used by Kaggle to evaluate our model's performance. Converts the results into the format needed for the competition submission file.
##### utility.py:
  Contains several helper functions, used for loading data and logging.

The scripts can be run within the same directory structure shown in the repo. The /data/train and /data/test folders must be populated with the .flac audio files from Kaggle. See text files in those folders for instructions on downloading the audio files. All other data needed is in the repo.
