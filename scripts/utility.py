import logging
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import yaml
from keras.utils import to_categorical
def parse_config(config_file):

    with open(config_file, "rb") as f:
        config = yaml.safe_load(f)
    return config


def set_logger(log_path):

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(log_path, mode="w")
    formatter = logging.Formatter(
        "%(asctime)s : %(levelname)s : %(name)s : %(message)s"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info("Finished logger configuration!")
    return logger



def load_data(processed_features, processed_labels, train_or_test_subsetter):
    """
    Returns:
        [tuple]: feature matrix and target variable
    """
    #get the row numbers you want to subset with (either the training or test set)
    with open(train_or_test_subsetter, 'rb') as f:
        subsetter = pickle.load(f)
    with open(processed_features, 'rb') as f:
        features = pickle.load(f)
    #convert features into a numpy array
    features = np.array(features).reshape(len(features), 224, 400, 1)
    #subset the array of features to include just the ones in the subsetter
    features = features[subsetter, :, :, :]

    with open(processed_labels, 'rb') as f:
        labels = pickle.load(f)
    labels_df = pd.DataFrame(labels, columns=['Label'])

    #subset your data
    labels_df = labels_df.iloc[subsetter, :]
    labels_df = to_categorical(labels_df)
    return features, labels_df, ['Spectrogram']


def load_submission_test_data(location):
    #create dataframe with submission test data
    #combine references for test into dataframe

    with open(str(location)+'//row_ids_test.pickle', 'rb') as f:
        row_ids_test = pickle.load(f)

    with open(str(location)+'//clips_test.pickle', 'rb') as f:
        clips_test = pickle.load(f)

    with open(str(location)+'//clip_segments_test.pickle', 'rb') as f:
        clip_segments_test = pickle.load(f)

    submission_clips_df = pd.DataFrame(columns=['Row_ID', 'Clip', 'Clip_Segment'])
    submission_clips_df['Row_ID'] = row_ids_test
    submission_clips_df['Clip'] = clips_test
    submission_clips_df['Clip_Segment'] = clip_segments_test

    with open(str(location)+'//spectrograms_test.pickle', 'rb') as f:
        spectrograms = pickle.load(f)
    spectrograms = np.array(spectrograms).reshape(len(spectrograms), 224, 400, 1)

    return submission_clips_df, spectrograms
