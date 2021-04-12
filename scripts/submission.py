# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 21:47:07 2021

@author: Ben
"""


import logging
from pathlib import Path

# from cloudpickle import dump
from pickle import dump

import click
import pandas as pd
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.python.keras import utils
from tensorflow import random
import keras.models
import numpy as np



from utility import parse_config, set_logger, load_submission_test_data 


@click.command()
@click.argument("config_file", type=str, default="config.yaml")
def create_submission(config_file):
    print('hello world')

    ##################
    # Load config from config file
    ##################
    #logger.info(f"Load config from {config_file}")
    config = parse_config(config_file)

    model_path = Path(config["submission"]["model_path"])
    processed_path = config["submission"]["processed_path"]
    submission_file = config["submission"]["submission_file"]
    submission_template = config["submission"]["submission_template"]
    
    ##################
    # Load trained model
    ##################
    # Load model
    #logger.info("-------------------Load the trained model-------------------")
        
    trained_model = keras.models.load_model(model_path)

    ##################
    # Load data
    ##################
    #get processed data
    submission_clips_df, X = load_submission_test_data(processed_path)
    #get template for submission
    the_template = pd.read_csv(submission_template)
    
    
    ##################
    # Make Predictions
    ##################  
    
    y_hat = trained_model.predict(X)
    
    
    test_set_predictions = pd.DataFrame(y_hat, columns = [x for x in the_template if x != 'recording_id'])
    test_set_predictions = pd.concat([submission_clips_df, test_set_predictions], axis=1)
    test_set_predictions = test_set_predictions.groupby('Row_ID').max()
    test_set_predictions = test_set_predictions[[x for x in test_set_predictions if x !='Row_ID' and x != 'Clip_Segment']]
    test_set_predictions = test_set_predictions.rename(columns={'Clip': 'recording_id'})
    test_set_predictions.to_csv(submission_file, index=False)
    
    

if __name__ == "__main__":
    create_submission()