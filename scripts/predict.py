# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 12:53:47 2021

@author: Ben
"""



import logging
from pathlib import Path


import click
import pandas as pd
import sklearn
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.python.keras import utils
import keras.models
import numpy as np


from utility import load_data, parse_config, set_logger


@click.command()
@click.argument("config_file", type=str, default="config.yaml")
def predict(config_file):
    """
    Main function that runs predictions
    Args:
        config_file [str]: path to config file
    Returns:
        None
    """
    ##################
    # configure logger
    ##################
    logger = set_logger("./log/predict.log")

    ##################
    # Load config from config file
    ##################
    logger.info(f"Load config from {config_file}")
    config = parse_config(config_file)

    model_path = Path(config["predict"]["model_path"])
    processed_path = config["predict"]["processed_path"]
    predicted_file = config["predict"]["predicted_file"]
    export_result = config["predict"]["export_result"]
    train_test_split_method = config["predict"]["train_test_split_method"]

    logger.info(f"config: {config['predict']}")

    ##################
    # Load model & test set
    ##################
    # Load model
    logger.info(f"-------------------Load the trained model-------------------")
    #with open(model_path, "rb") as f:
    #    trained_model = load(f)
    
    
    trained_model = keras.models.load_model(model_path)

    # Load test set
    logger.info(f"Load the test data from {processed_path}")
    
    if train_test_split_method == 'Simple':
        pf = str(processed_path)+'//'+'spectrograms.pickle'
        pl = str(processed_path)+'//'+'labels.pickle'
        tts = str(processed_path)+'//'+'test_rowids.pickle'    
    X, y, cols = load_data(processed_features=pf, processed_labels=pl, train_or_test_subsetter=tts)    
    
    logger.info(f"cols: {cols}")
    logger.info(f"X: {X.shape}")
    logger.info(f"y: {y.shape}")
    print('flag1')
    ##################
    # Make prediction and evaluate
    ##################
    logger.info(f"-------------------Predict and evaluate-------------------")
    #https://www.tensorflow.org/guide/keras/train_and_evaluate
    y_hat = trained_model.predict(X)
    y_classes = y_hat.argmax(axis=1)
    #logger.info(f"Classification report: \n {classification_report(y, y_hat)}")
    #convert each of the predictions to a label name, rather than a list of 24 probabilities
    #y_classes = y_hat.argmax(axis=-1)
    #labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    #predicted_label = sorted(labels)[y_classes]
    output = pd.DataFrame(y)
    output["prediction"] = y_classes
    if export_result:
        output.to_csv(predicted_file, index=False)
        logger.info(f"Export prediction to : {predicted_file}")
    


    # Evaluate the model on the test data using `evaluate`
    print("Evaluate on test data")
    results = trained_model.evaluate(X, y)
    print("test loss, test acc:", results)

    # Generate predictions (probabilities -- the output of the last layer)
    # on new data using `predict`
    #print("Generate predictions for 3 samples")
    #predictions = model.predict(x_test[:3])
    #print("predictions shape:", predictions.shape)


if __name__ == "__main__":
    predict()