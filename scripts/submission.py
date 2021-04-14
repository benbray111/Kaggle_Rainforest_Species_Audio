import os
import click
import pandas as pd
import keras.models
from utility import parse_config, set_logger, load_submission_test_data


@click.command()
@click.argument("config_file", type=str, default="config.yaml")
def create_submission(config_file):
    ##################
    # configure logger
    ##################
    logger = set_logger("./log/submission.log")

    ##################
    # Load config from config file
    ##################
    logger.info(f"Load config from {config_file}")
    config = parse_config(config_file)
    #get root directory of repository
    dirname = os.path.dirname(os.path.dirname(__file__))
    #set local variables from config file
    model_path = os.path.join(dirname, config["submission"]["model_path"])
    processed_path = os.path.join(dirname, config["submission"]["processed_path"])
    submission_file = os.path.join(dirname, config["submission"]["submission_file"])
    submission_template = os.path.join(dirname, config["submission"]["submission_template"])

    ##################
    # Load trained model
    ##################
    #logger.info("-------------------Load the trained model-------------------")
    trained_model = keras.models.load_model(model_path)

    ##################
    # Load data
    ##################
    #get processed data
    #this loads both information about each clip, and the spectrograms
    submission_clips_df, X = load_submission_test_data(processed_path)
    #get template for submission
    the_template = pd.read_csv(submission_template)

    ##################
    # Make Predictions
    ##################
    y_hat = trained_model.predict(X)
    test_set_predictions = pd.DataFrame(y_hat, columns = [x for x in the_template if x != 'recording_id'])
    test_set_predictions = pd.concat([submission_clips_df, test_set_predictions], axis=1)
    #adjusts for the fact that each clip can test positive for mult species
    test_set_predictions = test_set_predictions.groupby('Row_ID').max()
    test_set_predictions = test_set_predictions[[x for x in test_set_predictions if x != 'Row_ID' and x != 'Clip_Segment']]
    test_set_predictions = test_set_predictions.rename(columns={'Clip': 'recording_id'})
    #export file
    test_set_predictions.to_csv(submission_file, index=False)

if __name__ == "__main__":
    create_submission()
