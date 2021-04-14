import os
import click
import pandas as pd
import keras.models
from utility import load_data, parse_config, set_logger


@click.command()
@click.argument("config_file", type=str, default="config.yaml")
def predict(config_file):

    ##################
    # configure logger
    ##################
    logger = set_logger("./log/predict.log")

    ##################
    # Load config from config file
    ##################
    logger.info(f"Load config from {config_file}")
    config = parse_config(config_file)

    #get the root directory of the repository
    dirname = os.path.dirname(os.path.dirname(__file__))

    model_path = os.path.join(dirname, config["predict"]["model_path"])
    processed_path = os.path.join(dirname, config["predict"]["processed_path"])
    predicted_file = os.path.join(dirname, config["predict"]["predicted_file"])
    export_result = config["predict"]["export_result"]
    train_test_split_method = config["predict"]["train_test_split_method"]

    logger.info(f"config: {config['predict']}")

    ##################
    # Load model & test set
    ##################
    # Load model
    logger.info(f"-------------------Load the trained model-------------------")
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

    ##################
    # Make prediction and evaluate
    ##################
    logger.info(f"-------------------Predict and evaluate-------------------")
    y_hat = trained_model.predict(X)
    y_classes = y_hat.argmax(axis=1)
    output = pd.DataFrame(y)
    output["prediction"] = y_classes
    if export_result:
        output.to_csv(predicted_file, index=False)
        logger.info(f"Export prediction to : {predicted_file}")



    # Evaluate the model on the test data using `evaluate`
    logger.info("Evaluate on test data")
    results = trained_model.evaluate(X, y)
    logger.info("test loss, test acc:", results)




if __name__ == "__main__":
    predict()
