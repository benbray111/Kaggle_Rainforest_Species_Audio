import os
import click
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from utility import load_data, parse_config, set_logger


@click.command()
@click.argument("config_file", type=str, default="config.yaml")
def train(config_file):

    ##################
    # configure logger
    ##################
    logger = set_logger("./log/train.log")

    ##################
    # Load config from config file
    ##################
    logger.info(f"Load config from {config_file}")
    config = parse_config(config_file)
    #get path of root directory of repository
    dirname = os.path.dirname(os.path.dirname(__file__))
    processed_path = os.path.join(dirname, config["train"]["processed_path"])
    model_path = os.path.join(dirname, config["train"]["model_path"])

    train_test_split_method = config["train"]["train_test_split_method"]
    num_classes = config["train"]["num_classes"]
    num_epochs = config["train"]["num_epochs"]
    optimizer = config["train"]["optimizer"]
    lr = config["train"]["lr"]
    neur = config["train"]["neur"]
    dropout_rate = config["train"]["dropout_rate"]
    batch_size = config["train"]["batch_size"]
    logger.info(f"config: {config['train']}")

    ##################
    # Load data
    ##################
    if train_test_split_method == 'Simple':
        pf = str(processed_path)+'//'+'spectrograms.pickle'
        pl = str(processed_path)+'//'+'labels.pickle'
        tts = str(processed_path)+'//'+'train_rowids.pickle'
    logger.info("-------------------Load the processed data-------------------")

    X, y, cols = load_data(processed_features=pf, processed_labels=pl, train_or_test_subsetter=tts)
    logger.info(f"cols: {cols}")
    logger.info(f"X: {X.shape}")
    logger.info(f"y: {y.shape}")

    ##################
    # Set & train model
    ##################
    cnn_model = create_model(optimizer=optimizer, lr=lr, neur=neur,\
                             dropout_rate=dropout_rate, num_classes=num_classes)
    cnn_model.fit(X, y, batch_size=batch_size, epochs=num_epochs)

    ##################
    # Persist model
    ##################
    logger.info(f"-------------------Persist model-------------------")
    cnn_model.save(model_path)


def create_model(optimizer='adam', neur=64, lr=.01, dropout_rate=.1, num_classes=24):

    # Initiating an empty neural network
    cnn_model = Sequential(name='cnn_model')

    # Adding convolutional layer
    cnn_model.add(Conv2D(filters=16,
                         kernel_size=(3, 3),
                         activation='relu',
                         input_shape=(224, 400, 1)))

    # Adding max pooling layer
    cnn_model.add(MaxPooling2D(pool_size=(2, 4)))

    # Adding convolutional layer
    cnn_model.add(Conv2D(filters=32,
                         kernel_size=(3, 3),
                         activation='relu'))
    # Adding max pooling layer
    cnn_model.add(MaxPooling2D(pool_size=(2, 4)))
    # Adding a flattened layer to input our image data
    cnn_model.add(Flatten())


    # # Adding a dense layer with 64 neurons
    cnn_model.add(Dense(neur, activation='relu'))


    # Adding a dropout layer for regularization
    cnn_model.add(Dropout(0.25))

    # Adding an output layer
    cnn_model.add(Dense(num_classes, activation='softmax'))

    # Compiling our neural network
    cnn_model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])

    return cnn_model


if __name__ == "__main__":
    train()
