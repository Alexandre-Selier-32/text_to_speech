import tensorflow as tf
from app.model.Transformer import Transformer
import os
import numpy as np
from app.params import PATH_PADDED_TOKENS, PATH_PADDED_MELSPECS
from sklearn.model_selection import train_test_split

def load_data_from_directory(path):
    data = []
    for file_name in os.listdir(path):
        if file_name.endswith(".npy"):
            file_path = os.path.join(path, file_name)
            array = np.load(file_path)
            data.append(array)
    return data

def prepare_data():
    tokens_data = load_data_from_directory(PATH_PADDED_TOKENS)
    melspec_data = load_data_from_directory(PATH_PADDED_MELSPECS)
    
    # Split data into train, validation, and test sets
    tokens_train, tokens_temp, melspec_train, melspec_temp = train_test_split(tokens_data, melspec_data, test_size=0.2, random_state=42)
    tokens_val, tokens_test, melspec_val, melspec_test = train_test_split(tokens_temp, melspec_temp, test_size=0.5, random_state=42)
    
    return (tokens_train, melspec_train), (tokens_val, melspec_val), (tokens_test, melspec_test)


def initialize_model(config):
    model = Transformer(
        num_layers=config.num_layers,
        embedding_dim=config.embedding_dim,
        num_heads=config.num_heads,
        dff=config.dff,
        input_vocab_size=config.input_vocab_size,
        max_position_encoding=config.max_position_encoding,
        conv_kernel_size=config.conv_kernel_size,
        conv_filters=config.conv_filters,
        rate=config.rate
    )
    return model

# ajouter learning rate scheduler 
def compile_model(model):
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

def train_model(model, train_dataset, val_dataset):
    model.fit(train_dataset, validation_data=val_dataset, epochs=10)
    
