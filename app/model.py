import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from app.model.Transformer import Transformer
from app.model.CustomLearningRateScheduler import CustomLearningRateScheduler
from app.params import *

def load_data_from_directory(path, suffix):
    data_dict = {}
    for file_name in os.listdir(path):
        if file_name.endswith(".npy") and suffix in file_name:
            sequence_id = file_name.split(f"_{suffix}")[0]
            file_path = os.path.join(path, file_name)
            data_dict[sequence_id] = np.load(file_path)
    return data_dict

def get_test_train_val_split():
    tokens_data_dict = load_data_from_directory(PATH_PADDED_TOKENS, "tokens")
    melspec_data_dict = load_data_from_directory(PATH_PADDED_MELSPECS, "melspecs")
    
    assert tokens_data_dict.keys() == melspec_data_dict.keys()

    # Check qu'on fait bien correspondre les bons tokens aux bon melspecs
    sequence_ids = list(tokens_data_dict.keys())
    tokens_data = [tokens_data_dict[seq_id] for seq_id in sequence_ids]
    melspec_data = [melspec_data_dict[seq_id] for seq_id in sequence_ids]

    # Split data
    tokens_train, tokens_temp, melspec_train, melspec_temp = train_test_split(tokens_data, melspec_data, test_size=0.2, random_state=42)
    tokens_val, tokens_test, melspec_val, melspec_test = train_test_split(tokens_temp, melspec_temp, test_size=0.5, random_state=42)
    print(f"✅ Data split into:\n- Training: {len(tokens_train)/len(tokens_data)*100:.2f}%\n- Validation: {len(tokens_val)/len(tokens_data)*100:.2f}%\n- Test: {len(tokens_test)/len(tokens_data)*100:.2f}%")

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

def compile_model(model, config):
    lr_schedule = CustomLearningRateScheduler(embedding_dim=config.embedding_dim, warmup_steps=config.warmup_steps)
    optimizer = tf.keras.optimizers.Adam(lr_schedule, beta_1=config.beta_1, beta_2=config.beta_1, epsilon=config.epsilon)

    model.compile(optimizer=optimizer, loss='mean_squared_error')


def train_model(model, train_dataset, val_dataset):
    model.fit(train_dataset, validation_data=val_dataset, epochs=10)
    
