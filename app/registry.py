import os
import glob
import numpy as np
import tensorflow as tf
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import models

from keras import optimizers
from app.model import *
from app.params import *
import matplotlib.pyplot as plt



def run_pipeline():
    config = Config()
    # Load data
    train_dataset, val_dataset, test_dataset = get_test_train_val_split()

    latest_model_path = get_latest_model_path()
    
    if latest_model_path:
        model = load_model(latest_model_path)
        print(f'[LOAD MODEL]: ✅ Model loaded from last training ({latest_model_path})')
    else:
        # Initialize the model
        model = initialize_model(config)
        print('[LOAD MODEL]: No previous model saved. Training from scratch')

    # Compile the model
    compile_model(model, config)
    print('[COMPILE MODEL]:  ✅ Model Compiled')

    print('[TRAIN MODEL]: ✅ Model Training starting...')
    # Train the model
    history = train_model(model, train_dataset, val_dataset)
    print('[TRAIN MODEL]: ✅ Model Training completed !')

    save_model(model)
    print(f'[SAVE MODEL]: ✅ Model saved to in {PATH_FULL_MODEL}')

    # Evaluate the model
    loss = evaluate_model(model, test_dataset)
    print(f"[EVALUATE MODEL] Model loss on test data: {loss}")
    
    return model, history

def get_latest_model_path():
    local_model_paths = glob.glob(f"{PATH_FULL_MODEL}/*")
    if not local_model_paths:
        return None
    path_to_load_from = sorted(local_model_paths, key=os.path.getmtime)[-1]
    return path_to_load_from
    
def load_data_from_directory(path, suffix):
    data_dict = {}
    for file_name in os.listdir(path):
        if file_name.endswith(".npy") and suffix in file_name:
            sequence_id = file_name.split(f"_{suffix}")[0]
            file_path = os.path.join(path, file_name)
            data_dict[sequence_id] = np.load(file_path)
    return data_dict

def create_tensorflow_dataset(tokens, melspecs, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((tokens, melspecs))
    dataset = dataset.shuffle(buffer_size=len(tokens))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

def get_test_train_val_split():
    tokens_data_dict = load_data_from_directory(PATH_PADDED_TOKENS, "tokens")
    melspec_data_dict = load_data_from_directory(PATH_PADDED_MELSPECS, "melspecs")
    
    assert tokens_data_dict.keys() == melspec_data_dict.keys()

    # Pour s'assurer qu'on fait bien correspondre les bons tokens aux bon melspecs
    sequence_ids = list(tokens_data_dict.keys())
    tokens_data = [tokens_data_dict[seq_id] for seq_id in sequence_ids]
    melspec_data = [melspec_data_dict[seq_id] for seq_id in sequence_ids]

    # Split data
    tokens_train, tokens_temp, melspec_train, melspec_temp = train_test_split(tokens_data, melspec_data, test_size=0.2, random_state=42)
    tokens_val, tokens_test, melspec_val, melspec_test = train_test_split(tokens_temp, melspec_temp, test_size=0.5, random_state=42)
    print(f"✅ Data split into:\n- Training: {len(tokens_train)/len(tokens_data)*100:.2f}% - {len(tokens_train)} \
        \n- Validation: {len(tokens_val)/len(tokens_data)*100:.2f}% - {len(tokens_val)}\
        \n- Test: {len(tokens_test)/len(tokens_data)*100:.2f}% - {len(tokens_test)}")

    # Convert lists to tensors
    tokens_train = tf.convert_to_tensor(tokens_train, dtype=tf.int32)
    melspec_train = tf.convert_to_tensor(melspec_train, dtype=tf.float32)
    
    tokens_val = tf.convert_to_tensor(tokens_val, dtype=tf.int32)
    melspec_val = tf.convert_to_tensor(melspec_val, dtype=tf.float32)
    
    tokens_test = tf.convert_to_tensor(tokens_test, dtype=tf.int32)
    melspec_test = tf.convert_to_tensor(melspec_test, dtype=tf.float32)
    
    train_dataset = create_tensorflow_dataset(tokens_train, melspec_train, BATCH_SIZE)
    val_dataset = create_tensorflow_dataset(tokens_val, melspec_val, BATCH_SIZE)
    test_dataset = create_tensorflow_dataset(tokens_test, melspec_test, BATCH_SIZE)
    
    return train_dataset, val_dataset, test_dataset

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
        rate=config.rate,
        var_conv_filters= config.var_conv_filters, 
        var_conv_kernel_size= config.var_conv_kernel_size, 
        var_rate= config.var_rate
    )
    return model


def compile_model(model, config):    
    lr_schedule = CustomLearningRateScheduler(embedding_dim=config.embedding_dim, warmup_steps=config.warmup_steps)
    optimizer = tf.keras.optimizers.Adam(lr_schedule, beta_1=config.beta_1, beta_2=config.beta_1, epsilon=config.epsilon)
    
    model.compile(optimizer=optimizer, loss=tf.keras.losses.MeanAbsoluteError())

def train_model(model, train_dataset, val_dataset, epochs=N_EPOCHS):
    
    checkpoint_path = f"{PATH_MODEL_CHECKPOINTS}/model_at_epoch_{{epoch:02d}}.ckpt"

    checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_loss",  
        save_best_only=True,   
        save_weights_only=True,
        save_freq=300,
        verbose=1
    )
    
    es_callback = EarlyStopping(
        monitor="val_loss",
        patience=30,
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(train_dataset, 
                        validation_data=(val_dataset),
                        epochs=epochs,
                        callbacks=[checkpoint_callback, es_callback],
                        verbose=1)
    
    # Visualise training loss and validation loss
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')  # Added this line
    plt.title('Model Loss Over Time')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.show()
    
    return history

def save_model(model):
    """
    Saves the entire model to the specified path.
    """
    # Save the model structure + weights + optimizer
    model_path = f"{PATH_FULL_MODEL}/model_{int(time.time())}"  # Using timestamp to create a unique name
    model.save(model_path)

def load_model(model_path):
    """
    Loads the entire model from the specified path.
    """
    model = models.load_model(model_path)
    return model

def predict_melspec(model, input_tokens):
    input_tokens = tf.expand_dims(input_tokens, 0) 
    input_tokens = tf.convert_to_tensor(input_tokens, dtype=tf.int32)
    prediction = model.predict(input_tokens)
    return prediction

def evaluate_model(model, test_dataset):
    loss = model.evaluate(test_dataset)
    return loss