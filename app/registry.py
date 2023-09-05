import os
import glob
import numpy as np
import tensorflow as tf
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint

from app.model import *
from app.params import *
import matplotlib.pyplot as plt


def get_latest_model_path():
    local_model_paths = glob.glob(f"{PATH_FULL_MODEL}/*")
    if not local_model_paths:
        return None
    path_to_load_from = sorted(local_model_paths, key=os.path.getmtime)[-1]
    return path_to_load_from
    
def load_data_from_directory(path, suffix, data_fraction):
    data_dict = {}

    sorted_files = sorted(os.listdir(path))
    num_files_to_load = int(len(sorted_files) * data_fraction)

    for file_name in sorted_files[:num_files_to_load]:
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

def get_test_train_val_split(data_fraction=1.0):
    tokens_data_dict = load_data_from_directory(PATH_PADDED_TOKENS, "tokens", data_fraction)
    melspec_data_dict = load_data_from_directory(PATH_PADDED_MELSPECS, "melspecs", data_fraction)
    
    assert tokens_data_dict.keys() == melspec_data_dict.keys()

    # Pour s'assurer qu'on fait bien correspondre les bons tokens aux bon melspecs
    sequence_ids = list(tokens_data_dict.keys())
    num_samples = int(len(sequence_ids) * data_fraction)
    sequence_ids = sequence_ids[:num_samples] 
    
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

def train_model(model, train_dataset, val_dataset, epochs):

    checkpoint_path = f"{PATH_MODEL_CHECKPOINTS}/model_at_epoch_{{epoch:02d}}.ckpt"

    checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_loss",  
        save_best_only=True,   
        save_weights_only=True,
        save_freq='epoch',
        period=50,
        verbose=1
    )
    history = model.fit(train_dataset, 
                        validation_data=(val_dataset),
                        epochs=epochs,
                        callbacks=[checkpoint_callback],
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

def save_model_in_saved_models(model):
    """
    Saves the entire model to the specified path using tf.keras.
    """
    # Determine the path where the model will be saved
    model_path = f"{PATH_FULL_MODEL}/model_{int(time.time())}"
    
    # Use tf.keras.models.save_model to save the model
    tf.keras.models.save_model(model, model_path)

    print(f"Model saved at {model_path}")

def load_model_fom_saved_models(model_path):
    """
    Loads the entire model from the specified path.
    """
    model = tf.keras.models.load_model(model_path)
    return model


def load_latest_checkpoint_from_dir():
    """
    Load the latest checkpoint from the specified directory.
    
    Parameters:
    - config: The configuration object to initialize and compile the model.
    - checkpoint_dir: The directory where the checkpoints are saved.
    
    Returns:
    - The model with the loaded weights if a checkpoint is found.
    - None otherwise.
    """
    
    config= Config()
    
    checkpoints = [f for f in os.listdir(PATH_MODEL_CHECKPOINTS) if f.endswith(".ckpt.index")]
    
    if not checkpoints:
        print("aucun checkpoint trouvé")
        return None
    
    checkpoints.sort()
    latest_checkpoint_name = checkpoints[-1].replace(".index", "")  # Remove the .index extension to get the actual checkpoint name
    latest_checkpoint_path = os.path.join(PATH_MODEL_CHECKPOINTS, latest_checkpoint_name)
    
    model = initialize_model(config)
    compile_model(model, config)
    
    model.load_weights(latest_checkpoint_path)
    
    input_shape = (config.embedding_dim,)
    model.build(input_shape)
    
    return model

def predict_melspec(model, input_tokens):
    input_tokens = tf.expand_dims(input_tokens, 0) 
    input_tokens = tf.convert_to_tensor(input_tokens, dtype=tf.int32)
    prediction = model.predict(input_tokens)
    return prediction

def evaluate_model(model, test_dataset):
    loss = model.evaluate(test_dataset)
    return loss