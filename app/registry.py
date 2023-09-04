import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.losses import MeanAbsoluteError
from keras import optimizers
from app.model import *
from app.params import *
import matplotlib.pyplot as plt

def load_data_from_directory(path, suffix):
    data_dict = {}
    for file_name in os.listdir(path):
        if file_name.endswith(".npy") and suffix in file_name:
            sequence_id = file_name.split(f"_{suffix}")[0]
            file_path = os.path.join(path, file_name)
            data_dict[sequence_id] = np.load(file_path)
    return data_dict

def get_test_train_val_split(batch_size):
    tokens_data_dict = load_data_from_directory(PATH_PADDED_TOKENS, "tokens")
    melspec_data_dict = load_data_from_directory(PATH_PADDED_MELSPECS, "melspecs")
    
    assert tokens_data_dict.keys() == melspec_data_dict.keys()
    print("CREATION TEST TRAIN VAL SPLIT OK")

    # Pour s'assurer qu'on fait bien correspondre les bons tokens aux bon melspecs
    sequence_ids = list(tokens_data_dict.keys())
    tokens_data = [tokens_data_dict[seq_id] for seq_id in sequence_ids][:batch_size]
    melspec_data = [melspec_data_dict[seq_id] for seq_id in sequence_ids][:batch_size]

    # Split data
    tokens_train, tokens_temp, melspec_train, melspec_temp = train_test_split(tokens_data, melspec_data, test_size=0.2, random_state=42)
    tokens_val, tokens_test, melspec_val, melspec_test = train_test_split(tokens_temp, melspec_temp, test_size=0.5, random_state=42)
    print(f"✅ Data split into:\n- Training: {len(tokens_train)/len(tokens_data)*100:.2f}%\n- Validation: {len(tokens_val)/len(tokens_data)*100:.2f}%\n- Test: {len(tokens_test)/len(tokens_data)*100:.2f}%")
    
    # Convert lists to tensors
    tokens_train = tf.convert_to_tensor(tokens_train, dtype=tf.int32)
    melspec_train = tf.convert_to_tensor(melspec_train, dtype=tf.float32)
    
    tokens_val = tf.convert_to_tensor(tokens_val, dtype=tf.int32)
    melspec_val = tf.convert_to_tensor(melspec_val, dtype=tf.float32)
    
    tokens_test = tf.convert_to_tensor(tokens_test, dtype=tf.int32)
    melspec_test = tf.convert_to_tensor(melspec_test, dtype=tf.float32)
    
    print('tokens_train 0', tokens_train)

    return (tokens_train, melspec_train), (tokens_val, melspec_val), (tokens_test, melspec_test)

def get_config(): 
    config = Config()
    return config

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

def masked_mse_loss(y_true, y_pred):
    mask = tf.math.logical_not(tf.math.equal(y_true, MEL_SPEC_PADDING_VALUE))
    loss = tf.square(tf.subtract(y_pred, y_true))
    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask
    
    return tf.reduce_sum(loss) / tf.reduce_sum(mask)

def compile_model(model, config):    
    lr_schedule = CustomLearningRateScheduler(embedding_dim=config.embedding_dim, warmup_steps=config.warmup_steps)
    optimizer = tf.keras.optimizers.Adam(lr_schedule, beta_1=config.beta_1, beta_2=config.beta_1, epsilon=config.epsilon)
    
    model.compile(optimizer=optimizer, loss=masked_mse_loss)

def train_model(model, train_tokens, train_melspec, val_tokens, val_melspec, epochs=N_EPOCHS):
    checkpoint_callback = ModelCheckpoint(
        filepath=PATH_MODEL_PARAMS + "/checkpoint",
        monitor="val_loss",  
        save_best_only=True,
        save_weights_only=True,
        mode="auto",
        save_freq="epoch"
    )
    
    es_callback = EarlyStopping(
        monitor="val_loss",
        patience=15,
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        x=train_tokens, 
        y=train_melspec,
        validation_data=(val_tokens, val_melspec),
        epochs=epochs,
        batch_size= BATCH_SIZE,
        callbacks=[es_callback, checkpoint_callback]
    )
    # Visualisez l'erreur d'entraînement
    plt.plot(history.history['loss'])
    plt.title('Model Loss Over Time')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train'], loc='upper right')
    plt.show()
    
    return history


def predict_melspec(model, input_tokens):
    input_tokens = tf.expand_dims(input_tokens, 0)  # Add batch dimension
    input_tokens = tf.convert_to_tensor(input_tokens, dtype=tf.int32)
    prediction = model.predict(input_tokens)
    return prediction

def evaluate_model(model, test_tokens, test_melspec):
    loss = model.evaluate(test_tokens, test_melspec)
    return loss


def overfit_on_sample(model, train_tokens, train_melspec, epochs=150):
    history = model.fit(train_tokens, train_melspec, epochs=epochs, verbose=1)

    # plt.plot(history.history['loss'])
    # plt.title('Model Loss Over Time')
    # plt.ylabel('Loss')
    # plt.xlabel('Epoch')
    # plt.legend(['Train'], loc='upper right')
    # plt.show()
    
    return history
