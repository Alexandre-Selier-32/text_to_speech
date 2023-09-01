import tensorflow as tf
from app.model.Transformer import Transformer

def load_data():
    #TO DO 
    return 

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

def compile_model(model):
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

def train_model(model, train_dataset, val_dataset):
    # Entraînement
    model.fit(train_dataset, validation_data=val_dataset, epochs=10)
    
