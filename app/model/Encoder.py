import tensorflow as tf
from tensorflow.keras import layers
from app.model.EncoderLayer import EncoderLayer


class Encoder(layers.Layer):
    def __init__(self, num_layers, embedding_dim, num_heads, dff,
                 conv_kernel_size, conv_filters, rate):
        super(Encoder, self).__init__()

        self.num_layers = num_layers

        self.encoder_layers = [EncoderLayer(embedding_dim, num_heads, dff, conv_kernel_size, conv_filters, rate) 
                           for _ in range(num_layers)]

        self.dropout = layers.Dropout(rate)

    def call(self, input, training, mask):
        encoder_output = input
        for layer in range(self.num_layers):
            encoder_output = self.encoder_layers[layer](encoder_output, training, mask)
            
        return encoder_output