import tensorflow as tf
from tensorflow.keras import layers
from app.model.EncodecLayer import EncodecLayer


class Encoder(layers.Layer):
    def __init__(self, num_layers, embedding_dim, num_heads, dff,
                 conv_kernel_size, conv_filters, rate):
        super(Encoder, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.encoder_layers = [EncodecLayer(embedding_dim, num_heads, dff, conv_kernel_size, conv_filters, rate) 
                           for _ in range(num_layers)]

        self.dropout = layers.Dropout(rate)

    def call(self, input):
        encoder_output = input
        for layer in range(self.num_layers):
            encoder_output = self.encoder_layers[layer](encoder_output)
            
        return encoder_output