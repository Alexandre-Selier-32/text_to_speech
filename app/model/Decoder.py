import tensorflow as tf
from tensorflow.keras import layers
from app.model.EncodecLayer import EncodecLayer


class Decoder(layers.Layer):
    def __init__(self, num_layers, embedding_dim, num_heads, dff,
                conv_kernel_size, conv_filters, rate):
        super(Decoder, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.decoder_layers = [EncodecLayer(embedding_dim, num_heads, dff, conv_kernel_size, conv_filters, rate)
                           for _ in range(num_layers)]

        self.dropout = layers.Dropout(rate)

    def call(self, input, mask):
        decoder_output = input
        for layer in range(self.num_layers):
            decoder_output = self.decoder_layers[layer](decoder_output, mask)

        return decoder_output
