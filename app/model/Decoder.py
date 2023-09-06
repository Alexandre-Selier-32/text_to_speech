import tensorflow as tf
from tensorflow.keras import layers
from app.model.DecoderLayer import DecoderLayer


class Decoder(layers.Layer):
    def __init__(self, num_layers, embedding_dim, num_heads, dff,
                conv_kernel_size, conv_filters, rate):
        super(Decoder, self).__init__()

        self.num_layers = num_layers

        self.decoder_layers = [DecoderLayer(embedding_dim, num_heads, dff, conv_kernel_size, conv_filters, rate)
                           for _ in range(num_layers)]

        self.dropout = layers.Dropout(rate)

    def call(self, input):
        decoder_output = input
        for layer in range(self.num_layers):
            decoder_output = self.decoder_layers[layer](decoder_output)

        return decoder_output
