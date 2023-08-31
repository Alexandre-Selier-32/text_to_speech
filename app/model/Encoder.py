import tensorflow as tf
from tensorflow.keras import layers
from app.model.EncoderLayer import EncoderLayer


class Encoder(layers.Layer):
    def __init__(self, num_layers, embedding_dim, num_heads, dff, input_vocab_size,
                 maximum_position_encoding, conv_kernel_size, conv_filters, rate):
        super(Encoder, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.embedding = layers.TokenAndPositionEmbedding(input_vocab_size, embedding_dim, maximum_position_encoding)
        self.encoder_layers = [EncoderLayer(embedding_dim, num_heads, dff, conv_kernel_size, conv_filters, rate) 
                           for _ in range(num_layers)]

        self.dropout = layers.Dropout(rate)

    def call(self, input, mask):
        embedding_output = self.embedding(input)
        embedding_output = self.dropout(embedding_output)

        for layer in range(self.num_layers):
            encoder_output = self.encoder_layers[layer](embedding_output, mask)

        return encoder_output