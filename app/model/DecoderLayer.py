import tensorflow as tf
from tensorflow.keras import layers


class DecoderLayer(layers.Layer):
    def __init__(self, embedding_dim, dff, conv_kernel_size, conv_filters, rate):
        super().__init__()

        # Convolutions 1D 
        self.conv1 = layers.Conv1D(filters=conv_filters, kernel_size=conv_kernel_size, 
                                         strides=1, padding='same', activation='relu')
        self.conv2 = layers.Conv1D(filters=conv_filters, kernel_size=conv_kernel_size, 
                                         strides=1, padding='same', activation='relu')

        self.layernorm = layers.LayerNormalization(epsilon=1e-6)

        self.dropout = layers.Dropout(rate)

        # Feed Forward network
        self.ff = tf.keras.Sequential([
            layers.Dense(dff, activation='relu'), 
            layers.Dense(embedding_dim) 
        ])

    def call(self, input, training):
        # Convolutions 1D
        conv1 = self.conv1(input)
        conv1 = self.dropout(conv1, training=training)
        conv1 = self.layernorm(input + conv1)

        conv2 = self.conv2(conv1)
        conv2 = self.dropout(conv2, training=training)
        conv2 = self.layernorm(conv1 + conv2)

        # Feed Forward
        ff_output = self.ff(conv2)
        ff_output = self.dropout(ff_output, training=training)
        ff_output = self.layernorm(conv2 + ff_output)

        return ff_output
