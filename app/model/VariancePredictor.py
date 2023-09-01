import tensorflow as tf
from tensorflow.keras import layers
from app.model.Config import Config

'''
FastSpeech2 (https://arxiv.org/pdf/2006.04558.pdf)
The duration, pitch and energy predictors share similar model structure (but different
model parameters), which consists of a 2-layer 1D-convolutional network with ReLU activation,
each followed by the layer normalization and the dropout layer, and an extra linear layer to project
the hidden states into the output sequence.
'''
class VariancePredictor(layers.Layer):
    def __init__(self, embedding_dim, conv_filters, conv_kernel_size, rate):
        super(VariancePredictor, self).__init__()
        self.conv = layers.Conv1D(filters=conv_filters, kernel_size=conv_kernel_size, 
                                         strides=1, padding='same', activation='relu')

        self.dropout = layers.Dropout(rate=rate)
        
        self.layernorm = layers.LayerNormalization(epsilon=1e-6)

        self.dense_layer = layers.Dense(embedding_dim, activation='relu')
    
    def call(self, input):
        conv1_output = self.conv(input)
        conv1_output = self.layernorm(input + conv1_output)
        conv1_output = self.dropout(conv1_output)

        conv2_output = self.conv(conv1_output)
        conv2_output = self.layernorm(conv1_output + conv2_output)
        output = self.dropout(conv2_output)


        return self.dense_layer(output)
