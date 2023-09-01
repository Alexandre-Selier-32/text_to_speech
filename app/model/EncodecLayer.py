import tensorflow as tf
from tensorflow.keras import layers
from app.model.MultiHeadAttention import MultiHeadAttention

class EncodecLayer(layers.Layer):
    def __init__(self, embedding_dim, num_heads, dff, conv_kernel_size, conv_filters, rate):
        super(EncodecLayer, self).__init__()

        # Multihead Attention
        self.multihead_att = MultiHeadAttention(embedding_dim, num_heads)
        
        '''
        FastSpeech2 architecture (https://arxiv.org/pdf/2006.04558.pdf):
        -  2-layer 1D-convolutional network with ReLU activation,
        - each followed by the layer normalization and the dropout layer
        - and an extra linear layer to project the hidden states into the output sequence. 
        '''
        
        # Convolutions 1D 
        self.conv1 = layers.Conv1D(filters=conv_filters, kernel_size=conv_kernel_size, 
                                         strides=1, padding='same', activation='relu')
        self.conv2 = layers.Conv1D(filters=embedding_dim, kernel_size=conv_kernel_size, 
                                         strides=1, padding='same')
    

        self.layernorm = layers.LayerNormalization(epsilon=1e-6)

        self.dropout = layers.Dropout(rate)

        # Feed Forward network
        '''
        Attention is All You Need (https://arxiv.org/pdf/1706.03762.pdf)
        In addition to attention sub-layers, each of the layers in our encoder and decoder contains a fully
        connected feed-forward network, which is applied to each position separately and identically. This
        consists of two linear transformations with a ReLU activation in between.
        '''
        self.ff = tf.keras.Sequential([
            layers.Dense(dff, activation='relu'), # 1ère couche augmente la dim
            layers.Dense(embedding_dim) # 2eme couche ramène la dim à la dim de l'embedding
        ])
        
        
    def call(self, input, mask):
        attention_output, _ = self.multihead_att(input, input, input, mask)
        '''
        Attention is All You Need (https://arxiv.org/pdf/1706.03762.pdf)
        We apply dropout to the output of each sub-layer, before it is added to the
        sub-layer input and normalized.'''
        attention_output = self.dropout(attention_output)
        attention_output = self.layernorm(input + attention_output)
        
        # Convolutions 1D
        conv_output = self.conv1(attention_output)
        conv_output = self.conv2(conv_output)
        conv_output = self.dropout(conv_output)
        conv_output = self.layernorm(attention_output + conv_output)
        
        # Feed Forward 
        ff_output = self.ff(conv_output)
        ff_output = self.dropout(ff_output)
        ff_output = self.layernorm(conv_output + ff_output)

        return ff_output