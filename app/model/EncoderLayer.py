import tensorflow as tf
from tensorflow.keras import layers
from app.model.MultiHeadAttention import MultiHeadAttention

        
class EncoderLayer(layers.Layer):
    def __init__(self, embedding_dim, num_heads, dff, conv_kernel_size, conv_filters, rate):
        super().__init__()

        # Multihead Attention
        self.multihead_att = MultiHeadAttention(embedding_dim, num_heads)

        # Convolutions 1D 
        self.conv1 = layers.Conv1D(filters=conv_filters, kernel_size=conv_kernel_size, 
                                         strides=1, padding='same', activation='relu')
        self.conv2 = layers.Conv1D(filters=conv_filters, kernel_size=conv_kernel_size, 
                                         strides=1, padding='same', activation='relu')
    

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
        
       
    '''
    Attention is All You Need (https://arxiv.org/pdf/1706.03762.pdf)
    We apply dropout to the output of each sub-layer, before it is added to the
    sub-layer input and normalized.
    ''' 
    def call(self, input, training, mask=None):
        attention_output = self.multihead_att(input, input, input, mask)
        attention_output = self.dropout(attention_output, training=training)
        attention_output = self.layernorm(input + attention_output)

        # Convolutions 1D
        conv1 = self.conv1(attention_output)
        conv1 = self.dropout(conv1, training=training)
        conv1 = self.layernorm(attention_output + conv1)

        conv2 = self.conv2(conv1)
        conv2 = self.dropout(conv2, training=training)
        conv2 = self.layernorm(conv1 + conv2)

        # Feed Forward 
        ff_output = self.ff(conv2)
        ff_output = self.dropout(ff_output, training=training)
        ff_output = self.layernorm(conv2 + ff_output)

        return ff_output