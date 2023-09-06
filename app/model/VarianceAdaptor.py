import tensorflow as tf
from tensorflow.keras import layers
from app.params import *

'''
FastSpeech2 (https://arxiv.org/pdf/2006.04558.pdf)
The duration, pitch and energy predictors share similar model structure (but different
model parameters), which consists of a 2-layer 1D-convolutional network with ReLU activation,
each followed by the layer normalization and the dropout layer, and an extra linear layer to project
the hidden states into the output sequence.
'''
class VarianceAdaptor(layers.Layer):
    def __init__(self, embedding_dim, conv_filters, conv_kernel_size, var_rate):
        super(VarianceAdaptor, self).__init__()
        self.conv1 = layers.Conv1D(filters=conv_filters, kernel_size=conv_kernel_size, 
                                         strides=1, padding='same', activation='relu')
        self.conv2 = layers.Conv1D(filters=conv_filters, kernel_size=conv_kernel_size, 
                                         strides=1, padding='same', activation='relu')
    

        self.layernorm = layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout = layers.Dropout(rate=var_rate)
        
        self.dense_layer = layers.Dense(1)
    
    def call(self, input):
        x = input
        for conv, norm in zip(self.conv_layers, self.norm_layers):
            x = conv(x)
            x = norm(x)
            x = self.dropout(x)
        return self.dense_layer(x)

    
    def regulate_length(self, encoder_output, predicted_durations):
        # Clipping des durées prédites pour éviter de trop étirer ou compresser la séquence
        clipped_durations = tf.clip_by_value(predicted_durations, MIN_DURATION, MAX_DURATION)

        # Conversion des durées en int et suppression de la dernière dimension
        int_durations = tf.cast(tf.round(clipped_durations), tf.int32)
        int_durations = tf.squeeze(int_durations, axis=-1)
        
        # Fonction pour réguler la longueur d'une seule séquence
        def regulate_single_sequence(args):
            sequence, repeat_factors = args
            # Répétez chaque phonème selon sa durée prédite
            regulated_sequence = tf.repeat(sequence, repeat_factors, axis=0)
            # Ajustez la taille de sortie à 80
            regulated_sequence = regulated_sequence[:80]
            padding_size = 80 - tf.shape(regulated_sequence)[0]
            paddings = tf.zeros((padding_size, tf.shape(sequence)[-1]))
            return tf.concat([regulated_sequence, paddings], axis=0)

        # Appliquez cette opération sur chaque séquence du batch
        regulated_sequences = tf.map_fn(regulate_single_sequence, (encoder_output, int_durations), dtype=tf.float32)
        #print("[VarianceAdaptor] regulated_sequences",regulated_sequences)
        assert_shape = tf.TensorShape([None, 80, 256])
        tf.debugging.assert_shapes([(regulated_sequences, assert_shape)],
                                   message="[VarianceAdaptor] The shape of regulated_sequences does not match the expected shape (None, 80, 870)!")

        #print("[VarianceAdaptor] regulated_sequences shape after processing:", tf.shape(regulated_sequences))

        return regulated_sequences

