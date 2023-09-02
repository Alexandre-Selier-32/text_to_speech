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
    def __init__(self, embedding_dim, conv_filters, conv_kernel_size, rate):
        super(VarianceAdaptor, self).__init__()
        self.conv = layers.Conv1D(filters=conv_filters, kernel_size=conv_kernel_size, 
                                         strides=1, padding='same', activation='relu')
        
        self.dropout = layers.Dropout(rate=rate)
        
        self.layernorm = layers.LayerNormalization(epsilon=1e-6)

        self.dense_layer = layers.Dense(1, activation='relu')
    
    def call(self, input):
        conv1_output = self.conv(input)
        conv1_output = self.layernorm(input + conv1_output)
        conv1_output = self.dropout(conv1_output)

        conv2_output = self.conv(conv1_output)
        conv2_output = self.layernorm(conv1_output + conv2_output)
        output = self.dropout(conv2_output)


        return self.dense_layer(output)
    
    # Répète chaque token selon sa durée prédite.
    def regulate_length(self, encoder_output, predicted_durations):
        """
        Régule la longueur de la sortie de l'encodeur en fonction des durées prédites.
        La durée de prononciation de tous les phonèmes n'est pas la même, un 
        phonème peut s'étendre sur plusieurs melspec frames. On prend donc la sortie 
        de l'encodeur et on l'adapte pour prendre en compte la durée prédite pour chaque phonème.
        """
    # Clipping des durées prédites pour éviter de trop étirer ou compresser la séquence
        MAX_DURATION = 10  # par exemple
        MIN_DURATION = 1  # par exemple
        clipped_durations = tf.clip_by_value(predicted_durations, MIN_DURATION, MAX_DURATION)

        # conversion des durées en int pour tf.tile
        int_durations = tf.cast(tf.round(clipped_durations), tf.int32)
        int_durations = tf.squeeze(int_durations, axis=-1)  # on enlève la dernière dimension car predicted_durations est un tenseur 3D
                        
        expanded_output = []
        for i in range(tf.shape(encoder_output)[0]):
            expanded_segment = tf.repeat(encoder_output[i], int_durations[i], axis=0)
            expanded_output.append(expanded_segment)
                        
        max_length = max([tf.shape(segment)[0] for segment in expanded_output])
        
        
        # Pad each sequence to have the same length
        for i in range(len(expanded_output)):
            difference = max_length - tf.shape(expanded_output[i])[0]
            padding = tf.constant(TOKEN_PADDING_VALUE, shape=(difference, encoder_output.shape[-1]), dtype=tf.float32)
            expanded_output[i] = tf.concat([expanded_output[i], padding], axis=0)
        
        # Stack all the expanded and padded sequences
        reshaped_output = tf.stack(expanded_output, axis=0)
        
        # Calculate the target length based on the clipped durations
        target_length = tf.reduce_sum(int_durations, axis=1, keepdims=True)
        max_target_length = tf.reduce_max(target_length)

        # If the length of reshaped_output is different from the target length, add padding
        current_length = tf.shape(reshaped_output)[1]
        padding_amount = max_target_length - current_length

        # Ensure the padding amount is not negative
        padding_amount = tf.maximum(padding_amount, 0)

        padding_shape = [tf.shape(reshaped_output)[0], padding_amount, encoder_output.shape[-1]]
        padding = tf.zeros(padding_shape, dtype=tf.float32)

        reshaped_output = tf.concat([reshaped_output, padding], axis=1)

        # Ensure the final length of the sequences matches with the second dimension of the target tensors
        reshaped_output = reshaped_output[:, :80, :]

        return reshaped_output





