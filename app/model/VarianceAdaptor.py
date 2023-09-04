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
    
    def regulate_length(self, encoder_output, predicted_durations):
        """
        Régule la longueur de la sortie de l'encodeur en fonction des durées prédites.
        La durée de prononciation de tous les phonèmes n'est pas la même, un 
        phonème peut s'étendre sur plusieurs melspec frames. On prend donc la sortie 
        de l'encodeur et on l'adapte pour prendre en compte la durée prédite pour chaque phonème.
        """
        
        print("encoder_output shape at the start:", tf.shape(encoder_output))
        print("predicted_durations shape at the start:", tf.shape(predicted_durations))

        # Clipping des durées prédites pour éviter de trop étirer ou compresser la séquence
        clipped_durations = tf.clip_by_value(predicted_durations, MIN_DURATION, MAX_DURATION)
        print("clipped_durations shape:", tf.shape(clipped_durations))

        # conversion des durées en int pour tf.tile
        int_durations = tf.cast(tf.round(clipped_durations), tf.int32)
        int_durations = tf.squeeze(int_durations, axis=-1)  # on enlève la dernière dimension car predicted_durations est un tenseur 3D
        print("int_durations shape:", tf.shape(int_durations))

        print("Valeurs de int_durations:", int_durations)

        # Répéte chaque élément de encoder_output selon int_durations
        expanded_output = tf.repeat(encoder_output, int_durations, axis=1)

        # Trouve la longueur maximale après la répétition pour le padding
        max_length = tf.shape(expanded_output)[1]

        # Calcule la quantité de padding nécessaire pour chaque séquence
        padding_amounts = max_length - tf.reduce_sum(int_durations, axis=1)

        # Crée n tensor de padding
        paddings = tf.map_fn(lambda x: tf.zeros((x, tf.shape(encoder_output)[-1])), padding_amounts, dtype=tf.float32)

        # Concaténe le padding à la fin de chaque séquence
        expanded_output = tf.concat([expanded_output, paddings], axis=1)

        # pour s'assurer que la longueur finale des séquences correspond à la deuxième dimension des tenseurs cibles
        reshaped_output = expanded_output[:, :80, :]
        print("HEEEY")

        return reshaped_output

