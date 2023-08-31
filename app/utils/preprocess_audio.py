import numpy as np
from app.params import MEL_SPEC_PADDING_VALUE
import tensorflow as tf

def save_y_to_npy(y, path):
    np.save(path, arr= y)

def get_padded_melspecs(melspecs_dict):
    melspecs_lists = list(melspecs_dict.values())
    
    max_length = max(mel.shape[1] for mel in melspecs_lists)
    
    padded_mels = []
    for mel in melspecs_lists:
        padded_mel = np.pad(mel, ((0, 0), (0, max_length - mel.shape[1])), 'constant', constant_values=1)
        padded_mels.append(padded_mel)
    
    padded_melspecs_dict = {key: value for key, value in zip(melspecs_dict.keys(), padded_mels)}

    return padded_melspecs_dict


def create_melspecs_padding_mask_for(melspec_seq):
    """
    Crée un masque de padding pour la séquence de melspec.

    Les emplacements dans la séquence de melspec où il y a du padding MEL_SPEC_PADDING_VALUE (=1)
    auront une valeur de 1 dans le masque, les autres auront une valeur de 0.

    Params:
    - sequence de mel spectrogrammes : Un tensor de shape (batch_size, seq_len) contenant des mel spec .

    Return:
    - tf.Tensor: Un tensor de shape (batch_size, 1, 1, seq_len) représentant le masque de padding.
    """
    melspec_seq = tf.cast(tf.math.equal(melspec_seq, MEL_SPEC_PADDING_VALUE), tf.float32)
    return melspec_seq[:, tf.newaxis, tf.newaxis, :]
