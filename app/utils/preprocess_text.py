import numpy as np 
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from app.params import TOKEN_PADDING_VALUE

def save_X_to_npy(X, path):
    
    np.save(path, arr= X)

def get_padded_tokenized_transcripts(tokenized_transcriptions_dict):
    tokenized_transcriptions = list(tokenized_transcriptions_dict.values())
    
    padded_lists = pad_sequences(tokenized_transcriptions, padding='post', value=TOKEN_PADDING_VALUE)
    padded_tokens_dict = {key: value for key, value in zip(tokenized_transcriptions_dict.keys(), padded_lists)}

    return padded_tokens_dict 


def create_tokens_padding_mask_for(tokens_seq):
    """
    Crée un masque de padding pour la séquence de tokens.

    Les emplacements dans la séquence de tokens où le token est égal à la valeur de TOKEN_PADDING_VALUE (-10)
    auront une valeur de 1 dans le masque, les autres auront une valeur de 0.

    Params:
    - sequence de tokens : Un tensor de shape (batch_size, seq_len) contenant des tokens.

    Return:
    - tf.Tensor: Un tensor de shape (batch_size, 1, 1, seq_len) représentant le masque de padding.
    """
    tokens_seq = tf.cast(tf.math.equal(tokens_seq, TOKEN_PADDING_VALUE), tf.float32)
    return tokens_seq[:, tf.newaxis, tf.newaxis, :]
