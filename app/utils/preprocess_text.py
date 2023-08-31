import numpy as np 
from tensorflow.keras.preprocessing.sequence import pad_sequences


def save_X_to_npy(X, path):
    np.save(path, arr= X)

def get_padded_tokenized_transcripts(tokenized_transcriptions_dict):
    tokenized_transcriptions = list(tokenized_transcriptions_dict.values())
    
    padded_lists = pad_sequences(tokenized_transcriptions, padding='post', value=-10)
    padded_tokens_dict = {key: value for key, value in zip(tokenized_transcriptions_dict.keys(), padded_lists)}

    return padded_tokens_dict 

