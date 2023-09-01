import numpy as np

def get_padded_melspecs(melspecs_dict):
    melspecs_lists = list(melspecs_dict.values())
    
    max_length = max(mel.shape[1] for mel in melspecs_lists)
    
    padded_mels = []
    for mel in melspecs_lists:
        padded_mel = np.pad(mel, ((0, 0), (0, max_length - mel.shape[1])), 'constant', constant_values=1)
        padded_mels.append(padded_mel)
    
    padded_melspecs_dict = {key: value for key, value in zip(melspecs_dict.keys(), padded_mels)}

    return padded_melspecs_dict