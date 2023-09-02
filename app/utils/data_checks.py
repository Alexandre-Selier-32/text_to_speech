import numpy as np
import os
import soundfile as sf
import tensorflow as tf
from app.params import N_MELS

def have_same_shape(input_data):
    """
    Vérifie que toutes les valeurs d'un dictionnaire ont la même shape ou que tous les numpy arrays 
    dans un dossier ont la même shape.

    Parameters:
    - input_data: Dictionnaire dont les valeurs sont des arrays ou chemin vers un répertoire.

    Return:
    - Booléen indiquant si toutes les valeurs ont la même shape ou si tous les numpy arrays ont la même shape.
    """
    
    if isinstance(input_data, dict):
        shapes = [np.shape(value) for value in input_data.values()]
    
    elif isinstance(input_data, str) and os.path.isdir(input_data):
        shapes = []
        for filename in os.listdir(input_data):
            if filename.endswith('.npy'):
                arr = np.load(os.path.join(input_data, filename))
                if isinstance(arr, np.ndarray):
                    shapes.append(arr.shape)
                else:
                    print(f"{filename} does not contain a numpy array.")
                    return False
    else:
        raise ValueError("Input must be a dictionary or a path to a directory.")
    
    assert len(set(shapes)) == 1, "Not all elements have the same shape."
    print(f'✅ All elements have the same shape: {shapes[0]}')
    return True

def have_same_sample_rate(input_data):
    """
    Vérifie si tous les fichiers audio dans un répertoire ou dans un dictionnaire ont le même taux d'échantillonnage.

    Parameters:
    - input_data: Soit un chemin vers un répertoire contenant des fichiers audio, soit un dictionnaire dont les valeurs 
                  sont les chemins vers les fichiers audio.

    Returns:
    - Booléen indiquant si tous les fichiers audio ont le même taux d'échantillonnage.
    """
    
    if isinstance(input_data, dict):
        audio_files = list(input_data.values())

    elif isinstance(input_data, str) and os.path.isdir(input_data):
        audio_files = [os.path.join(input_data, file) for file in os.listdir(input_data) if file.endswith('.wav')]
    else:
        raise ValueError("Input must be a dictionary or a path to a directory.")
    
    # Récupère les sample rate de tous les fichiers audio
    sample_rates = [sf.info(audio_file).samplerate for audio_file in audio_files]
    unique_sample_rates = np.unique(sample_rates)
    
    if len(unique_sample_rates) == 1:
        print(f"Tous les fichiers ({len(audio_files)} fichiers) ont un taux d'échantillonnage de {unique_sample_rates[0]} Hz.")
        return True
    else:
        print("Les fichiers ont des taux d'échantillonnage différents.")
        assert False, "Not all audio files have the same sample rate."
        return False
    
def model_returns_the_right_shape(model, train_tokens, batch_size):
    tokens_seq_len = len(train_tokens[0])
    sample_tokens = tf.stack(train_tokens[:batch_size])
    
    predictions = model(sample_tokens)    

    assert predictions.shape == (batch_size, tokens_seq_len, N_MELS), \
        f"Wrong shape of predictions, we should have {(batch_size, tokens_seq_len, N_MELS)} \
            with {batch_size} = batch_size, {tokens_seq_len} = length of a sequence of tokens, {N_MELS} = number of filter banks"

    print("✅ Right shape of predictions:", predictions.shape)

    return