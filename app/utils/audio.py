import matplotlib.pyplot as plt
import librosa
import numpy as np
from app.params import *
from speechbrain.pretrained import HIFIGAN
import torch
import IPython
import tensorflow as tf


def display_mel_spectrogram(mel_spectrogram, sr=SAMPLE_RATE, hop_length=HOP_LENGTH):
    """
    Affiche une visualisation du mel spectrogramme 

    Paramètres:
    - mel_spectrogram (numpy array): Le mel spectrogramme à afficher.
    - sr: Taux d'échantillonnage. Par défaut à 16000.
    - hop_length: Longueur du saut entre les trames. Par défaut à 512.

    Retour:
    Aucun. Affiche le mel spectrogramme.
    """
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spectrogram, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.tight_layout()
    plt.show()

def waveform_to_mel_spectrogram_from_stft(audio_path, n_fft=N_FFT, hop_length=HOP_LENGTH):
    """
    Calcule le mel spectrogram (numpy array) d'un fichier audio à partir de librosa.stft
    Le mel spectrogram est une représentation de l'audio proche de la perception humaine.
    
    Parameters:
    - audio_path: path vers le fichier audio
    - sr: Taux d'échantillonnage. C'est le nombre d'échantillons de son pris chaque seconde. 16.000Hz par défaut
    - n_fft: Window size pour la transformée de Fourier. Une fenêtre plus grande donne plus de "détails" (comme un zoom)
    - hop_length: Décalage entre chaque fenêtre analysée. 
    - n_mels: Nombre de bandes de fréquences visbiles sur notre mel spectrogram. 

    Retour:
    - mel_spectrogram array
    """
    
    y, _ = librosa.load(audio_path, sr=None)
    stft = librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length)
    mel_spec = librosa.amplitude_to_db(stft, ref=np.max)
    
    return mel_spec

def waveform_to_mel_spectrogram_from_spectrum(audio_path, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS):
    """
    Calcule le mel spectrogram (numpy array) d'un fichier audio à partir de librosa.feature.melspectrogram
    Le mel spectrogram est une représentation de l'audio proche de la perception humaine.
    
    Parameters:
    - audio_path: path vers le fichier audio
    - sr: Taux d'échantillonnage. C'est le nombre d'échantillons de son pris chaque seconde. 16.000Hz par défaut
    - n_fft: Window size pour la transformée de Fourier. Une fenêtre plus grande donne plus de "détails" (comme un zoom)
    - hop_length: Décalage entre chaque fenêtre analysée. 
    - n_mels: Nombre de bandes de fréquences visbiles sur notre mel spectrogram. 

    Retour:
    - mel_spectrogram array
    """
    
    y, _ = librosa.load(audio_path, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    mel_spec = librosa.power_to_db(S, ref=np.max)
    
    return mel_spec


def get_melspecs_from_audio_files(audio_files_dict, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS):
    """
    Calcule le mel spectrogramme pour chaque fichier audio
    
    Parameters:
    - audio_files_dict: Dictionnaire avec les sequence_id comme clés et les path des fichiers audio comme valeurs.
    
    Return:
    - Dictionnaire avec les sequence_id comme clés et les mel spectrogrammes comme valeurs.
    """
    melspecs_dict = {}
    
    for sequence_id, audio_path in audio_files_dict.items():
        melspecs_dict[sequence_id] = waveform_to_mel_spectrogram_from_spectrum(audio_path, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    
    return melspecs_dict

def pad_single_melspec(mel, max_length):
    padding = max_length - mel.shape[1]
    if padding > 0:
        mel = np.pad(mel, ((0, 0), (0, padding)), mode='constant', constant_values=MEL_SPEC_PADDING_VALUE)
    return mel

def get_padded_melspecs_dict(melspecs_dict):
    melspecs_lists = list(melspecs_dict.values())
    
    max_length = np.max([mel.shape[1] for mel in melspecs_lists])
    padded_mels = [pad_single_melspec(mel, max_length) for mel in melspecs_lists]
    
    padded_melspecs_dict = {key: value for key, value in zip(melspecs_dict.keys(), padded_mels)}

    return padded_melspecs_dict

def get_melspec_by_sequence_id(sequence_id):
    file_name = f"{sequence_id}_melspecs.npy"
    file_path = os.path.join(PATH_PADDED_MELSPECS, file_name)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_name} not found in {PATH_PADDED_MELSPECS}.")
    return np.load(file_path)


def listen_to_audio(melspec): 
    hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech", savedir="tmpdir")
    
    if isinstance(melspec, tf.Tensor):
        melspec_numpy = melspec.numpy()
    elif isinstance(melspec, torch.Tensor):
        melspec_numpy = melspec.detach().cpu().numpy()
    elif isinstance(melspec, np.ndarray): 
        melspec_numpy = melspec
    else:
        raise ValueError("Unsupported melspec type. Expected tf.Tensor, torch.Tensor, or numpy array.")
  
    melspec_tensor = torch.tensor(melspec_numpy).float()
    
    if next(hifi_gan.parameters()).is_cuda:
        melspec_tensor = melspec_tensor.cuda()
    
    waveforms = hifi_gan.decode_batch(melspec_tensor)

    return IPython.display.Audio(waveforms, rate=SAMPLE_RATE)


def save_melspec(predicted_output):
    if not os.path.exists(PATH_PREDICTED_MELSPEC):
        os.makedirs(PATH_PREDICTED_MELSPEC)
    
    file_path_and_name = f"{PATH_PREDICTED_MELSPEC}/predicted_melspec.npy"

    stacked_predictions = np.stack(predicted_output)
    np.save(file_path_and_name, stacked_predictions)


def trim_melspectrogram(melspec, threshold=0):
    # Trouver les colonnes où le maximum est supérieur au seuil
    mask = np.max(melspec, axis=0) >= threshold
    
    # Trouver le premier indice où le masque est True
    idx = np.where(mask)[0][0] if mask.any() else melspec.shape[1]
    
    # Tronquer le melspectrogram à cet indice
    trimmed_melspec = melspec[:, :idx]
    
    return trimmed_melspec