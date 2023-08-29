import os
import soundfile as sf
import pandas as pd
from io import BytesIO
from pydub import AudioSegment
import soundfile as sf
from IPython.display import Audio, display
from text import get_phonem_tokens_from_directory, get_phonems_from_tokens
from audio import get_melspecs_from_audio_files

def display_data_by_df_row(row, show_transcript=True, show_seq=True, show_path=True, show_phonem=True, show_tokens=True, show_duration=True):
    """
    Affiche les informations d'une ligne du dataframe et un lecteur audio pour écouter l'extrait associé
    """
    if show_transcript:
        print("transcript:\n", row['transcription'])
    if show_seq:
        print("sequence_id:\n", row['sequence_id'])
    if show_path:
        print("audio_file:\n", row['audio_file'])
    if show_phonem:
        print("phonem:\n", row['phonems'])
    if show_tokens:
        print("phonem_tokens:\n", row['phonem_tokens'])
    if show_duration:
        print("duration:\n", row['duration'])

    # Temporary .wav as IPython.display doesn't handle the .flac files
    audio = AudioSegment.from_file(row['audio_file'], format="flac")
    buffer = BytesIO()
    audio.export(buffer, format="wav")
    
    display(Audio(buffer.getvalue()))

def get_audio_files_from_directory(directory_path):
    """
    Parcourt les dossiers imbriqués pour extraire les paths des fichiers audio.
    
    Parameters:
    - directory_path: path vers le répertoire contenant les fichiers audio.
    
    Return:
    - Dictionnaire avec les sequence_id comme clés et les path des fichiers audio comme valeurs.
    """
    audio_files = {}
    for root, _, files in os.walk(directory_path):
        for filename in files:
            if filename.endswith(".flac"):
                sequence_id = filename.strip('.flac')
                audio_path = os.path.join(root, filename)
                audio_files[sequence_id] = audio_path
    return audio_files

def get_transcriptions_from_directory(directory_path):
    """
    Parcourt les dossiers imbriqués pour extraire les transcriptions audio.
    
    Parameters:
    - directory_path: path vers le répertoire contenant les fichiers texte.
    
    Return:
    - Dictionnaire avec les sequence_id comme clés et les transcriptions comme valeurs.
    """

    def get_transcriptions_from_file(file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
            return {line.split()[0]: " ".join(line.split()[1:]) for line in lines}

    transcriptions = {}
    for root, _, files in os.walk(directory_path):
        transcription_files = [file for file in files if file.endswith(".txt")]
        if transcription_files:
            transcripts = get_transcriptions_from_file(os.path.join(root, transcription_files[0]))
            transcriptions.update(transcripts)
    return transcriptions

def get_audio_duration_from_directory(directory_path):
    """
    Parcourt les dossiers imbriqués pour calculer la duré des fichiers audio.
    
    Parameters:
    - directory_path: path vers le répertoire contenant les fichiers audio.
    
    Return:
    - Dictionnaire avec les sequence_id comme clés et les durées des audios comme valeurs.
    """
    
    def get_audio_duration(file_path):
        with sf.SoundFile(file_path) as f:
            return len(f) / f.samplerate

    audio_files = get_audio_files_from_directory(directory_path)
    durations = {sequence_id: get_audio_duration(file_path) for sequence_id, file_path in audio_files.items()}
    return durations

def make_dataframe(directory_path, tokenized_transcript_file, mapping_file):
    """
    Crée un DataFrame à partir des informations extraites des fonctions fournies.

    Parameters:
    - directory_path: Chemin vers le répertoire contenant les fichiers audio et texte.
    - tokenized_transcript_file: Chemin vers le fichier contenant une séquence de phonem tokens pour chaque sequence_id.
    - mapping_file: Chemin vers le fichier JSON contenant le mapping des phonèmes en tokens.

    Returns:
    - DataFrame contenant les colonnes: sequence_id, audio_file, transcription, duration, phonems, phonem_tokens, mel_spec.
    """

    audio_files = get_audio_files_from_directory(directory_path)
    transcriptions = get_transcriptions_from_directory(directory_path)
    durations = get_audio_duration_from_directory(directory_path)
    tokenized_transcripts = get_phonem_tokens_from_directory(tokenized_transcript_file, mapping_file)
    phonems = get_phonems_from_tokens(tokenized_transcripts, mapping_file)
    melspecs = get_melspecs_from_audio_files(audio_files)

    df = pd.DataFrame({
        'sequence_id': list(audio_files.keys()),
        'audio_file': list(audio_files.values()),
        'transcription': [transcriptions.get(seq_id, "") for seq_id in audio_files.keys()],
        'duration': [durations.get(seq_id, 0) for seq_id in audio_files.keys()],
        'phonem_tokens': [tokenized_transcripts.get(seq_id, []) for seq_id in audio_files.keys()],
        'phonem': [phonems.get(seq_id, []) for seq_id in audio_files.keys()],
        'mel_spec': [melspecs.get(seq_id, []) for seq_id in audio_files.keys()],
    })

    return df
