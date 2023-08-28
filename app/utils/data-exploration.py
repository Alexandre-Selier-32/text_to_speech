import os
import soundfile as sf
import pandas as pd

def get_data_from_directory(directory_path):
    """
    Parcourt des dossiers imbriqués contenant des fichiers audio (en .flac) et des transcriptions (en .txt).
    pour en extraire un sequence_id, une transcription, une durée et le path du fichier audio pour chaque extrait audio.
    
    Parameters:
    - dataset_path: path vers le répertoire contenant les fichiers audio et texte.

    Return:
    - Dataframe qui contient les colonnes 'sequence_id', 'transcription', 'duration', et 'audio_file'.
    """
    
    def get_transcriptions_from_file(file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
            return {line.split()[0]: " ".join(line.split()[1:]) for line in lines}
    
    def get_audio_duration(file_path):
        with sf.SoundFile(file_path) as f:
            return len(f) / f.samplerate
    
    data = {
        'sequence_id': [],
        'transcription': [],
        'duration': [],
        'audio_file': []
    }
    
    for root, _, files in os.walk(directory_path):
        transcription_files = [file for file in files if file.endswith(".txt")]
        
        transcripts = {}
        if transcription_files:
            transcripts = get_transcriptions_from_file(os.path.join(root, transcription_files[0]))

        for filename in files:
            if filename.endswith(".flac"):
                sequence_id = filename.strip('.flac')
                audio_path = os.path.join(root, filename)
                
                data['sequence_id'].append(sequence_id)
                data['audio_file'].append(audio_path)
                data['transcription'].append(transcripts.get(sequence_id, ""))
                data['duration'].append(get_audio_duration(audio_path))

    return pd.DataFrame(data)