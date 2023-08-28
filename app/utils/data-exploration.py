import os
import soundfile as sf
import pandas as pd
import json

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

def add_phonems_and_tokens_to_df(df, tokenized_transcript, mapping_file):
    """
    Ajoute les colonnes 'phonem_tokens' et 'phonems' au DataFrame en utilisant un fichier qui contient les séquences de 
    phonems de tokens associées à chaque transcription et un fichier de mapping de phonèmes (phoneme <=> int)
    
    Parameters:
    - DataFrame initial (colonne nécessaire : sequence_id)
    - tokenized_transcript : chemin vers le fichier contenant une séquence de phonem tokens pour chaque sequence_id.
    - mapping_file : chemin vers le fichier JSON contenant le mapping des phonèmes en tokens.
    
    Returns:
    - DataFrame initial avec les colonnes 'phonem_tokens' et 'phonems'.
    """
    
    with open(mapping_file, 'r') as file:
        phoneme_mapping = json.load(file)
    inverted_mapping = {int(value): key for key, value in phoneme_mapping.items()}
    
    phonem_tokens_dict = {}
    phonems_dict = {}

    with open(tokenized_transcript, 'r') as file:
        for line in file:
            tokens = line.strip().split()
            sequence_id = tokens[0]
            token_phonem_sequence = [int(token) for token in tokens[1:]]
            phonems = [inverted_mapping[token] for token in token_phonem_sequence]
            
            phonem_tokens_dict[sequence_id] = token_phonem_sequence
            phonems_dict[sequence_id] = phonems

    df['phonem_tokens'] = df['sequence_id'].map(phonem_tokens_dict)
    df['phonems'] = df['sequence_id'].map(phonems_dict)

    return df
