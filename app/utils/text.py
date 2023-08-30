import json
import string
from phonemizer import phonemize
from phonemizer.separator import Separator
import re

# Nécessaire pour les transcriptions que ne sont pas déjà tokenisées
def get_phonems_from_tokens(tokenized_transcripts_dict, mapping_file):
    """
    Convertit un dictionnaire de transcriptions tokenisées en un dictionnaire de transcriptions phonetisées

    Parameters:
    - tokenized_transcripts_dict : un dictionnaire avec les sequence_id comme clés et les listes de tokens comme valeurs.
    - mapping_file : path vers le fichier JSON contenant le mapping des phonèmes en tokens.

    Retour:
    - phonems_dict: Dictionnaire avec les sequence_id comme clés et les listes de phonèmes comme valeurs.
    """
    with open(mapping_file, 'r') as file:
        phoneme_mapping = json.load(file)
    inverted_mapping = {int(value): key for key, value in phoneme_mapping.items()}

    phonems_dict = {}
    for sequence_id, tokenized_transcript in tokenized_transcripts_dict.items():
        phonems = [inverted_mapping[token] for token in tokenized_transcript if token in inverted_mapping]
        phonems_dict[sequence_id] = phonems

    return phonems_dict

def get_phonem_tokens_from_directory(tokenized_transcript_file):
    """
    Récupère les transcription tokenisées des dossier imbriquées 
    
    Parameters:
    - tokenized_transcript : chemin vers le fichier contenant une séquence de phonem tokens pour chaque sequence_id.
    
    Returns:
    - phonem_tokens_dict: Dictionnaire avec les sequence_id comme clés et les listes de phonem_tokens comme valeurs.
    """
    
    phonem_tokens_dict = {}

    with open(tokenized_transcript_file, 'r') as file:
        for line in file:
            tokens = line.strip().split()
            sequence_id = tokens[0]
            token_phonem_sequence = [int(token) for token in tokens[1:]]
            phonem_tokens_dict[sequence_id] = token_phonem_sequence
            
    return phonem_tokens_dict

def get_tokens_from_phonems(phonemized_transcripts_dict, mapping_file):
    """
    Convertit un dictionnaire de transcriptions phonetisées en un dictionnaire de transcriptions tokenisées

    Parameters:
    - phonemized_transcripts_dict : un dictionnaire avec les sequence_id comme clés et les listes de phonèmes comme valeurs.
    - mapping_file : path vers le fichier JSON contenant le mapping des phonèmes en tokens.

    Retour:
    - tokens_dict: Dictionnaire avec les sequence_id comme clés et les listes de tokens comme valeurs.
    """
    with open(mapping_file, 'r') as file:
        phoneme_mapping = json.load(file)

    tokens_dict = {}

    for sequence_id, phonemized_transcripts_array in phonemized_transcripts_dict.items():
        token_phonem_sequence = [phoneme_mapping[phonem] for phonem in phonemized_transcripts_array if phonem in phoneme_mapping]
        tokens_dict[sequence_id] = token_phonem_sequence

    return tokens_dict

def get_cleaned_transcriptions(transcriptions_dict):
    """
    Clean les transcriptions pour pouvoir les processer
    
    params : un dictionnaire de transcripts
    
    returns : un dictionnaire de clean_transcripts
    """
    def clean_transcription(transcription):
        punc_to_remove = string.punctuation
        lower = transcription.lower()
        without_punc = lower.translate(str.maketrans('', '', punc_to_remove))
        without_extra_spaces = " ".join(without_punc.split())
        return without_extra_spaces
    
    clean_transcriptions_dict= {}
    for sequence_id, transcription in transcriptions_dict.items():
        clean_transcriptions_dict[sequence_id] = clean_transcription(transcription)
    return clean_transcriptions_dict
        
def phonemize_transcripts(clean_trancripts_dict, separator=Separator(phone=' ', word='/')):
    """
    Convertit les transcriptions en liste de phonèmes à l'aide de la librairie phonemizer

    Parameters:
    - transcripts_dict : un dictionnaire avec pour clés les sequence_id et pour valeurs les transcriptions textuelles.

    Retour:
    - phonems_dict: dictionnaire avec les sequence_id comme clés et une liste de phonèmes en valeur.
    """
    transcriptions = list(clean_trancripts_dict.values())
    
    phonemized_transcriptions = phonemize(transcriptions, separator=separator, strip=True)
        
    phonemized_lists = [transcription.replace('/', ' ').split(separator.phone) 
                        for transcription in phonemized_transcriptions]
    
    phonems_dict = {sequence_id: phonem_list for sequence_id, phonem_list in zip(clean_trancripts_dict.keys(), phonemized_lists)}
    
    return phonems_dict