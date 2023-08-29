import json
import eng_to_ipa as ipa

def text_to_phonems(sentence):     
    """
    Convertit une chaîne de caractères en une liste de phonèmes en utilisant la librairie eng_to_ipa

    Parameters:
    - sentence : une chaîne de caractères

    Retour:
    - phonems: Liste des phonèmes associée à cette chaîne de caractères
    """   
    ipa_text = ipa.convert(sentence, keep_punct=False, stress_marks=None)
    phonems_list = list(ipa_text.replace(" ", "")) 

    return phonems_list

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