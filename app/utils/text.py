import json
import string
from phonemizer import phonemize
from phonemizer.separator import Separator

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
        phoneme_mapping_unicode = json.load(file)
        
    phoneme_mapping = {key.encode('utf-8').decode('unicode_escape'): value for key, value in phoneme_mapping_unicode.items()}


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
        return without_extra_spaces.strip()
    
    clean_transcriptions_dict= {}
    for sequence_id, transcription in transcriptions_dict.items():
        clean_transcriptions_dict[sequence_id] = clean_transcription(transcription)
    return clean_transcriptions_dict
        
        
def handle_multi_char_phonem(phonem):
    '''
    méthode qui prend en entrée un phonem et qui en ressort une liste de phonems 
    qui contiendra 1 ou plusieurs éléments.
    Moche mais pragmatique.
    '''
    liste_44_phonems =  ['aɪ', 'aʊ', 'b', 'd', 'eɪ', 'f', 'h', 'i', 'iə', 'j', 'k', 'l', 'm', 'n', 'o', 'oʊ', 'p', 's', 't', 'uː', 'v', 'w', 'z', 'æ', 'ð', 'ŋ', 'ɑ', 'ɑː', 'ɔ', 'ɔɪ', 'ɔː', 'ə', 'ɚ', 'ɛ', 'ɡ', 'ɪ', 'ɹ', 'ɾ', 'ʃ', 'ʊ', 'ʊɹ', 'ʌ', 'ʒ', 'θ']
    
    # si la longueur du phonème est de 1, tu retournes une liste qui contient uniquement le phonème
    if len(phonem) == 1:
        return [phonem]
    # si la longueur du phonème est > à 2, tu retourne une liste de 2 éléments. Le premier élement contient les 2 premiers caractères, le 2ème élément contient le reste.
    elif len(phonem) > 2:
        return [phonem[:2], phonem[2:]]
    # si la longueur du phonème est égale à 2:
    elif len(phonem) == 2:
        # s'il existe dans liste_44_phonems, tu renvoies un liste qui contient uniquement le phonèmes
        if phonem in liste_44_phonems:
            return [phonem]
        # s'il est composé de 2 caractères qui existent individuellement dans liste_44_phonems alors tu le split en 2 et tu retournes une liste de 2 phonèmes
        elif phonem[0] in liste_44_phonems and phonem[1] in liste_44_phonems:
            return [phonem[0], phonem[1]]
        # si le 1ere élement du phonème existe dans liste_44_phonems, renvoie une liste où il y a uniquement le premier phonème 
        elif phonem[0] in liste_44_phonems:
            return [phonem[0]]
    # sinon renvoie une liste vide
    return []

def phonemize_transcripts(clean_trancripts_dict, separator=Separator(phone=' ', word='/')):
    """
    Convertit les transcriptions en liste de phonèmes à l'aide de la librairie phonemizer

    Parameters:
    - transcripts_dict : un dictionnaire avec pour clés les sequence_id et pour valeurs les transcriptions textuelles.

    Retour:
    - phonems_dict: dictionnaire avec les sequence_id comme clés et une liste de phonèmes en valeur.
    """
    transcriptions = clean_trancripts_dict.values()
        
    phonemized_transcriptions = phonemize(transcriptions, 
                                          backend='espeak',
                                          language='en-us', 
                                          separator=separator, 
                                          strip=True)
        
    phonemized_lists = [transcription.split(separator.phone) 
                        for transcription in phonemized_transcriptions]
    
    # split les phonèmes qui sont collés à cause de
    without_liaisons = []
    for phonemized_list in phonemized_lists:
        new_list = []
        for element in phonemized_list:
            split_elements = element.split(separator.word)
            new_list.extend(split_elements)
        without_liaisons.append(new_list)          
        
    
    phonems_dict = {sequence_id: phonem_list 
                    for sequence_id, phonem_list 
                    in zip(clean_trancripts_dict.keys(), without_liaisons)}
    
    return phonems_dict

def phonems_transcript_to_49(phonemized_transcriptions):
    """
    Gère les phonèmes multi-caractères dans les transcriptions pour n'avoir que 44 phonèmes

    Paramètres:
    - phonemized_transcriptions: Un dictionnaire avec sequence_id comme clés et une liste de phonèmes comme valeurs.

    Retour:
    - processed_transcripts: Un dictionnaire avec sequence_id comme clés et une liste traitée de phonèmes comme valeurs.
    """

    processed_transcripts = {}
    for sequence_id, phonem_list in phonemized_transcriptions.items():
        new_list = []
        for phonem in phonem_list:
            new_list.extend(handle_multi_char_phonem(phonem))
        processed_transcripts[sequence_id] = new_list
    
    return processed_transcripts

