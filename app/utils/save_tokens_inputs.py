import numpy as np 
from app.params import *
from app.utils import *

def save_tokens_to_npy(path_transcriptions_csv=PATH_LJ_CSV, path_mapping_phonem=PATH_PHONES_MAPPING_LJSPEECH):
    padded_tokens_dict = get_padded_tokenized_transcripts(path_transcriptions_csv, path_mapping_phonem)
    
    # Crée le dossier s'il n'existe pas
    if not os.path.exists(PATH_PADDED_TOKENS):
        os.makedirs(PATH_PADDED_TOKENS)
    
    # Save les tokens dans des fichiers .npy
    for sequence_id, value in padded_tokens_dict.items():
        file_path_and_name = f"{PATH_PADDED_TOKENS}/{sequence_id}_tokens.npy"
        np.save(file_path_and_name, value)
    
    print('✅ Tokens Files saved successfully')
