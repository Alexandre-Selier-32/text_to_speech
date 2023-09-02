import os

LOCAL_DATA_PATH =  os.path.dirname(os.path.dirname(__file__))

# PATHS LJSPEECH
PATH_LJ_AUDIOS = f"{LOCAL_DATA_PATH}/raw_data/LJSpeech/wavs"
PATH_LJ_CSV = f"{LOCAL_DATA_PATH}/raw_data/LJSpeech/metadata.csv"
PATH_PHONES_MAPPING_LJSPEECH = f"{LOCAL_DATA_PATH}/processed_data/phonem_mapping.json"

# PRE-PROCESSING PARAMS
TOKEN_PADDING_VALUE = -10
MEL_SPEC_PADDING_VALUE = 1

# MODEL INPUTS DIRECTORIES 
PATH_PADDED_TOKENS = f"{LOCAL_DATA_PATH}/processed_data/tokens"
PATH_PADDED_MELSPECS = f"{LOCAL_DATA_PATH}/processed_data/melspectrograms"

# AUDIO PARAMS
SAMPLE_RATE=22050
N_FFT=1024
HOP_LENGTH=256
N_MELS=80

