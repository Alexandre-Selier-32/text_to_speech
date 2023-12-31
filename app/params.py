import os
from argparse import Namespace


LOCAL_DATA_PATH =  os.path.dirname(os.path.dirname(__file__))

# PATHS LJSPEECH
PATH_LJ_AUDIOS = f"{LOCAL_DATA_PATH}/raw_data/LJSpeech/wavs"
PATH_LJ_CSV = f"{LOCAL_DATA_PATH}/raw_data/LJSpeech/metadata.csv"
PATH_PHONES_MAPPING_LJSPEECH = f"{LOCAL_DATA_PATH}/processed_data/phonem_mapping.json"

# PRE-PROCESSING PARAMS
TOKEN_PADDING_VALUE = 0
MEL_SPEC_PADDING_VALUE = 1

# FOR DURATION PREDICTION
MAX_DURATION = 9
MIN_DURATION = 2

# MODEL INPUTS DIRECTORIES
PATH_PADDED_TOKENS = f"{LOCAL_DATA_PATH}/processed_data/tokens"
PATH_PADDED_MELSPECS = f"{LOCAL_DATA_PATH}/processed_data/melspectrograms"

# PATH MODEL
PATH_MODEL_CHECKPOINTS = f"{LOCAL_DATA_PATH}/app/saved_models/checkpoints"
PATH_FULL_MODEL = f"{LOCAL_DATA_PATH}/app/saved_models/final_models"

PATH_PREDICTED_MELSPEC = f"{LOCAL_DATA_PATH}/processed_data/predicted_melspecs"

# PATHS BARK
PATH_Tacatron2_WAV = f"{LOCAL_DATA_PATH}/Tacotron2_model/wav"
PATH_Tacatron2_DUMMY_WAV = f"{LOCAL_DATA_PATH}/Tacotron2_model/dummy_wav"

# AUDIO PARAMS
SAMPLE_RATE=22050
N_FFT=1024
HOP_LENGTH=256
N_MELS=80

# INPUT OUTPUT PARAMS
MELSPEC_SHAPE= (80,870)
SEQ_TOKENS_LENGTH= 212

# MODEL TRAIN
BATCH_SIZE = 32
