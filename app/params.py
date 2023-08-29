import os

LOCAL_DATA_PATH =  os.path.dirname(os.path.dirname(__file__))

# PATHS 
PATH_1H = f"{LOCAL_DATA_PATH}/raw_data/librispeech_finetuning/1h"
PATH_9H = f"{LOCAL_DATA_PATH}/librispeech_finetuning/9h"
PATH_100H = f"{LOCAL_DATA_PATH}/raw_data/librispeech_finetuning/100h"
PATH_PHONES = f"{LOCAL_DATA_PATH}/raw_data/librispeech_finetuning/phones"
PATH_1h_PHONES = f"{PATH_PHONES}/1h_phones.txt"
PATH_10h_PHONES =  f"{PATH_PHONES}/10h_phones.txt"
PATH_PHONES_MAPPING = f"{PATH_PHONES}/phones_mapping.json"


