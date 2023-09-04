.DEFAULT_GOAL := default
#################### PACKAGE ACTIONS ###################
reinstall_package:
	@pip uninstall -y text-to-speek || :
	@pip install -e .

run_save_inputs:
	python -c 'from app.utils import save_tokens_to_npy; save_tokens_to_npy()'
	python -c 'from app.utils import process_all_wavs_in_folder; process_all_wavs_in_folder()'


# run_train:
# 	python -c 'from app.registry import get_config; get_config()'
# 	python -c 'from app.registry import initialize_model; initialize_model()'
# 	python -c 'from app.registry import compile_model; compile_model()'
# 	python -c 'from app.registry import train_model; train_model()'


