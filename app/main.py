from app.model.Config import Config
from app.registry import *

def run_pipeline(data_fraction=1.0, epochs=1000):
    config = Config()
    
    # Load data
    print(f'[LOAD DATA]: 🚧 Started loading data ...')
    train_dataset, val_dataset, test_dataset = get_test_train_val_split(data_fraction)
    print(f'[LOAD DATA]: ✅ Data loaded!')

    print(f"[DEBUG] Nombre d'échantillons dans le jeu de données de validation : {len(val_dataset)}")
    print(f"[DEBUG] validation dataset : {val_dataset}")

    latest_model_path = get_latest_model_path()
    
    print(f'[LOAD MODEL]: 🚧 searching for a saved model ...')
    if latest_model_path:
        model = load_model_fom_saved_models(latest_model_path)
        print(f'[LOAD MODEL]: ✅ Model loaded from last training ({latest_model_path})')
    else:
        # Initialize the model
        model = initialize_model(config)
        print('[LOAD MODEL]: ✅ No previous model saved. Training from scratch')

    print(f'[COMPILE MODEL]: 🚧 Model compiling ...')
    compile_model(model, config)
    print('[COMPILE MODEL]:  ✅ Model Compiled !')
    
    print('[TRAIN MODEL]: 🚧 Model Training ...')
    history = train_model(model, train_dataset, val_dataset, epochs=epochs)
    print('[TRAIN MODEL]: ✅ Model Training completed !')
    
    model.summary()

    try:
        save_model_in_saved_models(model)
        print(f'[SAVE MODEL]: ✅ Model saved to in {PATH_FULL_MODEL}')
    except Exception as e:
        print(f"[SAVE MODEL]: ⚠️ Failed to save model. Error: {e}")

    try:
        # Evaluate the model
        loss = evaluate_model(model, test_dataset)
        print(f"[EVALUATE MODEL] Model loss on test data: {loss}")
    except Exception as e:
        print(f"[EVALUATE MODEL]: ⚠️ Model evaluation failed. Error: {e}")
    
    return model, history
