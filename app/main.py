from app.model.Config import Config
from app.model import load_data, initialize_model, compile_model, train_model, evaluate_model

def main():
    config = Config()
    
    train_data, val_data, test_data = load_data()
    
    model = initialize_model(config)
    
    compile_model(model)
    
    train_model(model, train_data, val_data)
    
    loss = evaluate_model(model, test_data)

    return

if __name__ == "__main__":
    main()
