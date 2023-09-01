class Config():
    def __init__(self) -> None:
        # Model hyperparams
        self.embedding_dim = 256
        self.num_heads = 2  # (2 pour l'Encoder, 2 pour le Decoder)
        self.dff = 1024  # Feed forward hidden layer size
        self.max_position_encoding = 10000
        self.input_vocab_size = 49 # Nombre de tokens de phonems distincts
        
        # Encoder & Decoder hyperparams 
        self.num_layers = 4
        self.conv_kernel_size = 9
        self.conv_filters = 1024
        self.rate = 0.1

        # Variance Predictor hyperparams
        self.var_num_conv_layers = 2
        self.var_conv_filters = 256
        self.var_conv_kernel_size = 3
        self.var_rate = 0.5
        
        
        
config = Config()

'''
FastSpeech2 Paper (https://arxiv.org/pdf/2006.04558.pdf)
Hyperparameters:

Phoneme Embedding Dimension 256
Encoder Layers 4
Encoder Hidden 256
Encoder Conv1D Kernel 9
Encoder Conv1D Filter Size 1024 
Encoder Attention Heads 2
Mel-Spectrogram Decoder Layers 4
Mel-Spectrogram Decoder Hidden 256
Mel-Spectrogram Decoder Conv1D Kernel 9
Mel-Spectrogram Decoder Conv1D Filter Size 1024
Mel-Spectrogram Decoder Attention Headers 2
Encoder/Decoder Dropout 0.1
Variance Predictor Conv1D Kernel 3
Variance Predictor Conv1D Filter Size 256
Variance Predictor Dropout 0.5
Batch Size 48/48/12
Total Number of Parameters 23M/27M/28M
'''