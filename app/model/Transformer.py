from tensorflow.keras import Model, layers
from app.model.Encoder import Encoder
from app.model.Decoder import Decoder
from app.model.VariancePredictor import VariancePredictor



class Transformer(Model):
    def __init__(self, num_layers, embedding_dim, num_heads, dff, input_vocab_size,
                 target_vocab_size, max_position_encoding, conv_kernel_size, conv_filters, rate,
                 var_conv_filters, var_conv_kernel_size, var_num_layers, var_rate, mask_value):
        super(Transformer, self).__init__()
        
        # Ajouter positionnal embedding + embedding de tokens 
        self.embedding = ... 
        
        self.mask = create_tokens_padding_mask_for

        self.encoder = Encoder(num_layers, embedding_dim, num_heads, dff, 
                               input_vocab_size, max_position_encoding, 
                               conv_kernel_size, conv_filters, rate)

        self.decoder = Decoder(num_layers, embedding_dim, num_heads, dff, 
                               target_vocab_size, max_position_encoding, rate)
        
        self.duration_predictor = VariancePredictor(var_conv_filters, 
                                                    var_conv_kernel_size,
                                                    var_num_layers, 
                                                    var_rate)
        # Manque Energy and pitch predictors         

        self.final_layer = layers.Dense(target_vocab_size)


    # TODO ATTENTION : il manque probablement des informations du decoder
    def call(self, tokenized_phonems_input, enc_padding_mask): 
        self_emb 
        
        sel_mask
        
        encoder_output = self.encoder(tokenized_phonems_input, enc_padding_mask)  
        
        predicted_duration = self.duration_predictor(encoder_output)

        decoder_output, attention_weights = self.decoder(predicted_duration) 
        
        final_output = self.final_layer(decoder_output) 

        return final_output, attention_weights
