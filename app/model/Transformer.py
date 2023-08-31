from tensorflow.keras import Model, layers
from app.model.Encoder import Encoder
from app.model.Decoder import Decoder



class Transformer(Model):
    def __init__(self, num_layers, embedding_dim, num_heads, dff, input_vocab_size,
                 target_vocab_size, max_position_encoding,  conv_kernel_size, conv_filters, rate):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers, embedding_dim, num_heads, dff, 
                               input_vocab_size, max_position_encoding, 
                               conv_kernel_size, conv_filters, rate)

        self.decoder = Decoder(num_layers, embedding_dim, num_heads, dff, 
                               target_vocab_size, max_position_encoding, rate)

        self.final_layer = layers.Dense(target_vocab_size)


    # TODO ATTENTION : il manque probablement des informations du decoder
    def call(self, tokenized_phonems_input, enc_padding_mask): 
        encoder_output = self.encoder(tokenized_phonems_input, enc_padding_mask)  

        decoder_output, attention_weights = self.decoder(encoder_output) 

        final_output = self.final_layer(decoder_output) 

        return final_output, attention_weights
