from tensorflow.keras import Model, layers
import tensorflow as tf
import numpy as np
from app.model.Encoder import Encoder
from app.model.Decoder import Decoder
from app.model.VarianceAdaptor import VarianceAdaptor
from app.params import *

class Transformer(Model):
    def __init__(self, num_layers, embedding_dim, num_heads, dff, input_vocab_size,
                 max_position_encoding, conv_kernel_size, conv_filters, rate, 
                 var_conv_filters, var_conv_kernel_size, var_rate):
        
        super().__init__()
        
        self.embedding = layers.Embedding(input_dim=input_vocab_size, output_dim=embedding_dim, mask_zero=True)
        self.masked_embedding = layers.Masking(mask_value=TOKEN_PADDING_VALUE)        
        self.pos_encoding = self.positional_encoding(max_position_encoding, embedding_dim)     

        self.encoder = Encoder(num_layers, embedding_dim, num_heads, dff, 
                               conv_kernel_size, conv_filters, rate)
        

        self.duration_predictor = VarianceAdaptor(embedding_dim, var_conv_filters, 
                                                    var_conv_kernel_size, var_rate)
        
        self.duration_embedding = layers.Embedding(MAX_DURATION, embedding_dim)

        
        '''
        self.energy_predictor = VarianceAdaptor(embedding_dim, var_conv_filters, 
                                                  var_conv_kernel_size, var_rate)
        self.pitch_predictor = VarianceAdaptor(embedding_dim, var_conv_filters, 
                                                 var_conv_kernel_size, var_rate)
        '''
        
        self.decoder = Decoder(num_layers, embedding_dim, num_heads, dff, 
                               conv_kernel_size, conv_filters, rate)
                               
        
        self.final_layer = layers.Dense(870)
    
    def call(self, phoneme_tokens_input):  

        print("Input phoneme_tokens_input shape:", phoneme_tokens_input.shape)

        embedding_output = self.embedding(phoneme_tokens_input) 
        print("embedding_output shape:", embedding_output.shape)

        masked_embedding_output = self.masked_embedding(embedding_output)
        print("masked_embedding_output shape:", masked_embedding_output.shape)

        emb_dim = tf.shape(embedding_output)[2]
        tokens_seq_len = tf.shape(embedding_output)[1]

        seq_length = tf.shape(masked_embedding_output)[1]

        embedding_and_pos_output = masked_embedding_output + self.pos_encoding[:, :seq_length, :]
        print("embedding_and_pos_output shape:", embedding_and_pos_output.shape)

        encoder_output = self.encoder(embedding_and_pos_output) 
        print("encoder_output shape:", encoder_output.shape)

        duration_output = self.duration_predictor(encoder_output)
        print("duration_output shape:", duration_output.shape)

        regulated_output = self.duration_predictor.regulate_length(encoder_output, duration_output)
        print("regulated_output shape:", regulated_output.shape)

        decoder_output = self.decoder(regulated_output) 
        print("decoder_output shape:", decoder_output.shape)

        model_output = self.final_layer(decoder_output) 
        print("model_output shape:", model_output.shape)

        return model_output


    def positional_encoding(self, position, embedding_dim):
        def get_angles(pos, i, embedding_dim):
            angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(embedding_dim))
            return pos * angle_rates
        
        angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(embedding_dim)[np.newaxis, :],
                            embedding_dim)

        # apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]

        return tf.cast(pos_encoding, dtype=tf.float32)