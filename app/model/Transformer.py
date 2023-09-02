from tensorflow.keras import Model, layers
import tensorflow as tf
import numpy as np
from app.model.Encoder import Encoder
from app.model.Decoder import Decoder
from app.model.VarianceAdaptor import VarianceAdaptor
from app.params import TOKEN_PADDING_VALUE, N_MELS

class Transformer(Model):
    def __init__(self, num_layers, embedding_dim, num_heads, dff, input_vocab_size,
                 max_position_encoding, conv_kernel_size, conv_filters, rate, 
                 var_conv_filters, var_conv_kernel_size, var_rate):
        
        super(Transformer).__init__()
        
        self.embedding = layers.Embedding(input_vocab_size, embedding_dim)
        
        self.pos_encoding = self.positional_encoding(max_position_encoding, embedding_dim)
        
        self.padding_mask = self.create_tokens_padding_mask

        self.encoder = Encoder(num_layers, embedding_dim, num_heads, dff, 
                               conv_kernel_size, conv_filters, rate)
        

        self.duration_predictor = VarianceAdaptor(embedding_dim, var_conv_filters, 
                                                    var_conv_kernel_size, var_rate)
        
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
        tf.config.run_functions_eagerly(True)

        embedding_output = self.embedding(phoneme_tokens_input) 
        #print("Shape after embedding:", embedding_output.shape)
        emb_dim = tf.shape(embedding_output)[2] # Get the embedding dimension dynamically
        tokens_seq_len = tf.shape(embedding_output)[1] # Get the sequence length dynamically

        assert embedding_output.shape[1:] == (tokens_seq_len, emb_dim) # check des 2 dernières dim

        masked_embedding_output = self.padding_mask(embedding_output)
        #print("Shape after masked embedding:", masked_embedding_output.shape)
        assert masked_embedding_output.shape[1:] == (tokens_seq_len, emb_dim) # check des 2 dernières dim

        seq_length = tf.shape(masked_embedding_output)[1]
        #print("seq_length", seq_length)
            
        embedding_and_pos_output = masked_embedding_output + self.pos_encoding[:, :seq_length, :]
        #print("Shape after adding positional encoding:", embedding_and_pos_output.shape)
        assert embedding_and_pos_output.shape[1:] == (tokens_seq_len, emb_dim) # check des 2 dernières dim

        encoder_output = self.encoder(embedding_and_pos_output) 
        #print("Shape after encoder:", encoder_output.shape) 
        assert encoder_output.shape[1:] == (tokens_seq_len, emb_dim) # check des 2 dernières dim

        #### VARIANCE ADAPTOR BEGIN 
        duration_output = self.duration_predictor(encoder_output)
        #print("Shape after duration predictor:", duration_output.shape)
        assert duration_output.shape[1:] == (tokens_seq_len, 1) # check des 2 dernières dim

        # Regulate the length of the phoneme_tokens using the predicted durations.
        regulated_output = self.duration_predictor.regulate_length(encoder_output, duration_output)
        #print("Shape after regulate length:", regulated_output.shape)
        assert regulated_output.shape[1:] == (80, emb_dim) # check des 2 dernières dim

        #### VARIANCE ADAPTOR END 

        decoder_output = self.decoder(regulated_output) 
        #print("Shape after decoder:", decoder_output.shape)
        assert decoder_output.shape[1:] == (80, emb_dim) # check des 2 dernières dim

        model_output = self.final_layer(decoder_output) 
        #print("Shape after final layer:", model_output.shape)
        assert model_output.shape[1:] == (80, 870) # check des 2 dernières dim

        return model_output

    def create_tokens_padding_mask(self, input):
        """
        Crée un masque de padding pour la séquence de tokens.

        Les emplacements dans la séquence de tokens où le token est égal à la valeur de TOKEN_PADDING_VALUE
        auront une valeur de 1 dans le masque, les autres auront une valeur de 0.

        Params:
        - sequence de tokens : Un tensor de shape (batch_size, seq_len) contenant des tokens.

        Return:
        - tf.Tensor: Un tensor de shape (batch_size, 1, 1, seq_len) représentant le masque de padding.
        """
        tokens_seq = input
        tokens_seq = tf.cast(tf.math.equal(tokens_seq, TOKEN_PADDING_VALUE), tf.float32)
        return tokens_seq

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