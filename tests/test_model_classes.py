import tensorflow as tf
from app.model.Config import Config
from app.params import N_MELS
from app.model.Attention import Attention
from app.model.VariancePredictor import VariancePredictor
from app.model.Encoder import Encoder
from app.model.Decoder import Decoder
from app.model.MultiHeadAttention import MultiHeadAttention
from app.model.EncodecLayer import EncodecLayer
from app.model.Transformer import Transformer


config = Config()
batch_size = 32
seq_len = 10
n_mels = N_MELS
in_out_shape_req = (batch_size, seq_len, config.embedding_dim)

# TEST DE LA CLASSE ATTENTION
attention = Attention(embedding_dim=config.embedding_dim)
query = tf.random.uniform(in_out_shape_req)
key = tf.random.uniform(in_out_shape_req)
value = tf.random.uniform(in_out_shape_req)
attention_output = attention(query, key, value)

print(attention_output.shape)  
assert attention_output.shape == in_out_shape_req


# TEST DE LA CLASSE MULTI HEAD ATTENTION
multi_head_attention = MultiHeadAttention(embedding_dim=config.embedding_dim, num_heads=config.num_heads)
multi_head_attention_output = multi_head_attention(query, key, value)

print(multi_head_attention_output.shape) 
assert multi_head_attention_output.shape == in_out_shape_req

# TEST DE LA CLASSE ENCODEC LAYER 
encodec_layer = EncodecLayer(embedding_dim=config.embedding_dim, num_heads=config.num_heads, 
                             dff=config.dff, conv_kernel_size=config.conv_kernel_size, 
                             conv_filters=config.conv_filters, rate=config.rate)

encodec_input = tf.random.uniform(in_out_shape_req)
encodec_layer_output = encodec_layer(encodec_input)

print(encodec_layer_output.shape)  
assert encodec_layer_output.shape == in_out_shape_req

# TEST DE LA CLASSE ENCODER
encoder = Encoder(num_layers=config.num_layers, embedding_dim=config.embedding_dim, 
                  num_heads=config.num_heads, dff=config.dff, 
                  conv_kernel_size=config.conv_kernel_size, 
                  conv_filters=config.conv_filters, rate=config.rate)

encoder_input = tf.random.uniform(in_out_shape_req)
encoder_output = encoder(encoder_input)

print(encoder_output.shape)
assert encoder_output.shape == in_out_shape_req

# TEST DE LA CLASSE DECODER
decoder = Decoder(num_layers=config.num_layers, embedding_dim=config.embedding_dim, 
                  num_heads=config.num_heads, dff=config.dff, 
                  conv_kernel_size=config.conv_kernel_size, 
                  conv_filters=config.conv_filters, rate=config.rate)

decoder_input = tf.random.uniform(in_out_shape_req)
decoder_output = decoder(decoder_input)

print(decoder_output.shape)
assert decoder_output.shape == in_out_shape_req

# TEST DE LA CLASSE VARIANCE PREDICTOR
variance_predictor = VariancePredictor(embedding_dim= config.embedding_dim, conv_kernel_size=config.var_conv_kernel_size,
                                    conv_filters=config.var_conv_filters, rate=config.var_rate)

var_input = tf.random.uniform(in_out_shape_req)
var_input_encoder = variance_predictor(var_input)

print(var_input_encoder.shape)
assert var_input_encoder.shape == in_out_shape_req

# TEST DE LA CLASSE TRANSFORMER
transformer = Transformer(num_layers=config.num_layers, embedding_dim=config.embedding_dim, 
                          num_heads=config.num_heads, dff=config.dff, 
                          input_vocab_size=config.input_vocab_size, 
                          target_vocab_size=config.input_vocab_size, 
                          max_position_encoding=config.max_position_encoding, 
                          conv_kernel_size=config.conv_kernel_size, 
                          conv_filters=config.conv_filters, rate=config.rate,
                          var_conv_filters=config.var_conv_filters, 
                          var_conv_kernel_size=config.var_conv_kernel_size, 
                          var_rate=config.var_rate)

input_sequence = tf.random.uniform((batch_size, seq_len), dtype=tf.int32, minval=0, maxval=config.input_vocab_size)
transformer_output = transformer(input_sequence)

print(transformer_output.shape)  
assert transformer_output.shape == (batch_size, seq_len, n_mels)