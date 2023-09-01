import tensorflow as tf
from tensorflow.keras import layers

class Attention(layers.Layer):
    def __init__(self, embedding_dim) -> None:
        super(Attention, self).__init__()
        self.embedding_dim = embedding_dim

    def call(self, query, key, value):
        # Queries x Keys
        dot_product = tf.matmul(query, key, transpose_b=True, name = "dot_product_q_k")
        
        # Scale
        scaler = self.embedding_dim ** 0.5
        scaled_dot_product = dot_product / tf.math.sqrt(scaler)

        # Softmax
        softmaxed_attention_weights = tf.nn.softmax(scaled_dot_product, name = "apply_softmax")
        
        # Multiply by Values
        attention_output = tf.matmul(softmaxed_attention_weights, value,  name = "multiply_scores_w_value")
        
        return attention_output
    
