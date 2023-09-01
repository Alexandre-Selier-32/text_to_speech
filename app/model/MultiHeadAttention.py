from tensorflow.keras import layers
import tensorflow as tf
from app.model.Attention import Attention

class MultiHeadAttention(layers.Layer):
    def __init__(self, embedding_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        
        assert embedding_dim % self.num_heads == 0
        
        self.depth = embedding_dim // self.num_heads

        self.wq = layers.Dense(embedding_dim, name = "query_projection")
        self.wk = layers.Dense(embedding_dim, name = "key_projection")
        self.wv = layers.Dense(embedding_dim, name = "value_projection")

        self.attention = Attention(embedding_dim)

        self.dense = layers.Dense(embedding_dim)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention = self.attention(q, k, v)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3]) # (batch_size, seq_len_q, num_heads, depth)
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.embedding_dim)) # (batch_size, seq_len_q, embedding_dim)
        
        # attention_ouptut : c'est l'output après avoir combiné les résultats de toutes les têtes d'attention
        attention_ouptut = self.dense(concat_attention) # (batch_size, seq_len_q, embedding_dim)

        return attention_ouptut