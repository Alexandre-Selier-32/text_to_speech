import tensorflow as tf
from tensorflow.keras import optimizers

'''
Based on the learning rate optimizer from "Attention is All You Need" (https://arxiv.org/pdf/1706.03762.pdf)
'''
class CustomLearningRateScheduler(optimizers.schedules.LearningRateSchedule):
    def __init__(self, embedding_dim, warmup_steps=4000):
        super(CustomLearningRateScheduler, self).__init__()
        
        self.embedding_dim = tf.cast(embedding_dim, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step_float = tf.cast(step, tf.float32)  
        model_scale_factor = tf.math.rsqrt(self.embedding_dim)
        step_scale_factor = tf.math.rsqrt(step_float) 
        warmup_factor = step_float * (self.warmup_steps ** -1.5)
        return model_scale_factor * tf.math.minimum(step_scale_factor, warmup_factor)