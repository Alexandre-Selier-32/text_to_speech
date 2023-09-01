import tensorflow as tf
from tensorflow.keras import layers

class VariancePredictor(layers.Layer):
    def __init__(self, var_conv_filters, var_conv_kernel_size, num_conv_layers, var_rate):
        super(VariancePredictor, self).__init__()
        
        self.conv_layers = [layers.Conv1D(filters=var_conv_filters, 
                                          kernel_size=var_conv_kernel_size, 
                                          padding='same', 
                                          activation='relu') 
                            for _ in range(num_conv_layers)]
        
        self.dropout = layers.Dropout(rate=var_rate)
        
        self.dense_layer = layers.Dense(1, activation='relu')
    
    def call(self, input):
        for conv_layer in self.conv_layers:
            output = conv_layer(input)
            output = self.dropout(output)
        
        return self.dense_layer(output)
