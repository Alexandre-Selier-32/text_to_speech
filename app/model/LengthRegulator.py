import tensorflow as tf
from app.params import *

class LengthRegulator(tf.keras.layers.Layer):
    def __init__(self, hidden_dim, max_pos, SAMPLE_RATE, HOP_LENGTH, MAX_DURATION, **kwargs):
        super(LengthRegulator, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.SAMPLE_RATE = SAMPLE_RATE
        self.HOP_LENGTH = HOP_LENGTH
        self.MAX_DURATION = MAX_DURATION

    def expand(self, batch, predicted_durations):
        # Clipping des durées prédites
        clipped_durations = tf.clip_by_value(predicted_durations, MIN_DURATION, MAX_DURATION)

        # conversion des durées en int
        int_durations = tf.cast(tf.round(clipped_durations), tf.int32)
        int_durations = tf.squeeze(int_durations, axis=-1)

        expanded_output = []
        for i in range(tf.shape(batch)[0]):
            expanded_segment = tf.repeat(batch[i], int_durations[i], axis=0)
            expanded_output.append(expanded_segment)

        max_length = max([tf.shape(segment)[0] for segment in expanded_output])

        # Pad each sequence to have the same length
        for i in range(len(expanded_output)):
            difference = max_length - tf.shape(expanded_output[i])[0]
            padding = tf.zeros((difference, batch.shape[-1]))
            expanded_output[i] = tf.concat([expanded_output[i], padding], axis=0)

        out = tf.stack(expanded_output, axis=0)

        target_frames = (self.SAMPLE_RATE * self.MAX_DURATION) // self.HOP_LENGTH

        # Truncate or pad to match the target frames
        if tf.shape(out)[1] > target_frames:
            out = out[:, :target_frames, :]
        elif tf.shape(out)[1] < target_frames:
            padding_amount = target_frames - tf.shape(out)[1]
            padding = tf.zeros((tf.shape(out)[0], padding_amount, batch.shape[-1]))
            out = tf.concat([out, padding], axis=1)

        return out
