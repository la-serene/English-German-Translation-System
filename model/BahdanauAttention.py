import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Add, Activation, LayerNormalization


class BahdanauAttention(Layer):
    def __init__(self,
                 hidden_units):
        super(BahdanauAttention, self).__init__()
        self.Va = Dense(1)
        self.Wa = Dense(hidden_units)
        self.Ua = Dense(hidden_units)
        self.norm = LayerNormalization()
        self.tanh = Activation(tf.keras.activations.tanh)
        self.add = Add()

    def call(self,
             context, x):
        """
            Calculate the context vector based on all encoder hidden states and
            previous decoder state.

        :param: context: tensor, all encoder hidden states
        :param: x: tensor, previous state from Decoder
        :return:
            context_vector: tensor, the calculated context vector based on the
            input parameters
        """
        # Expand dims to ensure scores shape = [batch, Ty, Tx]
        context = tf.expand_dims(context, axis=1)
        x = tf.expand_dims(x, axis=2)

        scores = self.Va(self.tanh(self.add([self.Wa(context), self.Ua(x)])))
        scores = tf.squeeze(scores, axis=-1)
        attn_weights = tf.nn.softmax(scores, axis=-1)

        # NOTE: context shape = [batch, 1, Tx, feature] so that expand
        # dim of attention weights
        context_vector = tf.expand_dims(attn_weights, axis=-1) * context
        context_vector = tf.reduce_sum(context_vector, axis=-2)
        context_vector = self.norm(context_vector)
        context_vector = self.add([context_vector, tf.squeeze(x, -2)])

        return context_vector
