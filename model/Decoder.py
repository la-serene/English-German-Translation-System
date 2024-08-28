import tensorflow as tf
from tensorflow.keras.layers import Layer, Embedding, LSTM

from .BahdanauAttention import BahdanauAttention

DROPOUT = 0.3


@tf.keras.utils.register_keras_serializable()
class Decoder(Layer):
    def __init__(self,
                 embedding_size,
                 hidden_units,
                 vocab_size,
                 dropout=DROPOUT,
                 **kwargs):
        """
            Decoder Block in seq2seq

        :param embedding_size: dimensionality of the embedding layer
        :param hidden_units: dimensionality of the output
        :param vocab_size: vocabulary size
        :param dropout: dropout rate
        """

        super().__init__(**kwargs)
        self.embedding_size = embedding_size
        self.hidden_units = hidden_units
        self.vocab_size = vocab_size
        self.embedding = Embedding(input_dim=self.vocab_size,
                                   output_dim=embedding_size)
        self.rnn = LSTM(units=hidden_units,
                        dropout=dropout,
                        return_sequences=True,
                        return_state=True)
        self.attention = BahdanauAttention(hidden_units)

    def call(self,
             context, x,
             encoder_state,
             training=True,
             return_state=False):
        """
        :param context: all encoder states
        :param x: all initial decoder states
        :param encoder_state: last state from encoder
        :param training: training mode
        :param return_state:
        :return:
            logits: logits
            state_h: hidden state
            state_c: cell state
        """
        mask = tf.where(x != 0, True, False)
        x = self.embedding(x)
        decoder_outputs, state_h, state_c = self.rnn(x, initial_state=encoder_state,
                                                     mask=mask,
                                                     training=training)
        logits = self.attention(context, decoder_outputs)

        if return_state:
            return logits, state_h, state_c
        else:
            return logits

    def get_config(self):
        config = super().get_config()
        config.update({
            "embedding_size": self.embedding_size,
            "hidden_units": self.hidden_units,
            "vocab_size": self.vocab_size,
            "dropout": self.dropout
        })

        return {**config}
