import tensorflow as tf
from tensorflow.keras.layers import Layer, Embedding, LSTM, Dense

from .BahdanauAttention import BahdanauAttention

DROPOUT = 0.2


class Decoder(Layer):
    def __init__(self,
                 tokenizer,
                 embedding_size,
                 hidden_units,
                 dropout=DROPOUT):
        """
            Decoder Block in seq2seq

        :param tokenizer: tokenizer of the source language
        :param embedding_size: dimensionality of the embedding layer
        :param hidden_units: dimensionality of the output
        :param dropout: dropout rate
        """

        super().__init__()
        self.tokenizer = tokenizer
        self.embedding_size = embedding_size
        self.hidden_units = hidden_units
        self.vocab = tokenizer.get_vocabulary()
        self.vocab_size = tokenizer.vocabulary_size()
        self.embedding = Embedding(input_dim=self.vocab_size,
                                   output_dim=embedding_size)
        self.rnn = LSTM(units=hidden_units,
                        dropout=dropout,
                        return_sequences=True,
                        return_state=True)
        self.attention = BahdanauAttention(hidden_units)
        self.dense = Dense(self.vocab_size)

    def call(self,
             context, x,
             encoder_state,
             training=True,
             return_state=False):
        """
        :param context: all encoder states
        :param x: all initial decoder states
        :param encoder_state: last state from encoder
        :param training:
        :param return_state:

        :return:
            logits:
            state_h: hidden state
            state_c: cell state
        """
        mask = tf.where(x != 0, True, False)
        x = self.embedding(x)
        decoder_outputs, state_h, state_c = self.rnn(x, initial_state=encoder_state,
                                                     mask=mask,
                                                     training=training)
        dense_inputs = self.attention(context, decoder_outputs)
        logits = self.dense(dense_inputs)

        if return_state:
            return logits, state_h, state_c
        else:
            return logits
