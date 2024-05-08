import tensorflow as tf
from tensorflow.keras.layers import Layer, Embedding, Bidirectional, LSTM

DROPOUT = 0.3


class Encoder(Layer):
    def __init__(self,
                 tokenizer,
                 embedding_size,
                 hidden_units,
                 dropout=DROPOUT):
        """
            Encoder Block in seq2seq

        :param tokenizer: tokenizer of the source language
        :param embedding_size: dimensionality of the embedding layer
        :param hidden_units: dimensionality of the output
        :param dropout: dropout rate
        """

        super().__init__()
        self.tokenizer = tokenizer
        self.embedding_size = embedding_size
        self.hidden_units = hidden_units
        self.vocab_size = tokenizer.vocabulary_size()
        self.embedding = Embedding(input_dim=self.vocab_size,
                                   output_dim=embedding_size)
        self.rnn = Bidirectional(
            merge_mode="sum",
            layer=LSTM(units=hidden_units,
                       dropout=dropout,
                       return_sequences=True,
                       return_state=True))

    def call(self,
             x,
             training=True):
        """
        :param x: [batch, time_steps]
        :param training: is training or not
        :return:
            encoder_hidden_state: [batch, hidden_state_dim]
            state_h: [batch, hidden_state_dim]
            state_c: [batch, hidden_state_dim]
        """
        mask = tf.where(x != 0, True, False)
        x = self.embedding(x)
        x, forward_h, forward_c, backward_h, backward_c = self.rnn(x, mask=mask,
                                                                   training=training)

        return x, forward_h + backward_h, forward_c + backward_c
