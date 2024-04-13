import tensorflow as tf
from BahdanauAttention import BahdanauAttention
from tensorflow.keras.layers import Layer, Embedding, LSTM, Dense


class Decoder(Layer):
    @classmethod
    def add_method(cls, fun):
        setattr(cls, fun.__name__, fun)
        return fun

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
        """

        super(Decoder, self).__init__()
        self.hidden_units = hidden_units
        self.tokenizer = tokenizer
        self.vocab = tokenizer.get_vocabulary()
        self.vocab_size = tokenizer.vocabulary_size()
        self.embedding = Embedding(input_dim=self.vocab_size,
                                   output_dim=embedding_size)
        self.rnn = LSTM(units=hidden_units,
                        dropout=DROPOUT,
                        return_sequences=True,
                        return_state=True)
        self.attention = BahdanauAttention(hidden_units)
        self.dense = Dense(15000)

    def call(self,
            context, x,
            encoder_state,
            training=True,
            return_state=False):
        """
        :param trg: [batch, timesteps]
        :param previous_state: [batch, hidden_unit_dim]

        :return:
            prediction: [vocab_size, None]
        """
        mask = tf.where(x != 0, True, False)
        x = self.embedding(x)
        decoder_outputs, state_h, state_c = self.rnn(x, initial_state=encoder_state,
                                                     mask=mask, training=training)
        context_vector = self.attention(context, decoder_outputs)
        dense_inputs = tf.concat([decoder_outputs, context_vector], axis=-1)
        logits = self.dense(dense_inputs)

        if return_state:
            return logits, state_h, state_c
        else:
            return logits
