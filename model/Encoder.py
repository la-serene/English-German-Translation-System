import tensorflow as tf
from tensorflow.keras.layers import Layer, Embedding, Bidirectional, LSTM

DROPOUT = 0.2


class Encoder(Layer):
    def __init__(self,
                tokenizer,
                embedding_size,
                hidden_units):
        """
            Encoder Block in seq2seq

        :param tokenizer: tokenizer of the source language
        :param embedding_size: dimensionality of the embedding layer
        :param hidden_units: dimensionality of the output
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
                    dropout=DROPOUT,
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

    def get_config(self):
        config = super().get_config()
        config.update({
            "tokenizer": tf.keras.utils.serialize_keras_object(self.tokenizer),
            "hidden_units": self.hidden_units,
            "embedding_size": self.embedding_size
        })

        return {**config}

    @classmethod
    def from_config(cls, config):
        tokenizer_config = config.pop("tokenizer")
        tokenizer = tf.keras.utils.deserialize_keras_object(tokenizer_config)
        embedding_size = config["embedding_size"]
        hidden_units = config["hidden_units"]

        return cls(tokenizer, embedding_size, hidden_units)
