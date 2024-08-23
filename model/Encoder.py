import tensorflow as tf
from tensorflow.keras.layers import Layer, Embedding, Bidirectional, LSTM

DROPOUT = 0.3


@tf.keras.utils.register_keras_serializable()
class Encoder(Layer):
    def __init__(self,
                 embedding_size,
                 hidden_units,
                 vocab_size,
                 dropout=DROPOUT,
                 **kwargs):
        """
            Encoder Block in seq2seq

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

    def get_config(self):
        config = super().get_config()
        config.update({
            "embedding_size": self.embedding_size,
            "hidden_units": self.hidden_units,
            "vocab_size": self.vocab_size,
            "dropout": DROPOUT
        })

        return {**config}
