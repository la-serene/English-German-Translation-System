import tensorflow.keras.layers as layers


class Encoder(layers.Layer):
    def __init__(self,
                 hidden_units):
        """
            Encoder Block in seq2seq

        :param hidden_units: dimensionality of the output
        """

        super(Encoder, self).__init__()
        self.encoder_block = layers.LSTM(units=hidden_units,
                                         return_state=True)

    def call(self,
             src,
             **kwargs):
        """
            Calculate vector representation.

        :param src: [batch, timesteps]

        :return:
            encoder_hidden_state: [batch, hidden_state_dim]
            state_h: [batch, hidden_state_dim]
            state_c: [batch, hidden_state_dim]
        """

        encoder_outputs, state_h, state_c = self.encoder_block(inputs=src,
                                                               **kwargs)
        return encoder_outputs, state_h, state_c
