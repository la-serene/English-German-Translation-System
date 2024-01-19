import tensorflow.keras.layers as layers


class Decoder(tf.keras.Model):
    def __init__(self,
                 hidden_units):
        """
            Decoder Block in seq2seq

        :param hidden_units: dimensionality of the output
        """

        super(Decoder, self).__init__()
        self.decoder_block = layers.LSTM(units=hidden_units,
                                         return_sequences=True,
                                         return_state=True)

    def call(self,
             trg,
             previous_state,
             **kwargs):
        """
            Inputs:

        :param trg: [batch, timesteps]
        :param previous_state: [batch, hidden_unit_dim]

        :return:
            prediction: [vocab_size, None]
        """
        decoder_outputs, state_h, state_c = self.decoder_block(inputs=trg,
                                                               initial_state=previous_state,
                                                               **kwargs)

        return decoder_outputs, state_h, state_c
