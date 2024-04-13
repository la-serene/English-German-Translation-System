import tensorflow as tf

from Decoder import Decoder
from Encoder import Encoder
from tokenize import *


class NMT(tf.keras.Model):
    def __init__(self,
                 input_tokenizer,
                 output_tokenizer,
                 embedding_size,
                 hidden_units):
        """
        :param input_tokenizer: tokenizer of the input language
        :param output_tokenizer: tokenizer of the output language
        :param embedding_size: dimensionality of embedding layer
        :param hidden_units: dimensionality of the output
        """
        super(NMT, self).__init__()
        self.encoder = Encoder(input_tokenizer,
                               embedding_size,
                               hidden_units)
        self.decoder = Decoder(output_tokenizer,
                               embedding_size,
                               hidden_units)

    def call(self,
             inputs):
        encoder_inputs, decoder_inputs = inputs
        encoder_outputs, state_h, state_c = self.encoder(encoder_inputs)
        logits = self.decoder(encoder_outputs, decoder_inputs,
                              [state_h, state_c])

        return logits
