import tensorflow as tf
import numpy as np

from tokenizer import expand_contractions, en_contraction_map

from .Encoder import Encoder
from .Decoder import Decoder


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

    def predict(self,
                next_inputs,
                word_to_idx,
                maxlen=40):
        def sampling(_logits):
            probs = tf.nn.softmax(_logits)
            dist = probs.numpy().squeeze()
            idx = np.random.choice(range(self.decoder.vocab_size), p=dist)

            return idx

        translation = []
        next_inputs = expand_contractions(next_inputs, en_contraction_map)
        next_idx = np.asarray(self.encoder.tokenizer(next_inputs))

        while next_idx.ndim != 2:
            next_idx = tf.expand_dims(next_idx, axis=0)

        encoder_outputs, state_h, state_c = self.encoder(next_idx, training=False)

        next_inputs = "[START]"
        next_idx = np.asarray(word_to_idx[next_inputs])

        for i in range(maxlen):
            while next_idx.ndim != 2:
                next_idx = tf.expand_dims(next_idx, axis=0)

            logits, state_h, state_c = self.decoder(encoder_outputs, next_idx,
                                                    [state_h, state_c],
                                                    training=False,
                                                    return_state=True)
            next_idx = sampling(logits)
            next_inputs = self.decoder.vocab[next_idx]

            if next_inputs == "[END]":
                break
            elif next_inputs == "[UNK]":
                continue
            else:
                translation.append(next_inputs)

        return translation
