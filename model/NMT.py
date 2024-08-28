import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense

from clean_data import expand_contractions, en_contraction_map
from tokenizer import *
from .Decoder import Decoder
from .Encoder import Encoder


@tf.keras.utils.register_keras_serializable()
class NMT(Model):
    def __init__(self,
                 input_tokenizer,
                 output_tokenizer,
                 embedding_size,
                 hidden_units,
                 input_vocab_size,
                 output_vocab_size,
                 **kwargs):
        """
            Initialize an instance for Neural Machine Translation Task

        :param input_tokenizer: tokenizer of the input language
        :param output_tokenizer: tokenizer of the output language
        :param embedding_size: dimensionality of embedding layer
        :param hidden_units: dimensionality of the output
        :param input_vocab_size: size of the input vocabulary
        :param output_vocab_size: size of the output vocabulary
        """

        super().__init__(**kwargs)
        self.input_tokenizer = input_tokenizer
        self.output_tokenizer = output_tokenizer
        self.embedding_size = embedding_size
        self.hidden_units = hidden_units
        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size
        self.encoder = Encoder(embedding_size,
                               hidden_units,
                               input_vocab_size)
        self.decoder = Decoder(embedding_size,
                               hidden_units,
                               output_vocab_size)
        self.head = Dense(output_vocab_size)

    def call(self,
             inputs):
        encoder_inputs, decoder_inputs = inputs
        encoder_outputs, state_h, state_c = self.encoder(encoder_inputs)
        logits = self.decoder(encoder_outputs, decoder_inputs,
                              [state_h, state_c])
        logits = self.head(logits)

        return logits

    def translate(self, next_inputs,
                  maxlen=40):
        def sampling(head_logits):
            probs = tf.nn.softmax(head_logits)
            dist = probs.numpy().squeeze()
            idx = np.random.choice(range(len(ger_vocab)), p=dist)

            return idx

        translation = []
        next_inputs = expand_contractions(next_inputs.lower(), en_contraction_map)
        next_idx = np.asarray(en_vec(next_inputs))

        while next_idx.ndim != 2:
            next_idx = tf.expand_dims(next_idx, axis=0)

        encoder_outputs, state_h, state_c = self.encoder(next_idx, training=False)

        next_inputs = "[START]"
        next_idx = np.asarray(ger_word_to_idx[next_inputs])

        for i in range(maxlen):
            while next_idx.ndim != 2:
                next_idx = tf.expand_dims(next_idx, axis=0)

            logits, state_h, state_c = self.decoder(encoder_outputs, next_idx,
                                                    [state_h, state_c],
                                                    training=False,
                                                    return_state=True)
            logits = self.head(logits)
            next_idx = sampling(logits)
            next_inputs = ger_vocab[next_idx]

            if next_inputs == "[END]":
                break
            elif next_inputs == "[UNK]":
                continue
            else:
                translation.append(next_inputs)

        return " ".join(translation)

    def get_config(self):
        config = super().get_config()
        config.update({
            "input_tokenizer": tf.keras.utils.serialize_keras_object(self.input_tokenizer),
            "output_tokenizer": tf.keras.utils.serialize_keras_object(self.output_tokenizer),
            "embedding_size": self.embedding_size,
            "hidden_units": self.hidden_units,
            "input_vocab_size": self.input_vocab_size,
            "output_vocab_size": self.output_vocab_size
        })

        return {**config}
