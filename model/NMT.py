import numpy as np
import tensorflow.keras.layers as layers
from tqdm import tqdm

from Decoder import Decoder
from Encoder import Encoder
from tokenize import *


class NMT(tf.keras.Model):
    def __init__(self,
                 input_vocab_size,
                 output_vocab_size,
                 embedding_size,
                 hidden_units):
        """
            Initialize an instance for Neural Machine Translation Task

        :param input_vocab_size: number of unique word in source language
        :param output_vocab_size: number of unique word in target language
        :param embedding_size: dimensionality of embedding layer
        :param hidden_units: dimensionality of the output
        """

        super(NMT, self).__init__()
        self.e_embedding = layers.Embedding(input_dim=input_vocab_size,
                                            output_dim=embedding_size)
        self.d_embedding = layers.Embedding(input_dim=output_vocab_size,
                                            output_dim=embedding_size)
        self.encoder = Encoder(hidden_units)
        self.decoder = Decoder(hidden_units)
        self.dense = layers.Dense(output_vocab_size, activation='softmax')

    def call(self,
             context,
             target):
        e_mask = tf.not_equal(context, 0)
        d_mask = tf.not_equal(target, 0)

        embed_src = self.e_embedding(context)
        embed_trg = self.d_embedding(target)

        encoder_outputs, state_h, state_c = self.encoder.call(src=embed_src,
                                                              mask=e_mask,
                                                              training=True)

        decoder_outputs, state_h, state_c = self.decoder.call(trg=embed_trg,
                                                              previous_state=[state_h, state_c],
                                                              mask=d_mask,
                                                              training=True)
        prediction = self.dense(decoder_outputs)

        return prediction, state_h, state_c

    def train(self,
              dataset,
              loss_fn,
              optimizer,
              epochs=5,
              val_set=None):
        """
            Train the model.

        :param dataset: training dataset
        :param loss_fn: loss function
        :param optimizer: optimizer
        :param epochs: number of training epochs
        :param val_set: validation set
        """

        for epoch in range(epochs):
            loss_sum = 0
            for step, (context, target) in enumerate(tqdm(dataset)):
                tokenized_context = en_vec(context)
                tokenized_target = de_vec(target)

                # Apply Teacher Forcing (TF)
                TF_target = tf.map_fn(lambda x: x[1:], tokenized_target)
                tokenized_target = tf.map_fn(lambda x: x[:-1], tokenized_target)

                with tf.GradientTape() as tape:
                    prediction, _, _ = self.call(tokenized_context, tokenized_target)
                    loss = loss_fn(TF_target, prediction)
                    loss_sum += loss

                gradients = tape.gradient(loss, self.trainable_weights)
                optimizer.apply_gradients(zip(gradients, self.trainable_weights))

            val_loss_sum = 0
            if val_set is not None:
                for step, (context, target) in enumerate(tqdm(val_set)):
                    tokenized_context = en_vec(context)
                    tokenized_target = de_vec(target)

                    TF_target = tf.map_fn(lambda x: x[1:], tokenized_target)
                    tokenized_target = tf.map_fn(lambda x: x[:-1], tokenized_target)

                    prediction, _, _ = self.call(tokenized_context, tokenized_target)
                    loss = loss_fn(TF_target, prediction)
                    val_loss_sum += loss

            print("\nEpoch: {:1d}, loss = {:4f}, val_loss = {:4f}".format(epoch, loss_sum, val_loss_sum))

    def predict(self,
                inputs):
        """
            Generate translation from input.
        """
        translation = []

        tokenized_input = en_vec(inputs)
        embed_input = self.e_embedding(tokenized_input)
        embed_input = tf.expand_dims(embed_input, axis=0)

        next_word = "<sos>"
        de_vocab = de_vec.get_vocabulary()

        encoder_state, state_h, state_c = self.encoder.call(src=embed_input,
                                                            training=False)

        for i in range(30):
            if next_word != "<eos>":
                tokenized_word = de_vec(next_word)
                embed_word = self.d_embedding(tokenized_word)
                embed_word = tf.expand_dims(embed_word, axis=0)

                decoder_outputs, state_h, state_c = self.decoder.call(trg=embed_word,
                                                                      previous_state=[state_h, state_c],
                                                                      training=False)

                prediction = self.dense(decoder_outputs)

                dist = prediction.numpy().squeeze()
                idx = np.random.choice(range(len(de_vocab)), p=dist)

                next_word = de_vocab[tf.squeeze(idx)]

                if next_word == "[UNK]":
                    continue

                translation.append(next_word)
            else:
                break

        return translation
