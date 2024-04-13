import numpy as np
import tensorflow as tf

from tokenize import *

BUFFER_SIZE = 1024
BATCH_SIZE = 64


def prepare_dataset(path_to_dataset):
    english = []
    german = []

    with open(path_to_dataset) as f:
        for line in f:
            line = line.split("CC-BY")

            if len(line) > 0:
                sample = line[0]
                sample = sample.strip().split('\t')

                english.append(sample[0])
                german.append(sample[1])

    english = np.array(english)
    german = np.array(german)

    return english, german


def convert_to_tf_dataset(english, german):
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1

    train_mask = np.random.uniform(size=(len(english),)) < train_ratio
    val_mask = np.logical_and(~train_mask, np.random.uniform(size=(len(english),)) < val_ratio)
    test_mask = ~(train_mask | val_mask)

    train_raw = (
        tf.data.Dataset
        .from_tensor_slices((english[train_mask], german[train_mask]))
        .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.experimental.AUTOTUNE))

    val_raw = (
        tf.data.Dataset
        .from_tensor_slices((english[val_mask], german[val_mask]))
        .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.experimental.AUTOTUNE))

    test_raw = (
        tf.data.Dataset
        .from_tensor_slices((english[test_mask], german[test_mask]))
        .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.experimental.AUTOTUNE))

    return train_raw, val_raw, test_raw


def process_text(context, target):
    context = en_vec(context)
    target = ger_vec(target)
    targ_in = target[:, :-1]
    targ_out = target[:, 1:]
    return (context, targ_in), targ_out
