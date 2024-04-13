import numpy as np
import tensorflow as tf


contraction_mapping = {
    "Let's": "Let us",
    "'d better": " had better",
    "'s": " is",
    "'re": " are",
    "n't": " not",
    "'m": " am",
    "'ll": " will",
    "'d": " would",
    "'ve": " have",
    "won't": "will not",
    "cannot": "can not"
}

BUFFER_SIZE = 1024
BATCH_SIZE = 64


def expand_contraction(text, mapping=None):
    if mapping is None:
        mapping = contraction_mapping
    for contraction, expanded in mapping.items():
        text = text.replace(contraction, expanded)
    return text


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


def convert_to_tf_dataset(context, target):
    return (
        tf.data.Dataset
        .from_tensor_slices((context, target))
        .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )
