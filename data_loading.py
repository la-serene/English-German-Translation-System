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


def process_dataset(path_to_dataset):
    en_set = []
    de_set = []

    with open(path_to_dataset) as f:
        for line in f:
            en_de = line.split("CC-BY")
            if len(en_de) > 0:
                sample = en_de[0]
                sample = sample.strip().split('\t')
                en_set.append(expand_contraction(sample[0], contraction_mapping))
                de_set.append("<SOS> " + sample[1] + " <EOS>")

    en_set = np.array(en_set)
    de_set = np.array(de_set)

    return en_set, de_set


def convert_to_tf_dataset(context, target):
    return (
        tf.data.Dataset
        .from_tensor_slices((context, target))
        .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )
