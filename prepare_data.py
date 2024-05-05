import numpy as np
import tensorflow as tf

from clean_data import expand_contractions

BUFFER_SIZE = 1024
BATCH_SIZE = 64


def prepare_dataset(path_to_dataset):
    english = []
    german = []

    with open(path_to_dataset) as f:
        for line in f:
            if len(line) > 0:
                line = line.split("    ")
                english.append(expand_contractions(line[0], lang="en"))
                german.append(expand_contractions(line[1], lang="ger"))

    english = np.asarray(english)
    german = np.asarray(german)

    return english, german


def convert_to_tf_dataset(english, german):
    # Data splitting mask
    mask = np.full((len(english),), False)
    train_mask = np.copy(mask)
    train_mask[:int(len(english) * 0.8)] = True
    np.random.shuffle(train_mask)

    false_indices = np.where(train_mask is False)[0]
    np.random.shuffle(false_indices)
    border_idx = int(len(false_indices) / 2)

    val_mask = np.copy(mask)
    val_mask[false_indices[:border_idx]] = True

    test_mask = np.copy(mask)
    test_mask[false_indices[border_idx:]] = True

    # Create tf.data.Dataset
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


def tokenize_dataset(dataset, en_vec, ger_vec):
    def process_text(context, target):
        context = en_vec(context).to_tensor()
        target = ger_vec(target)
        targ_in = target[:, :-1].to_tensor()
        targ_out = target[:, 1:].to_tensor()

        return (context, targ_in), targ_out

    return dataset.map(process_text, tf.data.AUTOTUNE)
