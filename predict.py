import argparse

import tensorflow as tf

from model import NMT
from prepare_data import prepare_dataset, convert_to_tf_dataset
from tokenizer import en_vec, ger_vec, ger_word_to_idx
from utils import get_model_metadata


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="v9")

    return parser.parse_args()


def main():
    args = get_args()

    en_set, de_set = prepare_dataset("dataset/en_de.txt")
    train_raw, val_raw, test_raw = convert_to_tf_dataset(en_set, de_set)

    en_vec.adapt(train_raw.map(lambda src, tar: src))
    ger_vec.adapt(train_raw.map(lambda src, tar: tar))

    _, metadata = get_model_metadata(args.model_name)

    nmt = tf.keras.models.load_model(metadata["model_path"])

    while 1:
        sentence = input("Input the English sentence:")

        if sentence == "EXIT":
            break

        translation = nmt.translate(sentence, ger_word_to_idx)
        translation = " ".join(translation)
        print("{}\n".format(translation))

    return 0


if __name__ == "__main__":
    main()
