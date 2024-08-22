import argparse

import tensorflow as tf

from model import NMT
from prepare_data import prepare_dataset, convert_to_tf_dataset, tokenize_dataset
from tokenizer import en_vec, ger_vec
from utils import save_model_metadata


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="./dataset/en_de.txt")
    parser.add_argument('--embedding_size', type=int, default=128)
    parser.add_argument('--hidden_units', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--save_path', type=str, default="./weights/custom_v1.weights.h5")

    return parser.parse_args()


def main():
    args = get_args()
    en_set, de_set = prepare_dataset(args.data_path)
    train_raw, val_raw, test_raw = convert_to_tf_dataset(en_set, de_set)

    en_vec.adapt(train_raw.map(lambda src, tar: src))
    ger_vec.adapt(train_raw.map(lambda src, tar: tar))

    train_ds = tokenize_dataset(train_raw, en_vec, ger_vec)
    val_ds = tokenize_dataset(val_raw, en_vec, ger_vec)
    test_ds = tokenize_dataset(test_raw, en_vec, ger_vec)

    model = NMT(en_vec, ger_vec, args.embedding_size, args.hidden_units)
    model.compile(optimizer=tf.keras.optimizers.Adam(args.learning_rate),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=["accuracy"])
    print("Model is training...")
    model.fit(train_ds, epochs=args.epochs, validation_data=val_ds)
    print("Model is evaluating...")
    model.evaluate(test_ds)

    model_name = input("Enter model name: ")
    model.save(args.save_path)
    print("Model {} saved to {}".format(model_name, args.save_path))

    save_model_metadata(model_name, args.save_path, args.embedding_size, args.hidden_units)


if __name__ == "__main__":
    main()
