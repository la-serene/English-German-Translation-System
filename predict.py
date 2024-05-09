import argparse

from model import NMT
from prepare_data import prepare_dataset, convert_to_tf_dataset
from tokenizer import en_vec, ger_vec


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_size', type=int, default=128)
    parser.add_argument('--hidden_units', type=int, default=128)
    parser.add_argument('--weight_path', type=str, default="./weights/model_v8.weights.h5")

    return parser.parse_args()


def main():
    args = get_args()

    en_set, de_set = prepare_dataset("dataset/en_de.txt")
    train_raw, val_raw, test_raw = convert_to_tf_dataset(en_set, de_set)

    en_vec.adapt(train_raw.map(lambda src, tar: src))
    ger_vec.adapt(train_raw.map(lambda src, tar: tar))

    ger_voc = ger_vec.get_vocabulary()
    word_to_idx = {}

    for i in range(len(ger_voc)):
        word_to_idx[ger_voc[i]] = i

    nmt = NMT(en_vec,
              ger_vec,
              args.embedding_size,
              args.hidden_units)

    print("Loading the weight...")

    # Init model params
    nmt.translate("lorem ispum", word_to_idx)
    nmt.load_weights(args.weight_path)

    while 1:
        sentence = input("Input the English sentence:")

        if sentence == "EXIT":
            break

        translation = nmt.translate(sentence, word_to_idx)
        translation = " ".join(translation)
        print("{}\n".format(translation))

    return 0


if __name__ == "__main__":
    main()
