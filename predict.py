import sys

from prepare_data import *
from model import NMT
from tokenize import *

# Constant
hidden_units = 64
embedding_size = 32

en_set, de_set = prepare_dataset("dataset/en_de/deu.txt")
train_raw, val_raw, test_raw = convert_to_tf_dataset(en_set, de_set)

en_vec.adapt(train_raw.map(lambda src, tar: src))
ger_vec.adapt(train_raw.map(lambda src, tar: tar))

input_vocab_size = len(en_vec.get_vocabulary())
output_vocab_size = len(ger_vec.get_vocabulary())


def main():
    source_sentence = sys.argv[-1]

    nmt = NMT(en_vec,
              ger_vec,
              embedding_size,
              hidden_units)

    # Init model params
    nmt.predict("lorem ispum")
    nmt.load_weights("weights/model_v4.weights.h5")
    translation = nmt.predict(source_sentence)
    print(translation)

    return translation


if __name__ == "__main__":
    main()
