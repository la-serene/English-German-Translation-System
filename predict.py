import sys

from data_loading import *
from model import NMT
from tokenize import *

# Constant
hidden_units = 128
embedding_size = 256

en_set, de_set = process_dataset("dataset/en_de/deu.txt")
train_raw = convert_to_tf_dataset(en_set, de_set)

en_vec.adapt(train_raw.map(lambda src, tar: src))
de_vec.adapt(train_raw.map(lambda src, tar: tar))

input_vocab_size = len(en_vec.get_vocabulary())
output_vocab_size = len(de_vec.get_vocabulary())


def main():
    source_sentence = sys.argv[-1]

    nmt = NMT(input_vocab_size,
              output_vocab_size,
              embedding_size,
              hidden_units)

    nmt.load_weights("weights/model_v3")
    print(nmt.predict(source_sentence))

    return 0


if __name__ == "__main__":
    main()
