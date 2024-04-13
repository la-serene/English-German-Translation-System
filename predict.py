from model import NMT
from prepare_data import *
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
    nmt = NMT(en_vec,
              ger_vec,
              embedding_size,
              hidden_units)

    print("Loading the weight...")

    # Init model params
    nmt.predict("lorem ispum")
    nmt.load_weights("weights/model_v4.weights.h5")

    while 1:
        sentence = input("Input the English sentence:")

        if sentence == "" or "\n":
            break

        translation = nmt.predict(sentence)
        print(translation)

    return 0


if __name__ == "__main__":
    main()
