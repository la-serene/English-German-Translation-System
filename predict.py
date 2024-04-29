from model import NMT
from prepare_data import prepare_dataset, convert_to_tf_dataset
from tokenizer import en_vec, ger_vec


def main():
    # Constant
    hidden_units = 64
    embedding_size = 32

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
              embedding_size,
              hidden_units)

    print("Loading the weight...")

    # Init model params
    nmt.predict("lorem ispum", word_to_idx)
    # nmt.load_weights("weights/model_v4.weights.h5")

    while 1:
        sentence = input("Input the English sentence:")

        if sentence == "EXIT":
            break

        translation = nmt.predict(sentence, word_to_idx)
        translation = " ".join(translation)
        print(translation)

    return 0


if __name__ == "__main__":
    main()
