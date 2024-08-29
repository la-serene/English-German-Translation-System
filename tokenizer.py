import tensorflow as tf
import tensorflow_text as tf_text
from tensorflow.keras.layers import TextVectorization

max_vocab_size = 16000


def text_standardize(text):
    # Split accented characters.
    text = tf_text.normalize_utf8(text, 'NFKD')
    text = tf.strings.lower(text)

    # Keep space, a to z, and select punctuation.
    text = tf.strings.regex_replace(text, '[^ a-z.?!,¿]', '')

    # Add spaces around punctuation.
    text = tf.strings.regex_replace(text, '[.?!,¿]', r' \0 ')

    # Strip whitespace.
    text = tf.strings.strip(text)
    text = tf.strings.join(['[START]', text, '[END]'], separator=' ')

    return text


def get_word_to_idx(tokenizer):
    vocab = tokenizer.get_vocabulary()
    word_to_idx = {}
    for i in range(len(vocab)):
        word_to_idx[vocab[i]] = i
    return word_to_idx


def get_vocab(tokenizer):
    return tokenizer.get_vocabulary()


en_vec = TextVectorization(max_tokens=max_vocab_size,
                           standardize=text_standardize,
                           ragged=True)
ger_vec = TextVectorization(max_tokens=max_vocab_size,
                            standardize=text_standardize,
                            ragged=True)
