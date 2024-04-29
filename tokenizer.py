import tensorflow as tf
import tensorflow_text as tf_text
from tensorflow.keras.layers import TextVectorization

max_vocab_size = 15000

contraction_map = {
    # This should be wrapped as a JSON file.
    "Let's": "Let us",
    "'d better": " had better",
    "'s": " is",
    "'re": " are",
    "'m": " am",
    "'ll": " will",
    "'d": " would",
    "'ve": " have",
    "won't": "will not",
    "n't": " not",
    "cannot": "can not"
}


def expand_contractions(text, mapping=None):
    if mapping is None:
        mapping = contraction_map
    for key, value in mapping.items():
        text = tf.strings.regex_replace(text, key, value)
    return text


def tf_lower_and_split_punct(text):
    # Split accented characters.
    text = expand_contractions(text)
    text = tf_text.normalize_utf8(text, 'NFKD')
    text = tf.strings.lower(text)

    # Keep space, a to z, and select punctuation.
    text = tf.strings.regex_replace(text, '[^ a-z.?!,¿]', '')

    # Add spaces around punctuation.
    text = tf.strings.regex_replace(text, '[.?!,¿]', r' \0 ')

    # Strip whitespace.
    text = tf.strings.strip(text)

    return text


def tf_split_punct(text):
    # Split accented characters.
    text = tf_text.normalize_utf8(text, 'NFKD')
    text = tf.strings.lower(text)

    # Keep space, a to z, and select punctuation.
    text = tf.strings.regex_replace(text, '[^ a-z.?!,¿]', '')

    # Add spaces around punctuation.
    text = tf.strings.regex_replace(text, '[.?!,¿]', r' \0 ')

    # Strip whitespace and add special tokens
    text = tf.strings.strip(text)
    text = tf.strings.join(['[START]', text, '[END]'], separator=' ')

    return text


en_vec = TextVectorization(max_tokens=max_vocab_size,
                           standardize=tf_lower_and_split_punct)
ger_vec = TextVectorization(max_tokens=max_vocab_size,
                            standardize=tf_split_punct)
