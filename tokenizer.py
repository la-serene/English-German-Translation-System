import re

import tensorflow as tf
import tensorflow_text as tf_text
from tensorflow.keras.layers import TextVectorization

max_vocab_size = 20000

en_contraction_map = {
    "let's": "let us",
    "'d better": " had better",
    "'s": " is",
    "'re": " are",
    "'m": " am",
    "'ll": " will",
    "'d": " would",
    "'ve": " have",
    "'em": " them",
    "won't": "will not",
    "n't": " not",
    "cannot": "can not",
}

ger_contraction_map = {
    "'s": " ist",
    "ä": "ae",
    "ö": "oe",
    "ü": "ue",
    "ß": "ss",
    "'ne ": "eine ",
    "'n ": "ein ",
    "am ": "an dem ",
    "aufs ": "auf das ",
    "durchs ": "durch das ",
    "fuers ": "fuer das ",
    "hinterm ": "hinter dem ",
    "im ": "in dem ",
    "uebers ": "ueber das ",
    "ums ": "um das ",
    "unters ": "unter das ",
    "unterm ": "unter dem ",
    "vors ": "vor das ",
    "vorm ": "vor dem ",
    "zum ": "zu dem ",
    "ins ": "in das ",
    "ans ": "an das ",
    "vom ": "von dem",
    "beim ": "bei dem ",
    "zur  ": "zu der ",
}


def expand_contractions(text, mapping):
    for key, value in mapping.items():
        text = re.sub(key, value, text)
    return text


@tf.keras.utils.register_keras_serializable()
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


en_vec = TextVectorization(max_tokens=max_vocab_size,
                           standardize=text_standardize,
                           ragged=True)
ger_vec = TextVectorization(max_tokens=max_vocab_size,
                            standardize=text_standardize,
                            ragged=True)
