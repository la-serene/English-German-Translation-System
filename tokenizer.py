import tensorflow as tf
import tensorflow_text as tf_text
import re
from tensorflow.keras.layers import TextVectorization

max_vocab_size = 20000

en_contraction_map = {
    # This should be wrapped as a JSON file.
    "let's": "let us",
    "'d better": " had better",
    "'s": " is",
    "'re": " are",
    "'m": " am",
    "'ll": " will",
    "'d": " would",
    "'ve": " have",
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
    "am ": "an dem ",
    "ans ": "an das ",
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
    "vom ": "von dem ",
    "beim ": "bei dem ",
    "zur  ": "zu der ",
}


def expand_contractions(text, mapping):
    for key, value in mapping.items():
        text = re.sub(key, value, text)
    return text


def en_rule(text):
    # Split accented characters.
    text = tf_text.normalize_utf8(text, 'NFKD')

    # Keep space, a to z, and select punctuation.
    text = tf.strings.regex_replace(text, '[^ a-z.?!,¿]', '')

    # Add spaces around punctuation.
    text = tf.strings.regex_replace(text, '[.?!,¿]', r' \0 ')

    # Strip whitespace.
    text = tf.strings.strip(text)

    return text


def ger_rule(text):
    # Split accented characters.
    text = tf_text.normalize_utf8(text, 'NFKD')

    # Keep space, a to z, and select punctuation.
    text = tf.strings.regex_replace(text, '[^ a-z.?!,¿]', '')

    # Add spaces around punctuation.
    text = tf.strings.regex_replace(text, '[.?!,¿]', r' \0 ')

    # Strip whitespace and add special tokens
    text = tf.strings.strip(text)
    text = tf.strings.join(['[START]', text, '[END]'], separator=' ')

    return text


en_vec = TextVectorization(max_tokens=max_vocab_size,
                           standardize=en_rule)
ger_vec = TextVectorization(max_tokens=max_vocab_size,
                            standardize=ger_rule)
