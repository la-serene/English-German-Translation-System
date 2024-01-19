import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow_text as tf_text

max_vocab_size = 20000


def tf_lower_and_split_punct(text):
    # Split accented characters.
    text = tf_text.normalize_utf8(text, 'NFKD')
    text = tf.strings.lower(text)

    # Keep space, a to z, and select punctuation.
    text = tf.strings.regex_replace(text, r'[^ a-z.?!\\,<>]', '')

    # Add spaces around punctuation.
    text = tf.strings.regex_replace(text, r'[.?!,]', r' \0 ')

    # Strip whitespace.
    text = tf.strings.strip(text)

    return text


en_vec = layers.TextVectorization(max_tokens=max_vocab_size,
                                  standardize=tf_lower_and_split_punct)
de_vec = layers.TextVectorization(max_tokens=max_vocab_size,
                                  standardize=tf_lower_and_split_punct)
