import re


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


def expand_contractions(text, lang="en"):
    if lang == "en":
        mapping = en_contraction_map
    elif lang == "ger":
        mapping = ger_contraction_map

    for key, value in mapping.items():
        text = re.sub(key, value, text)

    return text
