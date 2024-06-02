# Introduction

[![Goolge Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1EKOm7ULFKEusvEFb8thSGOHTfRnRKru7?usp=sharing)

An English - German Neural Machine Translator informed by Bahdanau Attention paper by Bahdanau et al. (2014).

Note: Since this project is carried out for educational purpose, the overall performance may not be satisfying.

# Table of contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Training](#training)
- [Results](#results)
- [References](#references)

# Installation

```
pip install -r requirements.txt
```

# Usage

After replacing the `____` with the source English sentence and running the command, the translation should be printed
in console.

```
python predict.py ____
```

Here is a description on the command-line arguments:
```
--embedding_size: Size of the embedding layer. Default: 128.
--hidden_units: Number of hidden units in the model. Default: 128.
--weight_path: Path to the weights file for the custom model. Default: "./weights/model_v8.weights.h5".
```

Currently, there is no additional configurable command-line arguments. In the future, an optional argument
for `temperature` might be available.

# Data

The dataset is taken from Kaggle
Dataset [English To German](https://www.kaggle.com/datasets/kaushal2896/english-to-german), consisting of around 277,891
English-German pair of sentences and randomly split into 3 sets: training, evaluation and testing with ratio 0.8, 0.1,
0,1 respectively.

# Training

This project is now available for training on any English-German dataset. It's also noteworthy that the custom dataset 
should follow the same pattern as in dataset folder.

Here is a description on the command-line arguments:
```angular2html
--data_path: Path to the dataset file. Default: "./dataset/en_de.txt".
--embedding_size: Size of the embedding layer. Default: 128.
--hidden_units: Number of hidden units in the model. Default: 128.
--epochs: Number of epochs for training. Default: 5.
--learning_rate: Learning rate for the optimizer. Default: 0.001.
--save_path: Path to save the trained model weights. Default: "./weights/custom_model_v1.weights.h5".
```

# Results

Since the last version of this model, the apply of Bahdanau Attention on seq2seq has yielded significant improvement in
the overall performance. Translations have become more satisfying on long sentences. Repeatedly sampling a new translation
could be considered as a user-friendly solution to find the best translation as well as test the model quality.

Here are some translated examples. Note that the German version has been modified to follow German grammar.
As said, translations should not be expected to fully convey the meaning of the original sentence.

| English                                | German                                           |
|----------------------------------------|--------------------------------------------------|
| This is the first translation.         | Danke bitte wahrend das sonst fischen.           |
| I didn't go to his birthday yesterday. | Er arzt gleich den originals uber in den weisen. |
| He tries to read a book every week.    | Er schwierig eine, banane wir gestern zu einem.  |

# References

<a id="1" href="https://arxiv.org/abs/1409.0473">[1]</a>
Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio (2014).
Neural Machine Translation by Jointly Learning to Align and Translate.
ICLR 2015.

<a id="2" href="https://arxiv.org/abs/1409.3215">[2]</a>
Ilya Sutskever, Oriol Vinyals, Quoc V. Le (2014).
Sequence to Sequence Learning with Neural Networks.
NIPS 2014.
