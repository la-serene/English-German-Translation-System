# Introduction

An English - German Neural Machine Translator informed by the seq2seq paper by Sutskever et al. (2014).

Note: Since this project is carried out for educational purpose, the overall performance may not be satisfying.
Besides, there is no evaluation step despite the test set.

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
pip install requirements.txt
```

# Usage

After replacing the `____` with the source English sentence and running the command, the translation will be printed
in console.

```
python predict.py ____
```

Currently, there is no additional configurable command-line arguments. In the future, an optional argument
for `temperature` might be available.

# Data

The dataset is taken from Kaggle
Dataset [English To German](https://www.kaggle.com/datasets/kaushal2896/english-to-german), consisting of around 221,533
English-German pair of sentences and randomly split into 3 sets: training, evaluation and testing in which the last 2
contain 25,000 samples. Due to some reasons, test set is not used in any operation.

# Training

This project is not developed for training on a custom dataset. However, all training detail can be found in the
project's notebook.

# Results

Although the project has made it in implementing the seq2seq model proposed by Sutskever et al. (2014), model's
translations can not be considered decent enough. This could be due to various factors, namely the lack of dataset
diversity, model setting, etc. Repeatedly sampling a new translation could be considered as a user-friendly solution to find the
best translation as well as test the model quality.

Here are some translated examples. Note that the German version has been modified to follow German grammar.
As said, translations should not be expected to fully convey the meaning of the original sentence.

| English                                | German (edited)                               |
|----------------------------------------|-----------------------------------------------|
| This is the first translation.         | Das ist der schnellste Kapitel.               |
| I didn't go to his birthday yesterday. | Ich fuhr mit seinen Geburtstag nach drauen.   |
| He tries to read a book every week.    | Er hat einen woche eine Reicht fur sie horen. |

# References

<a id="1" href="https://arxiv.org/abs/1409.3215">[1]</a>
Ilya Sutskever, Oriol Vinyals, Quoc V. Le (2014).
Sequence to Sequence Learning with Neural Networks.
NIPS 2014.
