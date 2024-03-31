# Language Identifier

## Introduction
This is a deep learning project created with the help of Tensorflow that predicts the language of a given text snippet. Currently, this language prediction model supports a total of 22 languages as of now, which include: Arabic, Chinese, Dutch, English, Estonian, French, Hindi, Indonesian, Japanese, Korean, Latin, Persian, Portuguese, Pashto, Romanian, Russian, Spanish, Swedish, Tamil, Thai, Turkish, and Urdu.

## Dataset used in this project

The dataset used in this project is taken from kaggle: https://www.kaggle.com/datasets/zarajamshaid/language-identification-datasst

## Models used in this project

1) Vanilla Sequential model
2) TextCNN model
3) Bidirectional SimpleRNN model
4) Bidirectional LSTM model
5) Bidirectional GRU model
6) Ensemble Learning(Bidirectional LSTM + Bidirectional GRU) model

**Out of the all the above models, textCNN proved to be the most effective one with a training accuracy of around 78.99% and testing accuracy of around 73.65%**

## About the web application of the deep learning model

The deep learning model of this project is connected with an application created with Gradio for real time prediction and it is deployed on HuggingFace Spaces.

## Links

Live Preview: https://huggingface.co/spaces/som11/language-predictor

## Warning
While the model of this project can classify languages correctly, but in some cases, the model may misclassify languages, therefore, it is strongly advised not to rely solely on the output of this model.
