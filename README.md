# Reddit Comments Sentiment Analysis
## About the project

This project is a Recurrent + Convolutional Neural Network model that classifies the sentiment reddit comments into positive, neutral, and negative ones, with a validation accuracy of 77%. It is built with a 1D ConvNet and a stack of bidirectional LSTMs, and based on the GloVe embeddings.

### Details

**Keras.Models.Sequential Model**

| Layer   | Output Shape | Number of Parameters | Notes
| ----------- | ----------- | --- | ---|
| `Embedding(1000,50)`     | (None, 100, 50)    | 50,000| initialized with GloVe coefficients; trainable
| `keras.layers.Conv1D(32)`   | (None, 94, 32) | 11232 |
|`MaxPooling1D(3)`|(None, 31, 32)|0|
|`Bidirectional(LSTM(16))`|(None, 31, 32)|6272| regularized with L1; `return_sequences` set to `True` |
|`Bidirectional(LSTM(16))`|(None, 31, 32)|6272| regularized with L1; `return_sequences` set to `True` |
|`Bidirectional(LSTM(16))`|(None, 31, 32)|6272| regularized with L1; `return_sequences` set to `True` |
|`Bidirectional(LSTM(16))`|(None, 32)|6272| regularized with L1 |
|`Dense(3)`|(None, 3)|99| activated with softmax |


The model takes a sequence that represents a tokenized comment, which is 100 words in length (longer comments shall be truncated, and shorter ones padded)

The model is initialized with the GloVe representation matrix, which is then trained along with the rest of the network.

The notebook also builds a WhatsApp bot powered by Twilio that messages about what happens after each epoch, which is achieved via Keras Callbacks.

Frameworks: Tensorflow, Keras, Matplotlib, Twilio


## About the repository
* `nlp.ipynb`: the jupyter notebook used to build the model
* `BestModel.h5`: the model with accuracy of ~77% (in form of an h5 file), can be loaded with`keras.model.load('BestModel.h5')`
* the dataset and the GloVe embeddings are readily available via links in the credits, hence are not included.

## Credits
* the data source is the Kaggle dataset ["Twitter and Reddit Sentimental analysis Dataset"](https://www.kaggle.com/datasets/cosmos98/twitter-and-reddit-sentimental-analysis-dataset?resource=download)
* the [GloVe Representations](https://nlp.stanford.edu/projects/glove/) built by Stanford
* Some code are adapted from Francois Chollet's book *Deep Learning with Python (2018)*, and they are indicated in the notebook
