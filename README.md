# Sentiment Analysis using RNNs with Keras

This project demonstrates how to perform sentiment analysis using Recurrent Neural Networks (RNNs), particularly Long Short-Term Memory (LSTM) networks, in Keras. We utilize the IMDb movie reviews dataset provided by Keras, which is a popular benchmark dataset in natural language processing and sentiment analysis tasks.

## Dataset

The dataset comprises 50,000 movie reviews from IMDb, labeled as positive or negative. We use the built-in IMDb dataset from Keras, limiting the vocabulary to the top 5,000 words for simplicity.

```python
from keras.datasets import imdb

vocabulary_size = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocabulary_size)
# ML-Examples

Preprocessing

We inspect and preprocess the data, ensuring all reviews have the same length through padding, which is essential for training the RNN.
from keras.preprocessing import sequence

max_words = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)
Model Architecture

The model consists of an Embedding layer, an LSTM layer, and a Dense output layer with a sigmoid activation function. This architecture is suited for binary classification tasks such as sentiment analysis.

from keras import Sequential
from keras.layers import Embedding, LSTM, Dense

embedding_size = 32
model = Sequential()
model.add(Embedding(vocabulary_size, embedding_size, input_length=max_words))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
from keras import Sequential
from keras.layers import Embedding, LSTM, Dense

embedding_size = 32
model = Sequential()
model.add(Embedding(vocabulary_size, embedding_size, input_length=max_words))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))

Compilation

The model is compiled with the binary_crossentropy loss function, the adam optimizer, and accuracy as the evaluation metric.


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
Training

The model is trained on the dataset, using a validation split for monitoring the performance. Adjust the batch size and number of epochs as needed.


batch_size = 64
num_epochs = 3

X_valid, y_valid = X_train[:batch_size], y_train[:batch_size]
X_train2, y_train2 = X_train[batch_size:], y_train[batch_size:]

model.fit(X_train2, y_train2, validation_data=(X_valid, y_valid), batch_size=batch_size, epochs=num_epochs)
Evaluation

Finally, evaluate the model's performance on the test set.


scores = model.evaluate(X_test, y_test, verbose=0)
print('Test accuracy:', scores[1])
Conclusion

This README provides an overview of using RNNs, specifically LSTMs, for sentiment analysis with Keras. The process includes loading the data, preprocessing, modeling, training, and evaluation. Adjustments to the model architecture or training parameters can be made to enhance performance or address overfitting/underfitting issues.


Feel free to adjust the content as needed to fit the specifics of your project or to add any additional sections that might be relevant.
