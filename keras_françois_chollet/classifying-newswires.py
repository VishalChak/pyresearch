import keras
print(keras.__version__)


import numpy as np
from keras.datasets import  reuters
from keras import models, layers
import  matplotlib.pyplot as plt

(train_data, train_labels) , (test_data, test_labels) = reuters.load_data(num_words=10000)

words_index = reuters.get_word_index()
reverse_words_index = dict([(val, key) for (key, val) in words_index.items()])
decode_news_wire = " ".join([reverse_words_index.get(val-3, '?') for val in train_data[0]])


def vectorize_sequences(sequences, dimension = 10000):
    results = np.zeros((len(sequences), dimension))
    
    for i , label in enumerate(sequences):
        results[i, label] = 1

    return results

def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))

    for i, label in enumerate(labels):
        results[i, label] = 1
    return results


### there is a build-in way in keras for one hot encoding

from keras.utils.np_utils import  to_categorical
y_train_build_in = to_categorical(train_labels)
y_test_build_in = to_categorical(test_labels)





def network():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(46, activation='softmax'))

    model.compile(optimizer='rmsprop', loss = 'categorical_crossentropy', metrics=['accuracy'])


    return model

def fuc_plot_loss(epochs, loss, val_loss):
    plt.plot(epochs, loss, 'bo', label = "traning_loss")
    plt.plot(epochs, val_loss, 'b', label = "validation_loss")
    plt.xlabel("Epochs")
    plt.ylabel('Loss')
    plt.title("train validation loss")
    plt.legend()
    plt.show()

def fuc_plot_acc(epochs, acc, val_acc):
    plt.plot(epochs, acc, 'bo', label = "traning_acc")
    plt.plot(epochs, val_acc, 'b', label = "validation_acc")
    plt.xlabel("Epochs")
    plt.ylabel('accuracy')
    plt.title("train validation Accuracy")
    plt.legend()
    plt.show()


X_train = vectorize_sequences(train_data)
X_test = vectorize_sequences(test_data)

y_train = to_one_hot(train_labels)
y_test = to_one_hot(test_labels)

X = X_train[1000:]
X_val = X_train[:1000]

y = y_train[1000:]
y_val = y_train[:1000]

model = network()


history = model.fit(X, y, epochs=50, batch_size=512, validation_data=(X_val, y_val))
history_dict  = history.history

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']
epochs = range(1, len(loss) +1)


def print_data_etl():
    print(train_data.shape)
    print(type(train_data))
    print(train_data[0])
    print(train_labels.shape)
    print(train_labels[0])
    print(words_index)
    print(reverse_words_index)
    print(decode_news_wire)
    print(history_dict.keys())


# print_data_etl()
fuc_plot_acc(epochs, acc, val_acc)
fuc_plot_loss(epochs, loss, val_acc)