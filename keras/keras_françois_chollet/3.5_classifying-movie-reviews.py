import  keras
import tensorflow as tf

from keras.datasets import imdb

(train_data, train_labels) , (test_data, test_labels) = imdb.load_data(num_words= 10000)
print(train_data[0])
print(train_labels[0])

print(max ([max(sequence) for sequence in train_data]))

### 

word_index = imdb.get_word_index()
print(word_index.keys(), len(word_index))

reverse_word_index = dict ( [(val, key)  for (key, val) in word_index.items()])
print(reverse_word_index.keys())

decode_review = " ".join([reverse_word_index.get(i-3 , '?') for i in train_data[0]])
print(decode_review)


####  Preparing the data


import numpy as np

def vectorize_sequences (sequences , dimension = 10000):
    results = np.zeros((len(sequences), dimension))
    for i , sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results

X_train = vectorize_sequences(train_data)
print(X_train[0])

X_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

print(y_train[0])


from keras import  models
from keras import  layers

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape = (10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

#model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuary'])

# or we can use like this

from keras import  losses, metrics, optimizers

model.compile(optimizer= optimizers.RMSprop(lr=0.001), loss=losses.mean_absolute_error , metrics=[metrics.binary_accuracy])

X_val = X_train[:10000]
partial_X_train = X_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]


history = model.fit(partial_X_train,
                    partial_y_train,
                    epochs=4,
                    batch_size=512,
                    validation_data=(X_val, y_val))


history_dict = history.history

print(history_dict.keys())

acc = history_dict['binary_accuracy']
val_acc= history_dict['val_binary_accuracy']

loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) +1) 

import matplotlib.pyplot as plt


def plot_loss():
    plt.plot(epochs, loss, 'bo', label = 'traning_loss')
    plt.plot(epochs, val_loss, 'b' , label = "validation_loss")
    plt. title("traning and validation loss")
    plt.xlabel('Epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.show()

def plot_acc():
    plt.plot(epochs, acc, 'bo', label = "traning_acc")
    plt.plot(epochs, val_acc, 'b', label = 'validation_acc')
    plt.title("traning and validation acc")
    plt.xlabel('epochs')
    plt.ylabel('acc')
    plt.legend()
    plt.show()

#plot_acc()
#plot_loss()

results = model.evaluate(X_test, y_test)
pred_x_test = model.predict(X_test) 
print(results)
print( pred_x_test)