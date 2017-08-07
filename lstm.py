import os
import time
import warnings
import numpy as np
from keras import losses, optimizers
from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.models import load_model


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Hide messy TensorFlow warnings
warnings.filterwarnings("ignore")  # Hide messy Numpy warnings


def save_model(model, path):
    model.save(path)  # creates a HDF5 file 'my_model.h5'


def my_load_model(path):
    m = load_model(path)
    return m


def load_data(filename, seq_len, tr_pr):
    f = open(filename, 'rb').read()
    data = f.decode().split('\n')

    for i in range(len(data)):
        data[i] = float(data[i])
    my_min = min(data)
    my_max = max(data)
    t = (np.ones(len(data)))*(my_max - my_min)
    data = data/t

    sequence_length = seq_len + 1
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])

    result = np.array(result)

    row = tr_pr * result.shape[0]

    train = result[:int(row), :]
    x_train = train[:, :-1]
    y_train = train[:, -1]

    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    x_train.tofile('x_train')
    y_train.tofile('y_train')

    return [x_train, y_train, x_test, y_test]


def build_model(layers):
    model = Sequential()
    model.add( LSTM(
        input_shape=(layers[1], layers[0]),
        output_dim=layers[1],
        return_sequences=True))
    model.add(Dropout(0.2))

    model.add( LSTM(
        layers[2],
        return_sequences=False ) )
    model.add( Dropout( 0.2 ) )

    model.add(Dense(
        output_dim=layers[3]))
    model.add( Activation("linear"))

    start = time.time()
    model.compile(loss="mse", optimizer= optimizers.RMSprop(lr=0.0001))
    print("> Compilation Time : ", time.time() - start)
    return model


def predict_point_by_point(model, data):
    # Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
    predicted = model.predict(data)
    predicted = np.reshape(predicted, (predicted.size,))
    return predicted
