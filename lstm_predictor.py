import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.contrib import learn as tflearn
from tensorflow.contrib import layers as tflayers
from tensorflow.contrib import rnn
import warnings

warnings.filterwarnings("ignore")

def rnn_data(data, time_steps, labels=False):
    """
    creates new data frame based on previous observation
      * example:
        l = [1, 2, 3, 4, 5]
        time_steps = 2
        -> labels == False [[1, 2], [2, 3], [3, 4]] #Data frame for input with 2 timesteps
        -> labels == True [3, 4, 5] # labels for predicting the next timestep
    """
    rnn_df = []
    for i in range(len(data) - time_steps):
        if labels:
            try:
                rnn_df.append(data.iloc[i + time_steps].as_matrix())
            except AttributeError:
                rnn_df.append(data.iloc[i + time_steps])
        else:
            data_ = data.iloc[i: i + time_steps].as_matrix()
            rnn_df.append(data_ if len(data_.shape) > 1 else [[i] for i in data_])

    return np.array(rnn_df, dtype=np.float32)



def split_data(data, val_size=0.1, test_size=0.1):
    """
    splits data to training, validation and testing parts
    """
    ntest = int(round(len(data) * (1 - test_size)))
    nval = int(round(len(data.iloc[:ntest]) * (1 - val_size)))

    df_train, df_val, df_test = data.iloc[:nval], data.iloc[nval:ntest], data.iloc[ntest:]

    return df_train, df_val, df_test



def prepare_data(data, time_steps, labels=False, val_size=0.1, test_size=0.1):
    """
    Given the number of `time_steps` and some data,
    prepares training, validation and test data for an lstm cell.
    """
    df_train, df_val, df_test = split_data(data, val_size, test_size)
    return (rnn_data(df_train, time_steps, labels=labels),
            rnn_data(df_val, time_steps, labels=labels),
            rnn_data(df_test, time_steps, labels=labels))

def load_csvdata(rawdata, time_steps, seperate=False):
    # data = rawdata
    # if not isinstance(data, pd.DataFrame):
    #     data = pd.DataFrame(data)
    #
    # train_x, val_x, test_x = prepare_data(data['a'] if seperate else data, time_steps)
    # train_y, val_y, test_y = prepare_data(data['b'] if seperate else data, time_steps, labels=True)
    # return dict(train=train_x, val=val_x, test=test_x), dict(train=train_y, val=val_y, test=test_y)
    y = {}
    x = {}
    data = np.genfromtxt('my_data.dat', delimiter='\n')
    x_tr = data[0:2554]
    l = len(x_tr) - time_steps
    tt_x = np.zeros([l, time_steps])
    for i in range(l):
        tt_x[i, :] = x_tr[i:i + time_steps]

    x_val = data[2555:3103]
    l = len(x_val) - time_steps
    vv_x = np.zeros([l, time_steps])
    for i in range(l):
        vv_x[i, :] = x_val[i:i + time_steps]

    x_test = data[3104:3649]
    l = len(x_test) - time_steps
    tst_x = np.zeros([l, time_steps])
    for i in range(l):
        tst_x[i, :] = x_test[i:i + time_steps]

    x['train'] = tt_x
    x['val'] = vv_x
    x['test'] = tst_x

    y['train'] = x_tr[time_steps:len(x_tr)+time_steps]
    y['val'] = x_val[1:len(x_val)]
    y['test'] = x_test[time_steps+1:len(x_test)]

    return dict(train=x['train'], val=x['val'], test=x['test']), dict(train=y['train'], val=y['val'], test=y['test'])



def generate_data(fct, x, time_steps, seperate=False):
    """generates data with based on a function fct"""
    data = fct(x)
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    train_x, val_x, test_x = prepare_data(data['a'] if seperate else data, time_steps)
    train_y, val_y, test_y = prepare_data(data['b'] if seperate else data, time_steps, labels=True)
    return dict(train=train_x, val=val_x, test=test_x), dict(train=train_y, val=val_y, test=test_y)


def lstm_model(time_steps, rnn_layers, dense_layers=None, learning_rate=0.01, optimizer='Adagrad',learning_rate_decay_fn = None): # [Ftrl, Adam, Adagrad, Momentum, SGD, RMSProp]
    print(time_steps)
    #exit(0)
    """
        Creates a deep model based on:
            * stacked lstm cells
            * an optional dense layers
        :param num_units: the size of the cells.
        :param rnn_layers: list of int or dict
                             * list of int: the steps used to instantiate the `BasicLSTMCell` cell
                             * list of dict: [{steps: int, keep_prob: int}, ...]
        :param dense_layers: list of nodes for each layer
        :return: the model definition
        """

    def lstm_cells(layers):
        print('-------------------------sdsdsdsdssd---------------------------------------------',layers)
        if isinstance(layers[0], dict):
            return [rnn.DropoutWrapper(rnn.BasicLSTMCell(layer['num_units'],state_is_tuple=True),layer['keep_prob'])
                    if layer.get('keep_prob')
                    else rnn.BasicLSTMCell(layer['num_units'], state_is_tuple=True)
                    for layer in layers]

        return [rnn.BasicLSTMCell(steps, state_is_tuple=True) for steps in layers]

    def dnn_layers(input_layers, layers):
        if layers and isinstance(layers, dict):
            return tflayers.stack(input_layers, tflayers.fully_connected,
                                  layers['layers'],
                                  activation=layers.get('activation'),
                                  dropout=layers.get('dropout'))
        elif layers:
            return tflayers.stack(input_layers, tflayers.fully_connected, layers)
        else:
            return input_layers

    def _lstm_model(X, y):
        stacked_lstm = rnn.MultiRNNCell(lstm_cells(rnn_layers), state_is_tuple=True)
        x_ =  tf.unstack(X, num=time_steps, axis=1)


        output, layers = rnn.static_rnn(stacked_lstm, x_, dtype=dtypes.float32)
        output = dnn_layers(output[-1], dense_layers)
        prediction, loss = tflearn.models.linear_regression(output, y)
        train_op = tf.contrib.layers.optimize_loss(
            loss, tf.contrib.framework.get_global_step(), optimizer=optimizer,
            learning_rate = tf.train.exponential_decay(learning_rate, tf.contrib.framework.get_global_step(), decay_steps = 1000, decay_rate = 0.9, staircase=False, name=None))

        print('learning_rate',learning_rate)
        return prediction, loss, train_op

    # https://www.tensorflow.org/versions/r0.10/api_docs/python/train/decaying_the_learning_rate

    return _lstm_model