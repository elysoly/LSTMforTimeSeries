import lstm
import time
import matplotlib.pyplot as plt
import numpy as np


def get_inputs():
    print('Please Enter train, validation and test percentage respectively separated by comma:')
    [tr_, val_, ts_] = [float(x) for x in input().split(',')]
    print('Please Enter sequence length and max epoch number needed for prediction respectively separated by comma:')
    [seqlen, eps] = [int(x) for x in input().split(',')]
    print( 'Please enter your model file name IF you wanna use a pre-trained model:' )
    name = input()
    print( 'Please enter batch size (1 if you want recursive mode):' )
    b_size = int(input())

    return tr_, val_, ts_, seqlen, eps, name, b_size


def save_model(this_model):
    print('Enter a name to save your model:')
    model_saving_name = input()
    if model_saving_name != '':
        path = './results/' + model_saving_name
        lstm.save_model(this_model, path)
    else:
        return


def plot_results():

    x = range(len(y_train), len(y_train) + len(y_test))
    f, axarr = plt.subplots(2, sharex=True)
    axarr[0].plot(x, predicted, '-r', label='test prediction')
    axarr[0].plot(x, y_test, '-b', label='test True')

    tr_predicted = lstm.predict_point_by_point(model, X_train)
    x = range(len(y_train))
    axarr[0].plot(x, y_train, '-g', label='train prediction')
    axarr[0].plot(x, tr_predicted, '-k', label='train True')
    axarr[0].legend(loc='upper left')

    er1 = np.power(tr_predicted - y_train, 2) / 2
    x = range(len(y_train))
    axarr[1].plot(x, er1, '-g')
    er = np.power(predicted - y_test, 2) / 2
    x = range(len(y_train), len(y_train) + len(y_test))
    axarr[1].plot(x, er, '-r')
    axarr[0].set_ylabel('Series')
    axarr[1].set_ylabel('Error')
    axarr[1].set_xlabel('time')

    plt.show()
    plt.pause(1)


# Main Run Thread
while(True):
    epochs = 20
    seq_len = 5
    tr_pr = 0.7
    val_pr = 0.15
    ts_pr = 0.15
    my_batch_size = 10
    [tr_pr, val_pr, ts_pr, seq_len, epochs, model_name, my_batch_size] = get_inputs()
    tr_val_percentage = tr_pr+val_pr
    print('> Loading data... ')
    global_start_time = time.time()
    X_train, y_train, X_test, y_test = lstm.load_data('my_data.csv', seq_len, tr_val_percentage)
    print('> Data Loaded. Compiling...')

    # model_name = ''
    if model_name != '':
        model = lstm.my_load_model('./results/' + model_name)
    else:

        model = lstm.build_model([1, seq_len, 100, 1])
        h = model.fit(
            x=X_train,
            y=y_train,
            batch_size=my_batch_size,
            nb_epoch=epochs,
            validation_split=val_pr/(val_pr+tr_pr)
            )
        plt.figure()
        plt.plot(h.history['loss'])
        plt.plot(h.history['val_loss'])
        plt.xlabel('#epoch')
        plt.ylabel('MSE')
        plt.legend(['validation', 'train'], loc='upper right')

    predicted = lstm.predict_point_by_point(model, X_test)
    print('Training duration (s) : ', time.time() - global_start_time )
    a = y_test-predicted
    mse_test = sum(np.power(a, 2))/len(a)
    print('test mse: ')
    print(mse_test)
    plot_results()
    save_model(model)




