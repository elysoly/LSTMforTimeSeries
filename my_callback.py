import keras
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation


class Histories(keras.callbacks.Callback):

    def __init__(self, tr_data):
        self.train_data = tr_data

    def on_train_begin(self, logs={}):
        # fig1, ax = plt.subplots()
        # x = range( len( self.train_data[1] ) )
        # line1, = ax.plot( x, self.train_data[1] )
        # self.fig = fig1
        # self.line = line1
        # plt.show(block=False)
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        # x = range( len( self.train_data[1] ) )
        # y = self.model.predict( self.train_data[0] )
        # plt.plot(x, self.train_data[1], '-g', label='train true' )
        # plt.hold(True)
        # plt.plot(x, y, '-k', label='train prediction' )
        # # self.plot[0].legend( loc='upper left' )
        # # plt.show( block=False )
        # plt.hold(False)
        # # self.plot[1].hold( False )
        # # # er1 = np.power( tr_predicted - self.train_data[1], 2 ) / 2
        # # # x = range( len( self.train_data[1] ) )
        # # # self.plot[1].plot(er1, '-g' )
        # plt.pause(0.05)

        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return
