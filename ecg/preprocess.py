import wfdb
from wfdb import processing
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
import keras.utils as utils


def show_signal(X_set, title_name):
    plt.plot(X_set[3850:4850])
    plt.title(title_name)
    plt.ylabel('mV')
    # plt.legend([fields['sig_name'][0]])
    plt.show()
    return 0


'''-----Baseline removal (using median filters)-------'''


def Baseline_removal(X_set):    # X_set: two dimensions
    # Changeable parameters
    filter_size = np.array([217, 73])        # mini-seconds

    # Unchangeable parameters
    filter_time = filter_size / 0.36      # medium filter window size
    title_list = ['Original Signal', 'Baseline removed signal: ']

    # show_signal(X_set, title_list[0])

    for i in range(filter_time.shape[0]):
        print('... Baseline removing (filter: ' + str(int(filter_time[i])) + ' ms)')
        signal_smoothed = signal.medfilt(X_set.reshape((X_set.shape[0])), filter_size[i])
        X_set = X_set - signal_smoothed.reshape(X_set.shape)
        # show_signal(X_set, title_list[1] + str(int(filter_time[i])) + ' ms')

    return X_set


'''------------------Denoising-----------------'''


def denoising(X_set):   # X_set: two dimensions

    return X_set


'''--------------X: Normalization--------------'''


def normalizing(X_set):     # X_set: three dimensions
    '''
    plt.plot(X_set[1])
    plt.title('Original signal')
    plt.ylabel('mV')
    plt.show()
    '''

    X_set = utils.normalize(X_set, axis=1)

    '''
    xshow = X_set[1]
    plt.plot(xshow)
    plt.title('Normalized signal')
    plt.ylabel('mV')
    plt.show()
    '''

    return X_set


'''===========Call baseline removal and denoising============'''


def X_process(X_set):
    # Call baseline removal
    X_set = Baseline_removal(X_set)

    # Call denoising
    X_set = denoising(X_set)

    return X_set


'''
a = [[[1,2,3,2,1],[3,5,6,3,2]],[[10,20,30,20,10],[30,40,50,30,20]]]
b = np.array(a)
c = normalizing(b)
'''





